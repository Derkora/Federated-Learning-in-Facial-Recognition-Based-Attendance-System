import flwr as fl
import os
import io
import torch
import json
import math
import copy
import tempfile
import shutil
import numpy as np
import threading
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from app.db.db import SessionLocal
from app.db.models import FLRound, GlobalModel, UserGlobal, AttendanceRecap, Client
from app.utils.mobilefacenet import MobileFaceNet
from app.config import REGISTRY_PATH, ECONOMICS, TRAINING_PARAMS, FALLBACK_MODEL_PATH, CODECARBON_AVAILABLE
from app.utils.logging import init_logger, get_logger

if CODECARBON_AVAILABLE:
    from codecarbon import OfflineEmissionsTracker

def weighted_average(metrics: list) -> dict:
    logger = get_logger()
    logger.info(f"Mengagregasi hasil dari {len(metrics)} terminal...")
    examples = [m[0] for m in metrics]
    total_examples = sum(examples)
    if total_examples == 0: 
        logger.warn("Jumlah sampel data adalah 0, mengembalikan metrik kosong.")
        return {}

    try:
        aggregated = {
            "accuracy": sum([m[1].get("accuracy", 0.0) * m[0] for m in metrics]) / total_examples,
            "loss": sum([m[1].get("loss", 0.0) * m[0] for m in metrics]) / total_examples,
            "val_accuracy": sum([m[1].get("val_accuracy", 0.0) * m[0] for m in metrics]) / total_examples,
            "val_loss": sum([m[1].get("val_loss", 0.0) * m[0] for m in metrics]) / total_examples,
        }
        # Filter 0.0 values untuk log yang lebih bersih
        clean_metrics = {k: round(v, 4) for k, v in aggregated.items() if v != 0 or k == "accuracy"}
        logger.info(f"Hasil agregasi: {clean_metrics}")
        return aggregated
    except Exception as e:
        logger.error(f"Gagal mengagregasi metrik: {e}")
        return {}

# Strategi Penyimpanan Model Federated
# Bagian ini menangani penggabungan bobot model dari terminal-terminal (Aggregation)
# dan menyimpannya ke database global setelah setiap ronde selesai.
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, session_id: str, manager, target_version: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = session_id
        self.manager = manager
        self.target_version = target_version
        self.snapshots = [] # Untuk SWA/Snapshot Averaging
        self.logger = get_logger()

    def configure_fit(self, server_round: int, parameters, client_manager):
        self.round_start_time = datetime.now().timestamp()
        try:
            # Block until min_available_clients are connected to prevent the Round 1 selection race condition
            self.logger.info(f"Ronde {server_round}: Menunggu {self.min_available_clients} client terhubung ke gRPC server sebelum training...")
            client_manager.wait_for(num_clients=self.min_available_clients)
        except Exception as e:
            self.logger.error(f"Gagal menunggu client terhubung untuk training: {e}")
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round: int, parameters, client_manager):
        try:
            # Block until min_available_clients are connected to prevent evaluation sampling issues
            self.logger.info(f"Ronde {server_round}: Menunggu {self.min_available_clients} client terhubung ke gRPC server sebelum evaluasi...")
            client_manager.wait_for(num_clients=self.min_available_clients)
        except Exception as e:
            self.logger.error(f"Gagal menunggu client terhubung untuk evaluasi: {e}")
        return super().configure_evaluate(server_round, parameters, client_manager)

    # Agregasi Bobot Model Global (FedAvg)
    def aggregate_fit(self, server_round: int, results: list, failures: list):
        self.logger.info(f"Ronde {server_round} | Ringkasan Performa")
        if failures:
            self.logger.warn(f"Ronde {server_round} mendapati {len(failures)} kegagalan client.")

        # Laporan metrik performa masing-masing klien
        for i, (client_proxy, fit_res) in enumerate(results):
            cid = getattr(client_proxy, "cid", f"client-{i}")
            metrics = fit_res.metrics
            acc = metrics.get("accuracy", 0.0)
            loss = metrics.get("loss", 0.0)
            val_acc = metrics.get("val_accuracy", 0.0)
            self.logger.info(f"  > Client {cid}: Akurasi: {acc:.4f} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

        # Filter terminal client yang memiliki data latih valid (mencegah pembagian nol)
        valid_results = [(cp, fr) for cp, fr in results if fr.num_examples > 0]
        
        if not valid_results:
            self.logger.warn(f"Ronde {server_round} tidak memiliki data valid untuk diagregasi.")
            return None, {}

        # Agregasi parameter bobot model global dengan algoritma FedAvg terbobot
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, valid_results, failures)
        
        # Kumpulkan dan format metrik performa dari setiap klien
        clients_data = {}
        for i, (client_proxy, fit_res) in enumerate(valid_results):
            cid = fit_res.metrics.get("hostname") or getattr(client_proxy, "cid", f"client-{i}")
            clients_data[cid] = {
                "num_samples": fit_res.num_examples,
                "accuracy": fit_res.metrics.get("accuracy", 0.0),
                "loss": fit_res.metrics.get("loss", 0.0),
                "val_accuracy": fit_res.metrics.get("val_accuracy", 0.0),
                "val_loss": fit_res.metrics.get("val_loss", 0.0),
                "duration_s": fit_res.metrics.get("duration_s", 0.0),
                "epoch_history": fit_res.metrics.get("epoch_history", [])
            }
        
        if aggregated_metrics is None: aggregated_metrics = {}
        aggregated_metrics["clients"] = clients_data

        if aggregated_parameters is None:
            self.logger.warn(f"Agregasi Ronde {server_round} menghasilkan None (Tidak cukup data?).")
            return None, aggregated_metrics
            
        self.logger.success(f"Agregasi Ronde {server_round} Berhasil.")

        self.manager.current_round = server_round
        
        # Daftarkan ID klien unik baru ke dasbor server
        for i, (client_proxy, fit_res) in enumerate(results):
            cid = fit_res.metrics.get("hostname") or getattr(client_proxy, "cid", f"client-{i}")
            if cid not in self.manager.metrics["unique_client_ids"]:
                self.manager.metrics["unique_client_ids"].append(cid)
                self.logger.info(f"Klien baru terdeteksi: {cid}")

        # Lakukan penguraian bobot model federated yang ter-agregasi
        if aggregated_parameters is not None:
            if self.manager.tracker:
                try: self.manager.tracker.start()
                except: pass
            
            acc = aggregated_metrics.get('accuracy', 0)
            loss = aggregated_metrics.get('loss', 0)
            val_acc = aggregated_metrics.get('val_accuracy', 0)
            val_loss = aggregated_metrics.get('val_loss', 0)
            self.logger.success(f"Agregasi ronde {server_round} selesai | Akurasi: {acc:.4f} | Loss: {loss:.4f} | Val Akurasi: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

            # Koneksi ke basis data lokal server untuk penyimpanan
            try:
                db = SessionLocal()
                params_np = fl.common.parameters_to_ndarrays(aggregated_parameters)
                final_loss = aggregated_metrics.get("loss", 0.0)
                
                # Muat bobot model global sebelumnya sebagai basis (Preserve BN)
                global_model_instance = MobileFaceNet()
                global_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
                if global_model and global_model.weights:
                    sd = torch.load(io.BytesIO(global_model.weights), map_location="cpu")
                    self.logger.info("Menggunakan bobot model terakhir sebagai basis agregasi (Preserving BN).")
                else:
                    sd = global_model_instance.state_dict()
                    self.logger.info("Memulai agregasi dari model kosong (Seeding).")
                
                # Deteksi mode penyelarasan parameter (Full Backbone vs pFedFace)
                all_keys = list(sd.keys())
                conv_keys = [k for k in all_keys if not any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked'])]
                
                if len(params_np) == len(conv_keys):
                    self.logger.info(f"Ronde {server_round}: Mode pFedFace (Backbone saja) terdeteksi ({len(params_np)} parameter).")
                    target_keys = conv_keys
                elif len(params_np) == len(all_keys):
                    self.logger.info(f"Ronde {server_round}: Mode Sinkronisasi Penuh (Backbone + BN) terdeteksi ({len(params_np)} parameter).")
                    target_keys = all_keys
                elif len(params_np) > len(all_keys):
                    self.logger.info(f"Ronde {server_round}: Mode Sinkronisasi Penuh + Head terdeteksi ({len(params_np)} parameter).")
                    target_keys = all_keys
                else:
                    self.logger.warn(f"Panjang parameter tidak dikenal ({len(params_np)}). Mencoba pencocokan parsial...")
                    target_keys = all_keys[:len(params_np)]

                # Kelompokkan parameter menjadi Backbone vs statistik Batch Normalization (BN)
                bn_params = {}
                backbone_params = {}
                
                for i, k in enumerate(target_keys):
                    if i < len(params_np):
                        val = torch.from_numpy(params_np[i].copy())
                        if any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked']):
                            bn_params[k] = val
                        else:
                            backbone_params[k] = val
                
                # Perbarui state-dict model global dengan bobot backbone rata-rata terbaru
                sd.update(backbone_params)
                if bn_params:
                    self.logger.info(f"Ronde {server_round}: {len(bn_params)} parameter BN terdeteksi dan digabungkan.")
                    sd.update(bn_params)

                # Kelola bobot classifier head global jika disertakan
                num_target = len(target_keys)
                if len(params_np) > num_target:
                    head_params_np = params_np[num_target:]
                    self.logger.info(f"Ronde {server_round}: {len(head_params_np)} parameter Head terdeteksi.")
                    try:
                        head_sd = {"weight": torch.from_numpy(head_params_np[0].copy())}
                        tmp_head_path = "data/global_head.pth.tmp"
                        torch.save(head_sd, tmp_head_path)
                        os.replace(tmp_head_path, "data/global_head.pth")
                    except Exception as e:
                        self.logger.warn(f"Gagal menyimpan global_head secara atomik: {e}")

                # Rekam statistik konsumsi energi dan durasi ronde federated
                try:
                    if hasattr(self, 'round_start_time') and self.round_start_time > 0:
                        round_duration = round(datetime.now().timestamp() - self.round_start_time, 2)
                    else:
                        round_duration = round(datetime.now().timestamp() - self.manager.start_time, 2)
                    server_energy = 0
                    if self.manager.tracker and hasattr(self.manager.tracker, '_total_energy'):
                        server_energy = self.manager.tracker._total_energy.kWh
                    
                    aggregated_metrics["total_round_time_s"] = round_duration
                    aggregated_metrics["compute_energy_kwh"] = server_energy 
                    
                    backbone_size_mb = TRAINING_PARAMS.get("model_size_mb", 15.0)
                    num_clients = len(clients_data)
                    aggregated_metrics["backbone_sync_mb"] = num_clients * backbone_size_mb * 2
                    aggregated_metrics["registry_sync_mb"] = num_clients * 0.05
                    
                    new_round = FLRound(
                        session_id=self.session_id,
                        round_number=server_round,
                        loss=final_loss,
                        metrics=json.dumps(aggregated_metrics)
                    )
                    db.add(new_round)
                    
                    # Simpan bobot model ter-agregasi terbaru ke tabel GlobalModel
                    buf = io.BytesIO()
                    torch.save(sd, buf)
                    weights_bytes = buf.getvalue()
                    
                    global_model = db.query(GlobalModel).first()
                    if not global_model:
                        global_model = GlobalModel(version=self.manager.model_version, weights=weights_bytes)
                        db.add(global_model)
                    else:
                        global_model.weights = weights_bytes
                        global_model.version = self.manager.model_version
                        global_model.last_updated = datetime.now(timezone(timedelta(hours=7)))
                    
                    db.commit()
                    
                    # Sinkronkan data ekonomi dan waktu aktif ke manager dasbor
                    client_stats = aggregated_metrics.get("clients", {})
                    self.manager.record_round_data(server_round, aggregated_metrics, client_stats, db=db)
                    self.manager.metrics["total_round_time_s"] = round(datetime.now().timestamp() - self.manager.start_time, 2)
                    
                    if self.manager.tracker:
                        try:
                            self.manager.tracker.stop()
                            if hasattr(self.manager.tracker, '_total_energy'):
                                self.manager.metrics["server_energy_kwh"] = self.manager.tracker._total_energy.kWh
                        except: pass

                    self.manager.update_logs(f"SERVER LOG: Ronde {server_round} selesai. Akurasi: {aggregated_metrics.get('accuracy', 0):.4f} | Loss: {aggregated_metrics.get('loss', 0):.4f}")
                    self.logger.info(f"Ronde {server_round} tercatat di Dashboard.")

                except Exception as e:
                    self.logger.error(f"Gagal menyimpan hasil agregasi ke database: {e}")
                    db.rollback()
                finally:
                    db.close()
                    
                # Tulis berkas fisik backup backbone.pth secara atomik dan aman
                try:
                    os.makedirs("data", exist_ok=True)
                    tmp_pure_path = "data/backbone_pure.pth.tmp"
                    torch.save(sd, tmp_pure_path)
                    os.replace(tmp_pure_path, "data/backbone_pure.pth")
                    
                    bn_path = "data/global_bn_combined.pth"
                    if os.path.exists(bn_path):
                        try:
                            bn_params = torch.load(bn_path, map_location="cpu")
                            sd.update(bn_params)
                            self.logger.info("Backbone global kini diperkuat dengan statistik BN.")
                        except: pass
                        
                    with tempfile.NamedTemporaryFile(delete=False, dir="data") as tmp:
                        torch.save(sd, tmp.name)
                        shutil.move(tmp.name, "data/backbone.pth")
                except Exception as e:
                    self.logger.error(f"Gagal menyimpan file murni/BN: {e}")

                except Exception as e:
                    self.logger.error(f"Gagal menyimpan ronde {server_round} ke DB: {e}")
                    db.rollback()
                finally:
                    if results:
                        client_data = {}
                        for i, (client_proxy, fit_res) in enumerate(results):
                            cid = fit_res.metrics.get("hostname")
                            proxy_id = getattr(client_proxy, "cid", "")
                            
                            if not cid:
                                cid = proxy_id or f"client-{i}"
                                self.manager.logger.warn(f"Menggunakan Proxy ID {cid} karena hostname kosong.")
                            
                            m = fit_res.metrics.copy()
                            m["num_samples"] = fit_res.num_examples
                            eh = fit_res.metrics.get("epoch_history", "[]")
                            if isinstance(eh, str):
                                try: m["epoch_history"] = json.loads(eh)
                                except: m["epoch_history"] = []
                            else: m["epoch_history"] = eh
                            client_data[cid] = m
                        
                        self.manager.logger.info(f"Memanggil record_round_data untuk ronde {server_round}...")
                        self.manager.record_round_data(server_round, aggregated_metrics or {}, client_data, db=db)
                    
                    db.close()
            except Exception as e:
                self.manager.logger.error(f"Gagal memproses parameter agregasi: {e}")
        
        if aggregated_metrics:
            self.manager.update_metrics(aggregated_metrics)
            loss = aggregated_metrics.get('loss', 0.0)
            self.manager.update_logs(f"Ronde {server_round} selesai. Acc: {aggregated_metrics.get('accuracy', 0):.4f} | Loss: {loss:.4f}")


        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, results: list, failures: list):
        if not results:
            return None, {}
        valid_results = [(cp, er) for cp, er in results if er.num_examples > 0]
        if not valid_results:
            return None, {}
            
        loss, metrics = super().aggregate_evaluate(server_round, valid_results, failures)
        return loss, metrics

class FLServerManager:
    # Manajer Sesi dan Status Server Federated (FL)
    # Menyimpan informasi status pelatihan, metrik performa, dan log aktivitas.
    
    def __init__(self):
        self.lock = threading.RLock()
        self.session_id = None
        self.is_running = False
        self.current_phase = "idle"
        self.start_time = 0
        self.current_round = 0
        self.model_version = 0
        self.tracker = None
        
        # Inisialisasi Logger Terpusat (Global)
        self.log_path = "/app/data/server_training.log"
        init_logger(self.log_path, max_memory_logs=10000, tag="FL-SERVER")
        self.logger = get_logger()
        
        if CODECARBON_AVAILABLE:
            try:
                self.tracker = OfflineEmissionsTracker(
                    project_name="Federated_Learning_Server",
                    output_dir="/app/data/",
                    country_iso_code="IDN"
                )
            except: pass
        self.default_rounds = 10
        self.default_epochs = 1
        self.default_min_clients = 2
        self.default_batch_size = 32
        self.default_lr = 0.05
        self.default_mu = 0.05
        self.default_lambda = 0.1
        self.registered_clients = {}
        self.registry_submissions = {}
        self.ready_clients = set() 
        self.received_data = []
        self.discovery_clients = set()
        self.inference_threshold = 0.7
        self.metrics = {
            "accuracy": 0, "loss": 0, 
            "backbone_sync_mb": 0, "registry_sync_mb": 0,
            "transmission_cost_idr": 0,
            "training_duration_s": 0, "total_round_time_s": 0,
            "compute_energy_kwh": 0, "compute_cost_idr": 0,
            "round_history": [], # Riwayat data ronde dengan rincian client
            "unique_client_ids": [], # Pelacakan ID unik client yang berkontribusi
            "inference_logs": [] # Log detail untuk riset FAR/TAR
        }
        
        self.update_logs("=== Server Started / Restarted ===")
        
        self.settings_path = "/app/data/settings.json"
        self.inference_logs_path = "/app/data/inference_logs.json"
        self.load_settings()
        self._load_persistence()

    @property
    def current_logs(self):
        # Kompatibilitas dengan Dashboard UI agar tetap bisa membaca list log
        return self.logger.get_logs()

    @current_logs.setter
    def current_logs(self, value):
        # Memungkinkan pembersihan log (misal: self.current_logs = [])
        if isinstance(value, list) and len(value) == 0:
            self.logger.clear_logs()


    def update_logs(self, msg):
        """Menambahkan log baru ke memori dan file persisten (Wrapper ke Logger)."""
        if "[ERROR]" in msg:
            self.logger.error(msg.replace("[ERROR] ", ""))
        elif "[SUCCESS]" in msg:
            self.logger.success(msg.replace("[SUCCESS] ", ""))
        elif "[OK]" in msg:
            self.logger.success(msg.replace("[OK] ", ""))
        elif "[WARNING]" in msg:
            self.logger.warn(msg.replace("[WARNING] ", ""))
        else:
            self.logger.info(msg)

    def load_settings(self):
        """Memuat pengaturan server dari file JSON."""
        with self.lock:
            if os.path.exists(self.settings_path):
                try:
                    with open(self.settings_path, "r") as f:
                        settings = json.load(f)
                        self.inference_threshold = 0.7
                        self.default_batch_size = settings.get("batch_size", 32)
                        self.default_rounds = settings.get("rounds", 10)
                        self.default_epochs = settings.get("epochs", 1)
                        self.default_min_clients = settings.get("min_clients", 2)
                    self.logger.info(f"Pengaturan dimuat dari {self.settings_path} (Threshold dipaksa ke 0.7)")
                    self.save_settings()
                except Exception as e:
                    self.logger.error(f"Gagal memuat pengaturan: {e}")

    def save_inference_logs(self):
        """Menyimpan log inferensi ke file JSON agar persisten."""
        with self.lock:
            try:
                logs = self.metrics.get("inference_logs", [])
                with open(self.inference_logs_path, "w") as f:
                    json.dump(logs, f, indent=4)
            except Exception as e:
                self.logger.error(f"Gagal menyimpan log inferensi: {e}")

    def load_inference_logs(self):
        """Memuat log inferensi dari file JSON saat startup."""
        with self.lock:
            if os.path.exists(self.inference_logs_path):
                try:
                    with open(self.inference_logs_path, "r") as f:
                        self.metrics["inference_logs"] = json.load(f)
                    self.logger.info(f"Berhasil memulihkan {len(self.metrics['inference_logs'])} log inferensi.")
                except Exception as e:
                    self.logger.error(f"Gagal memuat log inferensi: {e}")

    def save_settings(self):
        """Menyimpan pengaturan server ke file JSON."""
        with self.lock:
            try:
                settings = {
                    "inference_threshold": self.inference_threshold,
                    "batch_size": getattr(self, 'default_batch_size', 32),
                    "rounds": getattr(self, 'default_rounds', 10),
                    "epochs": getattr(self, 'default_epochs', 1),
                    "min_clients": getattr(self, 'default_min_clients', 2)
                }
                with open(self.settings_path, "w") as f:
                    json.dump(settings, f, indent=4)
            except Exception as e:
                self.logger.error(f"Gagal menyimpan pengaturan: {e}")

    def _load_log_from_file(self):
        # Log sudah ditangani oleh class Logger
        pass

    def _load_persistence(self):
        """Memuat ulang status dari database dengan logika retry dan logging yang sangat detail."""
        with self.lock:
            self.logger.info("Memulai pemulihan status server Federated...")
            
            max_retries = 10
            retry_delay = 5
            
            for i in range(max_retries):
                db = SessionLocal()
                try:
                    # Muat Versi Model Terbaru
                    latest_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
                    if latest_model:
                        self.model_version = latest_model.version
                        self.logger.info(f"Database terhubung. Versi Model saat ini: v{self.model_version}")

                    # Muat riwayat ronde (Hanya sesi terbaru agar tidak duplikat)
                    latest_round = db.query(FLRound).order_by(FLRound.timestamp.desc()).first()
                    if latest_round:
                        latest_session_id = latest_round.session_id
                        rounds = db.query(FLRound).filter_by(session_id=latest_session_id).order_by(FLRound.round_number.asc()).all()
                        self.logger.info(f"Menemukan {len(rounds)} ronde federated dari sesi {latest_session_id}.")
                    else:
                        rounds = []
                        self.logger.info("Tidak ada riwayat ronde di database.")
                    
                    if rounds:
                        history = []
                        total_energy = 0
                        total_duration = 0
                        
                        for r in rounds:
                            try:
                                m = json.loads(r.metrics) if r.metrics else {}
                                clients = m.get("clients", {})
                                # Ensure epoch_history in each client is parsed if it's a string
                                for cid, c_data in clients.items():
                                    if "epoch_history" in c_data and isinstance(c_data["epoch_history"], str):
                                        try:
                                            c_data["epoch_history"] = json.loads(c_data["epoch_history"])
                                        except:
                                            pass

                                history.append({
                                    "round": r.round_number if r.round_number is not None else 0,
                                    "timestamp": r.timestamp.strftime("%H:%M:%S") if r.timestamp else "-",
                                    "server": m,
                                    "clients": clients
                                })
                                # Update global metrics
                                self.metrics["accuracy"] = float(m.get("accuracy", 0))
                                self.metrics["loss"] = float(m.get("loss", 0))
                                
                                # Agregasi Ekonomi (NEW)
                                s_energy = m.get("compute_energy_kwh", 0)
                                if s_energy > total_energy:
                                    total_energy = s_energy
                                total_duration += m.get("total_round_time_s", 0)
                            except Exception as e:
                                self.logger.warn(f"Gagal mengurai data ronde {r.round_number}: {e}")

                        self.metrics["round_history"] = history
                        self.metrics["compute_energy_kwh"] = total_energy
                        self.metrics["total_round_time_s"] = total_duration
                        
                        # Pemulihan data penggunaan bandwidth
                        total_backbone = 0
                        total_registry = 0
                        for r in history:
                            s = r.get("server", {})
                            total_backbone += s.get("backbone_sync_mb", 0)
                            total_registry += s.get("registry_sync_mb", 0)
                        
                        self.metrics["backbone_sync_mb"] = total_backbone
                        self.metrics["registry_sync_mb"] = total_registry
                        
                        # Pemulihan client id unik dari riwayat
                        unique_ids = set()
                        for r in history:
                            if "clients" in r:
                                unique_ids.update(r["clients"].keys())
                        self.metrics["unique_client_ids"] = list(unique_ids)
                        
                        # Update estimasi biaya/transmisi berdasarkan history yang dimuat
                        self.update_metrics({})
                        self.logger.info(f"Berhasil memulihkan {len(history)} ronde ke dashboard FL.")
                        self.logger.info(f"Ditemukan {len(unique_ids)} client aktif dari riwayat: {list(unique_ids)}")
                    
                    # Muat log inferensi dari file
                    self.load_inference_logs()
                    
                    db.close()
                    return # Berhasil
                except Exception as e:
                    self.logger.warn(f"Percobaan {i+1}/{max_retries} gagal: {e}")
                    db.close()
                    if i < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        self.logger.error("Gagal total memulihkan status FL. Dashboard mungkin akan kosong.")

    def increment_version(self):
        with self.lock:
            self.model_version += 1
            
            # Persistensi ke Database
            db = SessionLocal()
            try:
                global_model = db.query(GlobalModel).first()
                if global_model:
                    global_model.version = self.model_version
                    global_model.last_updated = datetime.utcnow()
                    db.commit()
                    self.logger.success(f"Database GlobalModel diperbarui ke v{self.model_version}")
            except Exception as e:
                self.logger.error(f"Gagal menyimpan kenaikan versi model: {e}")
                db.rollback()
            finally:
                db.close()

            self.update_logs(f"Versi Model Global naik ke v{self.model_version}")
            self.discovery_clients = set() 
            self.received_data = [] 

    def start_phase(self, phase_name):
        with self.lock:
            self.is_running = True
            self.current_phase = phase_name.lower().replace(" ", "_")
            if self.current_phase == "data_prep": self.start_time = datetime.now().timestamp()
            self.update_logs(f"Fase {phase_name} dimulai.")

    def end_phase(self, phase_name=None):
        with self.lock:
            if phase_name: self.update_logs(f"Fase {phase_name} selesai.")
            self.is_running = False
            self.current_phase = "idle"

    def update_metrics(self, new_data):
        # Memperbarui metrik performa dan estimasi biaya
        # Penggabungan riwayat ronde tanpa menimpa data yang lama
        with self.lock:
            if "round_history" in new_data:
                new_history = new_data.pop("round_history")
                existing_rounds = [h["round"] for h in self.metrics["round_history"]]
                for h in new_history:
                    if h["round"] not in existing_rounds:
                        self.metrics["round_history"].append(h)

            self.metrics.update(new_data)
            self.update_economics({})

    # Perhitungan Nilai Ekonomi
    def update_economics(self, new_data):
        # Menghitung lalu lintas data sinkronisasi backbone dan registry serta mengalikan dengan tarif kuota internet
        current_rounds = max(self.current_round, len(self.metrics.get("round_history", [])))
        num_clients = len(self.metrics.get("unique_client_ids", []))
        
        # Cek Ukuran File Asli
        bb_size = ECONOMICS["estimated_backbone_size_mb"]
        reg_size = ECONOMICS["estimated_registry_size_mb"]
        
        pure_bb_path = "data/backbone_pure.pth"
        if os.path.exists(pure_bb_path):
            bb_size = os.path.getsize(pure_bb_path) / (1024 * 1024)
        
        if os.path.exists(REGISTRY_PATH):
            reg_size = os.path.getsize(REGISTRY_PATH) / (1024 * 1024)

        self.metrics["backbone_sync_mb"] = round(current_rounds * num_clients * 2 * bb_size, 2)
        self.metrics["registry_sync_mb"] = round(num_clients * reg_size, 2)
        
        total_mb = self.metrics["backbone_sync_mb"] + self.metrics["registry_sync_mb"]
        self.metrics["transmission_cost_idr"] = round(total_mb * ECONOMICS["transmission_cost_per_mb"], 2)
        
        # Mengalkulasi total konsumsi daya komputasi dari server dan seluruh klien lalu dikonversi ke satuan rupiah (IDR)
        total_duration_s = 0
        max_server_energy = 0.0
        has_real_server_energy = False
        
        for r in self.metrics.get("round_history", []):
            s_metrics = r.get("server", {})
            total_duration_s += s_metrics.get("total_round_time_s", 0)
            s_energy = s_metrics.get("compute_energy_kwh", 0)
            if s_energy > 0:
                if s_energy > max_server_energy:
                    max_server_energy = s_energy
                has_real_server_energy = True
                
        if total_duration_s > 0:
            self.metrics["total_round_time_s"] = round(total_duration_s, 2)
            duration_s = total_duration_s
        else:
            duration_s = self.metrics.get("total_round_time_s", 0)
            if duration_s == 0:
                duration_s = current_rounds * 180
                self.metrics["total_round_time_s"] = duration_s
                
        duration_h = duration_s / 3600

        server_p = ECONOMICS["estimated_server_power_kw"]
        client_p = ECONOMICS["estimated_client_power_kw"]

        # Energi Server murni dari semua ronde jika tersedia, jika tidak menggunakan estimasi
        if has_real_server_energy:
            server_energy = max_server_energy
            self.metrics["server_energy_kwh"] = round(server_energy, 6)
        else:
            server_energy = self.metrics.get("server_energy_kwh", duration_h * server_p)
        
        # Energi Klien dari hasil agregasi data real yang dilaporkan
        client_energy_total = 0.0
        has_real_client_energy = False
        
        for r in self.metrics.get("round_history", []):
            for cid, cMetrics in r.get("clients", {}).items():
                if "energy_kwh" in cMetrics and cMetrics["energy_kwh"] > 0:
                    client_energy_total += cMetrics["energy_kwh"]
                    has_real_client_energy = True
        
        # Jika belum ada data real dari client, gunakan estimasi berbasis durasi
        if not has_real_client_energy:
            client_energy_total = duration_h * (num_clients * client_p)
            
        total_energy = server_energy + client_energy_total
        
        self.metrics["compute_energy_kwh"] = round(total_energy, 6)
        
        cost_per_kwh = ECONOMICS["compute_cost_per_kwh"]
        self.metrics["compute_cost_idr"] = round(total_energy * cost_per_kwh, 2)

    # Evaluasi & Konsolidasi Metrik Pelatihan (Akurasi & Loss)
    def record_round_data(self, round_num, server_metrics, client_metrics, db=None):
        with self.lock:
            # Mencegah redundansi jika fungsi ini dipanggil ulang untuk ronde yang sama
            existing_rounds = [h["round"] for h in self.metrics["round_history"]]
            if round_num in existing_rounds:
                self.logger.info(f"Ronde {round_num} sudah ada di history, memperbarui data...")
                for h in self.metrics["round_history"]:
                    if h["round"] == round_num:
                        h["server"] = server_metrics
                        h["clients"] = client_metrics
                return

            self.logger.info(f"Mencatat data ronde {round_num} dengan {len(client_metrics)} klien...")
            entry = {
                "round": round_num,
                "timestamp": datetime.now(timezone(timedelta(hours=7))).strftime("%H:%M:%S"),
                "server": server_metrics,
                "clients": client_metrics
            }
            self.metrics["round_history"].append(entry)
            
            # Update unique_client_ids dari rekaman data aktual
            new_cid = None
            for cid in client_metrics.keys():
                is_uuid = len(cid) >= 32 and cid.isalnum()
                if not is_uuid:
                    new_cid = cid
                
                if cid not in self.metrics["unique_client_ids"]:
                    self.metrics["unique_client_ids"].append(cid)
                    self.logger.info(f"Klien {cid} berhasil ditambahkan ke daftar unik klien")

            # Pembersihan berkala: Hanya simpan ID yang beneran ada di round_history
            all_active_ids = set()
            for r in self.metrics["round_history"]:
                all_active_ids.update(r["clients"].keys())
            
            # Sinkronkan unique_client_ids agar tidak ada ID sampah
            self.metrics["unique_client_ids"] = [cid for cid in self.metrics["unique_client_ids"] if cid in all_active_ids]
            
            # Sinkronkan data ekonomi
            for cid in client_metrics.keys():
                if cid not in self.metrics["unique_client_ids"]:
                    self.metrics["unique_client_ids"].append(cid)

    def _get_lr_for_round(self, server_round):
        # LR Schedule (Cosine vs Step)
        schedule_type = TRAINING_PARAMS.get("lr_schedule", "step")
        
        if schedule_type == "cosine":
            initial_lr = TRAINING_PARAMS.get("initial_lr", 0.1)
            min_lr = TRAINING_PARAMS.get("min_lr", 1e-4)
            total_rounds = self.default_rounds # Menggunakan settings saat ini
            
            # Cosine Annealing: min + 0.5*(initial-min)*(1 + cos(pi * round / total))
            # Round di-offset 1 agar ronde 1 dimulai dari initial_lr
            lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * (server_round-1) / total_rounds))
            return lr
        else:
            # Fallback ke Step-LR (Original logic)
            schedule = {1: 0.1, 5: 0.01, 8: 0.001} # Default jika tidak ada di config
            if isinstance(TRAINING_PARAMS.get("lr_schedule"), dict):
                schedule = TRAINING_PARAMS["lr_schedule"]
            
            lr = 0.1
            for threshold in sorted(schedule.keys()):
                if server_round >= threshold:
                    lr = schedule[threshold]
            return lr

    def get_status(self, db=None):
        # self.logger.info(f"get_status called. round_history len: {len(self.metrics.get('round_history', []))}")
        with self.lock:
            attendance_count = 0
            active_clients = []
            if db:
                attendance_count = db.query(AttendanceRecap).count()
                clients = db.query(Client).all()
                
                now_wib = datetime.now(timezone(timedelta(hours=7)))
                db_needs_commit = False
                
                for c in clients:
                    status = "offline"
                    wib_time_str = "-"
                    
                    if c.last_seen:
                        # Konversi waktu database naive (disimpan dalam UTC) ke WIB
                        dt = c.last_seen
                        if dt.tzinfo is None:
                            wib_dt = dt.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=7)))
                        else:
                            wib_dt = dt.astimezone(timezone(timedelta(hours=7)))
                        wib_time_str = wib_dt.strftime("%Y-%m-%d %H:%M:%S WIB")
                        
                        # Jika detak jantung (heartbeat) dikirim dalam 20 detik terakhir, terminal dinyatakan online
                        if (now_wib - wib_dt).total_seconds() <= 20:
                            status = "online"
                    
                    # Perbarui status di database jika terjadi perubahan
                    if c.status != status:
                        c.status = status
                        db.add(c)
                        db_needs_commit = True
                    
                    active_clients.append({
                        "id": c.edge_id,
                        "ip": c.ip_address,
                        "status": status.upper(),
                        "last_seen": wib_time_str
                    })
                    
                if db_needs_commit:
                    try:
                        db.commit()
                    except Exception as e:
                        db.rollback()
                        self.logger.error(f"Gagal commit status client terupdate: {e}")
                
            return {
                "is_running": self.is_running,
                "phase": self.current_phase,
                "session_id": self.session_id,
                "metrics": self.metrics,
                "current_logs": self.current_logs,
                "received_data": self.received_data,
                "model_version": self.model_version,
                "default_rounds": self.default_rounds,
                "default_epochs": self.default_epochs,
                "default_min_clients": self.default_min_clients,
                "inference_threshold": self.inference_threshold,
                "attendance_count": attendance_count,
                "uptime": int(datetime.now().timestamp() - self.start_time) if self.start_time > 0 else 0,
                "active_clients": active_clients,
                "inference_logs": self.metrics.get("inference_logs", [])
            }

    def ensure_model_seeded(self, db):
        # Memastikan tabel GlobalModel memiliki bobot awal (Seeding)
        latest_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
        if latest_model and latest_model.weights: return
            
        fallback_path = FALLBACK_MODEL_PATH
        if os.path.exists(fallback_path):
            self.logger.info(f"Memulai seeding GlobalModel dari {fallback_path}...")
            try:
                model = MobileFaceNet()
                loaded = torch.load(fallback_path, map_location="cpu")
                match_count = 0
                # Seeding harus konsisten dengan get_parameters (Full Sync untuk Parity)
                current_sd = model.state_dict()
                # Sertakan SEMUA parameter backbone (Weight, Bias, BN, Running Stats)
                shared_keys = [k for k in current_sd.keys() if 
                               any(x in k.lower() for x in ['weight', 'bias', 'bn', 'running_', 'num_batches_tracked'])]
                
                # Pastikan yang disimpan ke DB adalah STATE_DICT, bukan list
                final_sd = {}
                if isinstance(loaded, dict):
                    for key in shared_keys:
                        if key in loaded:
                            final_sd[key] = loaded[key]
                            match_count += 1
                        else:
                            final_sd[key] = current_sd[key]
                    self.logger.success(f"GlobalModel seeded dengan state_dict ({match_count} weights cocok).")
                else:
                    final_sd = current_sd
                    self.logger.success("GlobalModel seeded dengan default state_dict.")
                
                buf = io.BytesIO()
                torch.save(final_sd, buf)
                new_model = GlobalModel(version=0, weights=buf.getvalue())
                db.add(new_model)
                db.commit()
            except Exception as e:
                self.logger.error(f"Gagal inisialisasi GlobalModel: {e}")
                db.rollback()
                
    def get_label_map_from_db(self):
        # Mengambil daftar ID mahasiswa global sebagai acuan index untuk pelatihan
        db = SessionLocal()
        try:
            users = db.query(UserGlobal).order_by(UserGlobal.nrp).all()
            return [u.nrp for u in users]
        finally:
            db.close()

    def start_training(self, session_id: str, rounds: int = 10, min_clients: int = 2):
        with self.lock:
            self.session_id = session_id
            self.is_running = True
            self.default_rounds = rounds  # Track ronde untuk triggering increment versi
            self.start_phase("Training")
            self.update_logs(f"Pelatihan Federated dimulai: {rounds} ronde, {min_clients} client.")
            
            initial_parameters = None
            db = SessionLocal()
            try:
                self.ensure_model_seeded(db)
                latest_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
                if latest_model and latest_model.weights:
                    loaded = torch.load(io.BytesIO(latest_model.weights), map_location="cpu")
                    
                    # Konversi state_dict kembali ke ndarrays jika perlu
                    if isinstance(loaded, dict):
                        # Ekstrak params conv untuk Flower fit configuration
                        sd = loaded
                        # Sertakan BN agar model antar-client identik (Full Parity)
                        shared_keys = [k for k in sd.keys() if 
                                       any(x in k.lower() for x in ['weight', 'bias', 'bn', 'running_', 'num_batches_tracked'])]
                        weights_np = [sd[k].cpu().numpy() for k in shared_keys]
                    else:
                        weights_np = loaded
                    
                    initial_parameters = fl.common.ndarrays_to_parameters(weights_np)
            except Exception as e:
                self.logger.error(f"Gagal memuat parameter awal: {e}")
            finally:
                db.close()

            # Tentukan versi target untuk sesi ini (Stabilitas Versi: v15 -> v1)
            target_version = self.model_version + 1

            # Strategi pelatihan FedAvg dengan konfigurasi dinamis (mu=0.05 untuk stabilitas)
            strategy = SaveModelStrategy(
                session_id=session_id,
                manager=self,
                target_version=target_version,
                initial_parameters=initial_parameters,
                fraction_fit=1.0,
                min_fit_clients=max(1, min_clients // 2),
                min_available_clients=min_clients,
                min_evaluate_clients=max(1, min_clients // 2),
                fit_metrics_aggregation_fn=weighted_average,
                evaluate_metrics_aggregation_fn=weighted_average,
                on_fit_config_fn=lambda server_round: {
                    "round": server_round,
                    "total_rounds": rounds,
                    "local_epochs": self.default_epochs,
                    "lr": self._get_lr_for_round(server_round),
                    "mu": 0.05, 
                    "lambda": self.default_lambda,
                    "label_map": json.dumps(self.get_label_map_from_db())
                },
            )

        try:
            fl.server.start_server(
                server_address="0.0.0.0:8085",
                config=fl.server.ServerConfig(num_rounds=rounds),
                strategy=strategy,
            )
            
            with self.lock:
                self.update_logs(f"Pelatihan Federated ronde terakhir selesai.")
        finally:
            with self.lock:
                self.is_running = False
                self.end_phase("Training")


# Instansi Global
fl_manager = FLServerManager()
