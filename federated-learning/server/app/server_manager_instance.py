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
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from app.db.db import SessionLocal
from app.db.models import FLRound, GlobalModel, UserGlobal, AttendanceRecap, Client
from app.utils.mobilefacenet import MobileFaceNet
from app.config import ECONOMICS, TRAINING_PARAMS, FALLBACK_MODEL_PATH, CODECARBON_AVAILABLE
from app.utils.logging import init_logger, get_logger

if CODECARBON_AVAILABLE:
    from codecarbon import OfflineEmissionsTracker

# Fungsi rata-rata tertimbang untuk metrik
def weighted_average(metrics: list) -> dict:
    # Fungsi agregasi metrik dari seluruh terminal (Client)
    logger = get_logger()
    logger.info(f"Mengagregasi hasil dari {len(metrics)} terminal...")
    examples = [m[0] for m in metrics]
    total_examples = sum(examples)
    if total_examples == 0: 
        logger.warn("Total examples is 0, returning empty metrics.")
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


    def aggregate_fit(self, server_round: int, results: list, failures: list):
        self.logger.info(f"Ronde {server_round} | Ringkasan Performa")
        if failures:
            self.logger.warn(f"Ronde {server_round} mendapati {len(failures)} kegagalan client.")

        # Laporan masing-masing terminal
        for i, (client_proxy, fit_res) in enumerate(results):
            cid = getattr(client_proxy, "cid", f"client-{i}")
            metrics = fit_res.metrics
            acc = metrics.get("accuracy", 0.0)
            loss = metrics.get("loss", 0.0)
            val_acc = metrics.get("val_accuracy", 0.0)
            self.logger.info(f"  > Client {cid}: Akurasi: {acc:.4f} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

        # Filter client yang tidak punya data (mencegah division by zero di FedAvg)
        valid_results = [(cp, fr) for cp, fr in results if fr.num_examples > 0]
        
        if not valid_results:
            self.logger.warn(f"Ronde {server_round} tidak memiliki data valid untuk diagregasi.")
            return None, {}

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, valid_results, failures)
        
        # MANUALLY CAPTURE CLIENT METRICS (Because weighted_average only returns global averages)
        clients_data = {}
        for i, (client_proxy, fit_res) in enumerate(valid_results):
            cid = fit_res.metrics.get("hostname") or getattr(client_proxy, "cid", f"client-{i}")
            clients_data[cid] = {
                "num_samples": fit_res.num_examples,
                "accuracy": fit_res.metrics.get("accuracy", 0.0),
                "loss": fit_res.metrics.get("loss", 0.0),
                "val_accuracy": fit_res.metrics.get("val_accuracy", 0.0),
                "epoch_history": fit_res.metrics.get("epoch_history", [])
            }
        
        # Attach to aggregated_metrics so record_round_data can see it
        if aggregated_metrics is None: aggregated_metrics = {}
        aggregated_metrics["clients"] = clients_data

        if aggregated_parameters is None:
            self.logger.warn(f"Agregasi Ronde {server_round} menghasilkan None (Tidak cukup data?).")
            return None, aggregated_metrics
            
        self.logger.success(f"Agregasi Ronde {server_round} Berhasil.")


        self.manager.current_round = server_round
        
        # Populate unique_client_ids early to ensure client tables show up in UI
        for i, (client_proxy, fit_res) in enumerate(results):
            cid = fit_res.metrics.get("hostname") or getattr(client_proxy, "cid", f"client-{i}")
            if cid not in self.manager.metrics["unique_client_ids"]:
                self.manager.metrics["unique_client_ids"].append(cid)
                self.logger.info(f"New client detected: {cid}")

        if aggregated_parameters is not None:
            if self.manager.tracker:
                try: self.manager.tracker.start()
                except: pass
            
            acc = aggregated_metrics.get('accuracy', 0)
            loss = aggregated_metrics.get('loss', 0)
            val_acc = aggregated_metrics.get('val_accuracy', 0)
            val_loss = aggregated_metrics.get('val_loss', 0)
            self.logger.success(f"Agregasi ronde {server_round} selesai | Acc: {acc:.4f} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

            try:
                db = SessionLocal()
                params_np = fl.common.parameters_to_ndarrays(aggregated_parameters)
                final_loss = aggregated_metrics.get("loss", 0.0)
                
                # 1. KONVERSI KE STATE_DICT (Critical for Identification Consistency)
                # Gunakan model saat ini sebagai basis agar BN stats tidak terhapus (Preserve BN)
                global_model_instance = MobileFaceNet()
                global_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
                if global_model and global_model.weights:
                    sd = torch.load(io.BytesIO(global_model.weights), map_location="cpu")
                    self.logger.info("Menggunakan bobot model terakhir sebagai basis agregasi (Preserving BN).")
                else:
                    sd = global_model_instance.state_dict()
                    self.logger.info("Memulai agregasi dari model kosong (Seeding).")
                
                # Cek mode berdasarkan jumlah parameter
                # Full Backbone + BN (Parity) biasanya 173 keys (incl. running stats)
                # Backbone Only (pFedFace Legacy) biasanya ~60 keys
                all_keys = list(sd.keys())
                conv_keys = [k for k in all_keys if not any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked'])]
                
                if len(params_np) == len(conv_keys):
                    self.logger.info(f"Ronde {server_round}: Mode pFedFace (Backbone Only) terdeteksi ({len(params_np)} params).")
                    target_keys = conv_keys
                elif len(params_np) == len(all_keys):
                    self.logger.info(f"Ronde {server_round}: Mode Full Sync (Backbone + BN) terdeteksi ({len(params_np)} params).")
                    target_keys = all_keys
                elif len(params_np) > len(all_keys):
                    self.logger.info(f"Ronde {server_round}: Mode Full Sync + Head detected ({len(params_np)} params).")
                    target_keys = all_keys
                else:
                    self.logger.warn(f"Panjang parameter tidak dikenal ({len(params_np)}). Mencoba pencocokan parsial...")
                    target_keys = all_keys[:len(params_np)]

                # 2. IDENTIFIKASI PARAMETER (Orderly Filtering)
                bn_params = {}
                backbone_params = {}
                
                # Filter pFedFace Style: Pisahkan Backbone (Conv) dan BN
                for i, k in enumerate(target_keys):
                    if i < len(params_np):
                        val = torch.from_numpy(params_np[i].copy())
                        if any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked']):
                            bn_params[k] = val
                        else:
                            backbone_params[k] = val
                
                # Pasang ke model utama (Hanya yang disetujui untuk sinkronisasi)
                # Dalam pFedFace, bn_params biasanya kosong di sini karena tidak dikirim via Flower
                sd.update(backbone_params)
                if bn_params:
                    self.logger.info(f"Ronde {server_round}: {len(bn_params)} parameter BN terdeteksi dan digabungkan.")
                    sd.update(bn_params)

                # 3. HANDLE CLASSIFIER HEAD (Parameter tambahan di akhir payload)
                num_target = len(target_keys)
                if len(params_np) > num_target:
                    head_params_np = params_np[num_target:]
                    self.logger.info(f"Ronde {server_round}: {len(head_params_np)} parameter Head terdeteksi.")
                    try:
                        # Bungkus Head dalam state_dict teratur
                        head_sd = {"weight": torch.from_numpy(head_params_np[0].copy())}
                        torch.save(head_sd, "data/global_head.pth")
                    except Exception as e:
                        self.logger.warn(f"Gagal menyimpan global_head: {e}")

                # Simpan hasil ke database (db sudah diinisialisasi di awal try)
                try:
                    new_round = FLRound(
                        session_id=self.session_id,
                        round_number=server_round,
                        loss=final_loss,
                        metrics=json.dumps(aggregated_metrics)
                    )
                    db.add(new_round)
                    
                    # Simpan sebagai state_dict (Bukan list) agar API client konsisten
                    buf = io.BytesIO()
                    torch.save(sd, buf)
                    weights_bytes = buf.getvalue()
                    
                    global_model = db.query(GlobalModel).first()
                    if not global_model:
                        # Seeding awal jika belum ada
                        global_model = GlobalModel(version=self.manager.model_version, weights=weights_bytes)
                        db.add(global_model)
                    else:
                        # Update bobot DAN Versi (Sinkron dengan manager jika ronde terakhir)
                        global_model.weights = weights_bytes
                        global_model.version = self.manager.model_version
                        global_model.last_updated = datetime.utcnow()
                    
                    db.commit()
                    
                    # 4. PERSISTENSI KE MANAGER (Untuk Dashboard UI)
                    client_stats = aggregated_metrics.get("clients", {})
                    self.manager.record_round_data(server_round, aggregated_metrics, client_stats, db=db)
                    
                    # Update Total Waktu (Aktif)
                    self.manager.metrics["total_round_time_s"] = round(datetime.now().timestamp() - self.manager.start_time, 2)
                    
                    if self.manager.tracker:
                        try:
                            self.manager.tracker.stop()
                            # Simpan energi server saja (Real dari tracker)
                            if hasattr(self.manager.tracker, '_total_energy'):
                                self.manager.metrics["server_energy_kwh"] = self.manager.tracker._total_energy.kWh
                        except: pass

                    # LOGGING BENAR (Sesuai Permintaan)
                    self.manager.update_logs(f"SERVER LOG: Ronde {server_round} selesai. Akurasi: {aggregated_metrics.get('accuracy', 0):.4f} | Loss: {aggregated_metrics.get('loss', 0):.4f}")
                    self.logger.info(f"Ronde {server_round} tercatat di Dashboard.")

                except Exception as e:
                    self.logger.error(f"Gagal menyimpan hasil agregasi ke database: {e}")
                    db.rollback()
                finally:
                    db.close()
                    
                # Simpan salinan file fisik untuk backup & ONNX export stability
                try:
                    os.makedirs("data", exist_ok=True)
                    # Simpan versi murni (Shared Keys Only) untuk referensi internal jika perlu
                    torch.save(sd, "data/backbone_pure.pth")
                    
                    # Gabungkan BN jika sudah ada dari fase sebelumnya untuk konsistensi download client
                    bn_path = "data/global_bn_combined.pth"
                    if os.path.exists(bn_path):
                        try:
                            bn_params = torch.load(bn_path, map_location="cpu")
                            sd.update(bn_params)
                            self.logger.info("Backbone global kini diperkuat dengan statistik BN.")
                        except: pass
                        
                    # ATOMIC WRITE
                    with tempfile.NamedTemporaryFile(delete=False, dir="data") as tmp:
                        torch.save(sd, tmp.name)
                        shutil.move(tmp.name, "data/backbone.pth")
                except Exception as e:
                    self.logger.error(f"Gagal menyimpan file murni/BN: {e}")

                    # --- SWA / SNAPSHOT AVERAGING ---
                    if server_round >= TRAINING_PARAMS.get("swa_start_round", 8):
                        self.logger.info(f"Ronde {server_round}: Menyimpan snapshot backbone untuk perataan akhir.")
                        self.snapshots.append(copy.deepcopy(sd))

                    # Jika Ronde Terakhir, lakukan perataan (Averaging)
                    if server_round == self.manager.default_rounds and len(self.snapshots) > 1:
                        self.logger.info(f"Melakukan Snapshot Averaging dari {len(self.snapshots)} ronde terakhir...")
                        swa_sd = copy.deepcopy(self.snapshots[0])
                        for key in swa_sd:
                            for i in range(1, len(self.snapshots)):
                                swa_sd[key] += self.snapshots[i][key]
                            swa_sd[key] = torch.div(swa_sd[key], len(self.snapshots))
                        
                        # Simpan hasil SWA sebagai model final
                        buf_swa = io.BytesIO()
                        torch.save(swa_sd, buf_swa)
                        global_model.weights = buf_swa.getvalue()
                        db.commit()
                        
                        torch.save(swa_sd, "data/backbone.pth")
                        self.logger.success("Model final berhasil di-rata-ratakan (Snapshot Averaged). Akurasi stabil.")
                        self.manager.update_logs("Snapshot Averaging (SWA) berhasil diterapkan pada model final.")

                except Exception as e:
                    self.logger.error(f"Gagal menyimpan ronde {server_round} ke DB: {e}")
                    db.rollback()
                finally:
                    # PROSES DATA CLIENT DAN SIMPAN RONDE KE DB
                    if results:
                        client_data = {}
                        for i, (client_proxy, fit_res) in enumerate(results):
                            # Prioritaskan hostname dari metrik (ID asli client)
                            cid = fit_res.metrics.get("hostname")
                            proxy_id = getattr(client_proxy, "cid", "")
                            
                            # Jika hostname tidak ada, gunakan proxy_id tapi log peringatan
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

from app.utils.logging import Logger

class FLServerManager:
    # Manajer Sesi dan Status Server Federated (FL)
    # Menyimpan informasi status pelatihan, metrik performa, dan log aktivitas.
    
    def __init__(self):
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
            "convergence_round": None,
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
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, "r") as f:
                    settings = json.load(f)
                    self.inference_threshold = settings.get("inference_threshold", 0.7)
                    self.default_batch_size = settings.get("batch_size", 32)
                self.logger.info(f"Pengaturan dimuat dari {self.settings_path}")
            except Exception as e:
                self.logger.error(f"Gagal memuat pengaturan: {e}")

    def save_inference_logs(self):
        """Menyimpan log inferensi ke file JSON agar persisten."""
        try:
            logs = self.metrics.get("inference_logs", [])
            with open(self.inference_logs_path, "w") as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            self.logger.error(f"Gagal menyimpan log inferensi: {e}")

    def load_inference_logs(self):
        """Memuat log inferensi dari file JSON saat startup."""
        if os.path.exists(self.inference_logs_path):
            try:
                with open(self.inference_logs_path, "r") as f:
                    self.metrics["inference_logs"] = json.load(f)
                self.logger.info(f"Berhasil memulihkan {len(self.metrics['inference_logs'])} log inferensi.")
            except Exception as e:
                self.logger.error(f"Gagal memuat log inferensi: {e}")

    def save_settings(self):
        """Menyimpan pengaturan server ke file JSON."""
        try:
            settings = {
                "inference_threshold": self.inference_threshold,
                "batch_size": getattr(self, 'default_batch_size', 32)
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
        self.logger.info("Memulai pemulihan status server Federated...")
        
        max_retries = 10
        retry_delay = 5
        
        for i in range(max_retries):
            db = SessionLocal()
            try:
                # 1. Muat Versi Model Terbaru
                latest_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
                if latest_model:
                    self.model_version = latest_model.version
                    self.logger.info(f"Database terhubung. Versi Model saat ini: v{self.model_version}")

                # 2. Muat riwayat ronde (Hanya sesi terbaru agar tidak duplikat)
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
                                "server": m,
                                "clients": clients
                            })
                            # Update global metrics
                            self.metrics["accuracy"] = float(m.get("accuracy", 0))
                            self.metrics["loss"] = float(m.get("loss", 0))
                            
                            # Agregasi Ekonomi (NEW)
                            total_energy += m.get("compute_energy_kwh", 0)
                            total_duration += m.get("total_round_time_s", 0)
                            
                            if self.metrics["convergence_round"] is None and m.get("accuracy", 0) > 0.90:
                                self.metrics["convergence_round"] = r.round_number
                                
                        except Exception as e:
                            self.logger.warn(f"Failed to parse round {r.round_number}: {e}")

                    self.metrics["round_history"] = history
                    self.metrics["compute_energy_kwh"] = total_energy
                    self.metrics["total_round_time_s"] = total_duration
                    
                    # RECOVERY: Populate unique_client_ids from history
                    unique_ids = set()
                    for r in history:
                        if "clients" in r:
                            unique_ids.update(r["clients"].keys())
                    self.metrics["unique_client_ids"] = list(unique_ids)
                    
                    # 3. Update estimasi biaya/transmisi berdasarkan history yang dimuat
                    self.update_metrics({})
                    self.logger.info(f"Berhasil memulihkan {len(history)} ronde ke dashboard FL.")
                    self.logger.info(f"Ditemukan {len(unique_ids)} client aktif dari riwayat: {list(unique_ids)}")
                
                # Muat log inferensi dari file
                self.load_inference_logs()
                
                db.close()
                return # SUCCESS
            except Exception as e:
                self.logger.warn(f"Percobaan {i+1}/{max_retries} gagal: {e}")
                db.close()
                if i < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    self.logger.error("Gagal total memulihkan status FL. Dashboard mungkin akan kosong.")

    def increment_version(self):
        self.model_version += 1
        
        # Persistensi ke Database
        db = SessionLocal()
        try:
            global_model = db.query(GlobalModel).first()
            if global_model:
                global_model.version = self.model_version
                global_model.last_updated = datetime.utcnow()
                db.commit()
                self.logger.success(f"GlobalModel DB updated to v{self.model_version}")
        except Exception as e:
            self.logger.error(f"Failed to persist version increment: {e}")
            db.rollback()
        finally:
            db.close()

        self.update_logs(f"Versi Model Global naik ke v{self.model_version}")
        self.discovery_clients = set() 
        self.received_data = [] 

    def start_phase(self, phase_name):
        self.is_running = True
        self.current_phase = phase_name.lower().replace(" ", "_")
        if self.current_phase == "data_prep": self.start_time = datetime.now().timestamp()
        self.update_logs(f"Fase {phase_name} dimulai.")

    def end_phase(self, phase_name=None):
        if phase_name: self.update_logs(f"Fase {phase_name} selesai.")
        self.is_running = False
        self.current_phase = "idle"

    def update_metrics(self, new_data):
        # Memperbarui metrik performa dan estimasi biaya
        # KHUSUS: Jika ada round_history baru, jangan timpa yang lama, tapi gabungkan.
        if "round_history" in new_data:
            new_history = new_data.pop("round_history")
            existing_rounds = [h["round"] for h in self.metrics["round_history"]]
            for h in new_history:
                if h["round"] not in existing_rounds:
                    self.metrics["round_history"].append(h)

        self.metrics.update(new_data)
        self.update_economics({})

    def update_economics(self, new_data):
        # 1. Transmisi (Gunakan ukuran file asli jika tersedia, jika tidak gunakan estimasi)
        current_rounds = max(self.current_round, len(self.metrics.get("round_history", [])))
        num_clients = len(self.metrics.get("unique_client_ids", []))
        
        # Cek Ukuran File Asli
        bb_size = ECONOMICS["estimated_backbone_size_mb"]
        reg_size = ECONOMICS["estimated_registry_size_mb"]
        
        # Lokasi file backbone (Shared Keys)
        pure_bb_path = "data/backbone_pure.pth"
        if os.path.exists(pure_bb_path):
            bb_size = os.path.getsize(pure_bb_path) / (1024 * 1024)
        
        # Lokasi file registry
        from .config import REGISTRY_PATH
        if os.path.exists(REGISTRY_PATH):
            reg_size = os.path.getsize(REGISTRY_PATH) / (1024 * 1024)

        # Total BB = Ronde * Client * 2 (Upload + Download) * Ukuran Model
        self.metrics["backbone_sync_mb"] = round(current_rounds * num_clients * 2 * bb_size, 2)
        self.metrics["registry_sync_mb"] = round(num_clients * reg_size, 2)
        
        total_mb = self.metrics["backbone_sync_mb"] + self.metrics["registry_sync_mb"]
        self.metrics["transmission_cost_idr"] = round(total_mb * ECONOMICS["transmission_cost_per_mb"], 2)
        
        # 2. Komputasi: Total = Server (Real/Est) + Clients (Real/Est)
        duration_s = self.metrics.get("total_round_time_s", 0)
        if duration_s == 0:
            duration_s = current_rounds * 180
        duration_h = duration_s / 3600

        server_p = ECONOMICS["estimated_server_power_kw"]
        client_p = ECONOMICS["estimated_client_power_kw"]

        # A. Energi Server (Real via Tracker if available)
        server_energy = self.metrics.get("server_energy_kwh", duration_h * server_p)
        
        # B. Energi Client (Agregasi Data Real dari fit_res jika ada)
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

    def record_round_data(self, round_num, server_metrics, client_metrics, db=None):
        # Mencegah redundansi jika fungsi ini dipanggil ulang untuk ronde yang sama
        existing_rounds = [h["round"] for h in self.metrics["round_history"]]
        if round_num in existing_rounds:
            self.logger.info(f"Ronde {round_num} sudah ada di history, memperbarui data...")
            # Update data yang sudah ada (opsional, tapi lebih aman)
            for h in self.metrics["round_history"]:
                if h["round"] == round_num:
                    h["server"] = server_metrics
                    h["clients"] = client_metrics
            return

        self.logger.info(f"Recording round {round_num} data with {len(client_metrics)} clients...")
        entry = {
            "round": round_num,
            "server": server_metrics,
            "clients": client_metrics
        }
        self.metrics["round_history"].append(entry)
        
        # Update unique_client_ids from actual recorded data
        # Prefer meaningful IDs over random UUIDs if both exist
        new_cid = None
        for cid in client_metrics.keys():
            # Jika ID terlihat seperti UUID acak (panjang 32+), coba cari apakah ada ID lain
            is_uuid = len(cid) >= 32 and cid.isalnum()
            if not is_uuid:
                new_cid = cid
            
            if cid not in self.metrics["unique_client_ids"]:
                self.metrics["unique_client_ids"].append(cid)
                self.logger.info(f"Client {cid} added to unique_client_ids")

        # Pembersihan berkala: Hanya simpan ID yang beneran ada di round_history
        all_active_ids = set()
        for r in self.metrics["round_history"]:
            all_active_ids.update(r["clients"].keys())
        
        # Sinkronkan unique_client_ids agar tidak ada ID sampah
        self.metrics["unique_client_ids"] = [cid for cid in self.metrics["unique_client_ids"] if cid in all_active_ids]
        
        # PERSISTENSI KE DATABASE
        if db:
            try:
                new_fl_round = FLRound(
                    round_number=round_num,
                    loss=float(server_metrics.get('loss', 0.0)),
                    metrics=json.dumps(server_metrics),
                    timestamp=datetime.now(timezone(timedelta(hours=7)))
                )
                db.add(new_fl_round)
                db.commit()
                self.logger.success(f"Ronde {round_num} berhasil disimpan ke database.")
            except Exception as e:
                self.logger.error(f"Gagal menyimpan ronde ke database: {e}")
                db.rollback()

        # Memperbarui pelacakan client unik
        for cid in client_metrics.keys():
            if cid not in self.metrics["unique_client_ids"]:
                self.metrics["unique_client_ids"].append(cid)
        
        # Cek Konvergensi
        if self.metrics["convergence_round"] is None and server_metrics.get("accuracy", 0) > 0.90:
            self.metrics["convergence_round"] = round_num

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
        attendance_count = 0
        active_clients = []
        if db:
            attendance_count = db.query(AttendanceRecap).count()
            clients = db.query(Client).all()
            for c in clients:
                active_clients.append({
                    "id": c.edge_id,
                    "ip": c.ip_address,
                    "status": (c.status or "offline").upper(),
                    "last_seen": c.last_seen.strftime("%H:%M:%S") if c.last_seen else "-"
                })
            
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
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
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
            # NOTE: Versi tidak diupdate di sini agar siklus (Training + Registry) dianggap 1 kenaikan.
            # Versi akan diupdate oleh FLController melalui increment_version()
            
            self.update_logs(f"Pelatihan Federated ronde terakhir selesai.")
        finally:
            self.is_running = False
            self.end_phase("Training")


# Instansi Global
fl_manager = FLServerManager()
