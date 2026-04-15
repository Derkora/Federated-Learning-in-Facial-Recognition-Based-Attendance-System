import flwr as fl
import os
import io
import torch
import json
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from app.db.db import SessionLocal
from app.db.models import FLRound, GlobalModel, UserGlobal, AttendanceRecap, Client
from app.utils.mobilefacenet import MobileFaceNet
from app.config import ECONOMICS, TRAINING_PARAMS, FALLBACK_MODEL_PATH

# Fungsi rata-rata tertimbang untuk metrik
def weighted_average(metrics: list) -> dict:
    # Fungsi agregasi metrik dari seluruh terminal (Client)
    print(f"[METRICS] Aggregating {len(metrics)} client results...")
    examples = [m[0] for m in metrics]
    total_examples = sum(examples)
    if total_examples == 0: 
        print("[METRICS] Total examples is 0, returning empty.")
        return {}

    try:
        aggregated = {
            "accuracy": sum([m[1].get("accuracy", 0.0) * m[0] for m in metrics]) / total_examples,
            "loss": sum([m[1].get("loss", 0.0) * m[0] for m in metrics]) / total_examples,
            "val_accuracy": sum([m[1].get("val_accuracy", 0.0) * m[0] for m in metrics]) / total_examples,
            "val_loss": sum([m[1].get("val_loss", 0.0) * m[0] for m in metrics]) / total_examples,
        }
        print(f"[METRICS] Aggregated: {aggregated}")
        return aggregated
    except Exception as e:
        print(f"[METRICS ERROR] Failed to aggregate: {e}")
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

    def aggregate_fit(self, server_round: int, results: list, failures: list):
        print(f"\n[ROUND {server_round}] Ringkasan Performa")
        if failures:
            print(f"[WARN] Ronde {server_round} mendapati {len(failures)} kegagalan client.")

        # Laporan masing-masing terminal
        for i, (client_proxy, fit_res) in enumerate(results):
            cid = getattr(client_proxy, "cid", f"client-{i}")
            metrics = fit_res.metrics
            acc = metrics.get("accuracy", 0.0)
            loss = metrics.get("loss", 0.0)
            val_acc = metrics.get("val_accuracy", 0.0)
            print(f"  > Client {cid}: Akurasi: {acc:.4f} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

        # Filter client yang tidak punya data (mencegah division by zero di FedAvg)
        valid_results = [(cp, fr) for cp, fr in results if fr.num_examples > 0]
        
        if not valid_results:
            print(f"[WARN] Ronde {server_round} tidak memiliki data valid untuk diagregasi.")
            return None, {}

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, valid_results, failures)

        if aggregated_parameters is not None:
            print(f"[OK] Agregasi Ronde {server_round} Selesai: Akurasi Gabungan: {aggregated_metrics.get('accuracy', 0):.4f}")
            try:
                params_np = fl.common.parameters_to_ndarrays(aggregated_parameters)
                final_loss = aggregated_metrics.get("loss", 0.0)
                
                # 1. KONVERSI KE STATE_DICT (Critical for Identification Consistency)
                # Kita tidak lagi menyimpan list parameter mentah, melainkan file .pth yang siap pakai.
                global_model_instance = MobileFaceNet()
                sd = global_model_instance.state_dict()
                shared_keys = [k for k in sd.keys() if 
                               not any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked'])
                               and any(x in k.lower() for x in ['weight', 'bias'])]
                
                if len(params_np) == len(shared_keys):
                    for k, v in zip(shared_keys, params_np):
                        sd[k] = torch.from_numpy(v.copy())
                else:
                    print(f"[WARN] Key mismatch during saving: {len(params_np)} params vs {len(shared_keys)} keys.")

                # 2. MERGE GLOBAL BN (Jika tersedia dari fase registry sebelumnya)
                bn_path = "data/global_bn_combined.pth"
                if os.path.exists(bn_path):
                    try:
                        bn_params = torch.load(bn_path, map_location="cpu")
                        sd.update(bn_params)
                        print(f"[ROUND {server_round}] Merged Global BN stats into backbone.")
                    except Exception as bn_err:
                        print(f"[ROUND {server_round} WARN] Serializing BN failed: {bn_err}")

                # Simpan hasil ke database
                db = SessionLocal()
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
                        global_model = GlobalModel(version=self.target_version, weights=weights_bytes)
                        db.add(global_model)
                    else:
                        global_model.weights = weights_bytes
                        global_model.last_updated = datetime.utcnow()
                    
                    db.commit()
                    
                    # Simpan salinan file fisik untuk backup & ONNX export stability
                    try:
                        os.makedirs("data", exist_ok=True)
                        torch.save(sd, "data/backbone.pth")
                    except: pass
                        
                except Exception as e:
                    print(f"[ERROR] Gagal menyimpan ronde {server_round} ke DB: {e}")
                    db.rollback()
                finally:
                    db.close()
            except Exception as e:
                print(f"[ERROR] Gagal memproses parameter agregasi: {e}")
                
        if results:
            client_data = {}
            for i, (client_proxy, fit_res) in enumerate(results):
                # CID default atau gunakan hostname jika ada di metrik
                cid = fit_res.metrics.get("hostname") or getattr(client_proxy, "cid", f"client-{i}")
                m = fit_res.metrics.copy()
                m["num_samples"] = fit_res.num_examples
                eh = fit_res.metrics.get("epoch_history", "[]")
                if isinstance(eh, str):
                    try: m["epoch_history"] = json.loads(eh)
                    except: m["epoch_history"] = []
                else: m["epoch_history"] = eh
                client_data[cid] = m
            
            self.manager.record_round_data(server_round, aggregated_metrics or {}, client_data)
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
        self.session_id = None
        self.is_running = False
        self.current_phase = "idle"
        self.start_time = 0
        self.current_logs = []
        self.model_version = 0
        self.default_rounds = 15
        self.default_epochs = 3
        self.default_min_clients = 2
        self.default_lr = 1e-4
        self.default_mu = 0.05
        self.default_lambda = 0.1
        self.registered_clients = {}
        self.registry_submissions = {}
        self.ready_clients = set() 
        self.received_data = []
        self.discovery_clients = set()
        self.inference_threshold = 0.60
        self.metrics = {
            "accuracy": 0, "loss": 0, 
            "backbone_sync_mb": 0, "registry_sync_mb": 0,
            "transmission_cost_idr": 0,
            "training_duration_s": 0, "total_round_time_s": 0,
            "compute_energy_kwh": 0, "compute_cost_idr": 0,
            "round_history": [], # Riwayat data ronde dengan rincian client
            "unique_client_ids": [], # Pelacakan ID unik client yang berkontribusi
            "convergence_round": None
        }
        self._load_persistence()

    def _load_persistence(self):
        """Memuat ulang status dari database untuk persistensi log & versi."""
        print("[PERSISTENCE] Loading server state from database...")
        db = SessionLocal()
        try:
            # 1. Muat Versi Model Terbaru
            latest_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
            if latest_model:
                self.model_version = latest_model.version
                print(f"[PERSISTENCE] Loaded Model Version: v{self.model_version}")

            # 2. Muat Riwayat Ronde
            rounds = db.query(FLRound).order_by(FLRound.timestamp.asc()).all()
            if rounds:
                self.metrics["round_history"] = []
                self.metrics["unique_client_ids"] = []
                
                for r in rounds:
                    try:
                        m = json.loads(r.metrics)
                        # Reconstruct round data
                        # Note: clients metrics per round aren't fully stored in FLRound.metrics 
                        # but we can reconstruct the global history.
                        entry = {
                            "round": r.round_number,
                            "server": m,
                            "clients": m.get("clients", {}) # Fallback
                        }
                        self.metrics["round_history"].append(entry)
                        
                        # Update global metrics to the latest round
                        self.metrics["accuracy"] = m.get("accuracy", 0)
                        self.metrics["loss"] = m.get("loss", 0)
                        
                        if self.metrics["convergence_round"] is None and m.get("accuracy", 0) > 0.90:
                            self.metrics["convergence_round"] = r.round_number
                            
                    except Exception as e:
                        print(f"[PERSISTENCE WARN] Failed to parse round {r.round_number}: {e}")

                # 3. Update estimasi biaya/transmisi berdasarkan history yang dimuat
                self.update_metrics({})
                print(f"[PERSISTENCE] Loaded {len(rounds)} rounds from history.")
        except Exception as e:
            print(f"[PERSISTENCE ERROR] Failed to load history: {e}")
        finally:
            db.close()

    def increment_version(self):
        self.model_version += 1
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

    def update_logs(self, msg):
        # Gunakan Waktu Indonesia Barat (UTC+7)
        tz_wib = timezone(timedelta(hours=7))
        ts = datetime.now(tz_wib).strftime("%H:%M:%S")
        self.current_logs.append(f"[{ts}] {msg}")
        if len(self.current_logs) > 100: self.current_logs.pop(0)
        print(f"SERVER LOG: {msg}")

    def update_metrics(self, new_data):
        self.metrics.update(new_data)
        
        # 1. Transmisi (Estimatif berdasarkan Ronde & Client yang terhubung)
        current_rounds = len(self.metrics.get("round_history", []))
        num_clients = len(self.metrics.get("unique_client_ids", []))
        bb_size = ECONOMICS["estimated_backbone_size_mb"]
        reg_size = ECONOMICS["estimated_registry_size_mb"]
        
        # Total BB = Ronde * Client * 2 (Upload + Download) * Ukuran Model
        self.metrics["backbone_sync_mb"] = round(current_rounds * num_clients * 2 * bb_size, 2)
        self.metrics["registry_sync_mb"] = round(num_clients * reg_size, 2)
        
        total_mb = self.metrics["backbone_sync_mb"] + self.metrics["registry_sync_mb"]
        # Hitung biaya berdasarkan per-MB (Rp 3,25 / MB)
        self.metrics["transmission_cost_idr"] = round(total_mb * 3.25, 2)
        
        # 2. Komputasi (Server + Aggregate Clients)
        energy_kwh = self.metrics.get("compute_energy_kwh", 0)
        if energy_kwh == 0:
            duration_s = self.metrics.get("total_round_time_s", 0)
            if duration_s == 0:
                # Fallback jika waktu belum tercatat, ambil rata-rata 3 menit per ronde
                duration_s = current_rounds * 180
            
            duration_h = duration_s / 3600
            server_p = ECONOMICS["estimated_server_power_kw"]
            client_p = ECONOMICS["estimated_client_power_kw"]
            
            # Daya total = Daya Server + (Jumlah Client * Daya per Client)
            energy_kwh = duration_h * (server_p + num_clients * client_p)
            self.metrics["compute_energy_kwh"] = round(energy_kwh, 6)
            
        cost_per_kwh = ECONOMICS["compute_cost_per_kwh"] # Rp 1.444,70
        self.metrics["compute_cost_idr"] = round(energy_kwh * cost_per_kwh, 2)

    def record_round_data(self, round_num, server_metrics, client_metrics):
        print(f"[MANAGER] Recording round {round_num} data...")
        entry = {
            "round": round_num,
            "server": server_metrics,
            "clients": client_metrics
        }
        self.metrics["round_history"].append(entry)
        print(f"[MANAGER] Round history now has {len(self.metrics['round_history'])} entries.")
        
        # Memperbarui pelacakan client unik
        for cid in client_metrics.keys():
            if cid not in self.metrics["unique_client_ids"]:
                self.metrics["unique_client_ids"].append(cid)
        print(f"[MANAGER] Unique clients tracked: {len(self.metrics['unique_client_ids'])}")
        
        # Cek Konvergensi
        if self.metrics["convergence_round"] is None and server_metrics.get("accuracy", 0) > 0.90:
            # Ambang batas sederhana untuk konvergensi
            self.metrics["convergence_round"] = round_num

    def _get_lr_for_round(self, server_round):
        # Penyelarasan LR Schedule dari konfigurasi terpusat
        schedule = TRAINING_PARAMS["lr_schedule"]
        lr = 1e-4 # Nilai default
        for threshold in sorted(schedule.keys()):
            if server_round >= threshold:
                lr = schedule[threshold]
        return lr

    def get_status(self, db=None):
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
            "active_clients": active_clients
        }

    def ensure_model_seeded(self, db):
        # Memastikan tabel GlobalModel memiliki bobot awal (Seeding)
        latest_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
        if latest_model and latest_model.weights: return
            
        fallback_path = FALLBACK_MODEL_PATH
        if os.path.exists(fallback_path):
            print(f"Memulai seeding GlobalModel dari {fallback_path}...")
            try:
                model = MobileFaceNet()
                loaded = torch.load(fallback_path, map_location="cpu")
                current_sd = model.state_dict()
                weights_np = []
                if isinstance(loaded, dict):
                    for key in current_sd.keys():
                        if key in loaded: weights_np.append(loaded[key].cpu().numpy())
                        else: weights_np.append(current_sd[key].cpu().numpy())
                else: weights_np = loaded
                
                buf = io.BytesIO()
                torch.save(weights_np, buf)
                new_model = GlobalModel(version=0, weights=buf.getvalue())
                db.add(new_model)
                db.commit()
                print("[OK] GlobalModel berhasil diinisialisasi.")
            except Exception as e:
                print(f"[ERROR] Gagal inisialisasi GlobalModel: {e}")
                db.rollback()
                
    def get_label_map_from_db(self):
        # Mengambil daftar ID mahasiswa global sebagai acuan index untuk pelatihan
        db = SessionLocal()
        try:
            users = db.query(UserGlobal).order_by(UserGlobal.nrp).all()
            return [u.nrp for u in users]
        finally:
            db.close()

    def start_training(self, session_id: str, rounds: int = 20, min_clients: int = 2):
        self.session_id = session_id
        self.is_running = True
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
                    shared_keys = [k for k in sd.keys() if 
                                   not any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked'])
                                   and any(x in k.lower() for x in ['weight', 'bias'])]
                    weights_np = [sd[k].cpu().numpy() for k in shared_keys]
                else:
                    weights_np = loaded
                
                initial_parameters = fl.common.ndarrays_to_parameters(weights_np)
        except Exception as e:
            print(f"[ERROR] Gagal memuat parameter awal: {e}")
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
            # Update versi di Database dan Manager hanya setelah seluruh ronde sukses
            self.model_version = target_version
            
            # Persistensi Versi ke Database GlobalModel
            db_final = SessionLocal()
            try:
                global_model = db_final.query(GlobalModel).first()
                if global_model:
                    global_model.version = target_version
                    global_model.last_updated = datetime.utcnow()
                    db_final.commit()
                    print(f"[OK] Database GlobalModel updated to Version v{target_version}")
            except Exception as db_err:
                print(f"[ERROR] Gagal finalisasi versi di DB: {db_err}")
                db_final.rollback()
            finally:
                db_final.close()
                
            self.update_logs(f"Siklus Pelatihan Selesai. Versi Model Global naik ke v{self.model_version}")
        finally:
            self.is_running = False
            self.end_phase("Training")


# Instansi Global
fl_manager = FLServerManager()
