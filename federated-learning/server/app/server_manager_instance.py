import flwr as fl
import os
import io
import torch
import json
import numpy as np
from collections import OrderedDict
from datetime import datetime
from app.db.db import SessionLocal
from app.db.models import FLRound, GlobalModel, UserGlobal, AttendanceRecap
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
    def __init__(self, session_id: str, manager, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = session_id
        self.manager = manager

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

        if not results:
            return None, {}

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"[OK] Agregasi Ronde {server_round} Selesai: Akurasi Gabungan: {aggregated_metrics.get('accuracy', 0):.4f}")
            try:
                params_np = fl.common.parameters_to_ndarrays(aggregated_parameters)
                final_loss = aggregated_metrics.get("loss", 0.0)
                
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
                    
                    buf = io.BytesIO()
                    torch.save(params_np, buf)
                    
                    global_model = db.query(GlobalModel).first()
                    if not global_model:
                        global_model = GlobalModel(version=server_round, weights=buf.getvalue())
                        db.add(global_model)
                    else:
                        global_model.version = server_round
                        global_model.weights = buf.getvalue()
                        global_model.last_updated = datetime.utcnow()
                    
                    db.commit()
                    
                    # Simpan salinan file fisik untuk backup
                    try:
                        os.makedirs("data", exist_ok=True)
                        torch.save(params_np, "data/backbone.pth")
                    except: pass
                        
                except Exception as e:
                    print(f"[ERROR] Gagal menyimpan ronde {server_round} ke DB: {e}")
                    db.rollback()
                finally:
                    db.close()
            except Exception as e:
                print(f"[ERROR] Gagal memproses parameter agregasi: {e}")
                
        if aggregated_metrics or results:
            # Hitung Weight Divergence (Avg L2 Distance)
            weight_div = 0.0
            if results and aggregated_parameters:
                try:
                    target_params = fl.common.parameters_to_ndarrays(aggregated_parameters)
                    total_dist = 0
                    for _, fit_res in results:
                        client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                        dist = 0
                        for cp, tp in zip(client_params, target_params):
                            dist += np.linalg.norm(cp - tp)
                        total_dist += dist / len(client_params)
                    weight_div = round(total_dist / len(results), 6)
                except: pass
            
            if aggregated_metrics:
                aggregated_metrics["weight_divergence"] = weight_div
            
            # Mencatat rincian per-client
            client_data = {}
            for i, (client_proxy, fit_res) in enumerate(results):
                # Menghitung divergensi per-client
                client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                target_params = fl.common.parameters_to_ndarrays(aggregated_parameters) if aggregated_parameters else None
                
                c_div = 0.0
                if target_params:
                    c_dist = 0
                    for cp, tp in zip(client_params, target_params):
                        c_dist += np.linalg.norm(cp - tp)
                    c_div = round(c_dist / len(client_params), 6)
                
                # CID default atau gunakan hostname jika ada di metrik
                cid = fit_res.metrics.get("hostname") or getattr(client_proxy, "cid", f"client-{i}")
                
                # Menggabungkan semua metrik
                m = fit_res.metrics.copy()
                m["num_samples"] = fit_res.num_examples
                m["divergence"] = c_div
                
                # Dekoding epoch_history jika berupa string
                eh = fit_res.metrics.get("epoch_history", "[]")
                if isinstance(eh, str):
                    try:
                        m["epoch_history"] = json.loads(eh)
                    except:
                        m["epoch_history"] = []
                else:
                    m["epoch_history"] = eh
                    
                client_data[cid] = m
            
            print(f"[STRATEGY] Recording data for round {server_round} (Clients: {len(client_data)})")
            self.manager.record_round_data(server_round, aggregated_metrics or {}, client_data)
            if aggregated_metrics:
                self.manager.update_metrics(aggregated_metrics)
                loss = aggregated_metrics.get('loss', 0)
                self.manager.update_logs(f"Ronde {server_round} selesai. Acc: {aggregated_metrics.get('accuracy', 0):.4f} | Loss: {loss:.4f} | Div: {weight_div}")

        return aggregated_parameters, aggregated_metrics

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
        ts = datetime.now().strftime("%H:%M:%S")
        self.current_logs.append(f"[{ts}] {msg}")
        if len(self.current_logs) > 100: self.current_logs.pop(0)
        print(f"SERVER LOG: {msg}")

    def update_metrics(self, new_data):
        self.metrics.update(new_data)
        
        # 1. Transmisi
        bb_mb = self.metrics.get("backbone_sync_mb", 0)
        reg_mb = self.metrics.get("registry_sync_mb", 0)
        cost_per_mb = ECONOMICS["transmission_cost_per_mb"]
        self.metrics["transmission_cost_idr"] = round((bb_mb + reg_mb) * cost_per_mb, 2)
        
        # 2. Komputasi (Server + Aggregate Clients)
        energy_kwh = self.metrics.get("compute_energy_kwh", 0)
        if energy_kwh == 0:
            # Estimasi daya dari config
            duration_h = self.metrics.get("total_round_time_s", 0) / 3600
            server_p = ECONOMICS["estimated_server_power_kw"]
            client_p = ECONOMICS["estimated_client_power_kw"]
            
            # Asumsi 2 client aktif untuk estimasi daya total sistem
            energy_kwh = duration_h * (server_p + 2 * client_p)
            self.metrics["compute_energy_kwh"] = round(energy_kwh, 6)
            
        cost_per_kwh = ECONOMICS["compute_cost_per_kwh"]
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
        if db:
            attendance_count = db.query(AttendanceRecap).count()
            
        return {
            "is_running": self.is_running,
            "phase": self.current_phase,
            "session_id": self.session_id,
            "metrics": self.metrics,
            "phase_logs": self.current_logs,
            "received_data": self.received_data,
            "model_version": self.model_version,
            "default_rounds": self.default_rounds,
            "default_epochs": self.default_epochs,
            "default_min_clients": self.default_min_clients,
            "attendance_count": attendance_count,
            "uptime": int(datetime.now().timestamp() - self.start_time) if self.start_time > 0 else 0
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
                weights_np = torch.load(io.BytesIO(latest_model.weights))
                initial_parameters = fl.common.ndarrays_to_parameters(weights_np)
        except Exception as e:
            print(f"[ERROR] Gagal memuat parameter awal: {e}")
        finally:
            db.close()

        # Strategi pelatihan FedAvg dengan konfigurasi dinamis (mu=0.05 untuk stabilitas)
        strategy = SaveModelStrategy(
            session_id=session_id,
            manager=self,
            initial_parameters=initial_parameters,
            fraction_fit=1.0,
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
            fit_metrics_aggregation_fn=weighted_average,
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
        finally:
            self.is_running = False
            self.end_phase("Training")

# Instansi Global
fl_manager = FLServerManager()
