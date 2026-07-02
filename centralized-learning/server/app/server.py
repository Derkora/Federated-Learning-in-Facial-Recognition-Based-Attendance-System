import os
import time
import json
import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session
from .db.db import SessionLocal

from .db import models
from .db.models import Client
from .config import ECONOMICS, TRAINING_PARAMS
from .utils.logging import init_logger, get_logger

class CentralizedServerManager:
    # Manajer Utama Dashboard Server Terpusat
    # Melacak status pelatihan, fase aktif, dan metrik performa model global.
    
    def __init__(self):
        self.lock = threading.RLock()
        self.is_running = False
        self.running_tasks_count = 0
        self.current_phase = "Standby"
        self.upload_requested = False
        self.start_time = 0
        self.received_data = [] 
        self.uploader_map = {} 
        self.registered_clients = {}
        self.current_dataset = "students"
        self.model_version = 0
        
        # Inisialisasi Logger Terpusat (Global)
        self.log_path = "/app/data/server_training.log"
        init_logger(self.log_path, max_memory_logs=10000, tag="CL-SERVER")
        self.logger = get_logger()
        
        self.metrics = {
            "upload_volume_mb": 0,
            "download_volume_mb": 0,
            "transmission_cost_idr": 0,
            "training_duration_s": 0,
            "total_round_time_s": 0,
            "compute_energy_kwh": 0,
            "compute_cost_idr": 0,
            "preprocess_duration_s": 0.0,
            "preprocess_energy_kwh": 0.0,
            "preprocess_upload_mb": 0.0,
            "preprocess_download_mb": 0.0,
            "epoch_history": [], # Riwayat {"epoch": i, "loss": l, "accuracy": a}
            "inference_logs": [] # Log detail untuk riset FAR/TAR
        }
        self.settings_path = "data/settings_cl.json"
        self.inference_logs_path = "data/inference_logs.json"
        
        # Default dari config
        self.default_epochs = TRAINING_PARAMS["epochs"]
        self.default_batch_size = TRAINING_PARAMS["batch_size"]
        self.inference_threshold = 0.7
        self.current_db_version_id = None
        self.model_version_str = "v0"
        
        self.update_logs("=== Server Started / Restarted ===")
        
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


    def update_logs(self, message):
        """Menambahkan log baru ke memori dan file persisten (Wrapper ke Logger)."""
        self.logger.info(message)

    def _load_persistence(self):
        """Memuat ulang status dari database dengan logika retry dan logging yang sangat detail."""
        self.logger.info("Memulai pemulihan status server...")
        
        max_retries = 10
        retry_delay = 5
        
        for i in range(max_retries):
            db = SessionLocal()
            try:
                version_count = db.query(models.ModelVersion).count()
                self.model_version = version_count
                
                # Muat Versi Model String
                latest_version = db.query(models.ModelVersion).order_by(models.ModelVersion.version_id.desc()).first()
                if latest_version and latest_version.notes:
                    try:
                        notes = latest_version.notes
                        parts = [p.strip() for p in notes.split("|")]
                        dataset = "students"
                        epochs = 0
                        for p in parts:
                            if p.startswith("Dataset:"):
                                dataset = p.split(":")[1].strip()
                            elif p.startswith("Epochs:"):
                                epochs = p.split(":")[1].strip()
                        self.model_version_str = f"cl_{dataset}_{epochs}e"
                    except:
                        self.model_version_str = "v0"
                else:
                    self.model_version_str = "v0"
                self.logger.info("Database terhubung.")

                # 2. Muat Riwayat Pelatihan
                latest_version = db.query(models.ModelVersion).order_by(models.ModelVersion.version_id.desc()).first()
                if latest_version:
                    rounds = db.query(models.TrainingRound).filter_by(model_version_id=latest_version.version_id).order_by(models.TrainingRound.round_id.asc()).all()
                else:
                    rounds = db.query(models.TrainingRound).filter(models.TrainingRound.model_version_id == None).order_by(models.TrainingRound.round_id.asc()).all()
                
                if rounds:
                    history = []
                    total_duration = 0
                    total_energy = 0
                    total_upload = 0
                    total_download = 0
                    
                    for r in rounds:
                        time_str = "-"
                        if r.start_time:
                            dt = r.start_time
                            if dt.tzinfo is None:
                                wib_dt = dt.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=7)))
                            else:
                                wib_dt = dt.astimezone(timezone(timedelta(hours=7)))
                            time_str = wib_dt.strftime("%Y-%m-%d %H:%M:%S WIB")
                            
                        history.append({
                            "epoch": r.round_number if r.round_number is not None else 0,
                            "loss": float(r.global_loss) if r.global_loss is not None else 0.0,
                            "accuracy": float(r.global_accuracy) if r.global_accuracy is not None else 0.0,
                            "val_loss": float(r.val_loss) if r.val_loss is not None else None,
                            "val_accuracy": float(r.val_accuracy) if r.val_accuracy is not None else None,
                            "duration_s": float(r.training_duration_s) if r.training_duration_s is not None else 0.0,
                            "timestamp": time_str
                        })
                        total_duration += r.training_duration_s or 0
                        total_energy += r.compute_energy_kwh or 0
                        total_upload += r.upload_volume_mb or 0
                        total_download += r.download_volume_mb or 0
                        
                    self.metrics["epoch_history"] = history
                    self.metrics["training_duration_s"] = total_duration
                    self.metrics["compute_energy_kwh"] = total_energy
                    self.metrics["upload_volume_mb"] = total_upload
                    self.metrics["download_volume_mb"] = total_download
                    
                    # Update metrik terakhir
                    last = rounds[-1]
                    self.metrics["accuracy"] = float(last.global_accuracy) if last.global_accuracy is not None else 0.0
                    self.metrics["loss"] = float(last.global_loss) if last.global_loss is not None else 0.0
                    self.logger.info(f"Berhasil memulihkan {len(history)} baris riwayat ke dashboard.")

                # 4. Sinkronisasi Log Inferensi
                self.load_inference_logs()

                self.update_metrics({})
                self.logger.info("Sinkronisasi metrik selesai. Server Siap.")
                db.close()
                return # Berhasil
            except Exception as e:
                self.logger.warn(f"Percobaan {i+1}/{max_retries} gagal: {e}")
                db.close()
                if i < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    self.logger.error("Gagal total memulihkan status. Dashboard mungkin akan kosong.")

    def load_settings(self):
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, 'r') as f:
                    s = json.load(f)
                    self.default_epochs = s.get("epochs", self.default_epochs)
                    self.default_batch_size = s.get("batch_size", self.default_batch_size)
                    self.inference_threshold = 0.7
                    self.logger.info(f"Pengaturan berhasil dimuat dari {self.settings_path} (Threshold dipaksa ke 0.7)")
                self.save_settings({"epochs": self.default_epochs, "batch_size": self.default_batch_size, "threshold": 0.7})
            except Exception as e:
                self.logger.error(f"Gagal memuat pengaturan: {e}")

    def save_settings(self, new_settings: dict):
        with self.lock:
            try:
                self.default_epochs = int(new_settings.get("epochs", self.default_epochs))
                self.default_batch_size = int(new_settings.get("batch_size", self.default_batch_size))
                self.inference_threshold = float(new_settings.get("threshold", self.inference_threshold))
                
                os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
                with open(self.settings_path, 'w') as f:
                    json.dump({
                        "epochs": self.default_epochs,
                        "batch_size": self.default_batch_size,
                        "threshold": self.inference_threshold
                    }, f)
                self.update_logs("[OK] Pengaturan sistem berhasil diperbarui.")
                return True
            except Exception as e:
                self.update_logs(f"[ERROR] Gagal menyimpan pengaturan: {e}")
                return False

    def save_inference_logs(self):
        """Menyimpan log inferensi ke file JSON agar persisten."""
        with self.lock:
            try:
                logs = self.metrics.get("inference_logs", [])
                os.makedirs(os.path.dirname(self.inference_logs_path), exist_ok=True)
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

    def start_task(self):
        with self.lock:
            self.running_tasks_count += 1
            self.is_running = True

    def end_task(self):
        with self.lock:
            self.running_tasks_count = max(0, self.running_tasks_count - 1)
            if self.running_tasks_count == 0:
                self.is_running = False

    def start_phase(self, phase_name):
        # Menandai awal dari fase alur kerja penelitian
        with self.lock:
            self.is_running = True
            self.current_phase = phase_name
            if phase_name == "Import Data": self.start_time = time.time()
            self.update_logs(f"Fase {phase_name} dimulai.")

    def end_phase(self):
        # Mengembalikan status ke Standby setelah fase selesai
        with self.lock:
            if self.running_tasks_count == 0:
                self.is_running = False
            self.current_phase = "Standby"

    def update_logs(self, msg):
        """Menambahkan log baru ke memori dan file persisten (Wrapper ke Logger)."""
        if "[ERROR]" in msg:
            self.logger.error(msg.replace("[ERROR] ", ""))
        elif "[OK]" in msg:
            self.logger.success(msg.replace("[OK] ", ""))
        else:
            self.logger.info(msg)

    def update_received_data(self, upload_dir):
        # Mendata NRP mahasiswa yang datanya berhasil diterima dari terminal
        with self.lock:
            if os.path.exists(upload_dir):
                self.received_data = [d for d in os.listdir(upload_dir) if os.path.isdir(os.path.join(upload_dir, d))]

    def register_upload(self, edge_id, nrp_list):
        # Mencatat siapa yang mengunggah data apa untuk atribusi yang akurat
        with self.lock:
            for nrp in nrp_list:
                self.uploader_map[nrp] = edge_id

    def increment_version(self, dbs: Session):
        # Sinkronisasi versi model dari database untuk keakuratan dashboard
        with self.lock:
            try:
                version_count = dbs.query(models.ModelVersion).count()
                self.model_version = version_count
            except Exception as e:
                self.model_version += 1

    def update_metrics(self, new_data):
        # Memperbarui metrik performa dan estimasi biaya
        # Penggabungan riwayat epoch tanpa menimpa data yang lama
        with self.lock:
            if "epoch_history" in new_data:
                new_history = new_data.pop("epoch_history")
                existing_epochs = [h["epoch"] for h in self.metrics["epoch_history"]]
                for h in new_history:
                    if h["epoch"] not in existing_epochs:
                        self.metrics["epoch_history"].append(h)
            
            self.metrics.update(new_data)
            
            # Transmisi: Upload + Download
            upload_mb = self.metrics.get("upload_volume_mb", 0)
            download_mb = self.metrics.get("download_volume_mb", 0)
            total_mb = upload_mb + download_mb
            
            # Hitung biaya berdasarkan per-MB (Rp 3,25 / MB)
            self.metrics["transmission_cost_idr"] = round(total_mb * 3.25, 2)
            
            # Komputasi: kWh -> IDR
            energy_kwh = self.metrics.get("compute_energy_kwh", 0)
            if energy_kwh == 0:
                duration_s = self.metrics.get("training_duration_s", 0)
                if duration_s == 0:
                    duration_s = self.metrics.get("total_round_time_s", 0)
                
                duration_h = duration_s / 3600
                power_kw = ECONOMICS["estimated_server_power_kw"]
                energy_kwh = duration_h * power_kw
                self.metrics["compute_energy_kwh"] = round(energy_kwh, 6)
                
            cost_per_kwh = ECONOMICS["compute_cost_per_kwh"] # Rp 1.444,70
            self.metrics["compute_cost_idr"] = round(energy_kwh * cost_per_kwh, 2)

    def save_training_round(self, db, round_num, loss, accuracy, val_loss=None, val_accuracy=None, duration=0, energy=0, upload=0, download=0, start_time=None, model_version_id=None):
        # Menyimpan hasil ronde ke Database Postgres
        try:
            actual_version_id = model_version_id or self.current_db_version_id
            new_round = models.TrainingRound(
                round_number=round_num,
                global_loss=float(loss),
                global_accuracy=float(accuracy),
                val_loss=float(val_loss) if val_loss is not None else None,
                val_accuracy=float(val_accuracy) if val_accuracy is not None else None,
                training_duration_s=float(duration),
                compute_energy_kwh=float(energy),
                upload_volume_mb=float(upload),
                download_volume_mb=float(download),
                start_time=start_time or datetime.now(timezone(timedelta(hours=7))),
                model_version_id=actual_version_id
            )
            db.add(new_round)
            db.commit()
            self.logger.success(f"Ronde {round_num} berhasil disimpan ke database.")
        except Exception as e:
            self.logger.error(f"Gagal menyimpan ronde ke database: {e}")
            db.rollback()

    def save_version_metrics(self, version_str):
        with self.lock:
            try:
                # Preprocessing
                prep_duration = self.metrics.get("preprocess_duration_s", 0.0)
                prep_upload = self.metrics.get("preprocess_upload_mb", self.metrics.get("upload_volume_mb", 0.0))
                prep_download = 0.0
                prep_bandwidth = prep_upload + prep_download
                prep_transmission_cost = round(prep_bandwidth * ECONOMICS["transmission_cost_per_mb"], 2)
                
                prep_energy = self.metrics.get("preprocess_energy_kwh", 0.0)
                if prep_energy == 0.0 and prep_duration > 0.0:
                    prep_energy = (prep_duration / 3600.0) * ECONOMICS["estimated_server_power_kw"]
                prep_compute_cost = round(prep_energy * ECONOMICS["compute_cost_per_kwh"], 2)
                
                # Training
                train_duration = self.metrics.get("training_duration_s", 0.0)
                if train_duration == 0.0:
                    train_duration = self.metrics.get("total_round_time_s", 0.0)
                train_upload = 0.0
                train_download = self.metrics.get("download_volume_mb", 0.0)
                train_bandwidth = train_upload + train_download
                train_transmission_cost = round(train_bandwidth * ECONOMICS["transmission_cost_per_mb"], 2)
                
                train_energy = self.metrics.get("compute_energy_kwh", 0.0)
                if train_energy == 0.0 and train_duration > 0.0:
                    train_energy = (train_duration / 3600.0) * ECONOMICS["estimated_server_power_kw"]
                train_compute_cost = round(train_energy * ECONOMICS["compute_cost_per_kwh"], 2)
                
                # Combined
                comb_duration = prep_duration + train_duration
                comb_upload = prep_upload + train_upload
                comb_download = prep_download + train_download
                comb_bandwidth = comb_upload + comb_download
                comb_transmission_cost = round(comb_bandwidth * ECONOMICS["transmission_cost_per_mb"], 2)
                comb_energy = prep_energy + train_energy
                comb_compute_cost = round(comb_energy * ECONOMICS["compute_cost_per_kwh"], 2)
                
                self.metrics["preprocess"] = {
                    "duration_s": round(prep_duration, 2),
                    "upload_volume_mb": round(prep_upload, 2),
                    "download_volume_mb": round(prep_download, 2),
                    "bandwidth_mb": round(prep_bandwidth, 2),
                    "transmission_cost_idr": prep_transmission_cost,
                    "compute_energy_kwh": round(prep_energy, 6),
                    "compute_cost_idr": prep_compute_cost
                }
                
                self.metrics["training"] = {
                    "duration_s": round(train_duration, 2),
                    "upload_volume_mb": round(train_upload, 2),
                    "download_volume_mb": round(train_download, 2),
                    "bandwidth_mb": round(train_bandwidth, 2),
                    "transmission_cost_idr": train_transmission_cost,
                    "compute_energy_kwh": round(train_energy, 6),
                    "compute_cost_idr": train_compute_cost
                }
                
                self.metrics["combined"] = {
                    "duration_s": round(comb_duration, 2),
                    "upload_volume_mb": round(comb_upload, 2),
                    "download_volume_mb": round(comb_download, 2),
                    "bandwidth_mb": round(comb_bandwidth, 2),
                    "transmission_cost_idr": comb_transmission_cost,
                    "compute_energy_kwh": round(comb_energy, 6),
                    "compute_cost_idr": comb_compute_cost
                }

                import math
                from datetime import datetime
                def sanitize_floats(obj):
                    if isinstance(obj, float):
                        if math.isnan(obj) or math.isinf(obj):
                            return None
                        return obj
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, dict):
                        return {k: sanitize_floats(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [sanitize_floats(x) for x in obj]
                    return obj

                path = f"data/metrics_{version_str}.json"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                clean_metrics = sanitize_floats(self.metrics)
                with open(path, "w") as f:
                    json.dump(clean_metrics, f, indent=4)
                self.logger.info(f"Berhasil menyimpan metrik untuk versi {version_str} ke {path}")
            except Exception as e:
                self.logger.error(f"Gagal menyimpan metrik versi {version_str}: {e}")

    def load_version_metrics(self, version_str):
        with self.lock:
            path = f"data/metrics_{version_str}.json"
            if os.path.exists(path):
                try:
                    import math
                    def sanitize_floats(obj):
                        if isinstance(obj, float):
                            if math.isnan(obj) or math.isinf(obj):
                                return None
                            return obj
                        elif isinstance(obj, dict):
                            return {k: sanitize_floats(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [sanitize_floats(x) for x in obj]
                        return obj

                    with open(path, "r") as f:
                        data = json.load(f)
                        return sanitize_floats(data)
                except Exception as e:
                    self.logger.error(f"Gagal memuat metrik versi {version_str}: {e}")
            return None

    def reset_training_metrics(self):
        with self.lock:
            self.metrics["accuracy"] = 0.0
            self.metrics["loss"] = 0.0
            self.metrics["training_duration_s"] = 0.0
            self.metrics["compute_energy_kwh"] = 0.0
            self.metrics["download_volume_mb"] = 0.0
            self.metrics["epoch_history"] = []
            self.update_logs("[INFO] Metrik pelatihan di-reset untuk sesi pelatihan baru.")

    def get_status(self, db=None):
        # Mengembalikan status lengkap server untuk dashboard UI
        with self.lock:
            active_clients = []
            if db:
                clients = db.query(models.Client).all()
                
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
                    
                    c_data = self.registered_clients.get(c.edge_id, {})
                    dataset_name = c_data.get("dataset", "students")
                    is_prep = c_data.get("is_preprocessed", False)
                    avail_ds = c_data.get("available_datasets", {dataset_name: is_prep})
                    
                    active_clients.append({
                        "id": c.edge_id,
                        "ip": c.ip_address,
                        "status": status.upper(),
                        "last_seen": wib_time_str,
                        "dataset": dataset_name,
                        "is_preprocessed": is_prep,
                        "available_datasets": avail_ds
                    })
                    
                if db_needs_commit:
                    try:
                        db.commit()
                    except Exception as e:
                        db.rollback()
                        self.logger.error(f"Gagal commit status client terupdate: {e}")
                        
                try:
                    # Sinkronkan versi model dengan database
                    self.model_version = db.query(models.ModelVersion).count()
                    latest_version = db.query(models.ModelVersion).order_by(models.ModelVersion.version_id.desc()).first()
                    if latest_version and latest_version.notes:
                        notes = latest_version.notes
                        parts = [p.strip() for p in notes.split("|")]
                        dataset = "students"
                        epochs = 0
                        for p in parts:
                            if p.startswith("Dataset:"):
                                dataset = p.split(":")[1].strip()
                            elif p.startswith("Epochs:"):
                                epochs = p.split(":")[1].strip()
                        self.model_version_str = f"cl_{dataset}_{epochs}e"
                    else:
                        self.model_version_str = "v0"
                except: 
                    self.model_version_str = "v0"

            # Kumpulkan semua dataset unik yang dilaporkan oleh seluruh client
            all_datasets = set()
            for cid, c_data in self.registered_clients.items():
                avail = c_data.get("available_datasets", {})
                for ds in avail.keys():
                    all_datasets.add(ds)
                ds = c_data.get("dataset")
                if ds:
                    all_datasets.add(ds)
            all_datasets.add("students")
            all_datasets_list = sorted(list(all_datasets))

            return {
                "is_running": self.is_running,
                "current_phase": self.current_phase,
                "upload_requested": self.upload_requested,
                "metrics": self.metrics,
                "current_logs": self.current_logs,
                "received_data": self.received_data,
                "model_version": self.model_version_str,
                "default_epochs": self.default_epochs,
                "default_batch_size": self.default_batch_size,
                "inference_threshold": self.inference_threshold,
                "uptime": int(time.time() - self.start_time) if self.start_time > 0 else 0,
                "active_clients": active_clients,
                "available_datasets": all_datasets_list,
                "current_dataset": self.current_dataset,
                "inference_logs": self.metrics.get("inference_logs", [])
            }