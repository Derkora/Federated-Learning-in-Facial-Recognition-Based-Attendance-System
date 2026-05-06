import os
import time
import json
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session
from .db.db import SessionLocal

from .db import models
from .db.models import Client
from .config import ECONOMICS, TRAINING_PARAMS
from .utils.logging import init_logger, get_logger

class CentralizedServerManager:
# ... (rest of class)
    # Manajer Utama Dashboard Server Terpusat
    # Melacak status pelatihan, fase aktif, dan metrik performa model global.
    
    def __init__(self):
        self.is_running = False
        self.current_phase = "Standby"
        self.upload_requested = False
        self.start_time = 0
        self.received_data = [] 
        self.uploader_map = {} 
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
            "epoch_history": [], # Riwayat {"epoch": i, "loss": l, "accuracy": a}
            "inference_logs": [] # Log detail untuk riset FAR/TAR
        }
        self.settings_path = "data/settings_cl.json"
        self.inference_logs_path = "data/inference_logs.json"
        
        # Default dari config
        self.default_epochs = TRAINING_PARAMS["epochs"]
        self.default_batch_size = TRAINING_PARAMS["batch_size"]
        self.inference_threshold = 0.7
        
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
                # 1. Cek Koneksi & Muat Versi
                version_count = db.query(models.ModelVersion).count()
                self.model_version = version_count
                self.logger.info(f"Database terhubung. Ditemukan {version_count} versi model.")

                # 2. Muat Riwayat Pelatihan
                rounds = db.query(models.TrainingRound).order_by(models.TrainingRound.round_id.asc()).all()
                self.logger.info(f"Menemukan {len(rounds)} ronde pelatihan di database.")
                
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
        try:
            logs = self.metrics.get("inference_logs", [])
            os.makedirs(os.path.dirname(self.inference_logs_path), exist_ok=True)
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

    def start_phase(self, phase_name):
        # Menandai awal dari fase alur kerja penelitian
        self.is_running = True
        self.current_phase = phase_name
        if phase_name == "Import Data": self.start_time = time.time()
        self.update_logs(f"Fase {phase_name} dimulai.")

    def end_phase(self):
        # Mengembalikan status ke Standby setelah fase selesai
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
        if os.path.exists(upload_dir):
            self.received_data = [d for d in os.listdir(upload_dir) if os.path.isdir(os.path.join(upload_dir, d))]

    def register_upload(self, edge_id, nrp_list):
        # Mencatat siapa yang mengunggah data apa untuk atribusi yang akurat
        for nrp in nrp_list:
            self.uploader_map[nrp] = edge_id

    def increment_version(self, dbs: Session):
        # Sinkronisasi versi model dari database untuk keakuratan dashboard
        try:
            version_count = dbs.query(models.ModelVersion).count()
            self.model_version = version_count
            self.update_logs(f"Versi Model Global naik ke v{self.model_version}")
        except Exception as e:
            self.model_version += 1
            self.update_logs(f"Gagal sinkronisasi DB, fallback increment: v{self.model_version}")

    def update_metrics(self, new_data):
        # Memperbarui metrik performa dan estimasi biaya
        # Penggabungan riwayat epoch tanpa menimpa data yang lama
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

    def save_training_round(self, db, round_num, loss, accuracy, val_loss=None, val_accuracy=None, duration=0, energy=0, upload=0, download=0, start_time=None):
        # Menyimpan hasil ronde ke Database Postgres
        try:
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
                start_time=start_time or datetime.now(timezone(timedelta(hours=7)))
            )
            db.add(new_round)
            db.commit()
            self.logger.success(f"Ronde {round_num} berhasil disimpan ke database.")
        except Exception as e:
            self.logger.error(f"Gagal menyimpan ronde ke database: {e}")
            db.rollback()

    def get_status(self, db=None):
        # Mengembalikan status lengkap server untuk dashboard UI
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
                    
            try:
                # Sinkronkan versi model dengan database
                self.model_version = db.query(models.ModelVersion).count()
            except: pass

        return {
            "is_running": self.is_running,
            "current_phase": self.current_phase,
            "upload_requested": self.upload_requested,
            "metrics": self.metrics,
            "current_logs": self.current_logs,
            "received_data": self.received_data,
            "model_version": self.model_version,
            "default_epochs": self.default_epochs,
            "default_batch_size": self.default_batch_size,
            "inference_threshold": self.inference_threshold,
            "uptime": int(time.time() - self.start_time) if self.start_time > 0 else 0,
            "active_clients": active_clients,
            "inference_logs": self.metrics.get("inference_logs", [])
        }