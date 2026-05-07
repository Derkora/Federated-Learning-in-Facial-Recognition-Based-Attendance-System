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
        init_logger(self.log_path, tag="CL-SERVER")
        self.logger = get_logger()
        
        self.metrics = {
            "upload_volume_mb": 0,
            "download_volume_mb": 0,
            "transmission_cost_idr": 0,
            "training_duration_s": 0,
            "total_round_time_s": 0,
            "compute_energy_kwh": 0,
            "compute_cost_idr": 0,
            "epoch_history": [] # Riwayat {"epoch": i, "loss": l, "accuracy": a}
        }
        self.settings_path = "data/settings_cl.json"
        
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
                        history.append({
                            "epoch": r.round_number if r.round_number is not None else 0,
                            "loss": float(r.global_loss) if r.global_loss is not None else 0.0,
                            "accuracy": float(r.global_accuracy) if r.global_accuracy is not None else 0.0
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

                # 3. Sinkronisasi Metrik Ekonomi
                self.update_metrics({})
                self.logger.info("Sinkronisasi metrik selesai. Server Siap.")
                db.close()
                return # SUCCESS
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
                    self.inference_threshold = s.get("threshold", self.inference_threshold)
                    self.logger.info(f"Settings loaded from {self.settings_path}")
            except Exception as e:
                self.logger.error(f"Failed to load settings: {e}")

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
        # KHUSUS: Jika ada epoch_history baru, jangan timpa yang lama, tapi gabungkan.
        if "epoch_history" in new_data:
            new_history = new_data.pop("epoch_history")
            existing_epochs = [h["epoch"] for h in self.metrics["epoch_history"]]
            for h in new_history:
                if h["epoch"] not in existing_epochs:
                    self.metrics["epoch_history"].append(h)
        
        self.metrics.update(new_data)
        
        # 1. Transmisi: Upload + Download
        upload_mb = self.metrics.get("upload_volume_mb", 0)
        download_mb = self.metrics.get("download_volume_mb", 0)
        total_mb = upload_mb + download_mb
        
        # Hitung biaya berdasarkan per-MB (Rp 3,25 / MB)
        self.metrics["transmission_cost_idr"] = round(total_mb * 3.25, 2)
        
        # 2. Komputasi: kWh -> IDR
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

    def save_training_round(self, db, round_num, loss, accuracy, duration=0, energy=0, upload=0, download=0):
        # Menyimpan hasil ronde ke Database Postgres
        try:
            from .db import models
            new_round = models.TrainingRound(
                round_number=round_num,
                global_loss=float(loss),
                global_accuracy=float(accuracy),
                training_duration_s=float(duration),
                compute_energy_kwh=float(energy),
                upload_volume_mb=float(upload),
                download_volume_mb=float(download),
                start_time=datetime.now(timezone(timedelta(hours=7)))
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
            for c in clients:
                active_clients.append({
                    "id": c.edge_id,
                    "ip": c.ip_address,
                    "status": (c.status or "offline").upper(),
                    "last_seen": c.last_seen.strftime("%H:%M:%S") if c.last_seen else "-"
                })
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
            "active_clients": active_clients
        }