import os
import time
import json
from datetime import datetime, timedelta, timezone
from .db.db import SessionLocal
from .db import models
from .config import ECONOMICS, TRAINING_PARAMS

class CentralizedServerManager:
    # Manajer Utama Dashboard Server Terpusat
    # Melacak status pelatihan, fase aktif, dan metrik performa model global.
    
    def __init__(self):
        self.is_running = False
        self.current_phase = "Standby"
        self.upload_requested = False
        self.start_time = 0
        self.current_logs = []
        self.received_data = [] 
        self.uploader_map = {} 
        self.model_version = 0
        
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
        self.default_epochs = TRAINING_PARAMS["total_epochs"]
        self.default_batch_size = TRAINING_PARAMS["batch_size"]
        self.inference_threshold = 0.50
        self.load_settings()

    def load_settings(self):
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, 'r') as f:
                    s = json.load(f)
                    self.default_epochs = s.get("epochs", self.default_epochs)
                    self.default_batch_size = s.get("batch_size", self.default_batch_size)
                    self.inference_threshold = s.get("threshold", self.inference_threshold)
                    print(f"[OK] Settings loaded from {self.settings_path}")
            except Exception as e:
                print(f"[ERROR] Failed to load settings: {e}")

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
        # Mencatat aktivitas sistem ke dalam log dashboard (WIB Sync)
        tz_wib = timezone(timedelta(hours=7))
        ts = datetime.now(tz_wib).strftime('%H:%M:%S')
        self.current_logs.append(f"[{ts}] {msg}")
        if len(self.current_logs) > 100: self.current_logs.pop(0)
        print(f"SERVER LOG: {msg}")

    def update_received_data(self, upload_dir):
        # Mendata NRP mahasiswa yang datanya berhasil diterima dari terminal
        if os.path.exists(upload_dir):
            self.received_data = [d for d in os.listdir(upload_dir) if os.path.isdir(os.path.join(upload_dir, d))]

    def register_upload(self, edge_id, nrp_list):
        # Mencatat siapa yang mengunggah data apa untuk atribusi yang akurat
        for nrp in nrp_list:
            self.uploader_map[nrp] = edge_id

    def increment_version(self):
        # Meningkatkan versi model global setelah pelatihan selesai
        self.model_version += 1
        self.update_logs(f"Versi Model Global naik ke v{self.model_version}")

    def update_metrics(self, new_data):
        # Memperbarui metrik performa dan estimasi biaya
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

    def get_status(self):
        # Mengembalikan status lengkap server untuk dashboard UI
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
            "uptime": int(time.time() - self.start_time) if self.start_time > 0 else 0
        }