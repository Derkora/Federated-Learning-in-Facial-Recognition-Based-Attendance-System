import os
import time
from .db.db import SessionLocal
from .db import models
from .config import ECONOMICS

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
        # Mencatat aktivitas sistem ke dalam log dashboard
        ts = time.strftime('%H:%M:%S')
        self.current_logs.append(f"[{ts}] {msg}")
        if len(self.current_logs) > 100: self.current_logs.pop(0)
        print(f"SERVER LOG: {msg}")

    def update_received_data(self, upload_dir):
        # Mendata NRP mahasiswa yang datanya berhasil diterima dari terminal
        if os.path.exists(upload_dir):
            self.received_data = [d for d in os.listdir(upload_dir) if os.path.isdir(os.path.join(upload_dir, d))]

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
            "uptime": int(time.time() - self.start_time) if self.start_time > 0 else 0
        }