import os
import time
from .db.db import SessionLocal
from .db import models

class CentralizedServerManager:
    # Manajer Utama Dashboard Server Terpusat
    # Melacak status pelatihan, fase aktif, dan metrik performa model global.
    
    def __init__(self):
        self.is_busy = False
        self.current_phase = "Standby"
        self.upload_requested = False
        self.start_time = 0
        self.current_logs = []
        self.received_data = [] 
        self.model_version = 0
        
        self.metrics = {
            "accuracy": 0,
            "tar": 0,
            "far": 0,
            "eer": 0,
            "payload_size_mb": 0,
            "training_duration_s": 0,
            "total_round_time_s": 0,
            "cost_idr": 0
        }

    def start_phase(self, phase_name):
        # Menandai awal dari fase alur kerja penelitian
        self.is_busy = True
        self.current_phase = phase_name
        if phase_name == "Import Data": self.start_time = time.time()
        self.update_logs(f"Fase {phase_name} dimulai.")

    def end_phase(self):
        # Mengembalikan status ke Standby setelah fase selesai
        self.is_busy = False
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
        # Memperbarui metrik performa dan estimasi biaya komputasi
        self.metrics.update(new_data)
        payload = self.metrics.get("payload_size_mb", 0)
        duration = self.metrics.get("total_round_time_s", 0)
        # Estimasi biaya sederhana (contoh alokasi resource cloud)
        self.metrics["cost_idr"] = int((payload / 1024 * 5000) + (duration * 10))

    def get_status(self):
        # Mengembalikan status lengkap server untuk dashboard UI
        return {
            "is_busy": self.is_busy,
            "current_phase": self.current_phase,
            "upload_requested": self.upload_requested,
            "metrics": self.metrics,
            "current_logs": self.current_logs,
            "received_data": self.received_data,
            "model_version": self.model_version,
            "uptime": int(time.time() - self.start_time) if self.start_time > 0 else 0
        }