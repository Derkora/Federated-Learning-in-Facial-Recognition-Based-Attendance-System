import os
import time
from app.db.db import SessionLocal
from app.db import models

class CentralizedServerManager:
    def __init__(self):
        self.is_busy = False
        self.current_phase = "Standby"
        self.upload_requested = False
        self.start_time = 0
        self.current_logs = []
        self.received_data = [] # List of student NRPs
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
        self.is_busy = True
        self.current_phase = phase_name
        if phase_name == "Import Data": self.start_time = time.time()
        self.update_logs(f"Phase {phase_name} started.")

    def end_phase(self):
        self.is_busy = False
        self.current_phase = "Standby"

    def update_logs(self, msg):
        self.current_logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        # Keep last 100 logs
        if len(self.current_logs) > 100: self.current_logs.pop(0)

    def update_received_data(self, upload_dir):
        if os.path.exists(upload_dir):
            self.received_data = [d for d in os.listdir(upload_dir) if os.path.isdir(os.path.join(upload_dir, d))]

    def increment_version(self):
        self.model_version += 1
        self.update_logs(f"Global Model version incremented to {self.model_version}")

    def update_metrics(self, new_data):
        self.metrics.update(new_data)
        # Recalculate cost if payload or duration changes
        payload = self.metrics.get("payload_size_mb", 0)
        duration = self.metrics.get("total_round_time_s", 0)
        self.metrics["cost_idr"] = int((payload / 1024 * 5000) + (duration * 10))

    def get_status(self):
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