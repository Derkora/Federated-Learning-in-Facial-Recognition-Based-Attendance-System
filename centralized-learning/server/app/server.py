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

    def end_phase(self):
        self.is_busy = False
        self.current_phase = "Standby"

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
            "uptime": int(time.time() - self.start_time) if self.start_time > 0 else 0
        }