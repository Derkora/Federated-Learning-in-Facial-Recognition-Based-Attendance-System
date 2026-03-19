import os
import time
from app.db.db import SessionLocal
from app.db import models

class CentralizedServerManager:
    def __init__(self):
        self.is_training = False
        self.last_metrics = {
            "accuracy": 0,
            "tar": 0,
            "far": 0,
            "eer": 0,
            "payload_size_mb": 0,
            "total_duration_s": 0,
            "cost_idr": 0
        }
        self.start_time = 0

    def start_training_flow(self, training_controller):
        """Orchestrates the full centralized process."""
        self.is_training = True
        self.start_time = time.time()
        
        db = SessionLocal()
        try:
            # Trigger all clients first (Logic stays in training_controller for now or move here)
            # For consistency with FL, we can keep the high-level orchestration here
            result = training_controller.train_pipeline(db)
            self.last_metrics.update(result)
            return result
        finally:
            self.is_training = False
            db.close()

    def get_status(self):
        return {
            "is_training": self.is_training,
            "metrics": self.last_metrics,
            "uptime": int(time.time() - self.start_time) if self.start_time > 0 else 0
        }