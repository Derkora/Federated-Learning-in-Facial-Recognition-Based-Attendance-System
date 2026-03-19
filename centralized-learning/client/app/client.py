import os
import time
import threading
from app.utils.mobilefacenet import MobileFaceNet
from app.controllers.management import ManagementController
from app.controllers.attendance import AttendanceController

class CentralizedClientManager:
    def __init__(self):
        self.server_url = os.getenv("CL_SERVER_ADDRESS", "http://localhost:8000")
        self.client_id = os.getenv("HOSTNAME", "client-unknown")
        
        self.management = ManagementController(self.server_url, self.client_id)
        self.attendance = AttendanceController(self.server_url, self.client_id)
        
        self.model = MobileFaceNet()
        self.reference_embeddings = {}
        self.is_registered = False
        self.has_assets = False

    def start_background_tasks(self):
        threading.Thread(target=self._background_sync, daemon=True).start()

    def _background_sync(self):
        print(f"[{self.client_id}] Starting Background Sync...")
        while True:
            if not self.is_registered:
                if self.management.register_client("127.0.0.1"):
                    self.is_registered = True
                    print(f"[{self.client_id}] Client Registered.")
                else: 
                    time.sleep(5)
                    continue

            if not self.has_assets:
                success, refs = self.management.sync_assets(self.model)
                if success:
                    self.has_assets = True
                    self.reference_embeddings = refs
                    print(f"[{self.client_id}] Model and References Synced.")
                else:
                    time.sleep(10)
                    continue
            
            time.sleep(60)