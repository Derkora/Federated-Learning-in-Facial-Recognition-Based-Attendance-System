import os
import time
import threading
import requests
from app.utils.mobilefacenet import MobileFaceNet
from app.controllers.management import ManagementController
from app.controllers.attendance import AttendanceController

class CentralizedClientManager:
    def __init__(self):
        self.server_url = os.getenv("CL_SERVER_ADDRESS", "http://server-cl:8080")
        self.client_id = os.getenv("HOSTNAME", "client-unknown")
        
        self.management = ManagementController(self.server_url, self.client_id)
        self.attendance = AttendanceController(self.server_url, self.client_id)
        
        self.model = MobileFaceNet()
        self.reference_embeddings = {}
        self.is_registered = False
        self.has_assets = False
        self.current_model_version = 0

    def start_background_tasks(self):
        print(f"[{self.client_id}] Triggering Background Tasks Thread...", flush=True)
        threading.Thread(target=self._background_sync, daemon=True).start()

    def _background_sync(self):
        print(f"[{self.client_id}] Starting Background Sync Loop...", flush=True)
        while True:
            try:
                # 1. Registration
                if not self.is_registered:
                    if self.management.register_client(self.client_id):
                        self.is_registered = True
                        print(f"[{self.client_id}] Client Registered.", flush=True)
                    else: 
                        time.sleep(5)
                        continue

                # 2. Asset Sync / Check Version
                try:
                    res = requests.get(f"{self.server_url}/ping", timeout=3)
                    if res.status_code == 200:
                        server_info = res.json()
                        server_version = server_info.get("model_version", 0)
                        upload_requested = server_info.get("upload_requested", False)
                        
                        # Sync if never synced OR version mismatch
                        if not self.has_assets or server_version > self.current_model_version:
                            print(f"[{self.client_id}] Syncing Assets (Server v{server_version}, Local v{self.current_model_version})...", flush=True)
                            success, refs = self.management.sync_assets(self.model)
                            if success:
                                self.has_assets = True
                                self.reference_embeddings = refs
                                self.current_model_version = server_version
                                print(f"[{self.client_id}] Model and References Synced to v{server_version}.", flush=True)
                            else:
                                time.sleep(10)
                                continue
                        
                        # 3. Training Polling (using already fetched info)
                        if upload_requested:
                            print(f"[{self.client_id}] Server requested data. Packaging...", flush=True)
                            success, msg = self.management.package_and_upload()
                            if success:
                                print(f"[{self.client_id}] Data upload success. Waiting for training...", flush=True)
                                time.sleep(60)
                            else:
                                print(f"[{self.client_id}] Data upload failed: {msg}", flush=True)

                except Exception as e:
                    print(f"[{self.client_id}] Ping/Sync Error: {e}", flush=True)

            except Exception as e:
                print(f"[{self.client_id}] Error in background loop: {e}", flush=True)

            time.sleep(5)