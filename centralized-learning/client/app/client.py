import os
import time
import threading
import requests
from .utils.mobilefacenet import MobileFaceNet
from .controllers.management import ManagementController
from .controllers.attendance import AttendanceController

class CentralizedClientManager:
    # Manajer Utama Terminal Terpusat (Centralized)
    # Menangani proses registrasi, sinkronisasi model, dan pengunggahan dataset.
    
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
        # Menjalankan loop sinkronisasi di thread latar belakang
        threading.Thread(target=self._background_sync, daemon=True).start()

    def _background_sync(self):
        # Loop utama untuk memastikan terminal selalu sinkron dengan server
        print(f"[INFO] Memulai sinkronisasi latar belakang ({self.client_id}).")
        while True:
            try:
                # 1. Registrasi Terminal ke Server
                if not self.is_registered:
                    if self.management.register_client(self.client_id):
                        self.is_registered = True
                        print(f"[OK] Terminal berhasil terdaftar di server.")
                    else: 
                        time.sleep(5)
                        continue

                # 2. Cek Versi Model dan Sinkronisasi Aset
                try:
                    res = requests.get(f"{self.server_url}/ping", timeout=3)
                    if res.status_code == 200:
                        server_info = res.json()
                        server_version = server_info.get("model_version", 0)
                        upload_requested = server_info.get("upload_requested", False)
                        
                        # Sinkronisasi jika versi lokal tertinggal
                        if not self.has_assets or server_version > self.current_model_version:
                            print(f"[INFO] Sinkronisasi model (Lokal: v{self.current_model_version}, Server: v{server_version}).")
                            success, refs = self.management.sync_assets(self.model)
                            if success:
                                self.has_assets = True
                                self.reference_embeddings = refs
                                self.current_model_version = server_version
                                print(f"[OK] Model dan basis data referensi v{server_version} berhasil diperbarui.")
                            else:
                                time.sleep(10)
                                continue
                        
                        # 3. Menangani Permintaan Pengunggahan Dataset
                        if upload_requested:
                            print(f"[INFO] Server meminta unggah data dataset.")
                            success, msg = self.management.package_and_upload()
                            if success:
                                print(f"[OK] Unggah data berhasil. Menunggu proses training selesai...")
                                time.sleep(60) # Beri jeda lebih lama jika baru saja mengunggah
                            else:
                                print(f"[ERROR] Gagal unggah data: {msg}")

                except Exception as e:
                    print(f"[ERROR] Gagal komunikasi dengan server (Ping/Sync): {e}")

            except Exception as e:
                print(f"[ERROR] Terjadi kesalahan pada loop latar belakang: {e}")

            time.sleep(5)