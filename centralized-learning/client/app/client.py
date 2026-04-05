import os
import time
import threading
import requests
import cv2
import numpy as np
from PIL import Image
from app.utils.mobilefacenet import MobileFaceNet
from app.controllers.management import ManagementController
from app.controllers.attendance import AttendanceController

class CentralizedClientManager:
    # Manajer Utama Terminal Terpusat (Centralized)
    # Menangani proses registrasi, sinkronisasi model, dan pengunggahan dataset.
    
    def __init__(self):
        self.server_url = os.getenv("CL_SERVER_ADDRESS", "http://server-cl:8080")
        self.client_id = os.getenv("HOSTNAME", "client-unknown")
        self.raw_data_path = os.getenv("RAW_DATA_PATH", "raw_data")
        
        self.management = ManagementController(self.server_url, self.client_id)
        self.attendance = AttendanceController(self.server_url, self.client_id)
        
        self.model = MobileFaceNet()
        self.reference_embeddings = {}
        self.is_registered = False
        self.has_assets = False
        self.current_model_version = self._load_version()
        
        # Load local assets if available
        self._load_local_assets()
        
        # Headless Camera Support
        self.latest_frame = None
        self.latest_result = {"matched": "Standby", "confidence": 0, "latency_ms": 0, "is_virtual": False}
        self.is_camera_running = False

    def _load_version(self):
        v_path = os.path.join("app/data", "models", "model_version.txt")
        if os.path.exists(v_path):
            with open(v_path, "r") as f:
                try: return int(f.read().strip())
                except: return 0
        return 0

    def _save_version(self, v):
        os.makedirs("app/data/models", exist_ok=True)
        v_path = os.path.join("app/data", "models", "model_version.txt")
        with open(v_path, "w") as f:
            f.write(str(v))

    def _load_local_assets(self):
        path_m = "app/model/global_model.pth"
        path_r = "app/model/reference_embeddings.pth"
        if os.path.exists(path_m) and os.path.exists(path_r):
            try:
                from app.utils.image_processing import DEVICE
                self.model.load_state_dict(torch.load(path_m, map_location=DEVICE))
                self.model.eval()
                self.reference_embeddings = torch.load(path_r, map_location=DEVICE)
                self.has_assets = True
                print(f"[INIT] Loaded local model v{self.current_model_version} and references.")
            except Exception as e:
                print(f"[INIT ERROR] Failed to load local assets: {e}")

    def start_background_tasks(self):
        # Menjalankan loop sinkronisasi dan kamera di thread latar belakang
        threading.Thread(target=self._background_sync, daemon=True).start()
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        # Loop kamera mandiri (Headless Mode)
        print(f"[CAMERA] Menjalankan loop kamera otomatis...")
        
        # Coba buka kamera asli (Cek index 0 dan 1)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
            
        self.is_camera_running = True
        virtual_mode = not cap.isOpened()
        if virtual_mode:
            print("[CAMERA] Tidak ada hardware terdeteksi. Menggunakan Mode Virtual (Foto).")
            
        virtual_images = []
        virtual_idx = 0
        
        while self.is_camera_running:
            ret, frame = False, None
            if not virtual_mode:
                ret, frame = cap.read()
                if not ret:
                    print("[CAMERA ERROR] Gagal akses hardware. Beralih ke VIRTUAL CAMERA mode.")
                    virtual_mode = True
                    # Scan for virtual images
                    root_dir = os.path.join(self.raw_data_path, "students")
                    if os.path.exists(root_dir):
                        for root, dirs, files in os.walk(root_dir):
                            for f in files:
                                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    virtual_images.append(os.path.join(root, f))
                    if not virtual_images:
                        print("[VIRTUAL ERROR] Tidak ada dataset lokal untuk simulasi.")
            
            if virtual_mode and virtual_images:
                img_path = virtual_images[virtual_idx]
                frame = cv2.imread(img_path)
                if frame is not None:
                    ret = True
                    virtual_idx = (virtual_idx + 1) % len(virtual_images)
                    # Beri penanda virtual di frame
                    cv2.putText(frame, "VIRTUAL MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                time.sleep(1.0) # Lambatkan simulasi agar tidak spam
            
            if not ret:
                if not virtual_mode: print("[CAMERA ERROR] Gagal membaca frame dari kamera. Cek mapping device /dev/video0.")
                time.sleep(5)
                continue
            
            # Simpan frame terbaru untuk streaming MJPEG
            self.latest_frame = frame.copy()
            
            # Lakukan pemrosesan jika model sudah siap (Inference)
            if self.has_assets:
                start_time = time.time()
                try:
                    # Konversi ke PIL Image
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    matched, confidence = self.attendance.recognize_and_submit(
                        img_pil, self.model, self.reference_embeddings
                    )
                    
                    self.latest_result = {
                        "matched": matched,
                        "confidence": confidence,
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "model_version": self.current_model_version,
                        "is_virtual": virtual_mode
                    }
                except Exception as e:
                    print(f"[CAMERA ERROR] Pemrosesan gagal: {e}")
            
            # Beri jeda agar tidak memakan CPU terlalu tinggi (FPS ~2-5 cukup untuk presensi)
            if not virtual_mode: time.sleep(0.5)
        
        cap.release()

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
                                self._save_version(server_version)
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