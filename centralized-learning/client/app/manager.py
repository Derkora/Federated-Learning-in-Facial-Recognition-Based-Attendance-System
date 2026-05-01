import os
import time
import threading
import requests
import torch
import socket
import cv2
import numpy as np
import gc
from collections import deque
from PIL import Image
from app.utils.mobilefacenet import MobileFaceNet
from app.utils.preprocessing import image_processor, DEVICE
from app.utils.freezing import calibrate_bn
from app.controllers.management import ManagementController
from app.controllers.attendance import AttendanceController

class CLClientManager:
    # Manajer Utama Client Terpusat (Centralized)
    # Menangani proses registrasi, sinkronisasi model, dan pengunggahan dataset.
    
    def __init__(self):
        self.server_url = os.getenv("CL_SERVER_ADDRESS", "http://server-cl:8080")
        self.data_path = os.getenv("DATA_PATH", "/app/data")
        os.makedirs(os.path.join(self.data_path, "models"), exist_ok=True)
        
        # Muat atau Buat Identitas Persisten
        self.client_id = self._load_identity()
        
        self.raw_data_path = os.getenv("RAW_DATA_PATH", "/app/raw_data")
        self.camera_index = int(os.getenv("CAMERA_INDEX", "0"))
        
        self.management = ManagementController(self.server_url, self.client_id)
        self.attendance = AttendanceController(self)
        
        self.model = None 
        self.reference_embeddings = {}
        self.is_registered = False
        self.has_assets = False
        self.is_training_phase = False # Guard untuk manajemen sumber daya
        self.current_model_version = self._load_version()
        
        # EAGER LOADING: Muat model ke RAM segera saat inisialisasi
        print("[INIT] Memulai Eager Loading model...")
        self._reload_inference_models(force_reload=True)
        
        # Pengaturan Kamera
        self.device = DEVICE
        self.prediction_buffer = deque(maxlen=10)
        self.last_face_time = 0
        self.latest_frame = None
        self.latest_result = {"matched": "Standby", "confidence": 0, "latency_ms": 0, "is_virtual": False}
        self.is_camera_running = False
        self.threshold = 0.75

    def _load_identity(self):
        """Memuat atau membuat identitas unik client yang tersimpan di volume data."""
        id_path = os.path.join(self.data_path, "client_id.txt")
        if os.path.exists(id_path):
            with open(id_path, "r") as f:
                cid = f.read().strip()
                if cid:
                    print(f"[IDENTITY] Memuat ID persisten: {cid}")
                    return cid
        
        # Jika belum ada, gunakan HOSTNAME (Container ID) atau fallback
        new_id = os.getenv("HOSTNAME", f"client-{int(time.time())}")
        try:
            with open(id_path, "w") as f:
                f.write(new_id)
            print(f"[IDENTITY] Mendaftarkan ID persisten baru: {new_id}")
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan identitas: {e}")
        
        return new_id

    def _load_version(self):
        v_path = os.path.join(self.data_path, "models", "model_version.txt")
        if os.path.exists(v_path):
            with open(v_path, "r") as f:
                try: return int(f.read().strip())
                except: return 0
        return 0

    def _save_version(self, v):
        os.makedirs(os.path.join(self.data_path, "models"), exist_ok=True)
        v_path = os.path.join(self.data_path, "models", "model_version.txt")
        with open(v_path, "w") as f:
            f.write(str(v))

    def _ensure_models_loaded(self, force_reload=False):
        """Menjamin instance model tersedia di RAM (Tanpa paksa reload dari disk jika sudah ada)."""
        if self.model is not None and not force_reload:
            return True
            
        path_m = os.path.join(self.data_path, "models", "global_model.pth")
        path_r = os.path.join(self.data_path, "models", "reference_embeddings.pth")
        
        # 1. Inisialisasi instance objek jika belum ada
        if self.model is None:
            self.model = MobileFaceNet().to(DEVICE)
            self.model.eval()

        # 2. Muat dari disk jika diminta atau jika RAM belum terisi bobot
        if os.path.exists(path_m) and os.path.exists(path_r):
            try:
                self.model.load_state_dict(torch.load(path_m, map_location=DEVICE))
                self.model.eval()
                self.reference_embeddings = torch.load(path_r, map_location=DEVICE)
                self.has_assets = True
                return True
            except Exception as e:
                print(f"[ERROR] Gagal memuat bobot model dari disk: {e}")
        return False

    def _reload_inference_models(self, force_reload=False):
        """EAGER LOAD: Memuat ulang model ke RAM secara thread-safe."""
        try:
            if not force_reload and self.has_assets:
                return

            # 1. Muat ke variabel lokal
            new_model = MobileFaceNet().to(DEVICE).eval()
            new_refs = {}
            
            path_m = os.path.join(self.data_path, "models", "global_model.pth")
            path_r = os.path.join(self.data_path, "models", "reference_embeddings.pth")

            if os.path.exists(path_m):
                new_model.load_state_dict(torch.load(path_m, map_location=DEVICE))
            if os.path.exists(path_r):
                new_refs = torch.load(path_r, map_location=DEVICE)

            # 2. ATOMIC SWAP
            self.model = new_model
            self.reference_embeddings = new_refs
            self.has_assets = True
            
            gc.collect()
            print(f"[EAGER LOAD] Model CL v{self.current_model_version} Loaded into RAM")
        except Exception as e:
            print(f"[CRITICAL] Reload model CL gagal: {e}")

    def start_background_tasks(self):
        # Menjalankan loop sinkronisasi dan kamera di thread latar belakang
        threading.Thread(target=self.run_background_sync, daemon=True).start()
        threading.Thread(target=self.run_camera_loop, daemon=True).start()

    def run_camera_loop(self):
        # Loop kamera mandiri (Headless Mode)
        cam_idx = self.camera_index
        cam_width = int(os.getenv("CAMERA_WIDTH", 1280))
        cam_height = int(os.getenv("CAMERA_HEIGHT", 720))
        cam_format = os.getenv("CAMERA_FORMAT", "MJPG").upper()

        print(f"[CAMERA] Mencoba membuka kamera hardware (Index: {cam_idx}, {cam_width}x{cam_height})...")
        cap = cv2.VideoCapture(cam_idx)
        
        # Optimasi Jetson/Raspi: Paksa format MJPG jika dikonfigurasi (lebih ringan dari YUYV)
        if cam_format == "MJPG":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        
        # Beri waktu hardware inisialisasi
        time.sleep(1)
        
        if not cap.isOpened() and cam_idx == 0:
            print(f"[CAMERA] Gagal akses hardware pada index 0. Mencoba index 1...")
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
                    print("[ERROR] Gagal akses hardware. Beralih ke mode VIRTUAL CAMERA.")
                    virtual_mode = True
                    # Scan for virtual images
                    root_dir = os.path.join(self.raw_data_path, "students")
                    if os.path.exists(root_dir):
                        for root, dirs, files in os.walk(root_dir):
                            for f in files:
                                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    virtual_images.append(os.path.join(root, f))
                    if not virtual_images:
                        print("[ERROR] Tidak ada dataset lokal untuk simulasi.")
            
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
                if not virtual_mode: print("[ERROR] Gagal membaca frame dari kamera. Periksa /dev/video0.")
                time.sleep(5)
                continue
            
            # Update status training untuk mematikan beban kerja berat
            if self.is_training_phase:
                self.latest_result["matched"] = "TRAINING..."
                time.sleep(2)
                continue

            # Eager load sudah menjamin model terisi di RAM
            if not self.model and self.has_assets:
                self._reload_inference_models(force_reload=False)

            # Simpan frame terbaru untuk streaming MJPEG (Resized untuk efisiensi RAM/Bandwidth)
            self.latest_frame = cv2.resize(frame, (640, 480))
            
            # Lakukan pemrosesan jika model sudah siap (Inference)
            if self.model:
                start_time = time.time()
                try:
                    # Konversi ke PIL Image
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    matched, confidence = self.attendance.process_inference(
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
                    print(f"[ERROR] Pemrosesan kamera gagal: {e}")
            
            # Beri jeda agar tidak memakan CPU terlalu tinggi (FPS ~2-5 cukup untuk presensi)
            if not virtual_mode: time.sleep(0.5)
        
        cap.release()

    def run_background_sync(self):
        # Loop utama untuk memastikan terminal selalu sinkron dengan server
        print(f"[INFO] Memulai sinkronisasi latar belakang ({self.client_id}).")
        while True:
            try:
                # 1. Registrasi Terminal ke Server
                if not self.is_registered:
                    ip = socket.gethostbyname(socket.gethostname())
                    if self.management.register_client(ip):
                        self.is_registered = True
                        print(f"[SUCCESS] Client berhasil terdaftar pada server.")
                    else: 
                        time.sleep(5)
                        continue

                # 2. Cek Versi Model dan Sinkronisasi Aset
                try:
                    res = requests.get(f"{self.server_url}/ping", timeout=3)
                    if res.status_code == 200:
                        server_info = res.json()
                        server_version = server_info.get("model_version", 0)
                        server_threshold = server_info.get("inference_threshold", 0.75)
                        upload_requested = server_info.get("upload_requested", False)
                        
                        # Sinkronisasi threshold
                        if server_threshold != self.threshold:
                            print(f"[INFO] Threshold diperbarui: {server_threshold}")
                            self.threshold = server_threshold
                        
                        # Sinkronisasi jika versi lokal tertinggal
                        if not self.has_assets or server_version > self.current_model_version:
                            print(f"[INFO] Sinkronisasi model (Lokal: v{self.current_model_version}, Server: v{server_version}).")
                            
                            # Pastikan model sudah terinisialisasi (Lazy Load Guard)
                            if self.model is None:
                                self._ensure_models_loaded()
                                if self.model is None:
                                    print("[INFO] Inisialisasi model MobileFaceNet dasar untuk sinkronisasi...")
                                    self.model = MobileFaceNet().to(DEVICE)
                                    
                            success, refs = self.management.sync_assets(self.model)
                            if success:
                                self.has_assets = True
                                self.reference_embeddings = refs
                                self.current_model_version = server_version
                                self._save_version(server_version)
                                # EAGER RELOAD: Pastikan model terbaru masuk RAM
                                self._reload_inference_models(force_reload=True)
                                
                                # --- BN ADAPTATION (LOCAL CALIBRATION) ---
                                # Membuat model CL setara dengan pFedFace dalam hal adaptasi lingkungan.
                                if self.model:
                                    calibrate_bn(self.model, self.raw_data_path)
                            else:
                                time.sleep(10)
                                continue
                        
                        # 3. Menangani Permintaan Pengunggahan Dataset
                        if upload_requested:
                            self.is_training_phase = True # Masuk mode hemat daya
                            print(f"[INFO] Server meminta unggah dataset.")
                            success, msg = self.management.package_and_upload()
                            if success:
                                print(f"[SUCCESS] Unggah dataset berhasil. Menunggu proses pelatihan selesai...")
                                time.sleep(60) # Beri jeda lebih lama jika baru saja mengunggah
                            else:
                                print(f"[ERROR] Gagal mengunggah dataset: {msg}")
                        else:
                            self.is_training_phase = False # Kembali ke mode normal

                except Exception as e:
                    print(f"[ERROR] Gagal komunikasi dengan server (Ping/Sync): {e}")

            except Exception as e:
                print(f"[ERROR] Terjadi kesalahan fatal pada loop latar belakang: {e}")

            time.sleep(5)