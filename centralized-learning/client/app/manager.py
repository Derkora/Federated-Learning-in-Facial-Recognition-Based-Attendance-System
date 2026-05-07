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
from app.utils.logging import init_logger, get_logger
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
        
        # 1. Inisialisasi Logger (Gunakan ID mentah dulu untuk tag)
        temp_id = self._get_raw_id()
        self.log_path = os.path.join(self.data_path, "client_activity.log")
        init_logger(self.log_path, tag=f"CL-CLIENT-{temp_id}")
        self.logger = get_logger()
        
        # 2. Muat Identitas Persisten Resmi
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
        self.camera_enabled = False # Status logis (Desired State)
        self.is_camera_running = False # Status aktual thread hardware

        self.current_model_version = self._load_version()
        
        # EAGER LOADING: Muat model ke RAM segera saat inisialisasi
        self.logger.info("Memulai Eager Loading model...")
        self._reload_inference_models(force_reload=True)
        
        # Pengaturan Kamera
        self.device = DEVICE
        self.prediction_buffer = deque(maxlen=10)
        self.last_face_time = 0
        self.latest_frame = None
        self.latest_result = {"matched": "Standby", "confidence": 0, "latency_ms": 0, "is_virtual": False}
        self.is_camera_running = False
        self.threshold = 0.7
        
        self.logger.info("=== CL Client Started / Restarted ===")

    def _get_raw_id(self):
        """Ambil ID tanpa logging (untuk inisialisasi logger)."""
        id_path = os.path.join(self.data_path, "client_id.txt")
        if os.path.exists(id_path):
            with open(id_path, "r") as f:
                cid = f.read().strip()
                if cid: return cid
        return os.getenv("HOSTNAME", f"client-{int(time.time())}")

    def _log_to_file(self, message):

        """Wrapper log untuk kompatibilitas kode lama."""
        self.logger.info(message)

    def _load_identity(self):
        """Memuat atau membuat identitas unik client yang tersimpan di volume data."""
        id_path = os.path.join(self.data_path, "client_id.txt")
        if os.path.exists(id_path):
            with open(id_path, "r") as f:
                cid = f.read().strip()
                if cid:
                    self.logger.info(f"Memuat ID persisten: {cid}")
                    return cid
        
        # Jika belum ada, gunakan HOSTNAME (Container ID) atau fallback
        new_id = os.getenv("HOSTNAME", f"client-{int(time.time())}")
        try:
            with open(id_path, "w") as f:
                f.write(new_id)
            self.logger.info(f"Mendaftarkan ID persisten baru: {new_id}")
        except Exception as e:
            self.logger.error(f"Gagal menyimpan identitas: {e}")
        
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
                self.logger.error(f"Gagal memuat bobot model dari disk: {e}")
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
                # Muat bobot global secara penuh (termasuk BN stats)
                # Client CL tidak melakukan pelatihan, jadi lebih aman menggunakan stats server sebagai basis.
                loaded_sd = torch.load(path_m, map_location=DEVICE)
                new_model.load_state_dict(loaded_sd)
                self.logger.info(f"Bobot global v{self.current_model_version} (Full State) diterapkan.")

            if os.path.exists(path_r):
                new_refs = torch.load(path_r, map_location=DEVICE)

            # 2. ATOMIC SWAP & BN ADAPTATION
            self.model = new_model
            self.reference_embeddings = new_refs
            self.has_assets = True
            
            # Kalibrasi BN Lokal agar cocok dengan domain kamera perangkat ini
            try:
                from app.utils.freezing import calibrate_bn
                calibrate_bn(self.model, self.raw_data_path)
            except Exception as e:
                self.logger.warn(f"Gagal kalibrasi BN (CL): {e}")
            
            gc.collect()
            self.logger.success(f"Model CL v{self.current_model_version} berhasil dimuat ke RAM.")
        except Exception as e:
            self.logger.error(f"Reload model CL gagal: {e}")

    def start_background_tasks(self):
        # Menjalankan loop sinkronisasi dan kamera di thread latar belakang
        threading.Thread(target=self.run_background_sync, daemon=True).start()
        threading.Thread(target=self.run_camera_loop, daemon=True).start()

    def toggle_camera(self):
        """Menyalakan atau mematikan fungsi kamera (Hardware & Browser) secara dinamis."""
        if self.camera_enabled:
            self.logger.info("User mematikan fungsi kamera (OFF).")
            self.camera_enabled = False
            self.is_camera_running = False
            self.latest_frame = None 
            self.latest_result["matched"] = "CAMERA OFF"
            return False
        else:
            self.logger.info("User menyalakan fungsi kamera (ON).")
            self.camera_enabled = True
            
            # Jika hardware belum jalan, coba nyalakan threadnya
            if not (hasattr(self, "_camera_thread") and self._camera_thread.is_alive()):
                self.is_camera_running = True
                self._camera_thread = threading.Thread(target=self.run_camera_loop, args=(True,), daemon=True)
                self._camera_thread.start()
            return True


    def run_camera_loop(self, manual=False):
        # PENTING: Cek apakah kamera diaktifkan via .env
        if not self.camera_enabled:
            enable_camera = os.getenv("ENABLE_CAMERA", "false").lower() == "true"
            if not enable_camera:
                self.logger.info("Kamera dalam posisi PADAM (Default).")
                self.is_camera_running = False
                self.latest_result["matched"] = "CAMERA OFF"
                self.camera_enabled = False
                return
            else:
                self.camera_enabled = True

        # Loop kamera mandiri (Headless Mode)
        cam_idx = self.camera_index
        cam_width = int(os.getenv("CAMERA_WIDTH", 1280))
        cam_height = int(os.getenv("CAMERA_HEIGHT", 720))
        cam_format = os.getenv("CAMERA_FORMAT", "MJPG").upper()

        self.logger.info(f"Mencoba akses hardware kamera (Mulai dari Index {cam_idx})...")
        cap = cv2.VideoCapture(cam_idx)
        
        # LOGIKA SMART SCAN: Cari index 0 s/d 3 jika index utama gagal
        if not cap or not cap.isOpened():
            found = False
            for i in [0, 1, 2]:
                if i == cam_idx: continue
                if cap: cap.release()
                self.logger.info(f"Mencoba alternatif Index {i}...")
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cam_idx = i
                    found = True
                    break
            
            # FINAL FALLBACK: Mode Simulasi (Jika hardware benar-benar tidak ada di Docker)
            if not found:
                self.logger.warn("Hardware tidak ditemukan. Masuk ke MODE SIMULASI (Virtual).")
                test_video = os.path.join(self.data_path, "test_video.mp4")
                if os.path.exists(test_video):
                    cap = cv2.VideoCapture(test_video)
                    self.latest_result["is_virtual"] = True
                else:
                    if cap: cap.release()
                    self.logger.error("Tidak ada hardware maupun file simulasi.")
                    self.is_camera_running = False
                    self.latest_result["matched"] = "CAMERA ERROR"
                    return
        
        # Optimasi Jetson/Raspi: Paksa format MJPG jika dikonfigurasi (lebih ringan dari YUYV)
        if cam_format == "MJPG":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        
        # Beri waktu hardware inisialisasi
        time.sleep(1)

        if not cap.isOpened():
            self.logger.error(f"Gagal akses hardware pada index {cam_idx}.")
            self.is_camera_running = False
            self.latest_result["matched"] = "CAMERA ERROR"
            return

        self.is_camera_running = True
        self.logger.success(f"Akses hardware berhasil (Index {cam_idx}).")
        
        while self.is_camera_running:
            # Re-check enable_camera in case we want to stop
            if not self.is_camera_running: break

            ret, frame = cap.read()
            
            if not ret:
                self.logger.error(f"Gagal membaca frame dari kamera {cam_idx}. Mencoba ulang dlm 5 detik...")
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(cam_idx)
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
            
            # Lakukan pemrosesan jika model sudah siap dan sudah ditraining (v1+)
            if self.model and self.current_model_version > 0:
                start_time = time.time()
                try:
                    # Konversi ke PIL Image
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    matched, confidence, _ = self.attendance.process_inference(
                        img_pil, self.model, self.reference_embeddings
                    )
                    
                    # Cek lagi sebelum update result agar tidak menimpa status "CAMERA OFF"
                    if self.is_camera_running:
                        self.latest_result = {
                            "matched": matched,
                            "confidence": confidence,
                            "latency_ms": int((time.time() - start_time) * 1000),
                            "model_version": self.current_model_version,
                            "is_virtual": False
                        }

                    
                    # LOGGING: Catat hasil cocok
                    if matched != "Unknown" and matched != "Error":
                        self._log_to_file(f"INFERENCE SUCCESS: {matched} (Conf: {confidence:.4f})")
                except Exception as e:
                    pass
            elif self.current_model_version == 0:
                self.latest_result["matched"] = "MODEL NOT TRAINED (v0)"
                self.latest_result["model_version"] = "v0"
            
            # Explicit cleanup per loop to keep memory stable
            time.sleep(0.5)
            gc.collect()
        
        cap.release()

    def run_background_sync(self):
        # Loop utama untuk memastikan terminal selalu sinkron dengan server
        self.logger.info(f"Memulai sinkronisasi latar belakang ({self.client_id}).")
        while True:
            try:
                # 1. Registrasi Terminal ke Server
                if not self.is_registered:
                    ip = socket.gethostbyname(socket.gethostname())
                    if self.management.register_client(ip):
                        if not self.is_registered:
                            self._log_to_file(f"SUCCESS: Terminal terdaftar pada server (IP: {ip})")
                        self.is_registered = True
                        self.last_sync_success = True
                    else: 
                        time.sleep(5)
                        continue


                # 2. Cek Versi Model dan Sinkronisasi Aset
                try:
                    res = requests.get(f"{self.server_url}/ping", timeout=3)
                    if res.status_code == 200:
                        server_info = res.json()
                        server_version = server_info.get("model_version", 0)
                        server_threshold = server_info.get("inference_threshold", 0.7)
                        upload_requested = server_info.get("upload_requested", False)
                        
                        # Sinkronisasi threshold
                        if server_threshold != self.threshold:
                            self.logger.info(f"Threshold diperbarui: {server_threshold}")
                            self.threshold = server_threshold
                        
                        # Sinkronisasi jika versi lokal tertinggal
                        if not self.has_assets or server_version > self.current_model_version:
                            self._log_to_file(f"SYNC: Memperbarui model (Lokal: v{self.current_model_version}, Server: v{server_version})")
                            
                            # Pastikan model sudah terinisialisasi (Lazy Load Guard)
                            if self.model is None:
                                self._ensure_models_loaded()
                                if self.model is None:
                                    self.logger.info("Inisialisasi model MobileFaceNet dasar untuk sinkronisasi...")
                                    self.model = MobileFaceNet().to(DEVICE)
                                    
                            success, refs = self.management.sync_assets(self.model)
                            if success:
                                self.has_assets = True
                                self.reference_embeddings = refs
                                self.current_model_version = server_version
                                self._save_version(server_version)
                                # EAGER RELOAD: Pastikan model terbaru masuk RAM (Sudah termasuk BN Adaptation)
                                self._reload_inference_models(force_reload=True)
                            else:
                                time.sleep(10)
                                continue
                        
                        # 3. Menangani Permintaan Pengunggahan Dataset
                        if upload_requested:
                            if not self.is_training_phase:
                                self._log_to_file("UPLOAD: Menerima instruksi unggah dataset dari server.")
                            self.is_training_phase = True 
                            
                            success, msg = self.management.package_and_upload()
                            if success:
                                self._log_to_file("SUCCESS: Dataset berhasil diunggah ke server.")
                                time.sleep(60) 
                            else:
                                self._log_to_file(f"ERROR: Gagal mengunggah dataset: {msg}")
                        else:
                            self.is_training_phase = False # Kembali ke mode normal

                except Exception as e:
                    if getattr(self, 'last_sync_success', True):
                        self.logger.error(f"Gagal komunikasi dengan server (Ping/Sync): {e}")
                        self.last_sync_success = False
                    self.is_registered = False


            except Exception as e:
                self.logger.error(f"Terjadi kesalahan fatal pada loop latar belakang: {e}")

            time.sleep(5)