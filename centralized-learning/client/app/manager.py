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
from app.controllers.management import ManagementController, api_ping
from app.controllers.attendance import AttendanceController

class CLClientManager:
    # Manajer Utama Client Terpusat (Centralized)
    # Menangani proses registrasi, sinkronisasi model, dan pengunggahan dataset.
    
    def __init__(self):
        self.server_url = os.getenv("CL_SERVER_ADDRESS", "http://server-cl:8080")
        self.data_path = os.getenv("DATA_PATH", "/app/data")
        os.makedirs(os.path.join(self.data_path, "models"), exist_ok=True)
        
        # Inisialisasi Logger (Gunakan ID mentah dulu untuk tag)
        temp_id = self._get_raw_id()
        self.log_path = os.path.join(self.data_path, "client_activity.log")
        init_logger(self.log_path, tag=f"CL-CLIENT-{temp_id}")
        self.logger = get_logger()
        
        # Muat Identitas Persisten Resmi
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
        self.is_adapting = False # Status proses adaptasi lokal

        self.current_model_version = self._load_version()
        self.is_manual_lock = self._load_manual_lock()
        
        # Muat model ke RAM segera saat inisialisasi
        self.logger.info("Memulai Eager Loading model...")
        self._reload_inference_models(force_reload=True)
        
        # Pengaturan Kamera
        self.device = DEVICE
        self.prediction_buffer = deque(maxlen=5)
        self.last_face_time = 0
        self.latest_frame = None
        self.latest_result = {"matched": "Standby", "confidence": 0, "latency_ms": 0, "is_virtual": False}
        self.is_camera_running = False
        self.threshold = 0.7
        
        self.logger.info("=== Klien CL Dimulai / Dimulai Ulang ===")

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

    def get_processed_path(self, dataset_name: str):
        if dataset_name == "students":
            return os.path.join(self.data_path, "processed")
        return os.path.join(self.data_path, f"processed_{dataset_name}")

    def check_dataset_preprocessed(self, dataset_name: str):
        processed_dir = self.get_processed_path(dataset_name)
        if not os.path.exists(processed_dir):
            return False
        try:
            subdirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
            if not subdirs:
                return False
            for sd in subdirs:
                sd_path = os.path.join(processed_dir, sd)
                files = os.listdir(sd_path)
                if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                    return True
            return False
        except Exception:
            return False

    def get_available_datasets(self):
        datasets = {}
        try:
            if os.path.exists(self.raw_data_path):
                subdirs = sorted([
                    d for d in os.listdir(self.raw_data_path)
                    if os.path.isdir(os.path.join(self.raw_data_path, d)) and not d.startswith('.')
                ])
                for sd in subdirs:
                    datasets[sd] = self.check_dataset_preprocessed(sd)
            if not datasets:
                datasets["students"] = self.check_dataset_preprocessed("students")
        except Exception as e:
            self.logger.error(f"Error get_available_datasets: {e}")
            datasets["students"] = False
        return datasets

    def get_dataset_name(self):
        try:
            from app.main import current_dataset
            return current_dataset
        except Exception:
            return "students"

    def is_preprocessed(self):
        return self.check_dataset_preprocessed(self.get_dataset_name())

    def _load_version(self):
        v_path = os.path.join(self.data_path, "models", "model_version.txt")
        if os.path.exists(v_path):
            with open(v_path, "r") as f:
                content = f.read().strip()
                try: return int(content)
                except: return content
        return 0

    def _save_version(self, v):
        try:
            os.makedirs(os.path.join(self.data_path, "models"), exist_ok=True)
            v_path = os.path.join(self.data_path, "models", "model_version.txt")
            temp_path = v_path + ".tmp"
            with open(temp_path, "w") as f:
                f.write(str(v))
            os.replace(temp_path, v_path)
            self.logger.info(f"Berhasil menyimpan versi model terbaru: v{str(v).lstrip('v')}")
        except Exception as e:
            self.logger.error(f"Gagal menyimpan model_version.txt secara aman: {e}")

    def _load_manual_lock(self):
        lock_path = os.path.join(self.data_path, "models", "manual_lock.txt")
        if os.path.exists(lock_path):
            try:
                with open(lock_path, "r") as f:
                    return f.read().strip().lower() == "true"
            except: pass
        return False

    def _save_manual_lock(self, locked: bool):
        try:
            os.makedirs(os.path.join(self.data_path, "models"), exist_ok=True)
            lock_path = os.path.join(self.data_path, "models", "manual_lock.txt")
            temp_path = lock_path + ".tmp"
            with open(temp_path, "w") as f:
                f.write("true" if locked else "false")
            os.replace(temp_path, lock_path)
            self.logger.info(f"Berhasil menyimpan status manual lock: {locked}")
        except Exception as e:
            self.logger.error(f"Gagal menyimpan manual_lock.txt secara aman: {e}")

    def _ensure_models_loaded(self, force_reload=False):
        """Menjamin instance model tersedia di RAM (Tanpa paksa reload dari disk jika sudah ada)."""
        if self.model is not None and not force_reload:
            return True
        self._reload_inference_models(force_reload=True)
        return self.model is not None

    def _reload_inference_models(self, force_reload=False):
        """EAGER LOAD: Memuat ulang model ke RAM secara thread-safe."""
        try:
            if not force_reload and self.has_assets:
                return

            # Muat ke variabel lokal
            new_model = MobileFaceNet().to(DEVICE).eval()
            new_refs = {}
            
            version = self.current_model_version
            if version and version != "v0" and version != "0" and version != 0:
                path_m = os.path.join(self.data_path, "models", f"global_model_{version}.pth")
                path_r = os.path.join(self.data_path, "models", f"reference_embeddings_{version}.pth")
            else:
                path_m = os.path.join(self.data_path, "models", "global_model.pth")
                path_r = os.path.join(self.data_path, "models", "reference_embeddings.pth")

            # Fallback ke global_model jika berkas versi tidak ditemukan
            if not os.path.exists(path_m):
                path_m = os.path.join(self.data_path, "models", "global_model.pth")
                path_r = os.path.join(self.data_path, "models", "reference_embeddings.pth")

            # Muat global model dan reference
            if os.path.exists(path_m):
                loaded_sd = torch.load(path_m, map_location=DEVICE)
                new_model.load_state_dict(loaded_sd)
                self.logger.info(f"Bobot global v{str(self.current_model_version).lstrip('v')} (Full State) diterapkan.")
                if os.path.exists(path_r):
                    new_refs = torch.load(path_r, map_location=DEVICE)
                    self.logger.info("Menggunakan database referensi embedding dari server CL.")

            # Transisi update model inferensi ke RAM
            self.model = new_model
            self.reference_embeddings = new_refs
            self.has_assets = True
            
            gc.collect()
            self.logger.success(f"Model CL v{str(self.current_model_version).lstrip('v')} berhasil dimuat ke RAM.")
                
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
        # Cek status kamera aktif sebelum memulai proses capture
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

        self.is_camera_running = True
        cap = None
        is_virtual = False
        last_real_cam_check = 0

        while self.is_camera_running:
            # Re-check enable_camera in case we want to stop
            if not self.camera_enabled:
                self.logger.info("User mematikan fungsi kamera.")
                self.is_camera_running = False
                self.latest_result["matched"] = "CAMERA OFF"
                if cap:
                    cap.release()
                return

            now = time.time()
            # If camera is not opened or is virtual, try to find/probe real hardware camera every 10 seconds
            if cap is None or not cap.isOpened() or is_virtual:
                if cap is None or not cap.isOpened() or (now - last_real_cam_check > 10):
                    last_real_cam_check = now
                    # Scan indices: self.camera_index, then 0, 1, 2
                    indices = [self.camera_index]
                    for idx in [0, 1, 2]:
                        if idx not in indices:
                            indices.append(idx)

                    found_hardware = False
                    for i in indices:
                        self.logger.info(f"Mencoba memindai hardware kamera pada Index {i}...")
                        test_cap = cv2.VideoCapture(i)
                        if test_cap and test_cap.isOpened():
                            # We found a real hardware camera!
                            # If we were in virtual mode, release the virtual capture first
                            if cap:
                                cap.release()
                            cap = test_cap
                            cam_idx = i
                            self.camera_index = i
                            found_hardware = True
                            is_virtual = False
                            self.latest_result["is_virtual"] = False
                            self.logger.success(f"Akses hardware kamera berhasil dideteksi (Index {i}).")
                            
                            # Configure hardware properties
                            if cam_format == "MJPG":
                                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
                            time.sleep(1) # Hardware warmup
                            break
                        else:
                            if test_cap:
                                test_cap.release()
                    
                    if not found_hardware:
                        if cap is None or not cap.isOpened():
                            # No camera connected, and no simulation running yet: try to load simulation
                            test_video = os.path.join(self.data_path, "test_video.mp4")
                            if os.path.exists(test_video):
                                self.logger.warn("Hardware tidak ditemukan. Masuk ke MODE SIMULASI (Virtual).")
                                cap = cv2.VideoCapture(test_video)
                                is_virtual = True
                                self.latest_result["is_virtual"] = True
                            else:
                                self.logger.error("Tidak ada hardware maupun file simulasi. Mencoba mencari perangkat keras lagi dlm 5 detik...")
                                self.latest_result["matched"] = "CAMERA ERROR"
                                time.sleep(5)
                                continue

            # Read frame
            ret, frame = cap.read()
            if not ret:
                if is_virtual:
                    # Rewind simulation video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.logger.error(f"Gagal membaca frame dari kamera {cam_idx}. Mencoba memindai ulang dlm 5 detik...")
                    if cap:
                        cap.release()
                    cap = None
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
            
            # Lakukan pemrosesan jika model sudah siap dan sudah ditraining (v0+)
            if self.model and self.current_model_version >= 0:
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
                            "is_virtual": is_virtual
                        }

                    
                    # Catat log hanya jika hasil identifikasi cocok
                    if matched != "Unknown" and matched != "Error":
                        self._log_to_file(f"INFERENCE SUCCESS: {matched} (Conf: {confidence:.4f})")
                        
                    # Explicit PIL closure and dereferencing to prevent memory leaks
                    img_pil.close()
                    del img_rgb, img_pil
                except Exception as e:
                    pass
            elif self.current_model_version == 0:
                self.latest_result["matched"] = "MODEL NOT TRAINED (v0)"
                self.latest_result["model_version"] = "v0"
            
            # Explicit cleanup per loop to keep memory stable
            time.sleep(0.5)
            gc.collect()
        
        if cap:
            cap.release()

    def run_background_sync(self):
        # Loop utama untuk memastikan terminal selalu sinkron dengan server
        self.logger.info(f"Memulai sinkronisasi latar belakang ({self.client_id}).")
        while True:
            try:
                # Coba kirim ulang presensi dan log offline yang tertunda jika server online kembali
                try:
                    if hasattr(self, 'attendance'):
                        self.attendance.process_offline_queues()
                except Exception:
                    pass
                # Registrasi Terminal ke Server (Heartbeat)
                ip = socket.gethostbyname(socket.gethostname())
                if self.management.register_client(ip, self):
                    if not self.is_registered:
                        self._log_to_file(f"SUCCESS: Terminal terdaftar pada server (IP: {ip})")
                    self.is_registered = True
                    self.last_sync_success = True
                else: 
                    self.is_registered = False


                # Cek Versi Model dan Sinkronisasi Aset
                try:
                    res = requests.get(f"{self.server_url}{api_ping}", timeout=3)
                    if res.status_code == 200:
                        server_info = res.json()
                        server_version = server_info.get("model_version", 0)
                        server_threshold = server_info.get("inference_threshold", 0.7)
                        upload_requested = server_info.get("upload_requested", False)
                        
                        # Sinkronisasi threshold
                        if server_threshold != self.threshold:
                            self.logger.info(f"Threshold diperbarui: {server_threshold}")
                            self.threshold = server_threshold
                        
                        is_manual = self.is_manual_lock

                        # Normalisasi perbandingan versi dengan menghapus prefiks 'v' jika ada
                        def norm_v(v):
                            if v is None:
                                return ""
                            s = str(v).strip().lower()
                            return s[1:] if s.startswith("v") else s

                        norm_server = norm_v(server_version)
                        norm_local = norm_v(self.current_model_version)

                        # Sinkronisasi jika versi lokal tertinggal dan tidak dalam mode manual
                        if not self.has_assets or (not is_manual and norm_server != norm_local):
                            local_display = f"v{norm_local}" if norm_local else "v0"
                            server_display = f"v{norm_server}" if norm_server else "v0"
                            self._log_to_file(f"SYNC: Memperbarui model (Lokal: {local_display}, Server: {server_display})")
                            
                            # Pastikan model sudah terinisialisasi (Lazy Load Guard)
                            if self.model is None:
                                self._ensure_models_loaded()
                                if self.model is None:
                                    self.logger.info("Inisialisasi model MobileFaceNet dasar untuk sinkronisasi...")
                                    self.model = MobileFaceNet().to(DEVICE)
                                    
                            success, refs = self.management.sync_assets(self.model)
                            if success:
                                # Hapus model lokal lama karena versi baru sudah tersedia
                                path_local_m = os.path.join(self.data_path, "models", "local_calibrated_model.pth")
                                path_local_r = os.path.join(self.data_path, "models", "local_reference_embeddings.pth")
                                for p_file in [path_local_m, path_local_r]:
                                    if os.path.exists(p_file):
                                        try: os.remove(p_file)
                                        except: pass
                                
                                self.has_assets = True
                                self.reference_embeddings = refs
                                self.current_model_version = server_version
                                self._save_version(server_version)
                                self.is_manual_lock = False
                                self._save_manual_lock(False)
                                # Terapkan model terbaru langsung ke RAM
                                self._reload_inference_models(force_reload=True)
                            else:
                                time.sleep(10)
                                continue
                        
                        # Menangani Permintaan Pengunggahan Dataset
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

    def select_and_sync_version(self, version: str):
        # Mengunduh model versi spesifik dari server secara manual dan menerapkannya
        self.logger.info(f"Diminta memuat model versi: {version}")
        try:
            url_m = f"{self.server_url}/api/model?version={version}"
            url_r = f"{self.server_url}/api/reference?version={version}"
            
            res_m = requests.get(url_m, timeout=10)
            res_r = requests.get(url_r, timeout=10)
            
            if res_m.status_code != 200:
                return False, f"Gagal mengunduh model versi {version}: {res_m.text}"
            if res_r.status_code != 200:
                return False, f"Gagal mengunduh referensi versi {version}: {res_r.text}"
                
            path_m = os.path.join(self.data_path, "models", f"global_model_{version}.pth")
            path_r = os.path.join(self.data_path, "models", f"reference_embeddings_{version}.pth")
            
            os.makedirs(os.path.dirname(path_m), exist_ok=True)
            with open(path_m, "wb") as f:
                f.write(res_m.content)
            with open(path_r, "wb") as f:
                f.write(res_r.content)
                
            # Update file model_version.txt dengan string versi
            v_path = os.path.join(self.data_path, "models", "model_version.txt")
            with open(v_path, "w") as f:
                f.write(version)
                
            self.current_model_version = version
            self.is_manual_lock = True
            self._save_manual_lock(True)
            self._reload_inference_models(force_reload=True)
            
            return True, "Berhasil"
        except Exception as e:
            self.logger.error(f"Gagal memuat model versi {version}: {e}")
            return False, str(e)