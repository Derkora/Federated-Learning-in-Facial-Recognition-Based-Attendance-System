import os
import json
import torch
import threading
import collections
import requests
import traceback
import time
import socket
import io
import base64
import shutil
import gc
import cv2
import numpy as np
import flwr as fl
from PIL import Image
from app.utils.preprocessing import image_processor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from app.utils.logging import init_logger, get_logger
from app.utils.sync_utils import process_offline_queue, process_offline_inference_logs
from app.utils.freezing import calibrate_bn
from app.utils.security import encryptor
from app.utils.classifier import identify_user_globally
from app.utils.mobilefacenet import MobileFaceNet, ArcMarginProduct
from app.db.db import SessionLocal
from app.db.models import UserLocal, EmbeddingLocal
from app.recognition_client import FaceRecognitionClient
from app.controllers.attendance import AttendanceController

# Konfigurasi endpoint API
api_clients_register = "/api/clients/register"
api_clients_ready = "/api/clients/ready"
api_clients_discovery_done = "/api/clients/discovery_done"
api_training_identities = "/api/training/identities"
api_training_status = "/api/training/status"
api_training_get_label = "/api/training/get_label"
api_training_label_map = "/api/training/label_map"
api_training_registry_assets = "/api/training/registry_assets"
api_status = "/api/status"
api_model_backbone = "/api/model/backbone"
api_model_bn = "/api/model/bn"
api_model_registry = "/api/model/registry"

class FLClientManager:
    # Manajer Utama Client Federated (FL)
    # Menangani inti operasional client: sinkronisasi fase, manajemen model lokal, hingga alur kamera.
    def __init__(self):
        self.raw_data_path = os.getenv("RAW_DATA_PATH", "/app/raw_data")
        self.data_path = os.getenv("DATA_PATH", "/app/data")
        # Pastikan direktori tersedia
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(os.path.join(self.data_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, "processed"), exist_ok=True)

        self.device = torch.device("cpu")
        self.backbone = None 
        self.head = None     
        self.inference_backbone = None 
        self.inference_head = None     
        
        # Inisialisasi Logger (Gunakan ID mentah dulu untuk tag)
        temp_id = self._get_raw_id()
        self.log_path = os.path.join(self.data_path, "client_activity.log")
        init_logger(self.log_path, tag=f"FL-CLIENT-{temp_id}")
        self.logger = get_logger()
        
        # Muat Identitas Persisten Resmi
        self.client_id = self._load_identity()

        
        # Tentukan ukuran head dinamis dan versi model
        self.save_path = os.path.join(self.data_path, "models", "backbone.pth")
        self._models_loaded = False  # Cache flag: True berarti backbone sudah diload ke RAM, skip disk reload
        self.head_path = os.path.join(self.data_path, "models", "local_head.pth")
        self.version_path = os.path.join(self.data_path, "models", "model_version.txt")
        self.map_path = os.path.join(self.data_path, "models", "label_map.json")
        
        self.num_classes = 1000 # Default/Seeding awal (Akan menyusut otomatis saat sync)
        self.model_version = 0
        
        # Muat Versi (Prioritas Disk)
        if os.path.exists(self.version_path):
            try:
                with open(self.version_path, 'r') as f:
                    self.model_version = int(f.read().strip())
                self.logger.info(f"Terdeteksi Model Versi: v{self.model_version}")
            except: pass

        # Muat Jumlah Kelas dari Label Map (Sumber Kebenaran Utama)
        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, 'r') as f:
                    data = json.load(f)
                    self.num_classes = len(data)
                    self.logger.info(f"Terdeteksi {self.num_classes} identitas dari label map lokal.")
            except: pass
        elif os.path.exists(self.head_path):
            try:
                checkpoint = torch.load(self.head_path, map_location="cpu")
                if "weight" in checkpoint:
                    self.num_classes = checkpoint["weight"].shape[0]
                    self.logger.info(f"Terdeteksi {self.num_classes} kelas dari head yang tersimpan.")
            except: pass

        # Guard model initialization placeholder
        self.backbone = None # Lazy loaded
        self.head = None # Lazy loaded
        self.is_training_phase = False # Resource Guard
        self.camera_enabled = False # Status logis (Desired State)
        self.is_camera_running = False # Status aktual thread hardware


        self.fl_server_address = os.getenv("FL_SERVER_ADDRESS", "server-fl:8085")
        self.server_api_url = os.getenv("SERVER_API_URL", "http://server-fl:8080")
        
        self.client = FaceRecognitionClient(
            None, None, 
            data_path=self.data_path, 
            device=self.device
        )
        self.client.fl_manager = self
        
        self.logger.info("Memulai pemuatan ulang model inferensi (Eager Loading)...")
        self._reload_inference_models(force_reload=True)
        
        # Sinkronisasi Status Terakhir dari Persistensi
        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, 'r') as f:
                    data = json.load(f)
                    self.client.label_map = data
                    if hasattr(self.client, 'trainer'):
                        self.client.trainer.nrp_to_idx = {nrp: idx for idx, nrp in enumerate(data)}
                        self.logger.info(f"Label map trainer dipulihkan ({len(data)} identitas).")
            except: pass
        
        self.is_training = False
        self.current_phase = "idle"
        self.fl_status = "Online (Menunggu Instruksi)"
        self.fl_round = 0 
        self.last_phase = "idle"
        self.is_preprocessing_active = False
        
        self.is_registered = False
        self.last_register_attempt = 0
        self.register_retry_delay = 30
        
        self.prediction_buffer = collections.deque(maxlen=5)
        self.last_face_time = 0
        
        # Dukungan Kamera Headless
        self.camera_index = int(os.getenv("CAMERA_INDEX", 0))
        self.latest_frame = None
        self.latest_result = {"matched": "Standby", "confidence": 0, "latency_ms": 0, "is_virtual": False}
        self.is_camera_running = False
        
        # Ambang Batas Inferensi (Dinamis dari Server)
        self.inference_threshold = 0.7 # Parity with CL
        
        # Inisialisasi Kontroler Absensi (FIX: Agar tidak error di run_camera_loop)
        self.attendance = AttendanceController(self)
        
        self._log("=== FL Client Started / Restarted ===")

    def _log(self, message):
        """Wrapper log untuk kompatibilitas kode lama."""
        self.logger.info(message)

    def _log_to_file(self, message):
        """Wrapper log untuk kompatibilitas kode lama dengan Timezone WIB."""
        self.logger.info(message)

    def _safe_request(self, method, url, max_retries=3, **kwargs):
        """Helper to perform requests with retry logic."""
        for attempt in range(max_retries):
            try:
                # Set a default timeout if not provided
                if 'timeout' not in kwargs:
                    kwargs['timeout'] = 30
                
                response = requests.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except Exception as e:
                # self._log(f"[RETRY {attempt+1}/{max_retries}] Gagal {method} {url}: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2)
        return None

    def register_to_server(self):
        """Mendaftarkan client ke server API."""
        try:
            response = self._safe_request("POST", f"{self.server_api_url}{api_clients_register}", json={
                "client_id": self.client_id,
                "ip_address": socket.gethostbyname(socket.gethostname()),
                "port": int(os.getenv("CLIENT_PORT", 8080))
            })
            if response and response.status_code == 200:
                self._log(f"[SUCCESS] Client {self.client_id} berhasil terdaftar di server.")
                return True
        except Exception as e:
            self._log(f"[ERROR] Gagal registrasi ke server: {str(e)}")
        return False


    def _get_raw_id(self):
        """Ambil ID tanpa logging (untuk inisialisasi logger)."""
        id_path = os.path.join(self.data_path, "client_id.txt")
        if os.path.exists(id_path):
            with open(id_path, "r") as f:
                cid = f.read().strip()
                if cid: return cid
        return os.getenv("HOSTNAME", f"client-{int(time.time())}")

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


    def _sync_global_identities(self):
        """
        Sinkronisasi data NRP, Nama, dan Embedding Mahasiswa dari server.
        """
        try:
            self.logger.info("Sinkronisasi informasi identitas global dari server...")
            res = self._safe_request("GET", f"{self.server_api_url}{api_training_identities}")
            if res and res.status_code == 200:
                identities = res.json()
                db = SessionLocal()
                try:
                    sync_count = 0
                    emb_count = 0
                    for item in identities:
                        nrp = item['nrp']
                        name = item['name']
                        emb_b64 = item.get('embedding')
                        
                        user = db.query(UserLocal).filter_by(user_id=nrp).first()
                        if not user:
                            new_user = UserLocal(user_id=nrp, name=name)
                            db.add(new_user)
                            sync_count += 1
                        else:
                            if user.name != name:
                                user.name = name
                                sync_count += 1
                        
                        # Simpan/Update Embedding Global
                        if emb_b64:
                            emb_bytes = base64.b64decode(emb_b64)
                            # Cek apakah sudah ada embedding global untuk user ini
                            existing_emb = db.query(EmbeddingLocal).filter_by(user_id=nrp, is_global=True).first()
                            if not existing_emb:
                                new_emb = EmbeddingLocal(user_id=nrp, embedding_data=emb_bytes, is_global=True)
                                db.add(new_emb)
                                emb_count += 1
                            else:
                                existing_emb.embedding_data = emb_bytes
                                emb_count += 1
                                
                    db.commit()
                    self.logger.info(f"Berhasil sinkronisasi {sync_count} identitas dan {emb_count} embedding global.")
                except Exception as db_err:
                    self.logger.error(f"Gagal sinkronisasi basis data: {db_err}")
                    db.rollback()
                finally:
                    db.close()
        except Exception as e:
            self.logger.error(f"Gagal mengambil identitas dari server: {e}")

    def _apply_backbone_weights(self, loaded, ignore_bn=True, target_model=None):
        """Menerapkan bobot ke backbone secara aman dengan filter pFedFace (Local BN)."""
        try:
            model = target_model if target_model is not None else self.backbone
            if model is None:
                model = MobileFaceNet().to(self.device).eval()
                if target_model is None: self.backbone = model

            new_sd = model.state_dict()
            
            # Jika input adalah LIST (Flower/Numpy Arrays)
            if isinstance(loaded, list):
                all_keys = list(new_sd.keys())
                if ignore_bn:
                    shared_keys = [k for k in all_keys if not any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked']) and any(x in k.lower() for x in ['weight', 'bias'])]
                else:
                    shared_keys = all_keys
                
                if len(shared_keys) == len(loaded):
                    for k, v in zip(shared_keys, loaded):
                        new_sd[k] = torch.from_numpy(v).to(self.device)
                else:
                    self.logger.error(f"Ketidakcocokan bobot: {len(loaded)} vs {len(shared_keys)}")
                    return False
            
            # Jika input adalah DICT (State Dict dari .pth)
            else:
                for k, v in loaded.items():
                    if ignore_bn and any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked']):
                        continue # SKIP: Biarkan statistik BN tetap lokal
                    if k in new_sd:
                        new_sd[k] = v.to(self.device)
            
            model.load_state_dict(new_sd, strict=False)
            self.logger.success(f"Bobot backbone berhasil diterapkan (BN Diabaikan: {ignore_bn})")
            return True
        except Exception as e:
            self.logger.error(f"Gagal menerapkan bobot model: {e}")
            return False

    def _ensure_models_loaded(self, force_reload=False):
        """Menjamin instance model tersedia di RAM (Tanpa paksa reload dari disk jika sudah ada)."""
        # Selaraskan num_classes dari label map lokal (Sumber Kebenaran)
        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, 'r') as f:
                    self.num_classes = len(json.load(f))
            except: pass

        # Inisialisasi instance objek jika belum ada
        if self.backbone is None:
            self.backbone = MobileFaceNet().to(self.device)
            self.backbone.eval()
            
        if self.head is None:
            self.head = ArcMarginProduct(128, self.num_classes).to(self.device)
            self.head.eval()

        # Sinkronisasi referensi (PENTING untuk Flower Client dan Trainer)
        if hasattr(self, 'client'):
            self.client.model = self.backbone
            self.client.head = self.head
            if hasattr(self.client, 'trainer'):
                self.client.trainer.backbone = self.backbone
                self.client.trainer.head = self.head

        # Muat dari disk jika diminta atau jika RAM belum terisi bobot
        if not self._models_loaded or force_reload:
            if os.path.exists(self.save_path):
                try:
                    loaded = torch.load(self.save_path, map_location=self.device)
                    self._apply_backbone_weights(loaded, ignore_bn=True)
                except Exception as e:
                    self.logger.error(f"Ketidakcocokan berkas backbone: {e}. Menghapus file rusak.")
                    try:
                        os.remove(self.save_path)
                    except: pass
                    
            if os.path.exists(self.head_path):
                try:
                    self.head.load_state_dict(torch.load(self.head_path, map_location=self.device))
                except Exception as e:
                    # Jika mismatch, rekonstruksi lewat trainer
                    try:
                        checkpoint = torch.load(self.head_path, map_location=self.device)
                        disk_classes = checkpoint['weight'].shape[0]
                        self.num_classes = disk_classes
                        new_label_map = {nrp: idx for idx, nrp in enumerate(self.client.label_map)} if hasattr(self.client, 'label_map') and self.client.label_map else {}
                        self.head = self.client.trainer.update_head(self.num_classes, new_label_map)
                        self.head.load_state_dict(checkpoint)
                    except Exception as fallback_err:
                        self.logger.error(f"Berkas local head rusak atau tidak kompatibel: {fallback_err}. Menghapus berkas.")
                        try:
                            os.remove(self.head_path)
                        except: pass
            
            self._models_loaded = True
        
        return True

    def _reload_inference_models(self, force_reload=False):
        """EAGER LOAD: Memuat ulang model ke RAM secara thread-safe khusus untuk inferensi."""
        try:
            if not force_reload and self.inference_backbone is not None:
                return
                
            # Muat ke variabel lokal (Aman terhadap balapan camera loop)
            new_backbone = MobileFaceNet().to(self.device).eval()
            new_head = None
            if self.num_classes > 0:
                new_head = ArcMarginProduct(128, self.num_classes).to(self.device).eval()
            
            if os.path.exists(self.save_path):
                self.logger.info(f"Memuat backbone dari {self.save_path}...")
                loaded = torch.load(self.save_path, map_location=self.device)
                self._apply_backbone_weights(loaded, ignore_bn=False, target_model=new_backbone)
                if self.model_version > 0:
                    try:
                        calibrate_bn(new_backbone, self.raw_data_path, device=self.device, num_samples=50)
                    except Exception as e:
                        self.logger.warn(f"Gagal kalibrasi BN: {e}")
                else:
                    self.logger.info("Model adalah v0 (Pre-trained). Melewati kalibrasi BN untuk mempertahankan performa awal.")

            if new_head is not None and os.path.exists(self.head_path):
                try:
                    new_head.load_state_dict(torch.load(self.head_path, map_location=self.device))
                except: pass

            # Update model inferensi agar terhindar dari ketidakstabilan
            self.inference_backbone = new_backbone
            self.inference_head = new_head
            
            # Jika ini reload paksa atau awal, sinkronkan juga model training
            if force_reload or self.backbone is None:
                self.backbone = new_backbone
                self.head = new_head
                self._models_loaded = True
                if hasattr(self, 'client'):
                    self.client.model = self.backbone
                    self.client.head = self.head

            # Invalidate Cache Identitas & RAM Cleanup
            if hasattr(self, 'cached_refs'):
                self.cached_refs = {}
            if hasattr(self, 'prediction_buffer'):
                self.prediction_buffer.clear()
            self.last_cache_update = 0
            gc.collect()
            
            self.logger.success(f"Mode Inferensi: Global v{self.model_version} Siap.")
        except Exception as e:
            self.logger.error(f"Gagal melakukan Eager Load: {e}")
            self.logger.error(traceback.format_exc())

    def start_background_tasks(self):
        self.logger.info(f"Memulai tugas latar belakang untuk client: {self.client_id}")
        threading.Thread(target=self.run_background_sync, daemon=True).start()
        # Biarkan run_camera_loop mengecek status awal (PADAM/NYALA)
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
                self._camera_thread = threading.Thread(target=self.run_camera_loop, daemon=True)
                self._camera_thread.start()
            return True


    def run_camera_loop(self):
        # Cek status kamera aktif sebelum memulai proses capture
        if not self.camera_enabled:
            enable_camera = os.getenv("ENABLE_CAMERA", "false").lower() == "true"
            if not enable_camera:
                self.logger.info("Kamera dalam posisi PADAM (Default). Aktifkan via ENABLE_CAMERA=true atau tombol UI.")
                self.latest_result["matched"] = "CAMERA OFF"
                self.camera_enabled = False
                return
            else:
                self.camera_enabled = True

        self.is_camera_running = True

        # Loop kamera mandiri (Headless Mode)
        cam_idx = self.camera_index
        cam_width = int(os.getenv("CAMERA_WIDTH", 1280))
        cam_height = int(os.getenv("CAMERA_HEIGHT", 720))
        cam_format = os.getenv("CAMERA_FORMAT", "MJPG").upper()

        cap = None
        is_virtual = False
        last_real_cam_check = 0

        # Gunakan instance terpusat dari pengelola absensi
        attendance_engine = self.attendance
        
        while self.is_camera_running:
            # Re-check enable_camera in case it changes or we want to stop
            if not self.camera_enabled:
                self.logger.info("User mematikan fungsi kamera.")
                self.is_camera_running = False
                self.latest_result["matched"] = "CAMERA OFF"
                if cap:
                    cap.release()
                return

            # Training/Preprocessing Guard - Skip camera access during training
            if self.is_training_phase:
                if cap and cap.isOpened():
                    self.logger.info("Melepaskan hardware kamera untuk menghemat daya/RAM selama training.")
                    cap.release()
                    cap = None
                self.latest_result["matched"] = "TRAINING PHASE..."
                time.sleep(5)
                continue

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

            # Inisialisasi model hanya jika diperlukan (Lazy Load)
            if not self.backbone:
                self._ensure_models_loaded()

            # Simpan frame terbaru untuk streaming MJPEG
            self.latest_frame = frame.copy()
            
            # Lakukan pemrosesan jika model sudah siap dan sudah ditraining (v0+)
            # Gunakan inference_backbone agar tidak terganggu drift training ronde
            if self.inference_backbone and self.model_version >= 0:
                start_time = time.time()
                try:
                    # Konversi ke PIL Image
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    matched, confidence = self.attendance.recognize_directly(img_pil)
                    
                    status_str = f"{self.model_version}"
                    if self.is_training and self.fl_round > 0:
                        status_str += f" (R{self.fl_round}/15)"

                    # Cek lagi sebelum update result agar tidak menimpa status "CAMERA OFF"
                    if self.is_camera_running:
                        self.latest_result = {
                            "matched": matched,
                            "confidence": confidence,
                            "latency_ms": int((time.time() - start_time) * 1000),
                            "model_version": status_str,
                            "is_virtual": is_virtual
                        }

                    
                    # Catat log hanya jika hasil identifikasi cocok
                    if matched != "Unknown" and matched != "Error":
                        self._log_to_file(f"INFERENCE SUCCESS: {matched} (Conf: {confidence:.4f})")
                    
                    # Explicit PIL closure and dereferencing to prevent memory lingering
                    img_pil.close()
                    del img_rgb, img_pil
                except Exception as e:
                    pass
            elif self.model_version == 0:
                self.latest_result["matched"] = "MODEL NOT TRAINED (v0)"
                self.latest_result["model_version"] = "0"
            
            # Explicit cleanup per loop to keep memory stable
            time.sleep(0.5)
            gc.collect()
        
        if cap:
            cap.release()


    def report_status(self, status=None):
        if status:
            self.fl_status = status
            self.logger.info(f"STATUS UPDATE: {status}")
        now = time.time()
        
        try:
            self.last_register_attempt = now
            payload = {
                "id": self.client_id,
                "ip_address": socket.gethostbyname(socket.gethostname()),
                "port": int(os.getenv("CLIENT_PORT", 8080)),
                "fl_status": self.fl_status,
                "last_seen": now
            }
            response = self._safe_request("POST", f"{self.server_api_url}{api_clients_register}", json=payload, timeout=2)
            if response and response.status_code == 200:
                if not self.is_registered:
                    self.logger.success(f"Client {self.client_id} berhasil terdaftar di server.")
                self.is_registered = True
                self.last_sync_success = True

        except Exception as e:
            # self._log(f"[DEBUG] Heartbeat fail: {e}")
            self.is_registered = False

    def run_background_sync(self):
        self.logger.info(f"Layanan sinkronisasi (heartbeat) dimulai untuk {self.client_id}")
        while True:
            try:
                # Coba proses antrean offline tertunda jika server online kembali
                try:
                    process_offline_queue()
                except Exception:
                    pass
                try:
                    process_offline_inference_logs()
                except Exception:
                    pass

                self.report_status()
                
                # Cek Versi Model
                resp = self._safe_request("GET", f"{self.server_api_url}{api_training_status}")
                if resp and resp.status_code == 200:
                    data = resp.json()
                    phase = data.get("current_phase", "idle")
                    server_version = data.get("model_version", 0)
                    
                    # Sinkronisasi threshold dari server
                    if "inference_threshold" in data:
                        self.inference_threshold = float(data["inference_threshold"])
                    
                    # Update status ketersediaan update
                    if server_version > self.model_version:
                        self.fl_status = f"Update v{server_version} Tersedia"
                        # Jika sistem diam dan versi tertinggal, lakukan sinkronisasi akhir
                        if phase == "idle" and self.last_phase == "idle":
                             self.handle_phase_transition("completed")

                    if phase != self.last_phase:
                        self.handle_phase_transition(phase)
                        self.last_phase = phase
            except Exception as e:
                if getattr(self, 'last_sync_success', True):
                    self.logger.error(f"Masalah pada loop sinkronisasi (Server Down?): {e}")
                    self.last_sync_success = False
                self.is_registered = False
            time.sleep(5)


    def handle_phase_transition(self, phase):
        phase = phase.lower().strip().replace(" ", "_")
        if phase == self.last_phase:
            return
            
        self.current_phase = phase
        self.logger.info(f"Phase Transition: {self.last_phase} -> {phase}")
        
        if phase in ["discovery", "data_preparation"]:
            self.is_training_phase = True
            threading.Thread(target=self.run_discovery_phase).start()
        elif phase == "syncing":
            self.is_training_phase = True
            threading.Thread(target=self.run_sync_phase).start()
        elif phase in ["training", "training_phase"]:
            self.is_training_phase = True
            threading.Thread(target=self.start_fl, daemon=True).start()
        elif phase in ["registry_generation"]:
             self.is_training_phase = True
             threading.Thread(target=self.run_registry_phase).start()
        if phase == "idle" or phase == "completed":
            self.is_training_phase = False
            self.fl_status = "Online (Selesai)"
            
        # Menggunakan logika OR untuk menghindari ketidakselarasan transisi fase
        if (self.last_phase == "training" or self.last_phase == "registry generation" or self.last_phase == "idle") and (phase == "completed" or phase == "idle"):
            # Cek apakah memang ada kenaikan versi di server
            try:
                resp = self._safe_request("GET", f"{self.server_api_url}{api_status}")
                server_v = resp.json().get("model_version", self.model_version) if resp and resp.status_code == 200 else self.model_version
                if server_v <= self.model_version and not (phase == "completed"):
                    return # Tidak perlu sync jika sudah up to date dan bukan fase completion eksplisit
            except: pass

            self.logger.info("Pelatihan selesai atau Update tersedia. Memulai Sinkronisasi Final...")
            def update_task():
                try:
                    # Unduh Backbone global terakhir
                    if not self.download_backbone():
                        self.logger.error("Gagal mengunduh backbone final.")
                    
                    # Unduh BN global terakhir
                    if not self.download_bn():
                        self.logger.error("Gagal mengunduh BN final.")
                        
                    # Unduh Registry Centroid 
                    if not self.download_registry_assets():
                        self.logger.error("Gagal mengunduh registry final.")
                        
                    # Ambil versi terbaru dan update metadata
                    try:
                        resp = self._safe_request("GET", f"{self.server_api_url}{api_status}")
                        if resp and resp.status_code == 200:
                            v = resp.json().get("model_version", self.model_version)
                            self.logger.info(f"Menetapkan versi lokal ke v{v}")
                            self._save_version(v)
                    except Exception as e:
                        self.logger.warn(f"Gagal update nomor versi: {e}")

                    # Muat ulang model agar inferensi menggunakan versi terbaru
                    self._reload_inference_models(force_reload=True)
                    
                    # Selesaikan label map dan refresh lokal
                    self.sync_label_map()
                    self.refresh_local_embeddings()
                    self.logger.success("Siklus FL selesai, sistem siap untuk inferensi v-terbaru.")
                    
                except Exception as e:
                    self.logger.error(f"Sinkronisasi pasca-pelatihan gagal: {e}")
            
            threading.Thread(target=update_task, daemon=True).start()

    def _save_version(self, v):
        self.model_version = v
        try:
            v_path = os.path.join(self.data_path, "models", "model_version.txt")
            os.makedirs(os.path.dirname(v_path), exist_ok=True)
            temp_path = v_path + ".tmp"
            with open(temp_path, "w") as f:
                f.write(str(v))
            os.replace(temp_path, v_path)
            self.logger.info(f"Berhasil menyimpan versi model terbaru: v{v}")
        except Exception as e:
            self.logger.error(f"Gagal menyimpan model_version.txt secara aman: {e}")

    def download_backbone(self):
        """Mengambil StateDict Backbone hasil agregasi dari server."""
        try:
            res_bb = self._safe_request("GET", f"{self.server_api_url}{api_model_backbone}")
            if res_bb and res_bb.status_code == 200:
                save_path = os.path.join(self.data_path, "models", "backbone.pth")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    size = f.write(res_bb.content)
                self.logger.info(f"Berhasil mengunduh backbone ({size} bytes).")
                
                # Terapkan backbone global langsung ke RAM
                self._reload_inference_models(force_reload=True)
                return True
        except Exception as e:
            self.logger.error(f"Backbone fetch failed: {e}")
        return False

    def download_bn(self, max_wait=60):
        """Mengambil statistik BN (Running Mean/Var) hasil agregasi untuk konsistensi global."""
        path = os.path.join(self.data_path, "models", "global_bn_combined.pth")
        url = f"{self.server_api_url}{api_model_bn}"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                # Jangan biarkan raise_for_status() melempar eksepsi untuk 404/202 di sini
                # agar kita bisa menanganinya dengan log yang lebih sopan.
                res = requests.get(url, timeout=10)
                
                if res.status_code == 200:
                    tmp_path = path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        f.write(res.content)
                    os.replace(tmp_path, path)
                    
                    try:
                        # pFedFace: Simpan saja filenya, tapi JANGAN terapkan ke backbone aktif.
                        # Kita akan menggunakan calibrate_bn lokal sebagai gantinya.
                        self.logger.success("Parameter BN global diunduh (Disimpan untuk referensi, tidak diterapkan).")
                        return True
                    except Exception as e:
                        self.logger.error(f"Berkas BN tidak valid, mencoba kembali: {e}")
                elif res.status_code == 202 or res.status_code == 404:
                    # Log sebagai INFO saja, bukan ERROR, karena wajar jika belum ada di awal training
                    self.logger.info(f"BN global belum tersedia di server (Status {res.status_code}), menunggu...")
                else:
                    res.raise_for_status() # Lempar untuk kode error lain (500, dll)
            except Exception as e:
                # Kurangi kebisingan log untuk masalah koneksi/timeout sementara
                pass
            time.sleep(5)
        return False

    def download_registry_assets(self):
        """Unduh centroid identifikasi gabungan untuk inferensi offline."""
        try:
            res_reg = self._safe_request("GET", f"{self.server_api_url}{api_model_registry}")
            if res_reg and res_reg.status_code == 200:
                reg_path = os.path.join(self.data_path, "models", "global_embedding_registry.pth")
                os.makedirs(os.path.dirname(reg_path), exist_ok=True)
                with open(reg_path, "wb") as f:
                    f.write(res_reg.content)
                self.logger.success("Registry Centroid berhasil diperbarui.")
                return True
        except Exception as e:
            self.logger.error(f"Gagal memperbarui Registry: {e}")
        return False

    def run_sync_phase(self):
        self.report_status("Processing: Sinkronisasi Data...")
        self.download_backbone()
        self.sync_label_map()

        db = SessionLocal()
        try:
            res = self._safe_request("GET", f"{self.server_api_url}{api_training_identities}")
            if res and res.status_code == 200:
                global_users = res.json()
                for u in global_users:
                    user = db.query(UserLocal).filter_by(user_id=u['nrp']).first()
                    if not user:
                        try:
                            user = UserLocal(user_id=u['nrp'], name=u['name'])
                            db.add(user)
                            db.commit()
                        except Exception as inner_err:
                            db.rollback()
                            self.logger.warning(f"Gagal/duplikat saat menambahkan user global {u['nrp']}: {inner_err}")
            self.report_status("Siap Preprocess")
            self.logger.info("Sinkronisasi data selesai. Menjalankan tahap preprocessing secara otomatis...")
            self.run_preprocess_phase()
        except Exception as e:
            self.logger.error(f"{e}")
            self.report_status("Error: Sync Gagal")
        finally:
            db.close()

    def refresh_local_embeddings(self):
        # Memperbarui embedding lokal menggunakan backbone terbaru hasil sinkronisasi
        self._ensure_models_loaded()
        db = SessionLocal()
        try:
            users = db.query(UserLocal).all()
            processed_dir = os.path.join(self.data_path, "processed")
            
            for user in users:
                user_folder = os.path.join(processed_dir, user.user_id)
                if not os.path.exists(user_folder): continue
                
                # Pilih gambar terbaik (sudah berupa face crop 96x112 dari tahap preprocessing)
                selected_paths = image_processor.select_best_faces(user_folder, n=50)
                if not selected_paths:
                    self.logger.warn(f"User {user.user_id}: Tidak ada gambar tersedia.")
                    continue
                self.logger.info(f"User {user.user_id}: Menyeleksi {len(selected_paths)} gambar terbaik.")

                # Citra di folder processed sudah berupa potongan wajah 96x112
                # Tidak perlu memanggil deteksi MTCNN kembali.
                # Gunakan prepare_for_model langsung untuk optimasi.
                tensors = []
                valid_paths = []
                for img_name in selected_paths:
                    trial_path = os.path.join(user_folder, img_name)
                    try:
                        img_pil = Image.open(trial_path).convert('RGB')
                        input_tensor = image_processor.prepare_for_model(img_pil)
                        if input_tensor is not None:
                            tensors.append(input_tensor)
                            valid_paths.append(trial_path)
                    except Exception as e:
                        self.logger.error(f"Gagal memuat {img_name} saat refresh: {e}")
                        continue

                if not tensors:
                    self.logger.warn(f"User {user.user_id}: Semua gambar gagal diproses.")
                    continue

                self.logger.success(f"User {user.user_id}: Memproses {len(tensors)} gambar terbaik secara batched.")

                # Hitung rata-rata skor ketajaman untuk dokumentasi log
                blur_scores = []
                for p in valid_paths:
                    try:
                        blur_scores.append(image_processor.get_blur_score(p))
                    except: pass
                if blur_scores:
                    avg_blur = sum(blur_scores) / len(blur_scores)
                    self.logger.info(f"User {user.user_id}: Rata-rata skor ketajaman {avg_blur:.1f}")

                # Satukan seluruh tensor gambar menjadi satu batch besar
                batch_tensor = torch.cat(tensors, dim=0).to(self.device)
                
                with torch.no_grad():
                    self.backbone.eval()
                    # Proses flip trick alignment dengan inferensi batched
                    emb_orig = self.backbone(batch_tensor)
                    input_flipped = torch.flip(batch_tensor, dims=[3])
                    emb_mirror = self.backbone(input_flipped)
                    
                    # Gabungkan dan Normalisasi L2 untuk masing-masing gambar dalam batch
                    embeddings = torch.nn.functional.normalize((emb_orig + emb_mirror) / 2, p=2, dim=1)
                    
                    # Hitung Centroid (rata-rata representasi fitur wajah) dan normalisasi ulang ke unit sphere
                    centroid = torch.mean(embeddings, dim=0)
                    centroid_normalized = torch.nn.functional.normalize(centroid.unsqueeze(0), p=2, dim=1)
                    embedding_np = centroid_normalized.cpu().numpy()[0]
                    
                try:
                    encrypted_data, iv = encryptor.encrypt_embedding(embedding_np)
                    # Hapus entri global yang usang (is_global=True) untuk user ini
                    stale_global = db.query(EmbeddingLocal).filter_by(user_id=user.user_id, is_global=True).first()
                    if stale_global:
                        db.delete(stale_global)
                        self.logger.info(f"Menghapus entri global usang untuk {user.user_id}.")
                    
                    emb_record = db.query(EmbeddingLocal).filter_by(user_id=user.user_id, is_global=False).first()
                    if emb_record:
                        emb_record.embedding_data = encrypted_data
                        emb_record.iv = iv
                    else:
                        db.add(EmbeddingLocal(user_id=user.user_id, embedding_data=encrypted_data, iv=iv, is_global=False))
                    db.commit()
                except Exception as db_err:
                    db.rollback()
                    self.logger.error(f"Gagal memperbarui database lokal untuk user {user.user_id}: {db_err}")
                    continue
                
                # Cleanup per user untuk hemat RAM Jetson
                del batch_tensor, embeddings, centroid, centroid_normalized
                gc.collect()
                
                # Bagikan ke server
                try:
                    payload = {
                        "nrp": user.user_id, "name": user.name, "client_id": self.client_id,
                        "embedding": base64.b64encode(embedding_np.tobytes()).decode('utf-8')
                    }
                    self._safe_request("POST", f"{self.server_api_url}{api_training_get_label}", json=payload)
                except: pass
            
            self.logger.success("Local embeddings refresh complete.")
        except Exception as e:
            self.logger.error(f"Refresh Error: {e}")
        finally:
            db.close()


    def run_discovery_phase(self):
        self.report_status("Processing: Discovery Identitas...")
        try:
            # Tentukan path folder mahasiswa di raw_data/students
            student_path = os.path.join(self.raw_data_path, "students")
            if not os.path.exists(student_path):
                self._log(f"[DEBUG] Subfolder 'students' tidak ada, mencoba menggunakan root: {self.raw_data_path}")
                student_path = self.raw_data_path

            # Membaca daftar sub-direktori mahasiswa yang terdeteksi
            folders = [f for f in os.listdir(student_path) if os.path.isdir(os.path.join(student_path, f))]
            for folder in sorted(folders):
                # Memilah nama folder berformat "NRP_Nama" menjadi NRP dan Nama secara terpisah
                nrp = folder.split('_')[0] if "_" in folder else folder
                name = folder.split('_')[1] if "_" in folder else nrp
                
                # Kirim data identitas mentah ke REST API server pusat
                try:
                    self._safe_request("POST", f"{self.server_api_url}{api_training_get_label}", json={
                        "nrp": nrp, "name": name, "client_id": self.client_id
                    })
                except: pass
                
                # Daftarkan identitas subjek ke SQLite lokal perangkat secara aman
                db = SessionLocal()
                try:
                    nrp_clean = nrp.strip()
                    name_clean = name.strip()
                    user = db.query(UserLocal).filter_by(user_id=nrp_clean).first()
                    if not user:
                        db.add(UserLocal(user_id=nrp_clean, name=name_clean))
                        db.commit()
                except Exception as db_err:
                    db.rollback()
                    self.logger.warning(f"Gagal/duplikat saat mendaftarkan user {nrp} ke DB lokal: {db_err}")
                finally: 
                    db.close()
            
            # Beritahu server bahwa pencarian identitas di client ini telah selesai
            self._safe_request("POST", f"{self.server_api_url}{api_clients_discovery_done}", json={"client_id": self.client_id})
            self.report_status("Discovery Selesai: Menunggu Global Map...")
        except Exception as e:
            self.logger.error(f"Discovery Error: {e}")
            self.report_status("Error Discovery")

    def sync_label_map(self):
        try:
            self._ensure_models_loaded()
            # Request daftar label map terpadu dari REST API server pusat
            res = self._safe_request("GET", f"{self.server_api_url}{api_training_label_map}")
            if res and res.status_code == 200:
                self.client.label_map = res.json()
                self.num_classes = len(self.client.label_map)
                self.logger.success(f"Peta label global berhasil disinkronkan. Total identitas: {self.num_classes}")
                
                # Lakukan perluasan/ekspansi classifier head model jika kelas bertambah
                if self.head is not None:
                    current_head_classes = self.head.weight.shape[0]
                    if current_head_classes != self.num_classes:
                        self.logger.info(f"Memicu ekspansi classifier head segera ({current_head_classes} -> {self.num_classes})...")
                        new_label_map = {nrp: idx for idx, nrp in enumerate(self.client.label_map)}
                        self.head = self.client.trainer.update_head(self.num_classes, new_label_map)

                # Simpan peta label terpadu ke file JSON lokal (label_map.json)
                label_map_path = os.path.join(self.data_path, "models", "label_map.json")
                os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
                
                with open(label_map_path, "w") as f:
                    json.dump(self.client.label_map, f)
                return True
        except Exception as e:
            self.logger.error(f"Gagal sinkronisasi label map: {e}")
        return False

    def run_preprocess_phase(self):
        if getattr(self, "is_preprocessing_active", False):
            self.logger.info("Pra-pemrosesan wajah sudah aktif berjalan. Mengabaikan pemicuan ganda.")
            return
        
        self.is_preprocessing_active = True
        try:
            self._run_preprocess_phase_internal()
        finally:
            self.is_preprocessing_active = False

    def _run_preprocess_phase_internal(self):
        self.report_status("Processing: Ekstraksi Wajah (Laplacian Top 50)...")
        self._ensure_models_loaded()
        if not self.sync_label_map():
            self.report_status("Error: Gagal Sinkronisasi Global Map")
            return

        students_dir = os.path.join(self.raw_data_path, "students")
        if not os.path.exists(students_dir):
            self._log(f"[DEBUG] Subfolder 'students' tidak ada, mencoba menggunakan root: {self.raw_data_path}")
            students_dir = self.raw_data_path
            
        processed_dir = os.path.join(self.data_path, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        if not os.path.exists(students_dir) or not os.listdir(students_dir):
            self.report_status("Siap Training (No Data)")
            return

        folders = sorted([f for f in os.listdir(students_dir) if os.path.isdir(os.path.join(students_dir, f))])
        total_folders = len(folders)
        for idx, folder in enumerate(folders):
            nrp = folder.split('_')[0] if "_" in folder else folder
            target_folder = os.path.join(processed_dir, nrp)
            
            # Lewati apabila data untuk nrp ini sudah diproses sebelumnya
            if os.path.exists(target_folder) and len(os.listdir(target_folder)) >= 5:
                self.logger.info(f"  [SKIP] {nrp} sudah diproses sebelumnya.")
                continue

            self.logger.info(f"[{idx+1}/{total_folders}] Memilih 50 wajah terbaik untuk {nrp} (Laplacian Variance)...")
            self._log(f"[PREPROCESS] Memproses {nrp} ({idx+1}/{total_folders}): Seleksi Laplacian Variance...")
            self.report_status(f"Processing: Laplacian {nrp} ({idx+1}/{total_folders})...")
            
            # Gunakan folder sementara agar aman dari interupsi
            tmp_target = target_folder + ".tmp"
            if os.path.exists(tmp_target): shutil.rmtree(tmp_target)
            os.makedirs(tmp_target, exist_ok=True)
            
            source_path = os.path.join(students_dir, folder)
            top_images = image_processor.select_best_faces(source_path, n=50)
            
            count = 0
            for img_name in top_images:
                try:
                    img_path = os.path.join(source_path, img_name)
                    img_pil = Image.open(img_path).convert('RGB')
                    target_path = os.path.join(tmp_target, f"face_{count}.jpg")
                    
                    # Gunakan MTCNN dari image_processor (Portrait Cropping 96x112)
                    face_img, _, _ = image_processor.detect_face(img_pil, save_path=target_path)
                    if face_img is not None:
                        count += 1
                except: pass
            
            # Hanya rename ke folder asli jika proses selesai tanpa interupsi
            if os.path.exists(target_folder): shutil.rmtree(target_folder)
            os.rename(tmp_target, target_folder)
            self.logger.success(f"Preprocessing {nrp} selesai.")
            
            # Explicit Cleanup untuk mencegah OOM di Edge (Jetson/Pi)
            gc.collect()
            time.sleep(0.1)
        
        # Free MTCNN RAM after finishing all preprocessing to prepare for Training Phase
        image_processor.unload_detector()
        
        # Hentikan detector MTCNN untuk menghemat RAM
        image_processor.unload_detector()
        gc.collect()
        
        has_data = any(len(os.listdir(os.path.join(processed_dir, sub))) > 0 for sub in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, sub)))
        if has_data:
            # Unduh backbone dan parameter BN global terbaru sebelum training dimulai
            # Ini yang menjamin akurasi Ronde 1 bisa mencapai 0.9 karena model memulai 
            # dari titik optimal (Global Lens).
            self.logger.info("Menyiapkan bekal training: Mengunduh Backbone & BN Global...")
            self.download_backbone()
            self.download_bn(max_wait=10)
            
            # Pembersihan cache dan sampah memori sebelum komputasi
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            self.report_status("Siap Training")
            self.refresh_local_embeddings()
            
            try:
                self._log_to_file(f"[PREPROCESS] Client {self.client_id} is READY for training.")
                self._safe_request("POST", f"{self.server_api_url}{api_clients_ready}", json={"client_id": self.client_id})
            except: pass
            
            num_classes = sum(1 for sub in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, sub)) and len(os.listdir(os.path.join(processed_dir, sub))) > 0)
            
            # Update num_classes di manager agar sinkron
            if self.client.label_map and len(self.client.label_map) > num_classes:
                self.num_classes = len(self.client.label_map)
            else:
                self.num_classes = num_classes
                
            # Expand head dengan preservasi bobot
            new_label_map = {nrp: idx for idx, nrp in enumerate(self.client.label_map)} if self.client.label_map else {}
            self.head = self.client.trainer.update_head(self.num_classes, new_label_map)
            self.client.head = self.head
        else:
            self.report_status("Siap Training (No Data)")
            try:
                self._safe_request("POST", f"{self.server_api_url}{api_clients_ready}", json={"client_id": self.client_id})
            except: pass

    def run_registry_phase(self):
        """Ekstraksi Registry FINAL (Menggunakan Bobot Global Terpadu)."""
        if getattr(self, "is_sending_registry", False):
            self.logger.info("Registry generation already in progress, skipping duplicate trigger.")
            return

        self._ensure_models_loaded()
        self.is_sending_registry = True
        self.report_status("Processing: Finalisasi Registry Identitas...")
        try:
            # Sinkronisasi backbone terlebih dahulu
            self.logger.info("Fase: Mengunduh Backbone Global dari server...")
            self.download_backbone()
            # pFedFace: Kita tidak lagi memuat BN global, gunakan BN lokal yang sudah ada atau dikalibrasi.
            # self.download_bn(max_wait=10) 
            
            # Hitung centroid menggunakan calibrated backbone
            # Sangat krusial agar embedding di database (Registry) 
            # sinkron dengan embedding saat Live Inference (pFedFace).
            self.logger.info("Fase: Menghitung Centroid (menggunakan Calibrated Backbone)...")
            try:
                # Pastikan backbone trainer menggunakan bobot global terbaru namun BN lokal
                # (Ini sudah dihandle oleh download_backbone + load_state_dict dengan filter pFedFace)
                self.client.trainer.backbone.eval()
                self.logger.info("Trainer backbone siap (Mode Eval untuk Centroid).")
            except Exception as e:
                self.logger.warn(f"Inisialisasi centroid gagal: {e}")

            bn_params = self.client.trainer.get_bn_parameters()
            centroids = self.client.trainer.calculate_centroids(label_map=self.client.label_map)
            
            # Simpan cadangan lokal secara atomik
            local_registry_path = os.path.join(self.data_path, "models", "global_embedding_registry.pth")
            tmp_registry_path = local_registry_path + ".tmp"
            torch.save(centroids, tmp_registry_path)
            os.replace(tmp_registry_path, local_registry_path)
            
            # Sinkronisasi identitas global (nama dan nrp)
            self._sync_global_identities()
            
            # Kirim ke server
            self.logger.info("Fase: Mengirim aset lokal ke server...")
            serialized_centroids = {nrp: base64.b64encode(vec.tobytes()).decode('utf-8') for nrp, vec in centroids.items()}
            bn_buf = io.BytesIO()
            torch.save(bn_params, bn_buf)
            serialized_bn = base64.b64encode(bn_buf.getvalue()).decode('utf-8')
            
            payload = {
                "client_id": self.client_id, "bn_params": serialized_bn, "centroids": serialized_centroids
            }
            res = self._safe_request("POST", f"{self.server_api_url}{api_training_registry_assets}", json=payload, timeout=60)
            
            if res and res.status_code == 200:
                self._log_to_file("SUCCESS: Pengiriman aset lokal berhasil. Menunggu agregasi global...")
                self.report_status("Processing: Menunggu Agregasi Global...")
                
                # Tunggu dan unduh hasil global
                # Ini memastikan setiap client memiliki BN dan Registry yang SAMA
                if self.download_bn(max_wait=3600):
                    self._log_to_file("REGISTRY: Global BN Synced.")
                
                if self._download_global_registry(max_wait=3600):
                    self._log_to_file("REGISTRY: Global Registry Synced.")
                    
                    # Sinkronisasi versi secara otomatis
                    try:
                        resp = self._safe_request("GET", f"{self.server_api_url}{api_status}", timeout=5)
                        if resp and resp.status_code == 200:
                            v = resp.json().get("model_version", 1)
                            self.logger.info(f"Menyelaraskan versi lokal ke v{v}")
                            self._save_version(v)
                    except: pass

                    # Paksa reload backbone ke inferensi dengan versi global yang sudah disinkronisasi
                    self.logger.info("Finalisasi: Mereset backbone inferensi ke versi global agregasi...")
                    self._reload_inference_models(force_reload=True)
                    self.report_status("Siap Selesai")
                else:
                    self.logger.error("Unduhan registri global gagal atau habis waktu.")
                    self.report_status("Error: Registry Timeout")
            else:
                self.report_status(f"Error Submission: {res.status_code}")
                
        except Exception as e:
            self.logger.error(f"Kesalahan Registri: {e}")
            self.report_status("Error Registry")
        finally:
            self.is_sending_registry = False
            self.refresh_local_embeddings() # Final refresh with global BN

    def _download_global_registry(self, max_wait=3600):
        registry_url = f"{self.server_api_url}{api_model_registry}"
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                res = self._safe_request("GET", registry_url, timeout=15)
                if res and res.status_code == 200:
                    save_path = os.path.join(self.data_path, "models", "global_embedding_registry.pth")
                    tmp_save_path = save_path + ".tmp"
                    with open(tmp_save_path, "wb") as f:
                        f.write(res.content)
                    os.replace(tmp_save_path, save_path)
                    if hasattr(self, 'cached_refs'):
                        del self.cached_refs
                    self.logger.info(f"Global registry downloaded ({len(res.content)//1024} KB).")
                    return True
                elif res.status_code == 202:
                    self.logger.info("Server aggregating... waiting 5s")
                    time.sleep(5)
                else:
                    self.logger.error(f"Download failed: {res.status_code}")
                    return False
            except Exception as e:
                self.logger.error(f"Gagal mengunduh registry: {e}")
                time.sleep(5)
        self.logger.error("Batas waktu unduhan registry tercapai.")
        return False

    def start_fl(self):
        if self.is_training: return
        self.is_training = True
        def run_client():
            success = False
            retry_count = 0
            while self.current_phase in ["training", "training_phase"]:
                try:
                    self.report_status("Training: Flower FL...")
                    fl.client.start_client(server_address=self.fl_server_address, client=self.client.to_client())
                    success = True
                    self.logger.success("Pembelajaran Terfederasi (Flower) berhasil selesai.")
                    break
                except Exception as e:
                    err_msg = str(e)
                    retry_count += 1
                    self.logger.error(f"Gagal menghubungkan/menjalankan Flower client (Percobaan {retry_count}): {err_msg}")
                    # Deteksi error gRPC / Jaringan
                    if "Rendezvous" in err_msg or "UNAVAILABLE" in err_msg:
                        self.report_status(f"Error: Koneksi Putus (Coba-{retry_count})")
                    else:
                        self.report_status(f"Error: {err_msg[:20]} (Coba-{retry_count})")
                    
                    # Tunggu sebelum mencoba kembali jika fase masih aktif
                    time.sleep(5)
            
            self.is_training = False
            if success:
                self.report_status("Online (Selesai)")
            else:
                self.report_status("Online (Gagal)")
            
            # Reload model untuk menggunakan bobot global terbaru (EAGER RELOAD)
            self._reload_inference_models(force_reload=True)
        threading.Thread(target=run_client, daemon=True).start()

fl_manager = FLClientManager()
