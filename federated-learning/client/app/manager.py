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

from app.utils.security import encryptor
from app.utils.classifier import identify_user_globally
from app.utils.mobilefacenet import MobileFaceNet, ArcMarginProduct
from app.db.db import SessionLocal
from app.db.models import UserLocal, EmbeddingLocal
from app.recognition_client import FaceRecognitionClient
from app.controllers.attendance import AttendanceController

from app.utils.logging import init_logger, get_logger

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
        
        # 1. Inisialisasi Logger (Gunakan ID mentah dulu untuk tag)
        temp_id = self._get_raw_id()
        self.log_path = os.path.join(self.data_path, "client_activity.log")
        init_logger(self.log_path, tag=f"FL-CLIENT-{temp_id}")
        self.logger = get_logger()
        
        # 2. Muat Identitas Persisten Resmi
        self.client_id = self._load_identity()

        
        # PERSISTENSI: Tentukan ukuran head dinamis & versi
        self.save_path = os.path.join(self.data_path, "models", "backbone.pth")
        self._models_loaded = False  # Cache flag: True berarti backbone sudah diload ke RAM, skip disk reload
        self.head_path = os.path.join(self.data_path, "models", "local_head.pth")
        self.version_path = os.path.join(self.data_path, "models", "model_version.txt")
        self.map_path = os.path.join(self.data_path, "models", "label_map.json")
        
        self.num_classes = 1000 # Default/Seeding awal (Akan menyusut otomatis saat sync)
        self.model_version = 0
        
        # 1. Muat Versi (Prioritas Disk)
        if os.path.exists(self.version_path):
            try:
                with open(self.version_path, 'r') as f:
                    self.model_version = int(f.read().strip())
                self.logger.info(f"Terdeteksi Model Versi: v{self.model_version}")
            except: pass

        # 2. Muat Jumlah Kelas dari Label Map (Sumber Kebenaran Utama)
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

        # 3. Guard model initialization placeholder
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
        
        self.is_registered = False
        self.last_register_attempt = 0
        self.register_retry_delay = 30
        
        self.prediction_buffer = collections.deque(maxlen=10)
        self.last_face_time = 0
        
        # Dukungan Kamera Headless
        self.camera_index = int(os.getenv("CAMERA_INDEX", 0))
        self.latest_frame = None
        self.latest_result = {"matched": "Standby", "confidence": 0, "latency_ms": 0, "is_virtual": False}
        self.is_camera_running = False
        
        # Ambang Batas Inferensi (Dinamis dari Server)
        self.inference_threshold = 0.7
        
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
            response = self._safe_request("POST", f"{self.server_api_url}/api/clients/register", json={
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
            res = self._safe_request("GET", f"{self.server_api_url}/api/training/identities")
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
            
            # 1. Jika input adalah LIST (Flower/Numpy Arrays)
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
                    self.logger.error(f"Weight mismatch: {len(loaded)} vs {len(shared_keys)}")
                    return False
            
            # 2. Jika input adalah DICT (State Dict dari .pth)
            else:
                for k, v in loaded.items():
                    if ignore_bn and any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked']):
                        continue # SKIP: Biarkan statistik BN tetap lokal
                    if k in new_sd:
                        new_sd[k] = v.to(self.device)
            
            model.load_state_dict(new_sd, strict=False)
            self.logger.success(f"Backbone weights applied (BN Ignored: {ignore_bn})")
            return True
        except Exception as e:
            self.logger.error(f"Gagal menerapkan bobot model: {e}")
            return False

    def _ensure_models_loaded(self, force_reload=False):
        """Menjamin instance model tersedia di RAM (Tanpa paksa reload dari disk jika sudah ada)."""
        # 0. Selaraskan num_classes dari label map lokal (Sumber Kebenaran)
        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, 'r') as f:
                    self.num_classes = len(json.load(f))
            except: pass

        # 1. Inisialisasi instance objek jika belum ada
        if self.backbone is None:
            self.backbone = MobileFaceNet().to(self.device)
            self.backbone.eval()
            
        if self.head is None:
            self.head = ArcMarginProduct(128, self.num_classes).to(self.device)
            self.head.eval()

        # 2. Sinkronisasi referensi (PENTING untuk Flower Client dan Trainer)
        if hasattr(self, 'client'):
            self.client.model = self.backbone
            self.client.head = self.head
            if hasattr(self.client, 'trainer'):
                self.client.trainer.backbone = self.backbone
                self.client.trainer.head = self.head

        # 3. Muat dari disk jika diminta atau jika RAM belum terisi bobot
        if not self._models_loaded or force_reload:
            if os.path.exists(self.save_path):
                try:
                    loaded = torch.load(self.save_path, map_location=self.device)
                    self._apply_backbone_weights(loaded, ignore_bn=True)
                except Exception as e:
                    self.logger.error(f"Backbone file mismatch: {e}")
                    
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
                    except: pass
            
            self._models_loaded = True
        
        return True

    def _reload_inference_models(self, force_reload=False):
        """EAGER LOAD: Memuat ulang model ke RAM secara thread-safe khusus untuk inferensi."""
        try:
            if not force_reload and self.inference_backbone is not None:
                return
                
            # 1. Muat ke variabel lokal (Aman terhadap balapan camera loop)
            new_backbone = MobileFaceNet().to(self.device).eval()
            new_head = None
            if self.num_classes > 0:
                new_head = ArcMarginProduct(128, self.num_classes).to(self.device).eval()
            
            if os.path.exists(self.save_path):
                self.logger.info(f"Loading backbone from {self.save_path}...")
                loaded = torch.load(self.save_path, map_location=self.device)
                self._apply_backbone_weights(loaded, ignore_bn=True, target_model=new_backbone)
                
                # pFedFace FIX: Kalibrasi BN lokal alih-alih memuat statistik global yang tidak cocok
                try:
                    from .utils.freezing import calibrate_bn
                    calibrate_bn(new_backbone, self.raw_data_path, device=self.device, num_samples=50)
                except Exception as e:
                    self.logger.warn(f"Gagal kalibrasi BN: {e}")

            if new_head is not None and os.path.exists(self.head_path):
                try:
                    new_head.load_state_dict(torch.load(self.head_path, map_location=self.device))
                except: pass

            # 2. ATOMIC SWAP: Update hanya model inferensi agar tidak terganggu drift training
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

            # 3. Invalidate Cache Identitas & RAM Cleanup
            if hasattr(self, 'cached_refs'):
                self.cached_refs = {}
            if hasattr(self, 'prediction_buffer'):
                self.prediction_buffer.clear()
            self.last_cache_update = 0
            gc.collect()
            
            self.logger.success(f"Inference Mode: Global v{self.model_version} Ready.")
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
        # PENTING: Jika is_camera_running dipaksa True (via tombol), abaikan pengecekan env awal
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
        
        # Optimasi Jetson/Raspi: Paksa format MJPG jika didukung (lebih ringan dari YUYV)
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
        
        # PENTING: Gunakan instance terpusat dari manager
        attendance_engine = self.attendance
        
        while self.is_camera_running:
            # Re-check enable_camera in case it changes or we want to stop
            if not self.is_camera_running: break

            ret, frame = cap.read()
            
            if not ret:
                self.logger.error(f"Gagal membaca frame dari kamera {cam_idx}. Mencoba ulang dlm 5 detik...")
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(cam_idx)
                continue
            
            # Training/Preprocessing Guard
            if self.is_training_phase:
                if cap.isOpened():
                    self.logger.info("Melepaskan hardware kamera untuk menghemat daya/RAM selama training.")
                    cap.release()
                self.latest_result["matched"] = "TRAINING PHASE..."
                time.sleep(5)
                continue
            else:
                if not cap.isOpened():
                    self.logger.info("Mengaktifkan kembali hardware kamera...")
                    cap = cv2.VideoCapture(cam_idx)

            # Inisialisasi model hanya jika diperlukan (Lazy Load)
            if not self.backbone:
                self._ensure_models_loaded()

            # Simpan frame terbaru untuk streaming MJPEG
            self.latest_frame = frame.copy()
            
            # Lakukan pemrosesan jika model sudah siap dan sudah ditraining (v1+)
            # Gunakan inference_backbone agar tidak terganggu drift training ronde
            if self.inference_backbone and self.model_version > 0:
                start_time = time.time()
                try:
                    # Konversi ke PIL Image
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    matched, confidence = self.attendance.recognize_directly(img_pil)
                    
                    status_str = f"v{self.model_version}"
                    if self.is_training and self.fl_round > 0:
                        status_str += f" (R{self.fl_round}/15)"

                    # Cek lagi sebelum update result agar tidak menimpa status "CAMERA OFF"
                    if self.is_camera_running:
                        self.latest_result = {
                            "matched": matched,
                            "confidence": confidence,
                            "latency_ms": int((time.time() - start_time) * 1000),
                            "model_version": status_str,
                            "is_virtual": False
                        }

                    
                    # LOGGING: Catat hanya jika ada hasil cocok
                    if matched != "Unknown" and matched != "Error":
                        self._log_to_file(f"INFERENCE SUCCESS: {matched} (Conf: {confidence:.4f})")
                except Exception as e:
                    pass
            elif self.model_version == 0:
                self.latest_result["matched"] = "MODEL NOT TRAINED (v0)"
                self.latest_result["model_version"] = "v0"
            
            # Explicit cleanup per loop to keep memory stable
            time.sleep(0.5)
            gc.collect()
        
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
            response = self._safe_request("POST", f"{self.server_api_url}/api/clients/register", json=payload, timeout=2)
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
                self.report_status()
                
                # 2. Cek Versi Model
                resp = self._safe_request("GET", f"{self.server_api_url}/api/training/status")
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
                        # TRICK: Jika idle dan versi tertinggal, paksa re-sync final
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
            
        # PENTING: Gunakan OR untuk menangani race condition phase -> idle yang terlalu cepat
        if (self.last_phase == "training" or self.last_phase == "registry generation" or self.last_phase == "idle") and (phase == "completed" or phase == "idle"):
            # Cek apakah memang ada kenaikan versi di server
            try:
                resp = self._safe_request("GET", f"{self.server_api_url}/api/status")
                server_v = resp.json().get("model_version", self.model_version) if resp and resp.status_code == 200 else self.model_version
                if server_v <= self.model_version and not (phase == "completed"):
                    return # Tidak perlu sync jika sudah up to date dan bukan fase completion eksplisit
            except: pass

            self.logger.info("Pelatihan selesai atau Update tersedia. Memulai Sinkronisasi Final...")
            def update_task():
                try:
                    # 1. Unduh Backbone global terakhir
                    if not self.download_backbone():
                        self.logger.error("Gagal mengunduh backbone final.")
                    
                    # 2. Unduh BN global terakhir
                    if not self.download_bn():
                        self.logger.error("Gagal mengunduh BN final.")
                        
                    # 3. Unduh Registry Centroid 
                    if not self.download_registry_assets():
                        self.logger.error("Gagal mengunduh registry final.")
                        
                    # 4. Ambil versi terbaru dan update metadata
                    try:
                        resp = self._safe_request("GET", f"{self.server_api_url}/api/status")
                        if resp and resp.status_code == 200:
                            v = resp.json().get("model_version", self.model_version)
                            self.logger.info(f"Menetapkan versi lokal ke v{v}")
                            self._save_version(v)
                    except Exception as e:
                        self.logger.warn(f"Gagal update nomor versi: {e}")

                    # 5. FORCE RELOAD: Pastikan inferensi menggunakan v1 segera
                    self._reload_inference_models(force_reload=True)
                    
                    # 6. Selesaikan label map dan refresh lokal
                    self.sync_label_map()
                    self.refresh_local_embeddings()
                    self.logger.success("Siklus FL selesai, sistem siap untuk inferensi v-terbaru.")
                    
                except Exception as e:
                    self.logger.error(f"Sinkronisasi pasca-pelatihan gagal: {e}")
            
            threading.Thread(target=update_task, daemon=True).start()

    def _save_version(self, v):
        self.model_version = v
        v_path = os.path.join(self.data_path, "models", "model_version.txt")
        os.makedirs(os.path.dirname(v_path), exist_ok=True)
        with open(v_path, "w") as f:
            f.write(str(v))

    def download_backbone(self):
        """Mengambil StateDict Backbone hasil agregasi dari server."""
        try:
            res_bb = self._safe_request("GET", f"{self.server_api_url}/api/model/backbone")
            if res_bb and res_bb.status_code == 200:
                save_path = os.path.join(self.data_path, "models", "backbone.pth")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    size = f.write(res_bb.content)
                self.logger.info(f"Berhasil mengunduh backbone ({size} bytes).")
                
                # EAGER RELOAD: Terapkan backbone global ke RAM
                self._reload_inference_models(force_reload=True)
                return True
        except Exception as e:
            self.logger.error(f"Backbone fetch failed: {e}")
        return False

    def download_bn(self, max_wait=60):
        """Mengambil statistik BN (Running Mean/Var) hasil agregasi untuk konsistensi global."""
        path = os.path.join(self.data_path, "models", "global_bn_combined.pth")
        url = f"{self.server_api_url}/api/model/bn"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                # Jangan biarkan raise_for_status() melempar eksepsi untuk 404/202 di sini
                # agar kita bisa menanganinya dengan log yang lebih sopan.
                res = requests.get(url, timeout=10)
                
                if res.status_code == 200:
                    with open(path, "wb") as f:
                        f.write(res.content)
                    
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
            res_reg = self._safe_request("GET", f"{self.server_api_url}/api/model/registry")
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
            res = self._safe_request("GET", f"{self.server_api_url}/api/training/identities")
            if res and res.status_code == 200:
                global_users = res.json()
                for u in global_users:
                    user = db.query(UserLocal).filter_by(user_id=u['nrp']).first()
                    if not user:
                        user = UserLocal(user_id=u['nrp'], name=u['name'])
                        db.add(user)
                        db.commit()
            self.report_status("Siap Preprocess")
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

                # PENTING: Gambar di processed/ SUDAH berupa face crop 96x112.
                # Jangan panggil detect_face() lagi — MTCNN gagal mendeteksi wajah
                # di dalam crop yang sudah rapat (tidak ada margin dahi/leher).
                # Gunakan prepare_for_model() langsung sebagai penggantinya.
                input_tensor = None
                best_img_path = None
                for img_name in selected_paths:
                    trial_path = os.path.join(user_folder, img_name)
                    try:
                        img_pil = Image.open(trial_path).convert('RGB')
                        input_tensor = image_processor.prepare_for_model(img_pil)
                        if input_tensor is not None:
                            best_img_path = trial_path
                            self.logger.success(f"User {user.user_id}: Menggunakan crop langsung dari '{img_name}'.")
                            break
                    except Exception as e:
                        self.logger.error(f"Gagal memuat {img_name} saat refresh: {e}")
                        continue

                if input_tensor is None:
                    self.logger.warn(f"User {user.user_id}: Semua gambar gagal diproses.")
                    continue

                # Log kualitas gambar
                if best_img_path:
                    max_blur = image_processor.get_blur_score(best_img_path)
                    self.logger.info(f"User {user.user_id}: Skor ketajaman {max_blur:.1f}")

                input_tensor = input_tensor.to(self.device)
                
                with torch.no_grad():
                    self.backbone.eval()
                    # --- FLIP TRICK (Alignment dengan Inferensi) ---
                    emb_orig = self.backbone(input_tensor)
                    input_flipped = torch.flip(input_tensor, dims=[3])
                    emb_mirror = self.backbone(input_flipped)
                    
                    # Gabungkan dan Normalisasi L2
                    embedding_tensor = torch.nn.functional.normalize((emb_orig + emb_mirror) / 2, p=2, dim=1)
                    embedding_np = embedding_tensor.cpu().numpy()[0]
                    
                encrypted_data, iv = encryptor.encrypt_embedding(embedding_np)
                # Hapus entri global yang usang (is_global=True) untuk user ini
                # Embedding dari sesi training sebelumnya yang disimpan sebagai global
                # bisa menghasilkan orthogonal embedding dengan backbone saat ini.
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
                
                # Cleanup per user untuk hemat RAM Jetson
                del embedding_tensor
                gc.collect()
                
                # Bagikan ke server
                try:
                    payload = {
                        "nrp": user.user_id, "name": user.name, "client_id": self.client_id,
                        "embedding": base64.b64encode(embedding_np.tobytes()).decode('utf-8')
                    }
                    self._safe_request("POST", f"{self.server_api_url}/api/training/get_label", json=payload)
                except: pass
            
            self.logger.success("Local embeddings refresh complete.")
        except Exception as e:
            self.logger.error(f"Refresh Error: {e}")
        finally:
            db.close()


    def run_discovery_phase(self):
        """Pindai folder dan daftarkan ID ke server."""
        self.report_status("Processing: Discovery Identitas...")
        try:
            student_path = os.path.join(self.raw_data_path, "students")
            if not os.path.exists(student_path):
                # self.report_status("Error: Folder student tidak ditemukan")
                # return
                self._log(f"[DEBUG] Subfolder 'students' tidak ada, mencoba menggunakan root: {self.raw_data_path}")
                student_path = self.raw_data_path

            folders = [f for f in os.listdir(student_path) if os.path.isdir(os.path.join(student_path, f))]
            for folder in sorted(folders):
                nrp = folder.split('_')[0] if "_" in folder else folder
                name = folder.split('_')[1] if "_" in folder else nrp
                
                try:
                    self._safe_request("POST", f"{self.server_api_url}/api/training/get_label", json={
                        "nrp": nrp, "name": name, "client_id": self.client_id
                    })
                except: pass
                
                db = SessionLocal()
                try:
                    user = db.query(UserLocal).filter_by(user_id=nrp).first()
                    if not user:
                        db.add(UserLocal(user_id=nrp, name=name))
                        db.commit()
                finally: db.close()
            
            self._safe_request("POST", f"{self.server_api_url}/api/clients/discovery_done", json={"client_id": self.client_id})
            self.report_status("Discovery Selesai: Menunggu Global Map...")
        except Exception as e:
            self.logger.error(f"Discovery Error: {e}")
            self.report_status("Error Discovery")

    def sync_label_map(self):
        try:
            self._ensure_models_loaded()
            res = self._safe_request("GET", f"{self.server_api_url}/api/training/label_map")
            if res and res.status_code == 200:
                self.client.label_map = res.json()
                self.num_classes = len(self.client.label_map)
                self.logger.success(f"Peta label global berhasil disinkronkan. Total identitas: {self.num_classes}")
                
                # Expand head immediately if already loaded
                if self.head is not None:
                    current_head_classes = self.head.weight.shape[0]
                    if current_head_classes != self.num_classes:
                        self.logger.info(f"Triggering immediate head expansion ({current_head_classes} -> {self.num_classes})...")
                        new_label_map = {nrp: idx for idx, nrp in enumerate(self.client.label_map)}
                        self.head = self.client.trainer.update_head(self.num_classes, new_label_map)

                label_map_path = os.path.join(self.data_path, "models", "label_map.json")
                os.makedirs(os.path.dirname(label_map_path), exist_ok=True)
                
                with open(label_map_path, "w") as f:
                    json.dump(self.client.label_map, f)
                return True
        except Exception as e:
            self.logger.error(f"Gagal sinkronisasi label map: {e}")
        return False

    def run_preprocess_phase(self):
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
        for folder in folders:
            nrp = folder.split('_')[0] if "_" in folder else folder
            target_folder = os.path.join(processed_dir, nrp)
            
            # RESUME LOGIC: Jika sudah ada dan isinya cukup, lewati (save time)
            if os.path.exists(target_folder) and len(os.listdir(target_folder)) >= 5:
                self.logger.info(f"  [SKIP] {nrp} sudah diproses sebelumnya.")
                continue

            self.logger.info(f"Memilih 50 wajah terbaik untuk {nrp} (Laplacian Variance)...")
            self._log(f"[PREPROCESS] Memproses {nrp}: Seleksi Laplacian Variance...")
            self.fl_status = f"Processing: Laplacian {nrp}..."
            
            # ATOMIC LOGIC: Gunakan folder .tmp agar tidak corrupt jika mati mendadak
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
        
        # PENTING: Unload MTCNN setelah selesai untuk menghemat RAM (~150MB)
        image_processor.unload_detector()
        gc.collect()
        
        has_data = any(len(os.listdir(os.path.join(processed_dir, sub))) > 0 for sub in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, sub)))
        if has_data:
            # PENTING: Sebelum training, unduh Backbone & BN Global terbaru sebagai baseline.
            # Ini yang menjamin akurasi Ronde 1 bisa mencapai 0.9 karena model memulai 
            # dari titik optimal (Global Lens).
            self.logger.info("Menyiapkan bekal training: Mengunduh Backbone & BN Global...")
            self.download_backbone()
            self.download_bn(max_wait=10)
            
            # PENTING: Cleanup sebelum mulai hitung embedding/training
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            self.report_status("Siap Training")
            self.refresh_local_embeddings()
            
            try:
                self._log_to_file(f"[PREPROCESS] Client {self.client_id} is READY for training.")
                self._safe_request("POST", f"{self.server_api_url}/api/clients/ready", json={"client_id": self.client_id})
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
                self._safe_request("POST", f"{self.server_api_url}/api/clients/ready", json={"client_id": self.client_id})
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
            # 1. SINKRONISASI BACKBONE TERLEBIH DAHULU
            self.logger.info("Fase 1: Mengunduh Backbone Global dari server...")
            self.download_backbone()
            # pFedFace: Kita tidak lagi memuat BN global, gunakan BN lokal yang sudah ada atau dikalibrasi.
            # self.download_bn(max_wait=10) 
            
            # 2. HITUNG CENTROID menggunakan CALIBRATED BACKBONE
            # Sangat krusial agar embedding di database (Registry) 
            # sinkron dengan embedding saat Live Inference (pFedFace).
            self.logger.info("Fase 2: Menghitung Centroid (menggunakan Calibrated Backbone)...")
            try:
                # Pastikan backbone trainer menggunakan bobot global terbaru namun BN lokal
                # (Ini sudah dihandle oleh download_backbone + load_state_dict dengan filter pFedFace)
                self.client.trainer.backbone.eval()
                self.logger.info("Trainer backbone siap (Mode Eval untuk Centroid).")
            except Exception as e:
                self.logger.warn(f"Inisialisasi centroid gagal: {e}")

            bn_params = self.client.trainer.get_bn_parameters()
            centroids = self.client.trainer.calculate_centroids(label_map=self.client.label_map)
            
            # Simpan cadangan lokal segera
            local_registry_path = os.path.join(self.data_path, "models", "global_embedding_registry.pth")
            torch.save(centroids, local_registry_path)
            
            # SINKRONISASI IDENTITAS GLOBAL (Nama & NRP)
            self._sync_global_identities()
            
            # 3. KIRIM KE SERVER
            self.logger.info("Fase 3: Mengirim aset lokal ke server...")
            serialized_centroids = {nrp: base64.b64encode(vec.tobytes()).decode('utf-8') for nrp, vec in centroids.items()}
            bn_buf = io.BytesIO()
            torch.save(bn_params, bn_buf)
            serialized_bn = base64.b64encode(bn_buf.getvalue()).decode('utf-8')
            
            payload = {
                "client_id": self.client_id, "bn_params": serialized_bn, "centroids": serialized_centroids
            }
            res = self._safe_request("POST", f"{self.server_api_url}/api/training/registry_assets", json=payload, timeout=60)
            
            if res and res.status_code == 200:
                self._log_to_file("SUCCESS: Pengiriman aset lokal berhasil. Menunggu agregasi global...")
                self.report_status("Processing: Menunggu Agregasi Global...")
                
                # 4. TUNGGU & UNDUH HASIL GLOBAL
                # Ini memastikan setiap client memiliki BN dan Registry yang SAMA
                if self.download_bn(max_wait=3600):
                    self._log_to_file("REGISTRY: Global BN Synced.")
                
                if self._download_global_registry(max_wait=3600):
                    self._log_to_file("REGISTRY: Global Registry Synced.")
                    
                    # 5. ATOMIC VERSION SYNC: Pastikan versi diupdate SEBELUM reload
                    try:
                        resp = self._safe_request("GET", f"{self.server_api_url}/api/status", timeout=5)
                        if resp and resp.status_code == 200:
                            v = resp.json().get("model_version", 1)
                            self.logger.info(f"Syncing local version to v{v}")
                            self._save_version(v)
                    except: pass

                    # CRITICAL: Paksa reload backbone ke inferensi dengan versi global yang sudah disinkronisasi
                    self.logger.info("Finalisasi: Mereset backbone inferensi ke versi global agregasi...")
                    self._reload_inference_models(force_reload=True)
                    self.report_status("Siap Selesai")
                else:
                    self.logger.error("Global Registry download failed/timed out.")
                    self.report_status("Error: Registry Timeout")
            else:
                self.report_status(f"Error Submission: {res.status_code}")
                
        except Exception as e:
            self.logger.error(f"Registry Error: {e}")
            self.report_status("Error Registry")
        finally:
            self.is_sending_registry = False
            self.refresh_local_embeddings() # Final refresh with global BN

    def _download_global_registry(self, max_wait=3600):
        registry_url = f"{self.server_api_url}/api/model/registry"
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                res = self._safe_request("GET", registry_url, timeout=15)
                if res and res.status_code == 200:
                    save_path = os.path.join(self.data_path, "models", "global_embedding_registry.pth")
                    with open(save_path, "wb") as f:
                        f.write(res.content)
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
            self.report_status("Training: Flower FL...")
            success = False
            try:
                fl.client.start_client(server_address=self.fl_server_address, client=self.client.to_client())
                success = True
            except Exception as e:
                err_msg = str(e)
                # Deteksi error gRPC / Jaringan
                if "Rendezvous" in err_msg or "UNAVAILABLE" in err_msg:
                    self.report_status("Error: Koneksi Terputus")
                else:
                    self.report_status(f"Error: {err_msg[:30]}...")
            finally:
                self.is_training = False
                if success:
                    self.report_status("Online (Selesai)")
                # Reload model untuk menggunakan bobot global terbaru (EAGER RELOAD)
                self._reload_inference_models(force_reload=True)
        threading.Thread(target=run_client, daemon=True).start()

fl_manager = FLClientManager()
