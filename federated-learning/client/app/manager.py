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
                print(f"[INIT] Terdeteksi Model Versi: v{self.model_version}")
            except: pass

        # 2. Muat Jumlah Kelas dari Label Map (Sumber Kebenaran Utama)
        # Jika file ada, kita gunakan jumlah kelas nyata agar tidak terjadi '0 weights preserved' saat resize perdana
        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, 'r') as f:
                    data = json.load(f)
                    self.num_classes = len(data)
                    print(f"[INIT] Terdeteksi {self.num_classes} kelas dari label map lokal.")
            except: pass
        elif os.path.exists(self.head_path):
            try:
                checkpoint = torch.load(self.head_path, map_location="cpu")
                if "weight" in checkpoint:
                    self.num_classes = checkpoint["weight"].shape[0]
                    print(f"[INIT] Terdeteksi {self.num_classes} kelas dari head yang tersimpan.")
            except: pass

        # 3. Guard model initialization placeholder
        self.backbone = None # Lazy loaded
        self.head = None # Lazy loaded
        self.is_training_phase = False # Resource Guard

        self.fl_server_address = os.getenv("FL_SERVER_ADDRESS", "server-fl:8085")
        self.server_api_url = os.getenv("SERVER_API_URL", "http://server-fl:8080")
        
        # Load or Generate Persistent Identity
        self.client_id = self._load_identity()
        
        self.client = FaceRecognitionClient(
            None, None, 
            data_path=self.data_path, 
            device=self.device
        )
        self.client.fl_manager = self
        
        print("[INIT] Memulai Eager Loading model...")
        self._reload_inference_models(force_reload=True)
        
        # Sinkronisasi Final Status dari Persistence
        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, 'r') as f:
                    data = json.load(f)
                    self.client.label_map = data
                    if hasattr(self.client, 'trainer'):
                        self.client.trainer.nrp_to_idx = {nrp: idx for idx, nrp in enumerate(data)}
                        print(f"[INIT] Trainer label map restored ({len(data)} ids).")
            except: pass
        
        self.is_training = False
        self.current_phase = "idle"
        self.fl_status = "Online (Menunggu Instruksi)"
        self.fl_round = 0 # Track current round
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
        self.inference_threshold = 0.50

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

    def _sync_global_identities(self):
        """
        Sinkronisasi data NRP, Nama, dan Embedding Mahasiswa dari server.
        """
        try:
            print("[INFO] Sinkronisasi informasi identitas global dari server...")
            res = requests.get(f"{self.server_api_url}/api/training/identities", timeout=10)
            if res.status_code == 200:
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
                    print(f"[SUCCESS] Berhasil sinkronisasi {sync_count} identitas dan {emb_count} embedding global.")
                except Exception as db_err:
                    print(f"[ERROR] Gagal sinkronisasi basis data: {db_err}")
                    db.rollback()
                finally:
                    db.close()
        except Exception as e:
            print(f"[ERROR] Gagal mengambil identitas dari server: {e}")

    def _apply_backbone_weights(self, loaded, ignore_bn=True):
        """Menerapkan bobot ke backbone secara aman (mendukung state_dict dan list parameter)."""
        try:
            # Lazy Load Guard: Pastikan model sudah ada di RAM sebelum diisi bobot
            self._ensure_models_loaded()
            
            if self.backbone is None:
                # Fallback jika file checkpoint belum ada, buat model kosong
                print("[INFO] Inisialisasi backbone baru (Tanpa Checkpoint)...")
                self.backbone = MobileFaceNet().to(self.device)
                self.backbone.eval()
                if hasattr(self, 'client'):
                    self.client.model = self.backbone
                    self.client.trainer.backbone = self.backbone

            if isinstance(loaded, list):
                new_sd = self.backbone.state_dict()
                all_keys = list(new_sd.keys())
                
                # Identifikasi kunci konvolusi (untuk pFedFace/Flower NDArrays)
                if ignore_bn:
                    shared_keys = [k for k in all_keys if 
                                   not any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked'])
                                   and any(x in k.lower() for x in ['weight', 'bias'])]
                else:
                    shared_keys = all_keys
                
                if len(shared_keys) == len(loaded):
                    for k, v in zip(shared_keys, loaded):
                        new_sd[k] = torch.from_numpy(v).to(self.device)
                    print(f"[INFO] Berhasil menerapkan {len(loaded)} bobot konvolusi.")
                elif len(all_keys) == len(loaded):
                    for k, v in zip(all_keys, loaded):
                        new_sd[k] = torch.from_numpy(v).to(self.device)
                    print(f"[INFO] Berhasil menerapkan {len(loaded)} bobot full state_dict.")
                else:
                    print(f"[ERROR] Ketidakcocokan kunci: {len(loaded)} params vs {len(shared_keys)}/{len(all_keys)} keys.")
                
                self.backbone.load_state_dict(new_sd, strict=False)
            else:
                self.backbone.load_state_dict(loaded, strict=False)
                print(f"[SUCCESS] Backbone state-dict applied (BN ignored: {ignore_bn})")
            
            return True
        except Exception as e:
            print(f"[ERROR] Gagal menerapkan bobot model: {e}")
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
                    self.backbone.load_state_dict(torch.load(self.save_path, map_location=self.device))
                except Exception as e:
                    print(f"[LOAD ERROR] Backbone file mismatch: {e}")
                    
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
                print(f"[RELOAD] Loading backbone from {self.save_path}...")
                new_backbone.load_state_dict(torch.load(self.save_path, map_location=self.device))
                
                # PERSISTENCE FIX: Muat statistik BN hasil aggregasi jika ada
                bn_path = os.path.join(self.data_path, "models", "global_bn_combined.pth")
                if os.path.exists(bn_path):
                    try:
                        print(f"[RELOAD] Applying combined BN statistics from {bn_path}...")
                        bn_params = torch.load(bn_path, map_location=self.device)
                        new_backbone.load_state_dict(bn_params, strict=False)
                    except Exception as e:
                        print(f"[WARN] Gagal memuat BN stats: {e}")

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
            self.last_cache_update = 0
            gc.collect()
            
            print(f"[EAGER LOAD] Inference Mode: Global v{self.model_version} Ready.")
        except Exception as e:
            print(f"[ERROR] Gagal melakukan Eager Load: {e}")
            traceback.print_exc()

    def start_background_tasks(self):
        print(f"[INIT] Memulai tugas latar belakang untuk client: {self.client_id}")
        threading.Thread(target=self.run_background_sync, daemon=True).start()
        threading.Thread(target=self.run_camera_loop, daemon=True).start()

    def run_camera_loop(self):
        # Loop kamera mandiri (Headless Mode)
        cam_idx = self.camera_index
        cam_width = int(os.getenv("CAMERA_WIDTH", 1280))
        cam_height = int(os.getenv("CAMERA_HEIGHT", 720))
        cam_format = os.getenv("CAMERA_FORMAT", "MJPG").upper()

        print(f"[CAMERA] Menjalankan loop kamera otomatis (FL) pada index {cam_idx}...")
        cap = cv2.VideoCapture(cam_idx)
        
        # Optimasi Jetson/Raspi: Paksa format MJPG jika didukung (lebih ringan dari YUYV)
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
        
        attendance_engine = AttendanceController(self)
        
        while self.is_camera_running:
            ret, frame = False, None
            if not virtual_mode:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Gagal akses hardware. Beralih ke mode VIRTUAL CAMERA.")
                    virtual_mode = True
                    # Pindai gambar virtual
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
                    cv2.putText(frame, "VIRTUAL MODE (FL)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                time.sleep(1.0) # Lambatkan simulasi
            
            if not ret:
                if not virtual_mode: print("[ERROR] Gagal membaca frame dari kamera. Periksa /dev/video0.")
                time.sleep(5)
                continue
            
            # Training/Preprocessing Guard
            if self.is_training_phase:
                if cap.isOpened():
                    print("[CAMERA] Melepaskan hardware kamera untuk menghemat daya/RAM selama training.")
                    cap.release()
                self.latest_result["matched"] = "TRAINING PHASE..."
                time.sleep(5)
                continue
            else:
                if not cap.isOpened() and not virtual_mode:
                    print("[CAMERA] Mengaktifkan kembali hardware kamera...")
                    cap = cv2.VideoCapture(cam_idx)

            # Inisialisasi model hanya jika diperlukan (Lazy Load)
            if not self.backbone:
                self._ensure_models_loaded()

            # Simpan frame terbaru untuk streaming MJPEG
            self.latest_frame = frame.copy()
            
            # Lakukan pemrosesan jika model sudah siap (Inference)
            # Gunakan inference_backbone agar tidak terganggu drift training ronde
            if self.inference_backbone:
                start_time = time.time()
                try:
                    # Konversi ke PIL Image
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    matched, confidence = self.attendance.recognize_directly(img_pil)
                    
                    status_str = f"v{self.model_version}"
                    if self.is_training and self.fl_round > 0:
                        status_str += f" (R{self.fl_round}/15)"

                    self.latest_result = {
                        "matched": matched,
                        "confidence": confidence,
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "model_version": status_str,
                        "is_virtual": virtual_mode
                    }
                except Exception as e:
                    pass
            
            if not virtual_mode: time.sleep(0.5)
        
        cap.release()

    def report_status(self, status=None):
        if status: self.fl_status = status
        now = time.time()
        
        try:
            self.last_register_attempt = now
            payload = {
                "id": self.client_id,
                "ip_address": socket.gethostbyname(socket.gethostname()),
                "fl_status": self.fl_status,
                "last_seen": now
            }
            res = requests.post(f"{self.server_api_url}/api/clients/register", json=payload, timeout=2)
            if res.status_code == 200:
                self.is_registered = True
                print(f"[SUCCESS] Client {self.client_id} berhasil terdaftar di server.")
        except:
            self.is_registered = False

    def run_background_sync(self):
        print(f"[INFO] Layanan sinkronisasi (heartbeat) dimulai untuk {self.client_id}")
        while True:
            try:
                self.report_status()
                
                # 2. Cek Versi Model
                resp = requests.get(f"{self.server_api_url}/api/training/status", timeout=2)
                if resp.status_code == 200:
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
                print(f"[ERROR] Masalah pada loop sinkronisasi: {e}")
            time.sleep(5)

    def handle_phase_transition(self, phase):
        phase = phase.lower()
        print(f"[CLIENT] Phase Transition: {self.last_phase} -> {phase}")
        
        if phase == "discovery":
            self.is_training_phase = True
            threading.Thread(target=self.run_discovery_phase).start()
        elif phase == "syncing":
            self.is_training_phase = True
            threading.Thread(target=self.run_sync_phase).start()
        elif phase in ["training", "training phase"]:
            self.is_training_phase = True
            threading.Thread(target=self.start_fl, daemon=True).start()
        elif phase in ["registry generation", "registry_generation"]:
             self.is_training_phase = True
             threading.Thread(target=self.run_registry_phase).start()
        if phase == "idle" or phase == "completed":
            self.is_training_phase = False
            self.fl_status = "Online (Selesai)"
            
        # PENTING: Gunakan OR untuk menangani race condition phase -> idle yang terlalu cepat
        if (self.last_phase == "training" or self.last_phase == "registry generation" or self.last_phase == "idle") and (phase == "completed" or phase == "idle"):
            # Cek apakah memang ada kenaikan versi di server
            try:
                resp = requests.get(f"{self.server_api_url}/api/status", timeout=2)
                server_v = resp.json().get("model_version", self.model_version) if resp.status_code == 200 else self.model_version
                if server_v <= self.model_version and not (phase == "completed"):
                    return # Tidak perlu sync jika sudah up to date dan bukan fase completion eksplisit
            except: pass

            print("[INFO] Pelatihan selesai atau Update tersedia. Memulai Sinkronisasi Final...")
            def update_task():
                try:
                    # 1. Unduh Backbone global terakhir
                    if not self.download_backbone():
                        print("[ERROR] Gagal mengunduh backbone final.")
                    
                    # 2. Unduh BN global terakhir
                    if not self.download_bn():
                        print("[ERROR] Gagal mengunduh BN final.")
                        
                    # 3. Unduh Registry Centroid 
                    if not self.download_registry_assets():
                        print("[ERROR] Gagal mengunduh registry final.")
                        
                    # 4. Ambil versi terbaru dan update metadata
                    try:
                        resp = requests.get(f"{self.server_api_url}/api/status", timeout=5)
                        if resp.status_code == 200:
                            v = resp.json().get("model_version", self.model_version)
                            print(f"[SYNC] Menetapkan versi lokal ke v{v}")
                            self._save_version(v)
                    except Exception as e:
                        print(f"[WARNING] Gagal update nomor versi: {e}")

                    # 5. FORCE RELOAD: Pastikan inferensi menggunakan v1 segera
                    self._reload_inference_models(force_reload=True)
                    
                    # 6. Selesaikan label map dan refresh lokal
                    self.sync_label_map()
                    self.refresh_local_embeddings()
                    print("[SUCCESS] Siklus FL selesai, sistem siap untuk inferensi v-terbaru.")
                    
                except Exception as e:
                    print(f"[CRITICAL] Sinkronisasi pasca-pelatihan gagal: {e}")
            
            threading.Thread(target=update_task, daemon=True).start()

    def _save_version(self, v):
        self.model_version = v
        v_path = os.path.join(self.data_path, "models", "model_version.txt")
        with open(v_path, "w") as f:
            f.write(str(v))

    def download_backbone(self):
        """Mengambil StateDict Backbone hasil agregasi dari server."""
        try:
            url_bb = f"{self.server_api_url}/api/model/backbone"
            print(f"[INFO] Mengambil backbone global dari server...")
            res_bb = requests.get(url_bb, timeout=30)
            if res_bb.status_code == 200:
                save_path = os.path.join(self.data_path, "models", "backbone.pth")
                with open(save_path, "wb") as f:
                    size = f.write(res_bb.content)
                print(f"[SYNC] Berhasil mengunduh backbone ({size} bytes).")
                
                # EAGER RELOAD: Terapkan backbone global ke RAM
                self._reload_inference_models(force_reload=True)
                return True
            else:
                print(f"[SYNC ERROR] Server returned status code {res_bb.status_code} for backbone.")
        except Exception as e:
            print(f"[SYNC ERROR] Backbone fetch failed: {e}")
        return False

    def download_bn(self, max_wait=60):
        """Mengambil statistik BN (Running Mean/Var) hasil agregasi untuk konsistensi global."""
        path = os.path.join(self.data_path, "models", "global_bn_combined.pth")
        url = f"{self.server_api_url}/api/model/bn"
        
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    with open(path, "wb") as f:
                        f.write(res.content)
                    
                    # Verify binary
                    try:
                        bn_params = torch.load(path, map_location=self.device)
                        self.backbone.load_state_dict(bn_params, strict=False)
                        print(f"[SUCCESS] Parameter BN global berhasil diterapkan ke backbone.")
                        return True
                    except Exception as e:
                        print(f"[ERROR] Berkas BN tidak valid, mencoba kembali: {e}")
                elif res.status_code == 202 or res.status_code == 404:
                    print(f"[INFO] BN belum siap di server (Status {res.status_code}), menunggu...")
                else:
                    print(f"[ERROR] Unduhan BN gagal dengan status: {res.status_code}")
                    return False
            except Exception as e:
                print(f"[ERROR] Gagal mengunduh BN: {e}")
            time.sleep(5)
        return False

    def download_registry_assets(self):
        """Unduh centroid identifikasi gabungan untuk inferensi offline."""
        try:
            url_reg = f"{self.server_api_url}/api/model/registry"
            res_reg = requests.get(url_reg, timeout=20)
            if res_reg.status_code == 200:
                reg_path = os.path.join(self.data_path, "models", "global_embedding_registry.pth")
                with open(reg_path, "wb") as f:
                    f.write(res_reg.content)
                print("[SUCCESS] Registry Centroid berhasil diperbarui.")
                return True
            else:
                print(f"[INFO] Unduhan registry dilewati (Status {res_reg.status_code})")
        except Exception as e:
            print(f"[ERROR] Gagal memperbarui Registry: {e}")
        return False

    def run_sync_phase(self):
        self.report_status("Processing: Sinkronisasi Data...")
        self.download_backbone()
        self.sync_label_map()

        db = SessionLocal()
        try:
            res = requests.get(f"{self.server_api_url}/api/users/global", timeout=10)
            if res.status_code == 200:
                global_users = res.json()
                for u in global_users:
                    user = db.query(UserLocal).filter_by(user_id=u['nrp']).first()
                    if not user:
                        user = UserLocal(user_id=u['nrp'], name=u['name'])
                        db.add(user)
                        db.commit()
            self.report_status("Siap Preprocess")
        except Exception as e:
            print(f"[SYNC ERROR] {e}")
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
                
                # Muat daftar gambar yang tersedia
                imgs = [f for f in os.listdir(user_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if not imgs: continue
                
                # FUNCTIONAL PARITY: Pilih gambar terbaik menggunakan ImageProcessor
                # Coba hingga 5 gambar terbaik jika deteksi wajah gagal
                selected_paths = image_processor.select_best_faces(user_folder, n=5)
                print(f"[REFRESH] User {user.user_id}: Menyeleksi {len(selected_paths)} gambar terbaik untuk ekstraksi.")
                
                face_img = None
                best_img_path = None
                
                for i, img_name in enumerate(selected_paths):
                    trial_path = os.path.join(user_folder, img_name)
                    try:
                        img_pil = Image.open(trial_path).convert('RGB')
                        # FUNCTIONAL PARITY: Ambil crop wajah tertajam
                        face_img, _, _ = image_processor.detect_face(img_pil)
                        if face_img is not None:
                            best_img_path = trial_path
                            print(f"[REFRESH] [SUCCESS] Wajah user {user.user_id} terdeteksi (Percobaan {i+1}, File: {img_name}).")
                            break
                        else:
                            print(f"[REFRESH] [RETRY] Percobaan {i+1} gagal untuk {user.user_id} ({img_name}). Mencoba gambar lain...")
                    except Exception as e:
                        print(f"[REFRESH ERROR] Gagal memuat {img_name}: {e}")
                        continue

                if face_img is None:
                    print(f"[REFRESH] [FAILED] User {user.user_id}: Wajah tidak terdeteksi di {len(selected_paths)} gambar terbaik.")
                    continue

                # Ambil skor untuk logging (opsional)
                max_blur = image_processor.get_blur_score(best_img_path)
                print(f"[REFRESH] User {user.user_id}: Menggunakan gambar kualitas {max_blur:.1f}")

                input_tensor = image_processor.prepare_for_model(face_img).to(self.device)
                
                with torch.no_grad():
                    self.backbone.eval()
                    embedding_tensor = self.backbone(input_tensor)
                    embedding_tensor = torch.nn.functional.normalize(embedding_tensor, p=2, dim=1)
                    embedding_np = embedding_tensor.cpu().numpy()[0]
                    
                encrypted_data, iv = encryptor.encrypt_embedding(embedding_np)
                emb_record = db.query(EmbeddingLocal).filter_by(user_id=user.user_id, is_global=False).first()
                if emb_record:
                    emb_record.embedding_data = encrypted_data
                    emb_record.iv = iv
                else:
                    db.add(EmbeddingLocal(user_id=user.user_id, embedding_data=encrypted_data, iv=iv, is_global=False))
                db.commit()
                
                # Cleanup per user for Jetson RAM
                del embedding_tensor
                # RAM Cleanup per user (Pure CPU)
                gc.collect()
                
                # Bagikan ke server
                try:
                    payload = {
                        "nrp": user.user_id, "name": user.name, "client_id": self.client_id,
                        "embedding": base64.b64encode(embedding_np.tobytes()).decode('utf-8')
                    }
                    requests.post(f"{self.server_api_url}/api/training/get_label", json=payload, timeout=5)
                except: pass
            
            print("[REFRESH] Local embeddings refresh complete.")
        except Exception as e:
            print(f"[REFRESH ERROR] {e}")
        finally:
            db.close()


    def run_discovery_phase(self):
        """Pindai folder dan daftarkan ID ke server."""
        self.report_status("Processing: Discovery Identitas...")
        try:
            student_path = os.path.join(self.raw_data_path, "students")
            if not os.path.exists(student_path):
                self.report_status("Error: Folder student tidak ditemukan")
                return

            folders = [f for f in os.listdir(student_path) if os.path.isdir(os.path.join(student_path, f))]
            for folder in sorted(folders):
                nrp = folder.split('_')[0] if "_" in folder else folder
                name = folder.split('_')[1] if "_" in folder else nrp
                
                try:
                    requests.post(f"{self.server_api_url}/api/training/get_label", json={
                        "nrp": nrp, "name": name, "client_id": self.client_id
                    }, timeout=5)
                except: pass
                
                db = SessionLocal()
                try:
                    user = db.query(UserLocal).filter_by(user_id=nrp).first()
                    if not user:
                        db.add(UserLocal(user_id=nrp, name=name))
                        db.commit()
                finally: db.close()
            
            requests.post(f"{self.server_api_url}/api/clients/discovery_done", json={"client_id": self.client_id}, timeout=5)
            self.report_status("Discovery Selesai: Menunggu Global Map...")
        except Exception as e:
            print(f"[DISCOVERY ERROR] {e}")
            self.report_status("Error Discovery")

    def sync_label_map(self):
        try:
            self._ensure_models_loaded()
            res = requests.get(f"{self.server_api_url}/api/training/label_map", timeout=10)
            if res.status_code == 200:
                self.client.label_map = res.json()
                self.num_classes = len(self.client.label_map)
                print(f"[SYNC] Global label map synced. Total classes: {self.num_classes}")
                
                # Expand head immediately if already loaded
                if self.head is not None:
                    current_head_classes = self.head.weight.shape[0]
                    if current_head_classes != self.num_classes:
                        print(f"[SYNC] Triggering immediate head expansion ({current_head_classes} -> {self.num_classes})...")
                        new_label_map = {nrp: idx for idx, nrp in enumerate(self.client.label_map)}
                        self.head = self.client.trainer.update_head(self.num_classes, new_label_map)

                map_path = os.path.join(self.data_path, "models", "label_map.json")
                with open(map_path, "w") as f:
                    json.dump(self.client.label_map, f)
                return True
        except Exception as e:
            print(f"[SYNC ERROR] Gagal sinkronisasi label map: {e}")
        return False

    def run_preprocess_phase(self):
        self.report_status("Processing: Ekstraksi Wajah (Laplacian Top 50)...")
        self._ensure_models_loaded()
        if not self.sync_label_map():
            self.report_status("Error: Gagal Sinkronisasi Global Map")
            return

        students_dir = os.path.join(self.raw_data_path, "students")
        processed_dir = os.path.join(self.data_path, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        if not os.path.exists(students_dir):
            self.report_status("Siap Training (No Data)")
            return

        folders = sorted([f for f in os.listdir(students_dir) if os.path.isdir(os.path.join(students_dir, f))])
        for folder in folders:
            nrp = folder.split('_')[0] if "_" in folder else folder
            print(f"[INFO] Memilih 50 wajah terbaik untuk {nrp} menggunakan Laplacian Variance...")
            
            target_folder = os.path.join(processed_dir, nrp)
            if os.path.exists(target_folder):
                shutil.rmtree(target_folder, ignore_errors=True)
            os.makedirs(target_folder, exist_ok=True)
            
            source_path = os.path.join(students_dir, folder)
            top_images = image_processor.select_best_faces(source_path, n=50)
            
            count = 0
            for img_name in top_images:
                try:
                    img_path = os.path.join(source_path, img_name)
                    img_pil = Image.open(img_path).convert('RGB')
                    target_path = os.path.join(target_folder, f"face_{count}.jpg")
                    
                    # Gunakan MTCNN dari image_processor (Portrait Cropping 96x112)
                    face_img, _, _ = image_processor.detect_face(img_pil, save_path=target_path)
                    if face_img is not None:
                        count += 1
                except: pass
        
        # PENTING: Unload MTCNN setelah selesai untuk menghemat RAM (~150MB)
        image_processor.unload_detector()
        gc.collect()
        
        has_data = any(len(os.listdir(os.path.join(processed_dir, sub))) > 0 for sub in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, sub)))
        if has_data:
            self.report_status("Siap Training")
            self.refresh_local_embeddings()
            
            try:
                requests.post(f"{self.server_api_url}/api/clients/ready", json={"client_id": self.client_id}, timeout=5)
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
                requests.post(f"{self.server_api_url}/api/clients/ready", json={"client_id": self.client_id}, timeout=5)
            except: pass

    def run_registry_phase(self):
        """Ekstraksi Registry FINAL (Menggunakan Bobot Global Terpadu)."""
        if getattr(self, "is_sending_registry", False):
            print("[REGISTRY] Generation already in progress, skipping duplicate trigger.")
            return

        self._ensure_models_loaded()
        self.is_sending_registry = True
        self.report_status("Processing: Finalisasi Registry Identitas...")
        try:
            # 1. SINKRONISASI BACKBONE & BN TERLEBIH DAHULU (PENTING!)
            # Ini menjamin ekstraksi fitur menggunakan "lensa" global yang seragam.
            print("[INFO] Fase 1: Mengunduh Backbone & BN Global dari server...")
            self.download_backbone()
            # Catatan: Kita memanggil download_bn secara sinkron di sini
            # karena kita butuh stats BN yang teragregasi ROUND SEBELUMNYA sebagai baseline yang stabil.
            self.download_bn(max_wait=10) 
            
            # 2. HITUNG CENTROID (Menggunakan status model saat ini)
            print("[INFO] Fase 2: Menghitung Centroid Lokal (BN-Aligned)...")
            bn_params = self.client.trainer.get_bn_parameters()
            centroids = self.client.trainer.calculate_centroids(label_map=self.client.label_map)
            
            # Simpan cadangan lokal segera
            local_registry_path = os.path.join(self.data_path, "models", "global_embedding_registry.pth")
            torch.save(centroids, local_registry_path)
            
            # SINKRONISASI IDENTITAS GLOBAL (Nama & NRP)
            self._sync_global_identities()
            
            # 3. KIRIM KE SERVER
            print("[REGISTRY] Fase 3: Mengirim aset lokal ke server...")
            serialized_centroids = {nrp: base64.b64encode(vec.tobytes()).decode('utf-8') for nrp, vec in centroids.items()}
            bn_buf = io.BytesIO()
            torch.save(bn_params, bn_buf)
            serialized_bn = base64.b64encode(bn_buf.getvalue()).decode('utf-8')
            
            payload = {
                "client_id": self.client_id, "bn_params": serialized_bn, "centroids": serialized_centroids
            }
            res = requests.post(f"{self.server_api_url}/api/training/registry_assets", json=payload, timeout=60)
            
            if res.status_code == 200:
                print(f"[INFO] Pengiriman aset berhasil. Menunggu agregasi global...")
                self.report_status("Processing: Menunggu Agregasi Global...")
                
                # 4. TUNGGU & UNDUH HASIL GLOBAL
                # Ini memastikan setiap client memiliki BN dan Registry yang SAMA
                if self.download_bn(max_wait=120):
                    print("[REGISTRY] Global BN Synced.")
                
                if self._download_global_registry(max_wait=120):
                    print("[REGISTRY] Global Registry Synced.")
                    
                    # 5. ATOMIC VERSION SYNC: Pastikan versi diupdate SEBELUM reload
                    try:
                        resp = requests.get(f"{self.server_api_url}/api/status", timeout=5)
                        if resp.status_code == 200:
                            v = resp.json().get("model_version", 1)
                            print(f"[REGISTRY] Syncing local version to v{v}")
                            self._save_version(v)
                    except: pass

                    # CRITICAL: Paksa reload backbone ke inferensi dengan versi global yang sudah disinkronisasi
                    print("[INFO] Finalisasi: Mereset backbone inferensi ke versi global agregasi...")
                    self._reload_inference_models(force_reload=True)
                    self.report_status("Siap Selesai")
                else:
                    print("[REGISTRY] Global Registry download failed/timed out.")
                    self.report_status("Error: Registry Timeout")
            else:
                self.report_status(f"Error Submission: {res.status_code}")
                
        except Exception as e:
            print(f"[REGISTRY ERROR] {e}")
            self.report_status("Error Registry")
        finally:
            self.is_sending_registry = False
            self.refresh_local_embeddings() # Final refresh with global BN

    def _download_global_registry(self, max_wait=60):
        registry_url = f"{self.server_api_url}/api/model/registry"
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                res = requests.get(registry_url, timeout=15)
                if res.status_code == 200:
                    save_path = os.path.join(self.data_path, "models", "global_embedding_registry.pth")
                    with open(save_path, "wb") as f:
                        f.write(res.content)
                    if hasattr(self, 'cached_refs'):
                        del self.cached_refs
                    print(f"[REGISTRY] Global registry downloaded ({len(res.content)//1024} KB).")
                    return True
                elif res.status_code == 202:
                    print("[REGISTRY] Server aggregating... waiting 5s")
                    time.sleep(5)
                else:
                    print(f"[REGISTRY] Download failed: {res.status_code}")
                    return False
            except Exception as e:
                print(f"[ERROR] Gagal mengunduh registry: {e}")
                time.sleep(5)
        print("[ERROR] Batas waktu unduhan registry tercapai.")
        return False

    def start_fl(self):
        if self.is_training: return
        self.is_training = True
        def run_client():
            self.report_status("Training: Flower FL...")
            try:
                fl.client.start_client(server_address=self.fl_server_address, client=self.client.to_client())
            except Exception as e:
                self.report_status(f"Error: {str(e)[:30]}...")
            finally:
                self.is_training = False
                self.report_status("Online (Selesai)")
                # Reload model untuk menggunakan bobot global terbaru (EAGER RELOAD)
                self._reload_inference_models(force_reload=True)
        threading.Thread(target=run_client, daemon=True).start()

fl_manager = FLClientManager()
