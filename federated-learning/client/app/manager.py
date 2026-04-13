import os
import json
import torch
import threading
import collections
import requests
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
from app.utils.model_exporter import export_backbone_to_onnx

def add_phase_log(msg):
    print(f"[LOG] {msg}")

class FLClientManager:
    # Manajer Utama Client Federated (FL)
    # Menangani jantung operasional client mulai dari sinkronisasi fase,
    # manajemen model lokal, hingga alur kamera (inference).
    def __init__(self):
        self.raw_data_path = os.getenv("RAW_DATA_PATH", "/app/raw_data")
        self.data_path = os.getenv("DATA_PATH", "/app/data")
        # Pastikan direktori tersedia
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(os.path.join(self.data_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.data_path, "processed"), exist_ok=True)

        self.device = torch.device("cpu")
        self.backbone = MobileFaceNet().to(self.device)
        self.backbone.eval()
        
        # PERSISTENSI: Tentukan ukuran head dinamis
        save_path = os.path.join(self.data_path, "models", "backbone.pth")
        head_path = os.path.join(self.data_path, "models", "local_head.pth")
        
        self.num_classes = 1000 # Fallback default
        if os.path.exists(head_path):
            try:
                # Cegah korupsi file saat startup
                checkpoint = torch.load(head_path, map_location="cpu")
                if "weight" in checkpoint:
                    self.num_classes = checkpoint["weight"].shape[0]
                    print(f"[STARTUP] Detected {self.num_classes} classes from saved head.")
                del checkpoint
            except Exception as e:
                print(f"[STARTUP WARN] Failed to inspect head: {e}")
            
        self.head = ArcMarginProduct(128, self.num_classes).to(self.device)
        self.model_version = 0
        
        if os.path.exists(save_path):
            try:
                print(f"Loading existing backbone weights from {save_path}...")
                self.backbone.load_state_dict(torch.load(save_path, map_location=self.device))
                
                # MUAT BN GABUNGAN (Legacy logic restored)
                # Memastikan model memiliki statistik normalisasi global sejak awal
                bn_combined_path = os.path.join(self.data_path, "models", "global_bn_combined.pth")
                if os.path.exists(bn_combined_path):
                    print("Loading Combined Global BN statistics...")
                    try:
                        self.backbone.load_state_dict(torch.load(bn_combined_path, map_location=self.device), strict=False)
                    except: pass

                v_path = os.path.join(self.data_path, "models", "model_version.txt")
                if os.path.exists(v_path):
                    with open(v_path, "r") as f:
                        self.model_version = int(f.read().strip())
                        print(f"[STARTUP] Model Version v{self.model_version}")
                else:
                    self.model_version = int(os.path.getmtime(save_path)) % 1000
            except: pass

        self.fl_server_address = os.getenv("FL_SERVER_ADDRESS", "server-fl:8085")
        self.server_api_url = os.getenv("SERVER_API_URL", "http://server-fl:8080")
        self.client_id = os.getenv("HOSTNAME", "client-1")
        
        self.client = FaceRecognitionClient(
            self.backbone, self.head, 
            data_path=self.data_path, 
            device=self.device
        )
        self.client.fl_manager = self
        
        self.is_training = False
        self.current_phase = "idle"
        self.fl_status = "Online (Menunggu Instruksi)"
        self.last_phase = "idle"
        
        self.is_registered = False
        self.last_register_attempt = 0
        self.register_retry_delay = 30
        
        self.prediction_buffer = collections.deque(maxlen=10)
        self.last_face_time = 0
        
        # Dukungan Kamera Headless
        self.latest_frame = None
        self.latest_result = {"matched": "Standby", "confidence": 0, "latency_ms": 0, "is_virtual": False}
        self.is_camera_running = False
        
        # Ambang Batas Inferensi (Dinamis dari Server)
        self.inference_threshold = 0.5
        
        # Inisialisasi model inferensi
        self._reload_inference_models()

    def _sync_global_identities(self):
        """
        Sinkronisasi data NRP, Nama, dan Embedding Mahasiswa dari server untuk 
        memastikan seluruh terminal memiliki referensi identitas yang sama.
        """
        try:
            print("[SYNC] Sinkronisasi info identitas global dari server...")
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
                    print(f"[SYNC] Berhasil sinkronisasi {sync_count} identitas dan {emb_count} embedding global.")
                except Exception as db_err:
                    print(f"[SYNC ERROR] Database sync failed: {db_err}")
                    db.rollback()
                finally:
                    db.close()
        except Exception as e:
            print(f"[SYNC ERROR] Failed to fetch identities: {e}")

    def _apply_backbone_weights(self, loaded):
        """Menerapkan bobot ke backbone secara aman (mendukung state_dict dan list parameter)."""
        try:
            if isinstance(loaded, list):
                new_sd = self.backbone.state_dict()
                all_keys = list(new_sd.keys())
                
                # Identifikasi kunci konvolusi (untuk pFedFace/Flower NDArrays)
                shared_keys = [k for k in all_keys if 
                               not any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked'])
                               and any(x in k.lower() for x in ['weight', 'bias'])]
                
                if len(shared_keys) == len(loaded):
                    for k, v in zip(shared_keys, loaded):
                        new_sd[k] = torch.from_numpy(v).to(self.device)
                    print(f"[MODEL] Applied {len(loaded)} conv weights.")
                elif len(all_keys) == len(loaded):
                    for k, v in zip(all_keys, loaded):
                        new_sd[k] = torch.from_numpy(v).to(self.device)
                    print(f"[MODEL] Applied {len(loaded)} full state_dict weights.")
                else:
                    print(f"[MODEL ERROR] Key mismatch: {len(loaded)} params vs {len(shared_keys)}/{len(all_keys)} keys.")
                
                self.backbone.load_state_dict(new_sd, strict=False)
            else:
                self.backbone.load_state_dict(loaded, strict=False)
            
            return True
        except Exception as e:
            print(f"[MODEL ERROR] Failed to apply weights: {e}")
            return False

    def _reload_inference_models(self):
        """Muat kembali model dan ekspor ke ONNX untuk efisiensi RAM & CPU."""
        try:
            save_path = os.path.join(self.data_path, "models", "backbone.pth")
            head_path = os.path.join(self.data_path, "models", "local_head.pth")
            v_path = os.path.join(self.data_path, "models", "model_version.txt")
            
            # 1. Sync Versi Model
            if os.path.exists(v_path):
                try:
                    with open(v_path, "r") as f:
                        self.model_version = int(f.read().strip())
                except: pass

            # 2. Refresh Backbone
            if os.path.exists(save_path):
                print(f"[RELOAD] Refreshing backbone weights from {save_path} (v{self.model_version})")
                loaded = torch.load(save_path, map_location=self.device)
                self._apply_backbone_weights(loaded)
                del loaded
            
            # 3. Refresh Head (Dinamis Resizing)
            if os.path.exists(head_path):
                print(f"[RELOAD] Refreshing local head from {head_path}")
                try:
                    head_sd = torch.load(head_path, map_location=self.device)
                    if "weight" in head_sd:
                        new_classes = head_sd["weight"].shape[0]
                        if new_classes != self.head.weight.shape[0]:
                            print(f"[RELOAD] Resizing head: {self.head.weight.shape[0]} -> {new_classes}")
                            # Update label map agar trainer tahu index mana yang harus di-copy
                            new_label_map = {nrp: idx for idx, nrp in enumerate(self.client.label_map)} if self.client.label_map else {}
                            self.head = self.client.trainer.update_head(new_classes, new_label_map)
                            self.client.head = self.head # Sync ke client flower
                    
                    self.head.load_state_dict(head_sd)
                    del head_sd
                except Exception as head_err:
                    print(f"[RELOAD WARN] Head load failed (mismatch/corrupt): {head_err}")
            
            # PENTING: Segarkan registry agar vektor ciri (embeddings) selaras dengan backbone terbaru
            self.refresh_local_embeddings()
            
            # Ekspor ke ONNX secara background agar tidak menghambat sistem
            if os.path.exists(save_path):
                def bg_export():
                    try:
                        export_backbone_to_onnx(save_path, os.path.join(self.data_path, "models"), device=self.device)
                    except Exception as e:
                        print(f"[RELOAD ERROR] Background ONNX export failed: {e}")
                threading.Thread(target=bg_export, daemon=True).start()
                
        except Exception as e:
            print(f"[RELOAD ERROR] {e}")
        finally:
            gc.collect()

    def start_background_tasks(self):
        print(f"[STARTUP] Memulai tugas latar belakang untuk client: {self.client_id}")
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        # Loop kamera mandiri (Headless Mode)
        print(f"[CAMERA] Menjalankan loop kamera otomatis (FL)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
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
                    print("[CAMERA ERROR] Gagal akses hardware. Beralih ke VIRTUAL CAMERA mode.")
                    virtual_mode = True
                    # Pindai gambar virtual
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
                    cv2.putText(frame, "VIRTUAL MODE (FL)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                time.sleep(1.0) # Lambatkan simulasi
            
            if not ret:
                if not virtual_mode: print("[CAMERA ERROR] Gagal membaca frame dari kamera. Cek mapping device /dev/video0.")
                time.sleep(5)
                continue
            
            # Simpan frame terbaru untuk streaming MJPEG
            self.latest_frame = frame.copy()
            
            # Lakukan pemrosesan jika model sudah siap (Inference)
            # Pastikan registry sudah ada
            reg_path = os.path.join(self.data_path, "models", "global_embedding_registry.pth")
            if os.path.exists(reg_path):
                start_time = time.time()
                try:
                    # Konversi ke PIL Image
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    
                    matched, confidence = attendance_engine.recognize_directly(img_pil)
                    
                    self.latest_result = {
                        "matched": matched,
                        "confidence": confidence,
                        "latency_ms": int((time.time() - start_time) * 1000),
                        "model_version": self.model_version,
                        "is_virtual": virtual_mode
                    }
                except Exception as e:
                    # print(f"[CAMERA ERROR] {e}") # Terlalu berisik untuk log loop
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
                print(f"[OK] Client {self.client_id} berhasil terdaftar.")
        except:
            self.is_registered = False

    def heartbeat_loop(self):
        print(f"[CLIENT] Heartbeat service started for {self.client_id}")
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

                    if phase != self.last_phase:
                        self.handle_phase_transition(phase)
                        self.last_phase = phase
            except Exception as e:
                print(f"Heartbeat Error: {e}")
            time.sleep(5)

    def handle_phase_transition(self, phase):
        phase = phase.lower()
        print(f"[CLIENT] Phase Transition: {self.last_phase} -> {phase}")
        
        if phase == "discovery":
            threading.Thread(target=self.run_discovery_phase).start()
        elif phase == "syncing":
            threading.Thread(target=self.run_sync_phase).start()
        elif phase in ["training", "training phase"]:
            self.start_fl()
        elif phase in ["registry generation", "registry_generation"]:
             threading.Thread(target=self.run_registry_phase).start()
        elif phase == "idle" or phase == "completed":
            self.fl_status = "Online (Selesai)"
            
        if (self.last_phase == "training" or self.last_phase == "registry generation") and phase == "completed":
            print("[CLIENT] Pelatihan selesai. Mengunduh aset Registry Final...")
            def update_task():
                if self.download_registry_assets():
                    # Ambil versi terbaru dari server setelah download selesai
                    try:
                        resp = requests.get(f"{self.server_api_url}/api/status", timeout=2)
                        if resp.status_code == 200:
                            v = resp.json().get("model_version", self.model_version)
                            self._save_version(v)
                    except: pass
                    
                    self.sync_label_map()
                    self.refresh_local_embeddings() 
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
            print(f"[SYNC] Mengambil backbone global dari {url_bb}...")
            res_bb = requests.get(url_bb, timeout=30)
            if res_bb.status_code == 200:
                save_path = os.path.join(self.data_path, "models", "backbone.pth")
                with open(save_path, "wb") as f:
                    f.write(res_bb.content)
                
                try:
                    loaded = torch.load(save_path, map_location=self.device)
                    if self._apply_backbone_weights(loaded):
                        print("[SYNC] Global backbone applied successfully.")
                        return True
                except Exception as e:
                    print(f"[SYNC] Failed to apply backbone: {e}")
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
                        print(f"[CLIENT] Applied Global Combined BN to backbone.")
                        return True
                    except Exception as e:
                        print(f"[CLIENT] Downloaded BN was invalid, retrying: {e}")
                elif res.status_code == 202 or res.status_code == 404:
                    print(f"[CLIENT] BN not ready on server yet (Status {res.status_code}), waiting...")
                else:
                    print(f"[CLIENT ERROR] BN download unexpected status: {res.status_code}")
                    return False
            except Exception as e:
                print(f"[CLIENT ERROR] BN download failed: {e}")
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
                print("[RELOAD] Centroid Registry updated.")
                return True
            else:
                print(f"[RELOAD] Registry download skipped (Status {res_reg.status_code})")
        except Exception as e:
            print(f"[RELOAD ERROR] Registry Update failed: {e}")
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
        """Ekstraksi ulang embedding lokal menggunakan backbone terbaru yang sudah DISINKRONISASI."""
        db = SessionLocal()
        try:
            users = db.query(UserLocal).all()
            processed_dir = os.path.join(self.data_path, "processed")
            
            for user in users:
                user_folder = os.path.join(processed_dir, user.user_id)
                if not os.path.exists(user_folder): continue
                
                imgs = [f for f in os.listdir(user_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if not imgs: continue
                
                img_path = os.path.join(user_folder, imgs[0])
                img_pil = Image.open(img_path).convert('RGB')
                
                preprocess = transforms.Compose([
                    transforms.Resize((112, 96), interpolation=InterpolationMode.BILINEAR), 
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                
                input_tensor = preprocess(img_pil).unsqueeze(0).to(self.device)
                
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

    def get_blur_score(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None: return 0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except: return 0

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
            res = requests.get(f"{self.server_api_url}/api/training/label_map", timeout=10)
            if res.status_code == 200:
                self.client.label_map = res.json()
                map_path = os.path.join(self.data_path, "models", "label_map.json")
                with open(map_path, "w") as f:
                    json.dump(self.client.label_map, f)
                return True
        except: pass
        return False

    def run_preprocess_phase(self):
        self.report_status("Processing: Ekstraksi Wajah (Laplacian Top 50)...")
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
            print(f"[PREPROCESS] Memilih 50 wajah terbaik untuk {nrp} menggunakan Laplacian Variance...")
            
            target_folder = os.path.join(processed_dir, nrp)
            if os.path.exists(target_folder):
                shutil.rmtree(target_folder, ignore_errors=True)
            os.makedirs(target_folder, exist_ok=True)
            
            source_path = os.path.join(students_dir, folder)
            all_images = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            scored_images = sorted([(img, self.get_blur_score(os.path.join(source_path, img))) for img in all_images], key=lambda x: x[1], reverse=True)
            top_images = [img[0] for img in scored_images[:50]]
            
            count = 0
            for img_name in top_images:
                try:
                    img_path = os.path.join(source_path, img_name)
                    img_pil = Image.open(img_path).convert('RGB')
                    target_path = os.path.join(target_folder, f"face_{count}.jpg")
                    
                    # Gunakan MTCNN dari image_processor (lazy-loaded, bisa di-unload setelah selesai)
                    if image_processor.mtcnn(img_pil, save_path=target_path) is not None:
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
            # Expand head dengan preservasi bobot
            new_label_map = {nrp: idx for idx, nrp in enumerate(self.client.label_map)} if self.client.label_map else {}
            self.head = self.client.trainer.update_head(num_classes, new_label_map)
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

        self.is_sending_registry = True
        self.report_status("Processing: Finalisasi Registry Identitas...")
        try:
            # 1. SINKRONISASI BACKBONE & BN TERLEBIH DAHULU (Critical for consistency)
            print("[REGISTRY] Fase 1: Sinkronisasi Model & Stats Global...")
            self.download_backbone()
            # Coba ambil statistik BN global terbaru jika sudah tersedia
            self.download_bn(max_wait=5) 
            
            # 2. HITUNG CENTROID (Menggunakan status model saat ini)
            print("[REGISTRY] Fase 2: Menghitung Centroid Lokal (BN-Aligned)...")
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
                print(f"[REGISTRY] Pengiriman berhasil. Menunggu agregasi global...")
                self.report_status("Processing: Menunggu Agregasi Global...")
                
                # 4. TUNGGU & UNDUH HASIL GLOBAL
                # Ini memastikan setiap client memiliki BN dan Registry yang SAMA
                if self.download_bn(max_wait=120):
                    print("[REGISTRY] Global BN Synced.")
                
                if self._download_global_registry(max_wait=120):
                    print("[REGISTRY] Global Registry Synced.")
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
                print(f"[REGISTRY] Download error: {e}")
                time.sleep(5)
        print("[REGISTRY] Download timed out.")
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
                # Reload model untuk menggunakan bobot global terbaru
                self._reload_inference_models()
        threading.Thread(target=run_client, daemon=True).start()

fl_manager = FLClientManager()
