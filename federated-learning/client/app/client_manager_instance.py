from .client import FaceRecognitionClient
from .utils.mobilefacenet import MobileFaceNet
from .utils.trainer import ArcMarginProduct
import os
import torch
import threading
import flwr as fl

import requests
from PIL import Image
import time
import socket
from .utils.face_pipeline import face_pipeline

class FLClientManager:
    def __init__(self):
        self.data_path = os.getenv("DATA_PATH", "/app/data")
        self.device = torch.device("cpu")
        print(f"[HARDWARE] Forced device: {self.device}")
        
        self.backbone = MobileFaceNet().to(self.device).eval()
        
        # PERSISTENCE: Determine dynamic head size
        save_path = os.path.join(self.data_path, "backbone.pth")
        head_path = os.path.join(self.data_path, "local_head.pth")
        
        self.num_classes = 1000 # Default
        if os.path.exists(head_path):
            try:
                # Probe the head file to get the class count
                checkpoint = torch.load(head_path, map_location="cpu")
                if "weight" in checkpoint:
                    self.num_classes = checkpoint["weight"].shape[0]
                    print(f"[STARTUP] Detected {self.num_classes} classes from saved head.")
            except: pass
            
        self.head = ArcMarginProduct(128, self.num_classes).to(self.device)
        self.model_version = 0
        
        if os.path.exists(save_path):
            try:
                print(f"Loading existing backbone weights from {save_path}...")
                self.backbone.load_state_dict(torch.load(save_path, map_location=self.device))
                
                # Try loading persistent version first
                v_path = os.path.join(self.data_path, "model_version.txt")
                if os.path.exists(v_path):
                    with open(v_path, "r") as f:
                        self.model_version = int(f.read().strip())
                else:
                    self.model_version = int(os.path.getmtime(save_path)) % 1000
            except: pass

        self.fl_server_address = os.getenv("FL_SERVER_ADDRESS", "server-fl:8085")
        self.server_api_url = os.getenv("SERVER_API_URL", "http://server-fl:8080")
        self.client_id = os.getenv("HOSTNAME", "terminal-1")
        
        self.client = FaceRecognitionClient(
            self.backbone, self.head, 
            data_path=self.data_path, 
            device=self.device
        )
        
        self.is_training = False
        self.current_phase = "idle"
        self.fl_status = "Online (Menunggu Instruksi)"
        self.last_phase = "idle"

        # Start heartbeat thread
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()

    def report_status(self, status=None):
        if status: self.fl_status = status
        try:
            payload = {
                "id": self.client_id,
                "ip_address": socket.gethostbyname(socket.gethostname()),
                "fl_status": self.fl_status,
                "last_seen": time.time()
            }
            requests.post(f"{self.server_api_url}/api/clients/register", json=payload, timeout=2)
        except:
            pass

    def heartbeat_loop(self):
        print(f"[CLIENT] Heartbeat service started for {self.client_id}")
        while True:
            try:
                # 1. Report Status
                self.report_status()
                
                # 2. Get Phase from Server
                resp = requests.get(f"{self.server_api_url}/api/training/status", timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    phase = data.get("current_phase", "idle")
                    
                    if phase != self.last_phase:
                        self.handle_phase_transition(phase)
                        self.last_phase = phase
            except Exception as e:
                print(f"Heartbeat Error: {e}")
            time.sleep(5)

    def handle_phase_transition(self, phase):
        print(f"[CLIENT] Phase Transition: {self.last_phase} -> {phase}")
        
        if phase == "syncing":
            threading.Thread(target=self.run_sync_phase).start()
        elif phase == "preprocessing":
            threading.Thread(target=self.run_preprocess_phase).start()
        elif phase == "training":
            self.start_fl()
        elif phase == "idle" or phase == "completed":
            self.fl_status = "Online (Selesai)"
        
        # POST-TRAINING REFRESH: If we just finished training, refresh embeddings
        if self.last_phase == "training" and phase != "training":
            print("[CLIENT] Training finished. Refreshing local embeddings...")
            threading.Thread(target=self.refresh_local_embeddings, daemon=True).start()

    def run_sync_phase(self):
        self.report_status("Processing: Sinkronisasi Data...")
        from .db.db import SessionLocal
        from .db.models import UserLocal, EmbeddingLocal
        db = SessionLocal()
        
        try:
            # 1. Download Global Users & Embeddings
            res = requests.get(f"{self.server_api_url}/api/users/global", timeout=10)
            if res.status_code == 200:
                global_users = res.json()
                print(f"[SYNC] Memperoleh {len(global_users)} mahasiswa global.")
                
                for u in global_users:
                    # Sync UserLocal (Gunakan NRP sebagai user_id lokal agar konsisten)
                    user = db.query(UserLocal).filter_by(user_id=u['nrp']).first()
                    if not user:
                        user = UserLocal(user_id=u['nrp'], name=u['name'])
                        db.add(user)
                        db.commit()
                    
                    # Sync Global Embedding (Memory)
                    if u['embedding']:
                        # ALWAYS OVERWRITE: Backbone evolves, so global embeddings must be refreshed
                        import base64
                        import numpy as np
                        emb_bytes = base64.b64decode(u['embedding'])
                        
                        exists = db.query(EmbeddingLocal).filter_by(user_id=u['nrp'], is_global=True).first()
                        if exists:
                            exists.embedding_data = emb_bytes
                        else:
                            new_global_emb = EmbeddingLocal(
                                user_id=u['nrp'],
                                embedding_data=emb_bytes,
                                is_global=True,
                                iv=None
                            )
                            db.add(new_global_emb)
                        db.commit()
            
            # 2. Pastikan folder lokal sudah siap
            students_dir = os.path.join(self.data_path, "students")
            os.makedirs(students_dir, exist_ok=True)
            
            # 3. Label Mapping (Consistency check for folders)
            folders = [f for f in os.listdir(students_dir) if os.path.isdir(os.path.join(students_dir, f))]
            for folder in folders:
                nrp = folder.split('_')[0] if "_" in folder else folder
                name = folder.split('_')[1] if "_" in folder else folder
                requests.post(f"{self.server_api_url}/api/training/get_label", json={
                    "nrp": nrp, "name": name, "client_id": self.client_id
                }, timeout=5)

            self.report_status("Siap Preprocess")
        except Exception as e:
            print(f"[SYNC ERROR] {e}")
            self.report_status("Error: Sync Gagal")
        finally:
            db.close()

    def refresh_local_embeddings(self):
        """Re-extract embeddings for all local students using the latest backbone."""
        from .db.db import SessionLocal
        from .db.models import UserLocal, EmbeddingLocal
        from PIL import Image
        import io
        
        db = SessionLocal()
        try:
            users = db.query(UserLocal).all()
            processed_dir = os.path.join(self.data_path, "processed_faces")
            
            for user in users:
                user_folder = os.path.join(processed_dir, user.user_id)
                if not os.path.exists(user_folder): continue
                
                # Take first processed image
                imgs = [f for f in os.listdir(user_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if not imgs: continue
                
                img_path = os.path.join(user_folder, imgs[0])
                img_pil = Image.open(img_path).convert('RGB')
                
                # We don't need detections since it's already a processed face
                from torchvision import transforms
                preprocess = transforms.Compose([
                    transforms.Resize((112, 96)), # MobileFaceNet expected size (matching trainer)
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                
                input_tensor = preprocess(img_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    self.backbone.eval()
                    embedding_np = self.backbone(input_tensor).cpu().numpy()[0]
                    
                # Update DB
                from .utils.encryptor import encryptor
                encrypted_data, iv = encryptor.encrypt_embedding(embedding_np)
                
                emb_record = db.query(EmbeddingLocal).filter_by(user_id=user.user_id, is_global=False).first()
                if emb_record:
                    emb_record.embedding_data = encrypted_data
                    emb_record.iv = iv
                else:
                    new_emb = EmbeddingLocal(
                        user_id=user.user_id,
                        embedding_data=encrypted_data,
                        iv=iv,
                        is_global=False
                    )
                    db.add(new_emb)
                print(f"[REFRESH] Re-computed embedding for {user.user_id}")
            
            db.commit()
            print("[REFRESH] Local embeddings refresh complete.")
        except Exception as e:
            print(f"[REFRESH ERROR] {e}")
            db.rollback()
        finally:
            db.close()

    def run_preprocess_phase(self):
        self.report_status("Processing: Ekstraksi Wajah (MTCNN)...")
        students_dir = os.path.join(self.data_path, "students")
        processed_dir = os.path.join(self.data_path, "processed_faces")
        os.makedirs(processed_dir, exist_ok=True)
        
        if not os.path.exists(students_dir):
            self.report_status("Siap Training (No Data)")
            return

        folders = [f for f in os.listdir(students_dir) if os.path.isdir(os.path.join(students_dir, f))]
        for folder in folders:
            nrp = folder.split('_')[0] if "_" in folder else folder
            target_folder = os.path.join(processed_dir, nrp)
            
            # Skip if already processed
            if os.path.exists(target_folder) and len(os.listdir(target_folder)) >= 5:
                continue
                
            os.makedirs(target_folder, exist_ok=True)
            source_path = os.path.join(students_dir, folder)
            
            print(f"[PREPROCESS] Extracting faces for {nrp}...")
            count = 0
            img_list = os.listdir(source_path)
            for img_name in img_list:
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                try:
                    full_path = os.path.join(source_path, img_name)
                    img_pil = Image.open(full_path).convert('RGB')
                    
                    face_img = face_pipeline.detect_and_crop(img_pil)
                    if face_img is not None:
                        target_path = os.path.join(target_folder, f"face_{count}.jpg")
                        face_img.save(target_path)
                        count += 1
                    if count >= 100: break
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")
        
        # Check if we actually have data now
        has_data = False
        num_classes = 0
        if os.path.exists(processed_dir):
            for sub in os.listdir(processed_dir):
                sub_path = os.path.join(processed_dir, sub)
                if os.path.isdir(sub_path) and len(os.listdir(sub_path)) > 0:
                    has_data = True
                    num_classes += 1
        
        if has_data:
            self.report_status("Siap Training")
            # Update client head to match processed data (Only non-empty folders)
            print(f"[CLIENT] Updating training head for {num_classes} folders with data.")
            self.head = ArcMarginProduct(128, num_classes).to(self.device)
            self.client.head = self.head
            self.client.trainer.head = self.head
        else:
            self.report_status("Siap Training (No Data)")
            
        print("[CLIENT] Preprocessing phase complete.")

    def start_fl(self):
        if self.is_training: return
        self.is_training = True
        
        def run_client():
            self.report_status("Training: Flower FL...")
            print(f"Starting Flower client, connecting to {self.fl_server_address}...")
            try:
                fl.client.start_client(
                    server_address=self.fl_server_address, 
                    client=self.client.to_client()
                )
            except Exception as e:
                print(f"FL Client Error: {e}")
            finally:
                self.is_training = False
                self.report_status("Online (Selesai Training)")
                print("FL Client session ended.")
            
        threading.Thread(target=run_client, daemon=True).start()

fl_manager = FLClientManager()
