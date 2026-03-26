from .client import FaceRecognitionClient
from .utils.mobilefacenet import get_model, verify_architecture_match, ArcMarginProduct
import os
import torch
import threading
import flwr as fl

import requests
from PIL import Image
import time
import socket

from .utils.face_pipeline import face_pipeline
from .utils.security import encryptor
from .utils.voting import TemporalVoter

class FLClientManager:
    def __init__(self):
        self.data_path = os.getenv("DATA_PATH", "/app/data")
        self.artifacts_path = os.getenv("ARTIFACTS_PATH", "/app/artifacts")
        # Ensure directories exist
        os.makedirs(self.artifacts_path, exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_path, "processed"), exist_ok=True)

        self.device = torch.device("cpu")
        print(f"[HARDWARE] Forced device: {self.device}")
        
        self.backbone = get_model().to(self.device)
        verify_architecture_match(self.backbone) # Fail-fast local check
        self.backbone.eval()
        
        # PERSISTENCE: Determine dynamic head size
        save_path = os.path.join(self.artifacts_path, "models", "backbone.pth")
        head_path = os.path.join(self.artifacts_path, "models", "local_head.pth")
        
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
                v_path = os.path.join(self.artifacts_path, "models", "model_version.txt")
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
            artifacts_path=self.artifacts_path, 
            device=self.device
        )
        
        self.is_training = False
        self.current_phase = "idle"
        self.fl_status = "Online (Menunggu Instruksi)"
        self.last_phase = "idle"
        
        # Stability tracking
        self.is_registered = False
        self.last_register_attempt = 0
        self.register_retry_delay = 30 # seconds

        # Inference state
        self.voter = TemporalVoter(window_size=10, majority_threshold=7)

        # Start heartbeat thread
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()

    def report_status(self, status=None):
        if status: self.fl_status = status
        
        # Rate limit registration attempts if not successful
        now = time.time()
        if self.is_registered and not status: # Routine heartbeat is fine
            pass # We still want to send heartbeat, let's keep it simple
        elif not self.is_registered and (now - self.last_register_attempt < self.register_retry_delay):
            return

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
        except:
            self.is_registered = False

    def heartbeat_loop(self):
        print(f"[CLIENT] Heartbeat service started for {self.client_id}")
        while True:
            try:
                self.report_status()
                
                # Get Phase from Server
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
            
        if self.last_phase == "training" and phase != "training":
            print("[CLIENT] Training finished. Downloading latest model and refreshing local embeddings...")
            def update_task():
                if self.download_global_model():
                    self.refresh_local_embeddings()
            threading.Thread(target=update_task, daemon=True).start()

    def download_global_model(self):
        """Atomic Update: Download backbone parameters from server and reload."""
        try:
            url = f"{self.server_api_url}/api/model/backbone"
            print(f"[RELOAD] Fetching latest backbone from {url}...")
            res = requests.get(url, timeout=30)
            if res.status_code == 200:
                # Save as ndarrays bytes
                model_dir = os.path.join(self.artifacts_path, "models")
                tmp_path = os.path.join(model_dir, "backbone_tmp.pth")
                save_path = os.path.join(model_dir, "backbone.pth")
                
                with open(tmp_path, "wb") as f:
                    f.write(res.content)
                
                # Atomic swap
                os.replace(tmp_path, save_path)
                
                # Load list of ndarrays from the .pth file (saved by server)
                parameters = torch.load(save_path, map_location=self.device)
                
                # Apply using the trainer's robust mapping logic (pFedFace compliant)
                print(f"[RELOAD] Applying {len(parameters)} parameters via refined trainer logic...")
                self.client.trainer.set_backbone_parameters(parameters, personalized=True)
                
                print(f"[RELOAD] Backbone updated atomically on {self.device}")
                return True
            else:
                print(f"[RELOAD] Server returned {res.status_code} during model fetch.")
        except Exception as e:
            print(f"[RELOAD ERROR] {e}")
        return False

    def run_sync_phase(self):
        self.report_status("Processing: Sinkronisasi Data...")
        from .db.db import SessionLocal
        from .db.models import UserLocal, EmbeddingLocal
        db = SessionLocal()
        
        try:
            # Download Global Users & Embeddings
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
                print(f"[SYNC] Sinkronisasi selesai. Total {len(global_users)} mahasiswa global di DB.")
            else:
                print(f"[SYNC] Server returned {res.status_code} during sync.")
            
            # Pastikan folder lokal sudah siap
            students_dir = os.path.join(self.data_path, "students")
            os.makedirs(students_dir, exist_ok=True)
            
            # Label Mapping is now done during registration in preprocessing
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
            processed_dir = os.path.join(self.artifacts_path, "processed")
            
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
                from torchvision.transforms import InterpolationMode
                preprocess = transforms.Compose([
                    transforms.Resize((112, 112), interpolation=InterpolationMode.BILINEAR), 
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                
                input_tensor = preprocess(img_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    self.backbone.eval()
                    embedding_tensor = self.backbone(input_tensor)
                    # L2 NORMALIZATION (Expert requirement)
                    embedding_tensor = torch.nn.functional.normalize(embedding_tensor, p=2, dim=1)
                    embedding_np = embedding_tensor.cpu().numpy()[0]
                    
                # Update DB
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
                print(f"[REFRESH] Re-computed embedding for {user.user_id}")
                
                # Register to Server (Knowledge Sharing)
                try:
                    import base64
                    payload = {
                        "nrp": user.user_id,
                        "name": user.name,
                        "client_id": self.client_id,
                        "embedding": base64.b64encode(embedding_np.tobytes()).decode('utf-8')
                    }
                    requests.post(f"{self.server_api_url}/api/training/get_label", json=payload, timeout=5)
                except Exception as e:
                    print(f"[REFRESH] Server registration failed for {user.user_id}: {e}")
            
            print("[REFRESH] Local embeddings refresh and server registration complete.")
        except Exception as e:
            print(f"[REFRESH ERROR] {e}")
            db.rollback()
        finally:
            db.close()

    def run_preprocess_phase(self):
        self.report_status("Processing: Ekstraksi Wajah (MTCNN)...")
        students_dir = os.path.join(self.data_path, "students")
        processed_dir = os.path.join(self.artifacts_path, "processed")
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
            
            # Register discovered user to local DB
            if count > 0:
                from .db.db import SessionLocal
                from .db.models import UserLocal
                db = SessionLocal()
                try:
                    user = db.query(UserLocal).filter_by(user_id=nrp).first()
                    if not user:
                        name = folder.split('_')[1] if "_" in folder else nrp
                        user = UserLocal(user_id=nrp, name=name)
                        db.add(user)
                        db.commit()
                        print(f"[PREPROCESS] Added {nrp} to local database.")
                finally:
                    db.close()
        
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
            # Refresh embeddings to ensure they are registered to server
            self.refresh_local_embeddings()
            
            # Update client head
            print(f"[CLIENT] Updating training head for {num_classes} folders with data.")
            self.head = ArcMarginProduct(128, num_classes).to(self.device)
            self.client.head = self.head
            self.client.trainer.head = self.head
        else:
            self.report_status("Siap Training (No Data)")
            
        print("[CLIENT] Preprocessing phase complete.")

    def start_fl(self):
        if self.is_training:
            print("[CLIENT] Flower client is already running. Skipping duplicate call.")
            return
        self.is_training = True
        
        def run_client():
            self.report_status("Training: Flower FL...")
            print(f"Starting Flower client, connecting to {self.fl_server_address}...")
            try:
                fl.client.start_client(
                    server_address=self.fl_server_address, 
                    client=self.client.to_client()
                )
                print("[CLIENT] Flower session completed normally.")
            except Exception as e:
                error_msg = f"FL Client Connection Error: {e}"
                print(f"[ERROR] {error_msg}")
                self.report_status(f"Error: {str(e)[:30]}...")
            finally:
                self.is_training = False
                if self.last_phase == "training":
                    self.report_status("Online (Gagal/Diskonek)")
                else:
                    self.report_status("Online (Selesai)")
                print("FL Client session ended.")
            
        threading.Thread(target=run_client, daemon=True).start()

fl_manager = FLClientManager()
