import os
import json
import torch
import threading
import collections
import flwr as fl
import requests
from PIL import Image
from facenet_pytorch import MTCNN
import time
import socket
import io
import base64
import numpy as np
import cv2
import shutil

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .utils.security import encryptor
from .utils.classifier import identify_user_globally
from .utils.mobilefacenet import MobileFaceNet, ArcMarginProduct
from .db.db import SessionLocal
from .db.models import UserLocal, EmbeddingLocal
from .client import FaceRecognitionClient

# Helper to bridge to fl_manager (if called from global context)
def add_phase_log(msg):
    # This matches the server's helper style for consistency in logs
    print(f"[LOG] {msg}")

class FLClientManager:
    def __init__(self):
        self.data_path = os.getenv("DATA_PATH", "/app/data")
        self.raw_data_path = os.getenv("RAW_DATA_PATH", "/app/raw_data")
        self.artifacts_path = os.getenv("ARTIFACTS_PATH", "/app/data/artifacts")
        # Ensure directories exist
        os.makedirs(self.artifacts_path, exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.artifacts_path, "processed"), exist_ok=True)

        self.device = torch.device("cpu")
        self.backbone = MobileFaceNet().to(self.device)
        self.backbone.eval()
        self.detector = MTCNN(image_size=112, margin=20, keep_all=False, device=self.device, post_process=False)
        
        # PERSISTENCE: Determine dynamic head size
        save_path = os.path.join(self.artifacts_path, "models", "backbone.pth")
        head_path = os.path.join(self.artifacts_path, "models", "local_head.pth")
        
        self.num_classes = 1000 # Default fallback
        if os.path.exists(head_path):
            try:
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
                
                # Load Combined BN if exists (Universal Mode)
                bn_combined_path = os.path.join(self.artifacts_path, "models", "global_bn_combined.pth")
                if os.path.exists(bn_combined_path):
                    print("Loading Combined BN statistics...")
                    self.backbone.load_state_dict(torch.load(bn_combined_path, map_location=self.device), strict=False)

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
        
        self.is_registered = False
        self.last_register_attempt = 0
        self.register_retry_delay = 30 # seconds
        
        # Temporal Voting Buffer (10 frames)
        self.prediction_buffer = collections.deque(maxlen=10)
        self.last_face_time = 0

    def start_background_tasks(self):
        print(f"[STARTUP] Starting background tasks for client: {self.client_id}")
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()

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
            print("[CLIENT] Training finished. Downloading Final Registry assets...")
            def update_task():
                if self.download_registry_assets():
                    self.sync_label_map()
                    self.refresh_local_embeddings() 
            threading.Thread(target=update_task, daemon=True).start()

    def download_backbone(self):
        """Fetches the aggregated Backbone StateDict from server."""
        try:
            url_bb = f"{self.server_api_url}/api/model/backbone"
            print(f"[SYNC] Fetching global backbone from {url_bb}...")
            res_bb = requests.get(url_bb, timeout=30)
            if res_bb.status_code == 200:
                save_path = os.path.join(self.artifacts_path, "models", "backbone.pth")
                with open(save_path, "wb") as f:
                    f.write(res_bb.content)
                
                try:
                    loaded = torch.load(save_path, map_location=self.device)
                    if isinstance(loaded, list):
                        new_sd = self.backbone.state_dict()
                        all_keys = list(new_sd.keys())
                        
                        shared_keys = [k for k in all_keys if 
                                       not any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked'])
                                       and any(x in k.lower() for x in ['weight', 'bias'])]
                        
                        if len(shared_keys) == len(loaded):
                            for k, v in zip(shared_keys, loaded):
                                new_sd[k] = torch.from_numpy(v).to(self.device)
                            print(f"[SYNC] Applied {len(loaded)} pFedFace conv weights.")
                        elif len(all_keys) == len(loaded):
                            for k, v in zip(all_keys, loaded):
                                new_sd[k] = torch.from_numpy(v).to(self.device)
                            print(f"[SYNC] Applied {len(loaded)} full state_dict weights (incl. BN).")
                        else:
                            print(f"[SYNC ERROR] Unexpected param count: {len(loaded)} (conv={len(shared_keys)}, full={len(all_keys)})")
                        
                        self.backbone.load_state_dict(new_sd, strict=False)
                    else:
                        self.backbone.load_state_dict(loaded, strict=False)
                    
                    print("[SYNC] Global backbone applied successfully.")
                    return True
                except Exception as e:
                    print(f"[SYNC] Failed to apply backbone: {e}")
        except Exception as e:
            print(f"[SYNC ERROR] Backbone fetch failed: {e}")
        return False

    def download_bn(self):
        """Fetches the aggregated BN stats (Running Mean/Var) for global consistency."""
        path = os.path.join(self.artifacts_path, "models", "global_bn_combined.pth")
        try:
            res = requests.get(f"{self.server_api_url}/api/model/bn", timeout=10)
            if res.status_code == 200:
                with open(path, "wb") as f:
                    f.write(res.content)
                
                # LOAD BN stats into backbone in memory
                bn_params = torch.load(path, map_location=self.device)
                self.backbone.load_state_dict(bn_params, strict=False)
                print(f"[CLIENT] Applied Global Combined BN to backbone.")
                return True
        except Exception as e:
            print(f"[CLIENT ERROR] BN download failed: {e}")
        return False

    def download_registry_assets(self):
        """Download combined identification centroids for offline inference."""
        try:
            url_reg = f"{self.server_api_url}/api/model/registry"
            res_reg = requests.get(url_reg, timeout=10)
            if res_reg.status_code == 200:
                reg_path = os.path.join(self.artifacts_path, "models", "global_embedding_registry.pth")
                with open(reg_path, "wb") as f:
                    f.write(res_reg.content)
                print("[RELOAD] Centroid Registry updated.")
                return True
        except Exception as e:
            print(f"[RELOAD ERROR] Registry Update skipped: {e}")
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
        """Re-extract local embeddings using the latest SYNCED backbone."""
        db = SessionLocal()
        try:
            users = db.query(UserLocal).all()
            processed_dir = os.path.join(self.artifacts_path, "processed")
            
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
                
                # Share with server
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
        """Scan folders and register IDs with the server."""
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
                map_path = os.path.join(self.artifacts_path, "models", "label_map.json")
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
        processed_dir = os.path.join(self.artifacts_path, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        if not os.path.exists(students_dir):
            self.report_status("Siap Training (No Data)")
            return

        folders = sorted([f for f in os.listdir(students_dir) if os.path.isdir(os.path.join(students_dir, f))])
        for folder in folders:
            nrp = folder.split('_')[0] if "_" in folder else folder
            print(f"[PREPROCESS] Selecting Top 50 faces for {nrp} using Laplacian Variance...")
            
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
                    img_pil = Image.open(os.path.join(source_path, img_name)).convert('RGB')
                    target_path = os.path.join(target_folder, f"face_{count}.jpg")
                    if self.detector(img_pil, save_path=target_path) is not None: count += 1
                except: pass
        
        has_data = any(len(os.listdir(os.path.join(processed_dir, sub))) > 0 for sub in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, sub)))
        if has_data:
            self.report_status("Siap Training")
            self.refresh_local_embeddings()
            
            try:
                requests.post(f"{self.server_api_url}/api/clients/ready", json={"client_id": self.client_id}, timeout=5)
            except: pass
            
            num_classes = sum(1 for sub in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, sub)) and len(os.listdir(os.path.join(processed_dir, sub))) > 0)
            self.head = ArcMarginProduct(128, num_classes).to(self.device)
            self.client.head = self.head
            self.client.trainer.head = self.head
        else:
            self.report_status("Siap Training (No Data)")
            try:
                requests.post(f"{self.server_api_url}/api/clients/ready", json={"client_id": self.client_id}, timeout=5)
            except: pass

    def run_registry_phase(self):
        """FINAL Registry Extraction (Uses Unified Global Weights)."""
        if getattr(self, "is_sending_registry", False):
            print("[REGISTRY] Generation already in progress, skipping duplicate trigger.")
            return

        self.is_sending_registry = True
        self.report_status("Processing: Finalisasi Registry Identitas...")
        try:
            # FORCE SYNC WITH GLOBAL MODEL (Critical Fix for similarity scores)
            print("[REGISTRY] Pre-syncing with Unified Global Backbone...")
            self.download_backbone()
            self.download_bn() # Fetch global BN stats if available
            
            # Extract Centroids from the perspective of the Global Model
            bn_params = self.client.trainer.get_bn_parameters()
            centroids = self.client.trainer.calculate_centroids(label_map=self.client.label_map)
            
            # ensures inference works for local students even if server aggregate is not ready
            local_registry_path = os.path.join(self.artifacts_path, "models", "global_embedding_registry.pth")
            torch.save(centroids, local_registry_path)
            
            # SUBMIT: Upload to server for global aggregation
            serialized_centroids = {nrp: base64.b64encode(vec.tobytes()).decode('utf-8') for nrp, vec in centroids.items()}
            bn_buf = io.BytesIO()
            torch.save(bn_params, bn_buf)
            serialized_bn = base64.b64encode(bn_buf.getvalue()).decode('utf-8')
            
            payload = {
                "client_id": self.client_id, "bn_params": serialized_bn, "centroids": serialized_centroids
            }
            res = requests.post(f"{self.server_api_url}/api/training/registry_assets", json=payload, timeout=60)
            if res.status_code == 200:
                print(f"[REGISTRY] Submitted {len(centroids)} identities.")
                self.report_status("Siap Selesai")
                self._download_global_registry()
                self.download_bn() 
            else:
                self.report_status(f"Error Registry: {res.status_code}")
        except Exception as e:
            print(f"[REGISTRY ERROR] {e}")
            self.report_status("Error Registry")
        finally:
            self.is_sending_registry = False

    def _download_global_registry(self, max_wait=60):
        import time
        registry_url = f"{self.server_api_url}/api/model/registry"
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                res = requests.get(registry_url, timeout=15)
                if res.status_code == 200:
                    save_path = os.path.join(self.artifacts_path, "models", "global_embedding_registry.pth")
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
        threading.Thread(target=run_client, daemon=True).start()

fl_manager = FLClientManager()
