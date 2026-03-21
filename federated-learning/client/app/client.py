import flwr as fl
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import requests
import os
import socket
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import OrderedDict
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix, classification_report

# Import DB & Models Local
from app.db.db import SessionLocal
from app.db.models import UserLocal, Embedding
from app.utils.security import EmbeddingEncryptor
from app.utils.classifier import load_backbone, save_backbone, LocalClassifierHead
from app.utils.mobilefacenet import MobileFaceNet 
from app.config import config

CLIENT_ID = os.getenv("HOSTNAME", "client-unknown")
# SERVER_URL removed, will use config.get_server_url()
MAX_USERS_CAPACITY = 100
EMBEDDING_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CLIENT] Running on device: {DEVICE}")

def get_global_label(nrp: str, name: str = "", client_id: str = ""): # Menambahkan client_id
    try:
        payload = {"nrp": nrp, "name": name, "registered_edge_id": client_id} # Kirim CLIENT_ID
        response = requests.post(f"{config.get_server_url()}/api/training/get_label", json=payload, timeout=5)
        if response.status_code == 200:
            return response.json() # Mengembalikan seluruh data (termasuk label & registered_edge_id)
    except Exception as e:
        print(f"[SYNC ERROR] {e}")
    return None

def sync_users_from_server():
    print("[SYNC] Service Sinkronisasi User Berjalan (Setiap 30 detik)...")
    while True:
        try:
            # print("[SYNC] Mengecek update user dari server...")
            response = requests.get(f"{config.get_server_url()}/api/training/global_users", timeout=5)
            if response.status_code == 200:
                global_users = response.json() 
                db = SessionLocal()
                
                # Tambah User Baru
                for u_data in global_users:
                    nrp = u_data['nrp']
                    name = u_data['name']
                    
                    if u_data.get('registered_edge_id') and u_data.get('registered_edge_id') != CLIENT_ID:
                        # Skip user yang terdaftar di edge lain
                        continue

                    exists = db.query(UserLocal).filter(UserLocal.nrp == nrp).first()
                    if not exists:
                        print(f"[SYNC] Menambahkan user baru dari server: {name}")
                        new_user = UserLocal(name=name, nrp=nrp)
                        db.add(new_user)
                
                if len(global_users) > 0: 
                    valid_nrps = {u['nrp'] for u in global_users}
                    
                    local_users = db.query(UserLocal).all()
                    
                    for lu in local_users:
                        if lu.nrp not in valid_nrps:
                            print(f"[SYNC] Menghapus user lokal {lu.name} ({lu.nrp}) karena telah dihapus di server.")
                            db.query(Embedding).filter(Embedding.user_id == lu.user_id).delete()
                            db.delete(lu)
                
                db.commit()
                db.close()
            
        except Exception as e:
            print(f"[SYNC FAILED] {e}")
            
        time.sleep(30) # Cek setiap 30 detik
        

# Phase tracking to avoid double-runs
LAST_PROCESSED_PHASE = "idle"
CURRENT_CLIENT_STATUS = "Standby (Siap Training)"

def report_status(status_msg=None, metrics=None):
    global CURRENT_CLIENT_STATUS, LAST_PROCESSED_PHASE
    if status_msg:
        CURRENT_CLIENT_STATUS = status_msg
    
    try:
        # 1. Report Status
        payload = {
            "id": CLIENT_ID,
            "ip_address": socket.gethostbyname(socket.gethostname()),
            "fl_status": CURRENT_CLIENT_STATUS,
            "last_seen": time.time()
        }
        if metrics: payload["metrics"] = metrics
        
        resp = requests.post(f"{config.get_server_url()}/api/clients/register", json=payload, timeout=2)
        
        # 2. Poll Server Phase (Heartbeat)
        if resp.status_code == 200:
            status = resp.json()
            # If server provided status in register response (convenience) or we poll /status
            # For now, let's assume we need to poll /status separately for detail
            res = requests.get(f"{config.get_server_url()}/api/training/status", timeout=2)
            if res.status_code == 200:
                status_data = res.json()
                phase = status_data.get("current_phase", "idle")
                
                if phase != LAST_PROCESSED_PHASE:
                    if phase == "syncing":
                        print(f"[LIFECYCLE] Entering Sync Phase...")
                        auto_volume_import()
                        LAST_PROCESSED_PHASE = "syncing"
                    elif phase == "preprocessing":
                        print(f"[LIFECYCLE] Entering Preprocessing Phase...")
                        # Run the heavy preprocessing once
                        run_persistent_preprocessing()
                        LAST_PROCESSED_PHASE = "preprocessing"
                    elif phase == "idle":
                        LAST_PROCESSED_PHASE = "idle"
                
                # Report detailed status during heavy phases
                if phase == "preprocessing":
                     # No-op here, run_persistent_preprocessing handles its own status
                     pass
                elif phase == "syncing":
                     report_status("Syncing Volumes...")
                
    except Exception as e:
        print(f"[CLIENT REPORT] Error: {e}")

def heartbeat_service():
    """Background service to keep server informed of client presence."""
    print("[HEARTBEAT] Client Heartbeat Service Started.")
    while True:
        try:
            report_status() # Reports CURRENT_CLIENT_STATUS
        except: pass
        time.sleep(5)

def run_persistent_preprocessing():
    """Performs MTCNN face detection and resizing once and saves to processed folder."""
    import os
    import torchvision.transforms as T
    from PIL import Image
    from app.utils.face_pipeline import face_pipeline
    
    students_dir = "/app/data/students"
    processed_dir = "/app/data/processed_faces"
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(students_dir): return

    print(f"[PREPROCESS] Starting Persistent Preprocessing...")
    db = SessionLocal()
    try:
        folders = [f for f in os.listdir(students_dir) if os.path.isdir(os.path.join(students_dir, f))]
        for folder in folders:
            nrp = folder.split("_")[0] if "_" in folder else folder
            name = folder.split("_")[1] if "_" in folder else folder
            
            # 1. Ensure Global Label (Registration)
            lbl_data = get_global_label(nrp, name, client_id=CLIENT_ID)
            if not lbl_data: continue
            
            target_folder = os.path.join(processed_dir, nrp)
            if os.path.exists(target_folder) and len(os.listdir(target_folder)) > 0:
                # Already processed, skip
                # print(f"[PREPROCESS] NRP {nrp} already processed. Skipping.")
                continue
                
            # Report progress for this user
            report_status(f"Preprocessing {name}...")

            os.makedirs(target_folder, exist_ok=True)
            source_path = os.path.join(students_dir, folder)
            
            print(f"[PREPROCESS] Processing {nrp} - {name}...")
            count = 0
            for img_name in os.listdir(source_path):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                try:
                    img = Image.open(os.path.join(source_path, img_name)).convert('RGB')
                    face_img, _, _ = face_pipeline._detect_and_crop(img)
                    
                    if face_img:
                        face_img.save(os.path.join(target_folder, f"face_{count}.jpg"))
                        count += 1
                except Exception as e:
                    print(f"[PREPROCESS ERROR] {img_name}: {e}")
            
            print(f"[PREPROCESS] Done {nrp}: {count} faces saved.")
            
        db.commit()
    finally:
        db.close()
    print(f"[PREPROCESS] Persistent Preprocessing Completed.")

def auto_volume_import():
    """Scans /app/data/students/ and registers them to server."""
    report_status("Syncing Volumes...")
    students_dir = "/app/data/students"
    if not os.path.exists(students_dir): return

    print(f"[IMPORT] Scanning volume {students_dir}...")
    db = SessionLocal()
    try:
        for folder in os.listdir(students_dir):
            if "_" in folder:
                nrp, name = folder.split("_", 1)
            else:
                nrp, name = folder, folder
            
            # Sync with server (Register student globally)
            get_global_label(nrp, name, client_id=CLIENT_ID)
            
            # Update local DB
            exists = db.query(UserLocal).filter(UserLocal.nrp == nrp).first()
            if not exists:
                new_user = UserLocal(name=name, nrp=nrp)
                db.add(new_user)
        db.commit()
    except Exception as e:
        print(f"[IMPORT ERROR] {e}")
    finally:
        db.close()

def save_confusion_matrix(y_true, y_pred):
    """Menyimpan Confusion Matrix sebagai gambar untuk debugging"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {CLIENT_ID}')
        plt.ylabel('Label Asli')
        plt.xlabel('Prediksi')
        
        # Simpan di folder static agar bisa diakses web (opsional)
        output_path = "app/static/confusion_matrix.png"
        os.makedirs("app/static", exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"[METRICS] Confusion Matrix disimpan di {output_path}")
    except Exception as e:
        print(f"[METRICS] Gagal simpan confusion matrix: {e}")

def validate_model(backbone, head, test_loader):
    """Validasi model (Backbone + Head Lokal) + Generate Confusion Matrix"""
    criterion = nn.CrossEntropyLoss()
    backbone.eval()
    head.eval()
    
    val_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # Ekstraksi embedding
            embeddings = backbone(images)
            # Klasifikasi dengan head lokal
            output = head(embeddings) 
            
            loss = criterion(output, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    if len(test_loader) > 0:
        val_loss /= len(test_loader)
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Simpan CM jika ini adalah tahap evaluasi
    save_confusion_matrix(all_labels, all_preds)
    
    return val_loss, accuracy

def train_model(backbone, head, train_loader, val_loader=None, epochs=5, patience=3):
    """Training Backbone + Head Lokal dengan Early Stopping & Scheduler"""
    criterion = nn.CrossEntropyLoss()
    
    # Membekukan 50% parameter awal backbone untuk mengurangi beban komputasi
    total_layers = len(list(backbone.named_parameters()))
    freeze_split = int(total_layers * 0.5)
    
    for i, (name, param) in enumerate(backbone.named_parameters()):
        if i < freeze_split:
            param.requires_grad = False
        else:
            param.requires_grad = True # Pastikan layer akhir bisa dilatih

    for module in backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval() 

    # Optimasi Adam sesuai model.ipynb
    trainable_params = [p for p in backbone.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = optim.Adam(trainable_params, lr=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) # Optional
    
    backbone.to(DEVICE)
    head.to(DEVICE)
    backbone.train() 
    # Force BN back to eval after .train() call
    for module in backbone.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            
    head.train()
    
    best_loss = float('inf')
    patience_counter = 0
    final_loss = None 
    final_acc = 0.0
    
    # Simpan state awal MobileFaceNet
    best_state = backbone.state_dict()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = backbone(images)
            output = head(embeddings)

            loss = criterion(output, labels)
            
            if torch.isnan(loss):
                print(f"[TRAIN ERROR] NaN Loss detected at Epoch {epoch+1}!")
                continue

            loss.backward()
            
            # Gradient Clipping untuk mencegah Exploding Gradient
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        epoch_loss = float(epoch_loss) # Ensure standard float
        epoch_acc = correct / total if total > 0 else 0.0
        
        print(f"[CLIENT TRAIN] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.1%}")
        
        if epoch_loss < 0.0001:
             print("[TRAIN INFO] Skipping Metric Update for Invalid/Zero Loss")
             continue 

        current_val_loss = epoch_loss
        if val_loader:
            v_loss, _ = validate_model(backbone, head, val_loader)
            current_val_loss = v_loss
            backbone.train() 
            head.train()

        if current_val_loss < best_loss:
            best_loss = current_val_loss
            final_loss = epoch_loss
            final_acc = epoch_acc
            # Hanya simpan state dict dari MobileFaceNet (backbone)
            best_state = backbone.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"[EARLY STOPPING] Berhenti di epoch {epoch+1}")
            break
    
    # Fallback jika final_loss masih None (tidak pernah improve dari inf)
    if final_loss is None:
        final_loss = epoch_loss
        final_acc = epoch_acc

    # Kembalikan bobot terbaik yang ditemukan ke backbone
    backbone.load_state_dict(best_state)
    return final_loss, final_acc

def load_data_from_disk():
    """Loads pre-processed images from /app/data/processed_students/."""
    import os
    import torchvision.transforms as T
    from PIL import Image
    
    db = SessionLocal()
    processed_dir = "/app/data/processed_faces"
    
    X_list, y_list = [], []
    
    try:
        if not os.path.exists(processed_dir):
            print(f"[DISK DATA] Directory {processed_dir} not found. Running preprocessing...")
            run_persistent_preprocessing()
            if not os.path.exists(processed_dir): return None, None

        # Get local users mapping (already synced in Phase 1/2)
        users = db.query(UserLocal).all()
        user_to_label = {}
        for u in users:
            # We don't call server here to be fast, assume already mapped
            # If not mapped, we'll try a quick sync
            lbl_data = get_global_label(u.nrp, client_id=CLIENT_ID)
            if lbl_data:
                user_to_label[u.nrp] = lbl_data['label']

        for nrp_folder in os.listdir(processed_dir):
            path = os.path.join(processed_dir, nrp_folder)
            if not os.path.isdir(path): continue
            
            if nrp_folder not in user_to_label:
                continue
            
            label = user_to_label[nrp_folder]
            
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    # Already 112x96, just normalize
                    face_np = np.array(img).astype(np.float32) / 255.0
                    face_tensor = torch.tensor(face_np).permute(2, 0, 1)
                    face_tensor = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(face_tensor)
                    
                    X_list.append(face_tensor.numpy())
                    y_list.append(label)
                except Exception as e:
                    print(f"[DISK DATA] Error loading {img_path}: {e}")
                    continue
        
        if not X_list: 
            print("[DISK DATA] No valid samples found.")
            return None, None
            print("[DISK DATA] No valid samples found.")
            return None, None
        
        print(f"[DISK DATA] Loaded {len(X_list)} samples from disk.")
        X_tensor = torch.tensor(np.array(X_list))
        y_tensor = torch.tensor(np.array(y_list))
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Use batch size 32 from model.ipynb
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        if test_size == 0 and len(dataset) > 0:
            train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            return train_loader, None
            
        train_ds, test_ds = random_split(dataset, [train_size, test_size])
        return DataLoader(train_ds, batch_size=32, shuffle=True), DataLoader(test_ds, batch_size=32)
        
    except Exception as e:
        print(f"[DISK DATA ERROR] {e}")
        return None, None
    finally:
        db.close()

class RealClient(fl.client.NumPyClient):
    def __init__(self):
        # Menggunakan MobileFaceNet sebagai model FL (Backbone)
        self.backbone = MobileFaceNet(embedding_size=EMBEDDING_SIZE).to(DEVICE)
        # Head classifier lokal
        self.local_head = LocalClassifierHead(num_classes=MAX_USERS_CAPACITY).to(DEVICE)

    def get_parameters(self, config):
        return [val.cpu().numpy().astype(np.float16) for _, val in self.backbone.state_dict().items()]

    def fit(self, parameters, config):
        try:
            rnd = config.get('round', '?')
            print(f"[CLIENT] Round {rnd} dimulai.")
            report_status(f"Sedang Training (Ronde {rnd})")

            # Load Global Model Backbone
            # Harus sesuai dengan urutan parameter di MobileFaceNet
            params_dict = zip(self.backbone.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.backbone.load_state_dict(state_dict, strict=True)
            
            # Simpan backbone yang baru di-load untuk digunakan di /attendance
            save_backbone(self.backbone, path="local_backbone.pth")

            # Load & Split Data From Disk
            trainloader, valloader = load_data_from_disk()
            if trainloader is None:
                # Mengembalikan parameter yang sama jika tidak ada data
                return self.get_parameters({}), 0, {}

            # Training Backbone + Head Lokal
            loss, accuracy = train_model(self.backbone, self.local_head, trainloader, valloader, epochs=5, patience=3)

            # Simpan Model Backbone Lokal 
            save_backbone(self.backbone, path="local_backbone.pth")
            
            print("[CLIENT] Menambahkan DP Noise...")
            params = self.get_parameters(config={})
            noise_multiplier = 0.005 # Tingkat noise
            noisy_params = []
            for p in params:
                noise = np.random.normal(0, noise_multiplier, p.shape)
                noisy_params.append(p + noise)
            
            status_msg = f"Selesai (L:{loss:.3f}, A:{accuracy:.1%})"
            report_status(status_msg, metrics={"loss": loss, "accuracy": accuracy})
            
            # Mengembalikan parameter MobileFaceNet yang sudah ditambahkan noise
            return noisy_params, len(trainloader.dataset), {"loss": loss, "accuracy": accuracy}
        except Exception as e:
            print(f"[CLIENT FIT ERROR] {e}")
            traceback.print_exc()
            return [], 0, {}  # Return empty to avoid crash propagation

    def evaluate(self, parameters, config):
        try:
            # Load Global Model Backbone
            params_dict = zip(self.backbone.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.backbone.load_state_dict(state_dict, strict=True)
            
            # Inisialisasi Head lokal (untuk evaluasi)
            local_head_eval = LocalClassifierHead(num_classes=MAX_USERS_CAPACITY)

            _, testloader = load_data_from_disk()
            
            if testloader is None or len(testloader.dataset) == 0:
                return 0.0, 0, {"accuracy": 0.0}

            # Evaluasi dengan Head Lokal yang baru diinisialisasi
            loss, accuracy = validate_model(self.backbone, local_head_eval, testloader)
            
            print(f"[CLIENT] Evaluasi (Test Set). Loss: {loss:.4f}, Acc: {accuracy:.1%}")
            return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}
        except Exception as e:
            print(f"[CLIENT EVAL ERROR] {e}")
            traceback.print_exc()
            return 0.0, 0, {"accuracy": 0.0}

def start_flower_client():
    report_status("Online (Menunggu Server)")
    while True:
        try:
            print("[CLIENT] Connecting to FL Server...")
            server_addr = config.get_fl_server_address()
            fl.client.start_client(server_address=server_addr, client=RealClient().to_client())
            print("[CLIENT] FL Session ended or Server stopped. Returning to standby.")
            report_status("Online (Menunggu Server)")
            time.sleep(5)
        except Exception as e:
            print(f"[CLIENT ERROR] Connection failed or ended with error: {e}")
            print(f"[CLIENT] Standby... Waiting for Training Signal from Server Dashboard.")
            report_status("Online (Menunggu Server)")
            time.sleep(5)