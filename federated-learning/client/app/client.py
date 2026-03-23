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
from app.utils.classifier import load_backbone, save_local_model, LocalClassifierHead
from app.utils.mobilefacenet import MobileFaceNet, ArcMarginProduct
from app.config import config

CLIENT_ID = os.getenv("HOSTNAME", "client-unknown")
MAX_USERS_CAPACITY = 100
EMBEDDING_SIZE = 128
DEVICE = torch.device("cpu")
print(f"[CLIENT] Running on device: {DEVICE}")

# Phase tracking to avoid double-runs
LAST_PROCESSED_PHASE = "idle"
CURRENT_CLIENT_STATUS = "Standby (Siap Training)"
CURRENT_RESET_COUNTER = -1 # State tracking from user reference

# Prep tracking to avoid infinite reprocessing
HAS_SYNCED = False
HAS_PREPROCESSED = False

def get_global_label(nrp: str, name: str = "", client_id: str = ""):
    try:
        payload = {"nrp": nrp, "name": name, "registered_edge_id": client_id}
        response = requests.post(f"{config.get_server_url()}/api/training/get_label", json=payload, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"[SYNC ERROR] {e}")
    return None

def sync_users_from_server():
    print("[SYNC SERVICE] Service Sinkronisasi User Berjalan (Setiap 30 detik)...")
    while True:
        try:
            response = requests.get(f"{config.get_server_url()}/api/training/global_users", timeout=5)
            if response.status_code == 200:
                global_users = response.json() 
                db = SessionLocal()
                for u_data in global_users:
                    nrp = u_data['nrp']
                    name = u_data['name']
                    lbl = u_data.get('label')
                    
                    exists = db.query(UserLocal).filter(UserLocal.nrp == nrp).first()
                    if not exists:
                        print(f"[SYNC] Menambahkan user baru dari server: {name} (Label: {lbl})")
                        new_user = UserLocal(name=name, nrp=nrp, global_label=lbl)
                        db.add(new_user)
                    else:
                        # Update label if missing or changed
                        if exists.global_label != lbl:
                            exists.global_label = lbl
                db.commit()
                db.close()
        except Exception as e:
            print(f"[SYNC FAILED] {e}")
        time.sleep(30)

def report_status(status_msg=None, metrics=None):
    """Notify server of client presence and current message."""
    global CURRENT_CLIENT_STATUS, CURRENT_RESET_COUNTER
    if status_msg:
        CURRENT_CLIENT_STATUS = status_msg
    try:
        payload = {
            "id": CLIENT_ID,
            "ip_address": socket.gethostbyname(socket.gethostname()),
            "fl_status": CURRENT_CLIENT_STATUS,
            "last_seen": time.time()
        }
        if metrics: payload["metrics"] = metrics
        resp = requests.post(f"{config.get_server_url()}/api/clients/register", json=payload, timeout=2)
        
        if resp.status_code == 200:
            data = resp.json()
            server_counter = data.get("server_reset_counter", 0)
            
            # User Reference Logic: Reset local model if server was reset
            if CURRENT_RESET_COUNTER == -1:
                CURRENT_RESET_COUNTER = server_counter
            elif server_counter > CURRENT_RESET_COUNTER:
                print(f"[CLIENT] Detected Server Reset (Counter: {server_counter}). Resetting Local Model...")
                if os.path.exists("local_backbone.pth"):
                    os.remove("local_backbone.pth")
                    print("[CLIENT] local_backbone.pth deleted.")
                CURRENT_RESET_COUNTER = server_counter
    except:
        pass

def get_current_status():
    return CURRENT_CLIENT_STATUS

def heartbeat_service():
    """Background service to keep server informed of client presence."""
    print("[HEARTBEAT] Client Heartbeat Service Started.")
    while True:
        report_status()
        time.sleep(5)

def run_persistent_preprocessing():
    """Performs MTCNN face detection and resizing once."""
    from app.utils.face_pipeline import face_pipeline
    from PIL import Image
    
    students_dir = "/app/data/students"
    processed_dir = "/app/data/processed_faces"
    os.makedirs(processed_dir, exist_ok=True)
    
    if not os.path.exists(students_dir): 
        print(f"[PREPROCESS ERROR] Students directory {students_dir} not found!")
        return

    print(f"[PREPROCESS] Checking for facial images to process...")
    try:
        folders = [f for f in os.listdir(students_dir) if os.path.isdir(os.path.join(students_dir, f))]
        
        for folder in folders:
            nrp = folder.split("_")[0] if "_" in folder else folder
            name = folder.split("_")[1] if "_" in folder else folder
            
            res = get_global_label(nrp, name, client_id=CLIENT_ID)
            if res:
                db_l = SessionLocal()
                try:
                    u_l = db_l.query(UserLocal).filter(UserLocal.nrp == nrp).first()
                    if u_l:
                        u_l.global_label = res.get("label")
                        db_l.commit()
                finally: db_l.close()
            
            target_folder = os.path.join(processed_dir, nrp)
            if os.path.exists(target_folder) and len(os.listdir(target_folder)) > 0:
                continue
                
            report_status(f"Processing local face data for {name}...")
            os.makedirs(target_folder, exist_ok=True)
            source_path = os.path.join(students_dir, folder)
            
            print(f"[PREPROCESS] Extracting faces from {nrp}...")
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
            print(f"[PREPROCESS] Success: {count} faces extracted for {nrp}.")
    except Exception as e:
        print(f"[PREPROCESS FATAL] {e}")

def auto_volume_import():
    """Scans /app/data/students/ and registers them to server."""
    students_dir = "/app/data/students"
    if not os.path.exists(students_dir): return

    print(f"[IMPORT] Scanning for students in {students_dir}...")
    db = SessionLocal()
    try:
        folders = os.listdir(students_dir)
        print(f"[IMPORT] Found {len(folders)} folders in students directory.")
        for folder in folders:
            if "_" in folder:
                nrp, name = folder.split("_", 1)
            else:
                nrp, name = folder, folder
            res = get_global_label(nrp, name, client_id=CLIENT_ID)
            print(f"[IMPORT DEBUG] {nrp} label: {res.get('label') if res else 'None'}")
            exists = db.query(UserLocal).filter(UserLocal.nrp == nrp).first()
            if not exists:
                new_user = UserLocal(name=name, nrp=nrp, global_label=res.get("label") if res else None)
                db.add(new_user)
            elif res:
                exists.global_label = res.get("label")
        db.commit()
    except Exception as e:
        print(f"[IMPORT ERROR] {e}")
    finally:
        db.close()

def save_confusion_matrix(y_true, y_pred):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {CLIENT_ID}')
        plt.ylabel('Label Asli')
        plt.xlabel('Prediksi')
        output_path = "app/static/confusion_matrix.png"
        os.makedirs("app/static", exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    except: pass

def validate_model(model, metric_fc, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    metric_fc.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            output = metric_fc(model(img), label) 
            loss = criterion(output, label)
            val_loss += loss.item()
            _, pred = torch.max(output.data, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
    if len(test_loader) > 0: val_loss /= len(test_loader)
    accuracy = correct / total if total > 0 else 0.0
    save_confusion_matrix(all_labels, all_preds)
    return val_loss, accuracy

def train_model(model, metric_fc, train_loader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.parameters()}, 
        {'params': metric_fc.parameters()}
    ], lr=0.001)
    
    model.to(DEVICE)
    metric_fc.to(DEVICE)
    model.train() 
    metric_fc.train()
    
    best_loss = float('inf')
    final_loss, final_acc = None, 0.0
    best_model_state = model.state_dict()
    best_head_state = metric_fc.state_dict()

    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for img, label in train_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            output = metric_fc(model(img), label)
            loss = criterion(output, label)
            if torch.isnan(loss): continue
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, pred = torch.max(output.data, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
        
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        print(f"[CLIENT TRAIN] Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.1%}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            final_loss, final_acc = epoch_loss, epoch_acc
            best_model_state = model.state_dict()
            best_head_state = metric_fc.state_dict()
    
    model.load_state_dict(best_model_state)
    metric_fc.load_state_dict(best_head_state)
    return final_loss, final_acc

from torchvision import transforms, datasets
def load_data_from_disk():
    processed_dir = "/app/data/processed_faces"
    if not os.path.exists(processed_dir) or len(os.listdir(processed_dir)) == 0:
        return None, None
    transform = transforms.Compose([
        transforms.Resize((112, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = datasets.ImageFolder(processed_dir, transform=transform)
    if len(dataset) < 2: return None, None
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader

class RealClient(fl.client.NumPyClient):
    def __init__(self):
        super().__init__()
        self.model = MobileFaceNet(embedding_size=EMBEDDING_SIZE).to(DEVICE)
        self.metric_fc = ArcMarginProduct(128, MAX_USERS_CAPACITY).to(DEVICE)
        if os.path.exists("local_backbone.pth"):
            self.model.load_state_dict(torch.load("local_backbone.pth", map_location=DEVICE))
    def get_parameters(self, config):
        all_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        all_params += [val.cpu().numpy() for _, val in self.metric_fc.state_dict().items()]
        return all_params
    def set_parameters(self, parameters):
        backbone_keys = list(self.model.state_dict().keys())
        head_keys = list(self.metric_fc.state_dict().keys())
        backbone_params = parameters[:len(backbone_keys)]
        head_params = parameters[len(backbone_keys):]
        model_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(backbone_keys, backbone_params)})
        self.model.load_state_dict(model_dict, strict=True)
        if len(head_params) > 0:
            head_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(head_keys, head_params)})
            self.metric_fc.load_state_dict(head_dict, strict=True)
    def fit(self, parameters, config):
        rnd = config.get("round", 0)
        try:
            self.set_parameters(parameters)
            report_status(f"Training Round {rnd}")
            trainloader, _ = load_data_from_disk()
            if trainloader is None:
                print(f"[CLIENT] Round {rnd}: No local data found or insufficient samples. Skipping training.")
                return self.get_parameters({}), 0, {}
            
            print(f"[CLIENT] Round {rnd}: Starting training with {len(trainloader.dataset)} samples.")
            loss, accuracy = train_model(self.model, self.metric_fc, trainloader, epochs=1)
            save_local_model(self.model, self.metric_fc)
            num_samples = len(trainloader.dataset)
            report_status(f"Finished Round {rnd}", metrics={"loss": loss, "accuracy": accuracy})
            return self.get_parameters({}), num_samples, {"loss": loss, "accuracy": accuracy}
        except Exception as e:
            traceback.print_exc()
            return [], 0, {}
    def evaluate(self, parameters, config):
        try:
            self.set_parameters(parameters)
            _, testloader = load_data_from_disk()
            if testloader is None or len(testloader.dataset) == 0: return 0.0, 0, {"accuracy": 0.0}
            loss, accuracy = validate_model(self.model, self.metric_fc, testloader)
            return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}
        except Exception as e:
            return 0.0, 0, {"accuracy": 0.0}

def start_flower_client():
    """Main client loop: Strictly reactive to server phases."""
    global LAST_PROCESSED_PHASE
    print("[CLIENT] Federated Lifecycle Service Started (Reactive Mode).")
    
    # 0. INITIAL REGISTRATION 
    print(f"[CLIENT] Registering with ID: {CLIENT_ID}...")
    registered = False
    while not registered:
        try:
            report_status("Online (Menunggu Instruksi)")
            registered = True
            print("[CLIENT] Registration successful.")
        except Exception as e:
            print(f"[CLIENT] Registration failed: {e}. Retrying in 5s...")
            time.sleep(5)

    last_seen_phase = "idle"

    while True:
        try:
            # 1. POLL SERVER FOR PHASE
            resp = requests.get(f"{config.get_server_url()}/api/training/status", timeout=5)
            if resp.status_code == 200:
                status_data = resp.json()
                current_phase = status_data.get("current_phase", "idle")
                
                # REACTION LOGIC
                if current_phase == "syncing" and last_seen_phase != "syncing":
                    print("[CLIENT] Phase 1: Received sync command. Importing students...")
                    report_status("Processing: Sync Volume...")
                    auto_volume_import()
                    report_status("Standby (Siap Preprocess)") # Signal server Phase 1 is done for us
                    print("[CLIENT] Phase 1 Complete.")
                    last_seen_phase = "syncing"

                elif current_phase == "preprocessing" and last_seen_phase != "preprocessing":
                    print("[CLIENT] Phase 2: Received preprocessing command. Extracting faces...")
                    report_status("Processing: Face Extraction...")
                    run_persistent_preprocessing()
                    report_status("Standby (Siap Training)") # Signal server Phase 2 is done for us
                    print("[CLIENT] Phase 2 Complete.")
                    last_seen_phase = "preprocessing"

                elif current_phase == "training" and last_seen_phase != "training":
                    print("[CLIENT] Phase 3: Received training command. Connecting to Flower...")
                    try:
                        report_status("Training: Flower Round...")
                        fl.client.start_client(server_address=config.get_fl_server_address(), client=RealClient().to_client(), insecure=True)
                        print("[CLIENT] FL Session completed.")
                        report_status("Selesai (Siap Siklus Baru)")
                    except Exception as e:
                        print(f"[CLIENT] Flower Connection Error: {e}. Retrying in 10s...")
                        time.sleep(10)
                    last_seen_phase = "training"
                
                elif current_phase == "idle" or current_phase == "completed":
                    if last_seen_phase != "idle":
                        print("[CLIENT] System idle. Ready for next cycle.")
                        last_seen_phase = "idle"
                    # Simple heartbeat status
                    if current_phase == "completed":
                        report_status("Selesai (Siap Siklus Baru)")
                    else:
                        report_status("Online (Menunggu Instruksi)")

        except Exception as e:
            print(f"[LIFECYCLE ERROR] {e}")
        time.sleep(5)