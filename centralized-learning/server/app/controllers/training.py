import os
import time
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as T
from PIL import Image

from app.utils.face_utils import face_handler, DEVICE
from app.utils.mobilefacenet import MobileFaceNet, ArcMarginProduct
from app.db import models

# Constants
UPLOAD_DIR = "data/students"
PROCESSED_DATA = "data/datasets_processed"
BALANCED_DATA = "data/datasets_balanced"
MODEL_DIR = "app/model"
MODEL_PATH = f"{MODEL_DIR}/global_model.pth"
PRETRAINED_PATH = "app/model/global_model_v0.pth"

class TrainingController:
    def __init__(self):
        os.makedirs(PROCESSED_DATA, exist_ok=True)
        os.makedirs(BALANCED_DATA, exist_ok=True)

    def train_pipeline(self, dbs):
        """Full training pipeline with evaluation (Ref: model.ipynb)."""
        start_time = time.time()
        
        # --- 0. Pre-Evaluation: Measure Payload ---
        # Untuk simulasi, kita hitung total ukuran UPLOAD_DIR
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(UPLOAD_DIR):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        payload_mb = round(total_size / (1024 * 1024), 2)
        
        # 1. Preprocessing (MTCNN)
        pre_start = time.time()
        print("[TRAIN] Step 1: Preprocessing...")
        for nrp_folder in os.listdir(UPLOAD_DIR):
            src = os.path.join(UPLOAD_DIR, nrp_folder)
            if not os.path.isdir(src): continue
            dst = os.path.join(PROCESSED_DATA, nrp_folder)
            os.makedirs(dst, exist_ok=True)
            for img_name in os.listdir(src):
                face_handler.detect_and_save(os.path.join(src, img_name), os.path.join(dst, img_name))
        pre_end = time.time()

        # 2. Balancing (Smart Balance)
        print("[TRAIN] Step 2: Balancing...")
        TARGET_COUNT = 100
        aug_transform = T.Compose([
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomRotation(degrees=20),
            T.RandomHorizontalFlip(p=0.5)
        ])
        for nrp_folder in os.listdir(PROCESSED_DATA):
            src, dst = os.path.join(PROCESSED_DATA, nrp_folder), os.path.join(BALANCED_DATA, nrp_folder)
            os.makedirs(dst, exist_ok=True)
            files = os.listdir(src)
            if not files: continue
            for f in files: shutil.copy(os.path.join(src, f), os.path.join(dst, f))
            if len(files) < TARGET_COUNT:
                for i in range(TARGET_COUNT - len(files)):
                    base = random.choice(files)
                    aug = aug_transform(Image.open(os.path.join(src, base)))
                    aug.save(os.path.join(dst, f"aug_{i}_{base}"))

        # 3. Training Loop
        train_start = time.time()
        print("[TRAIN] Step 3: Training MobileFaceNet...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        train_dataset = datasets.ImageFolder(BALANCED_DATA, transform=transform)
        num_classes = len(train_dataset.classes)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model = MobileFaceNet().to(DEVICE)
        if os.path.exists(PRETRAINED_PATH):
            model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
        
        metric_fc = ArcMarginProduct(128, num_classes).to(DEVICE)
        optimizer = optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        acc = 0
        for epoch in range(10): # Sesuai mood research, kita percepat dikit atau full 20
            correct = 0
            total = 0
            for img, label in train_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                optimizer.zero_grad()
                output = metric_fc(model(img), label)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            acc = round(100 * correct / total, 2)
        torch.save(model.state_dict(), MODEL_PATH)
        train_end = time.time()

        # 4. Reference Embeddings & Biometric Eval
        print("[TRAIN] Step 4: Generating Reference & Evaluating Biometrics...")
        model.eval()
        ref_db = {}
        # Simple Validation Set (last 10 images of each class)
        val_samples = []
        with torch.no_grad():
            for nrp in os.listdir(BALANCED_DATA):
                p = os.path.join(BALANCED_DATA, nrp)
                if not os.path.isdir(p): continue
                all_files = sorted(os.listdir(p))
                train_files = all_files[:-10]
                val_files = all_files[-10:]
                
                # Build Ref from train files
                embs = [face_handler.get_embedding(model, Image.open(os.path.join(p, f)).convert('RGB')) for f in train_files[:5]]
                if embs: ref_db[nrp] = torch.mean(torch.stack(embs), dim=0)
                
                # Collect val samples
                for vf in val_files:
                    v_emb = face_handler.get_embedding(model, Image.open(os.path.join(p, vf)).convert('RGB'))
                    val_samples.append((v_emb, nrp))
        
        torch.save(ref_db, f"{MODEL_DIR}/reference_embeddings.pth")
        
        # Calculate TAR / FAR / EER (Simplified)
        tars, fars = [], []
        thresholds = [i/100 for i in range(0, 101, 5)]
        for th in thresholds:
            ta, fa, tr, fr = 0, 0, 0, 0
            for v_emb, v_nrp in val_samples:
                best_sim = -1
                best_match = "Unknown"
                for r_nrp, r_emb in ref_db.items():
                    sim = torch.nn.functional.cosine_similarity(v_emb, r_emb).item()
                    if sim > best_sim:
                        best_sim, best_match = sim, r_nrp
                
                if best_sim > th:
                    if best_match == v_nrp: ta += 1
                    else: fa += 1
                else:
                    if best_match == v_nrp: fr += 1
                    else: tr += 1
            
            tars.append(ta / (ta + fr) if (ta + fr) > 0 else 0)
            fars.append(fa / (fa + tr) if (fa + tr) > 0 else 0)

        # find EER (where FAR matches 1-TAR approx)
        eer = 0
        for i in range(len(fars)):
            if fars[i] >= (1 - tars[i]):
                eer = round(fars[i] * 100, 2)
                break

        total_duration = round(time.time() - start_time, 2)
        cost_idr = int((payload_mb / 1024 * 5000) + (total_duration * 10))

        return {
            "status": "success", 
            "accuracy": acc,
            "tar": round(tars[thresholds.index(0.65)] * 100, 2) if 0.65 in thresholds else 0,
            "far": round(fars[thresholds.index(0.65)] * 100, 2) if 0.65 in thresholds else 0,
            "eer": eer,
            "payload_size_mb": payload_mb,
            "total_duration_s": total_duration,
            "cost_idr": cost_idr
        }

training_controller = TrainingController()
