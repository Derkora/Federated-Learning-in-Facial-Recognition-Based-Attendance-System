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

from ..utils.face_utils import face_handler, DEVICE
from ..utils.mobilefacenet import MobileFaceNet, ArcMarginProduct
from ..db import models
from ..server_manager_instance import cl_manager

# Konfigurasi Jalur Data (Dataset Paths)
UPLOAD_DIR = "data/students"
PROCESSED_DATA = "data/datasets_processed"
BALANCED_DATA = "data/datasets_balanced"
MODEL_DIR = "app/model"
MODEL_PATH = f"{MODEL_DIR}/global_model.pth"
PRETRAINED_PATH = "app/model/global_model_v0.pth"

class TrainingController:
    # Kontroler Utama untuk Siklus Pelatihan Terpusat (Centralized Training).
    # Menangani penerimaan data, pra-pemrosesan, hingga evaluasi biometric.
    
    def __init__(self):
        os.makedirs(PROCESSED_DATA, exist_ok=True)
        os.makedirs(BALANCED_DATA, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

    def fetch_data(self, wait_timeout=600, expected_clients=None):
        # Tahap 1: Sinkronisasi dan Menunggu Unggahan Data dari Terminal
        print(f"[FASE 1] Memulai pemantauan unggahan data di {UPLOAD_DIR}...", flush=True)
        if expected_clients:
            print(f"[FASE 1] Menunggu data dari minimal {expected_clients} terminal.", flush=True)
            
        start_time = time.time()
        last_img_count = -1
        stable_since = None
        STABILIZATION_TIME = 15 # Detekasi kestabilan data (tidak ada perubahan jumlah berkas)
        
        while (time.time() - start_time) < wait_timeout:
            try:
                subdirs = [d for d in os.listdir(UPLOAD_DIR) if os.path.isdir(os.path.join(UPLOAD_DIR, d))]
                img_count = 0
                if len(subdirs) > 0:
                    img_count = sum([len(os.listdir(os.path.join(UPLOAD_DIR, d))) for d in subdirs])
                
                # Cek progres unggahan
                if img_count > 0:
                    if img_count != last_img_count:
                        last_img_count = img_count
                        stable_since = time.time()
                        msg = f"Progres: {len(subdirs)} kelas, {img_count} gambar terdeteksi. Menunggu stabil..."
                        print(f"[FASE 1] {msg}", flush=True)
                        cl_manager.update_logs(msg)
                        cl_manager.update_received_data(UPLOAD_DIR)
                    else:
                        elapsed_stable = time.time() - stable_since
                        if elapsed_stable >= STABILIZATION_TIME:
                            msg = f"Data telah stabil! Total akhir: {len(subdirs)} kelas, {img_count} gambar."
                            print(f"[FASE 1] {msg}", flush=True)
                            cl_manager.update_logs(msg)
                            break
                        else:
                            print(f"[FASE 1] Menunggu kestabilan data... ({int(STABILIZATION_TIME - elapsed_stable)} detik lagi)", flush=True)
                else:
                    print(f"[FASE 1] Menunggu unggahan pertama... (Durasi: {int(time.time() - start_time)} detik)", flush=True)
            except Exception as e:
                print(f"[FASE 1] Kesalahan saat pengecekan: {e}", flush=True)
            
            time.sleep(5)
        
        if last_img_count <= 0:
            return {"status": "error", "message": "Tidak ada data yang diterima hingga batas waktu."}
            
        # Hitung Ukuran Payload
        total_size = 0
        for dirpath, _, filenames in os.walk(UPLOAD_DIR):
            for f in filenames: total_size += os.path.getsize(os.path.join(dirpath, f))
        
        return {
            "status": "success", 
            "payload_mb": round(total_size / (1024 * 1024), 2),
            "classes": len(subdirs),
            "images": last_img_count
        }

    def preprocess_and_balance(self):
        # Tahap 2: Penyelarasan Wajah (MTCNN) dan Balancing Dataset menggunakan Augmentasi
        try:
            print("[FASE 2] Memulai Tahap Pra-pemrosesan...", flush=True)
            if os.path.exists(PROCESSED_DATA): shutil.rmtree(PROCESSED_DATA)
            os.makedirs(PROCESSED_DATA, exist_ok=True)
            
            # 1. Deteksi dan Cropping Wajah (MTCNN)
            folders = [f for f in os.listdir(UPLOAD_DIR) if os.path.isdir(os.path.join(UPLOAD_DIR, f))]
            total_folders = len(folders)
            for i, nrp_folder in enumerate(folders):
                src = os.path.join(UPLOAD_DIR, nrp_folder)
                dst = os.path.join(PROCESSED_DATA, nrp_folder)
                os.makedirs(dst, exist_ok=True)
                
                msg = f"Memproses wajah mahasiswa {i+1}/{total_folders}: {nrp_folder}"
                print(f"[FASE 2] {msg}", flush=True)
                cl_manager.update_logs(msg)
                
                for img_name in os.listdir(src):
                    face_handler.detect_and_save(os.path.join(src, img_name), os.path.join(dst, img_name))
            
            # 2. Balancing (Menyeimbangkan jumlah data tiap kelas)
            msg = "Menyeimbangkan dataset dengan augmentasi data..."
            print(f"[FASE 2] {msg}", flush=True)
            cl_manager.update_logs(msg)
            if os.path.exists(BALANCED_DATA): shutil.rmtree(BALANCED_DATA)
            os.makedirs(BALANCED_DATA, exist_ok=True)
            
            TARGET_COUNT = 100 # Target jumlah gambar per mahasiswa
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
                # Salin berkas asli
                for f in files: shutil.copy(os.path.join(src, f), os.path.join(dst, f))
                # Tambahkan augmentasi jika kurang dari target
                if len(files) < TARGET_COUNT:
                    for i in range(TARGET_COUNT - len(files)):
                        base = random.choice(files)
                        aug = aug_transform(Image.open(os.path.join(src, base)))
                        aug.save(os.path.join(dst, f"aug_{i}_{base}"))
            
            return {"status": "success", "message": "Pra-pemrosesan dan penyeimbangan data selesai."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def train_model(self, epochs=10):
        # Tahap 3: Pelatihan Model MobileFaceNet
        try:
            print(f"[FASE 3] Memulai Pelatihan ({epochs} epoch)...", flush=True)
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
            final_acc = 0
            start_time = time.time()
            for epoch in range(epochs):
                correct, total = 0, 0
                for img, label in train_loader:
                    img, label = img.to(DEVICE), label.to(DEVICE)
                    optimizer.zero_grad()
                    output = metric_fc(model(img), label)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()
                    _, pred = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (pred == label).sum().item()
                final_acc = round(100 * correct / total, 2)
                msg = f"Epoch {epoch+1}/{epochs} - Akurasi: {final_acc}%"
                print(f"[FASE 3] {msg}", flush=True)
                cl_manager.update_logs(msg)
            
            torch.save(model.state_dict(), MODEL_PATH)
            cl_manager.update_logs("Pelatihan selesai. Model berhasil disimpan.")
            return {
                "status": "success", 
                "accuracy": final_acc, 
                "duration_s": round(time.time() - start_time, 2)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def generate_reference_and_eval(self):
        # Tahap 4: Pembuatan Basis Data Referensi Wajah dan Evaluasi Biometrik
        try:
            print("[FASE 4] Menghasilkan basis data referensi dan evaluasi...", flush=True)
            model = MobileFaceNet().to(DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            
            ref_db = {}
            val_samples = []
            with torch.no_grad():
                for nrp in os.listdir(BALANCED_DATA):
                    p = os.path.join(BALANCED_DATA, nrp)
                    if not os.path.isdir(p): continue
                    all_files = sorted(os.listdir(p))
                    train_files, val_files = all_files[:-10], all_files[-10:]
                    
                    # Membuat embedding referensi dari data latih
                    embs = [face_handler.get_embedding(model, Image.open(os.path.join(p, f)).convert('RGB')) for f in train_files[:5]]
                    if embs: ref_db[nrp] = torch.mean(torch.stack(embs), dim=0)
                    # Sampel validasi dari data sisa
                    for vf in val_files:
                        val_samples.append((face_handler.get_embedding(model, Image.open(os.path.join(p, vf)).convert('RGB')), nrp))
            
            torch.save(ref_db, f"{MODEL_DIR}/reference_embeddings.pth")
            
            # Pengujian Biometrik (Threshold Sweeping)
            tars, fars = [], []
            thresholds = [i/100 for i in range(0, 101, 5)]
            for th in thresholds:
                ta, fa, tr, fr = 0, 0, 0, 0
                for v_emb, v_nrp in val_samples:
                    best_sim, best_match = -1, "Unknown"
                    for r_nrp, r_emb in ref_db.items():
                        sim = torch.nn.functional.cosine_similarity(v_emb, r_emb).item()
                        if sim > best_sim: best_sim, best_match = sim, r_nrp
                    if best_sim > th:
                        if best_match == v_nrp: ta += 1
                        else: fa += 1
                    else:
                        if best_match == v_nrp: fr += 1
                        else: tr += 1
                tars.append(ta / (ta + fr) if (ta + fr) > 0 else 0)
                fars.append(fa / (fa + tr) if (fa + tr) > 0 else 0)

            eer = 0
            for i in range(len(fars)):
                if fars[i] >= (1 - tars[i]):
                    eer = round(fars[i] * 100, 2)
                    break

            return {
                "status": "success",
                "tar": round(tars[thresholds.index(0.65)] * 100, 2) if 0.65 in thresholds else 0,
                "far": round(fars[thresholds.index(0.65)] * 100, 2) if 0.65 in thresholds else 0,
                "eer": eer
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

training_controller = TrainingController()
