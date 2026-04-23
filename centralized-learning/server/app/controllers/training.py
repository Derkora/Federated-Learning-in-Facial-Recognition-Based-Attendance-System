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
import cv2
import numpy as np

from app.utils.face_utils import face_handler, DEVICE
from app.utils.mobilefacenet import MobileFaceNet, ArcMarginProduct
from app.db import models
from app.server_manager_instance import cl_manager
from app.config import (
    UPLOAD_DIR, PROCESSED_DATA, MODEL_DIR, MODEL_PATH, 
    PRETRAINED_PATH, EMISSIONS_DIR, TRAINING_PARAMS, CODECARBON_AVAILABLE
)

if CODECARBON_AVAILABLE:
    try:
        from codecarbon import OfflineEmissionsTracker
    except ImportError:
        pass

class TrainingController:
    # Kontroler Utama untuk Siklus Pelatihan Terpusat (Centralized Training).
    # Menangani penerimaan data, pra-pemrosesan, hingga evaluasi biometric.
    
    def __init__(self):
        os.makedirs(PROCESSED_DATA, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

    def get_blur_score(self, image_path):
        # Mengukur tingkat ketajaman gambar menggunakan variansi Laplacian.
        try:
            img = cv2.imread(image_path)
            if img is None: return 0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except: return 0

    def fetch_data(self, wait_timeout=600, expected_clients=None):
        # Tahap 1: Sinkronisasi dan Menunggu Unggahan Data dari Terminal
        print(f"[INFO] Memulai pemantauan unggahan data di {UPLOAD_DIR}...", flush=True)
        if expected_clients:
            print(f"[INFO] Menunggu data dari minimal {expected_clients} terminal.", flush=True)
            
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
                            print(f"[INFO] Menunggu kestabilan data... ({int(STABILIZATION_TIME - elapsed_stable)} detik lagi)", flush=True)
                else:
                    print(f"[INFO] Menunggu unggahan pertama... (Durasi: {int(time.time() - start_time)} detik)", flush=True)
            except Exception as e:
                print(f"[ERROR] Kesalahan saat pengecekan: {e}", flush=True)
            
            time.sleep(5)
        
        if last_img_count <= 0:
            return {"status": "error", "message": "Tidak ada data yang diterima hingga batas waktu."}
            
        # Hitung Ukuran Payload
        total_size = 0
        for dirpath, _, filenames in os.walk(UPLOAD_DIR):
            for f in filenames: total_size += os.path.getsize(os.path.join(dirpath, f))
        
        return {
            "status": "success", 
            "upload_volume_mb": round(total_size / (1024 * 1024), 2),
            "classes": len(subdirs),
            "images": last_img_count
        }

    def preprocess_and_balance(self):
        # Tahap 2: Penyelarasan Wajah (MTCNN) dan Seleksi Kualitas (Laplacian)
        try:
            print("[INFO] Memulai tahap pra-pemrosesan dan seleksi kualitas...", flush=True)
            if os.path.exists(PROCESSED_DATA): shutil.rmtree(PROCESSED_DATA)
            os.makedirs(PROCESSED_DATA, exist_ok=True)
            
            # Deteksi dan Cropping Wajah (MTCNN)
            folders = [f for f in os.listdir(UPLOAD_DIR) if os.path.isdir(os.path.join(UPLOAD_DIR, f))]
            total_folders = len(folders)
            for i, nrp_folder in enumerate(folders):
                src = os.path.join(UPLOAD_DIR, nrp_folder)
                dst = os.path.join(PROCESSED_DATA, nrp_folder)
                os.makedirs(dst, exist_ok=True)
                
                # Seleksi berbasis Blur Detection (Laplacian) - Ambil Top 50 Tertajam
                all_images = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                scored_images = sorted([(img, self.get_blur_score(os.path.join(src, img))) for img in all_images], key=lambda x: x[1], reverse=True)
                
                # Ambil maksimal 50 gambar terbaik
                top_images = [img[0] for img in scored_images[:50]]
                
                msg = f"Memproses {len(top_images)} wajah terbaik untuk {nrp_folder} ({i+1}/{total_folders})"
                print(f"[INFO] {msg}", flush=True)
                cl_manager.update_logs(msg)
                
                for img_name in top_images:
                    face_handler.detect_and_save(os.path.join(src, img_name), os.path.join(dst, img_name))
            
            # Augmentasi sekarang dilakukan secara dinamis (on-the-fly) saat training
            return {"status": "success", "message": "Pra-pemrosesan dan seleksi kualitas (Top 50 Laplacian) selesai."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _get_lr(self, epoch):
        # Penyelarasan LR Schedule dari konfigurasi terpusat
        schedule = TRAINING_PARAMS["lr_schedule"]
        # Ambil LR berdasarkan epoch tertinggi yang sudah dilewati
        lr = 1e-4 # Default
        for threshold in sorted(schedule.keys()):
            if epoch >= threshold:
                lr = schedule[threshold]
        return lr

    def train_model(self, epochs=None):
        if epochs is None:
            epochs = cl_manager.default_epochs
            
        # Tahap 3: Pelatihan Model MobileFaceNet
        tracker = None
        if CODECARBON_AVAILABLE:
            try:
                tracker = OfflineEmissionsTracker(
                    country_iso_code="IDN", 
                    log_level="error", 
                    save_to_file=True, 
                    output_dir=EMISSIONS_DIR
                )
                tracker.start()
            except: pass

        try:
            print(f"[INFO] Memulai pelatihan dengan augmentasi dinamis ({epochs} epoch)...", flush=True)
            transform = transforms.Compose([
                transforms.Resize((112, 96)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                transforms.RandomErasing(p=0.1)
            ])
            full_dataset = datasets.ImageFolder(PROCESSED_DATA, transform=transform)
            num_classes = len(full_dataset.classes)
            
            # Split 80/20 untuk training & validation (Menyelaraskan dengan FL)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
            
            batch_size = cl_manager.default_batch_size
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            model = MobileFaceNet().to(DEVICE)
            if os.path.exists(PRETRAINED_PATH):
                model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
            
            metric_fc = ArcMarginProduct(128, num_classes).to(DEVICE)
            
            # 3. Adam Optimizer dengan Label Smoothing dari Config
            smoothing = TRAINING_PARAMS["label_smoothing"]
            criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)
            
            optimizer = optim.Adam([
                {'params': model.parameters()}, 
                {'params': metric_fc.parameters()}
            ], lr=self._get_lr(0))
            
            epoch_history = []
            final_acc = 0
            start_time = time.time()
            
            for epoch in range(epochs):
                # 4. Perbarui Learning Rate sesuai Jadwal
                current_lr = self._get_lr(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                model.train()
                correct, total, total_loss = 0, 0, 0.0
                for img, label in train_loader:
                    img, label = img.to(DEVICE), label.to(DEVICE)
                    optimizer.zero_grad()
                    output = metric_fc(model(img), label)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, pred = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (pred == label).sum().item()
                
                # 5. Validation Loop (Menyelaraskan dengan metrik FL)
                model.eval()
                val_correct, val_total, val_loss = 0, 0, 0.0
                with torch.no_grad():
                    for img, label in val_loader:
                        img, label = img.to(DEVICE), label.to(DEVICE)
                        output = metric_fc(model(img), label)
                        loss = criterion(output, label)
                        val_loss += loss.item()
                        _, pred = torch.max(output.data, 1)
                        val_total += label.size(0)
                        val_correct += (pred == label).sum().item()
                
                val_acc = round(100 * val_correct / val_total, 2)
                val_avg_loss = round(val_loss / len(val_loader), 4)
                
                epoch_acc = round(100 * correct / total, 2)
                avg_loss = round(total_loss / len(train_loader), 4)
                epoch_history.append({
                    "epoch": epoch + 1, 
                    "loss": avg_loss, 
                    "accuracy": epoch_acc,
                    "val_loss": val_avg_loss,
                    "val_accuracy": val_acc
                })
                
                msg = f"Epoch {epoch+1}/{epochs} | Acc: {epoch_acc}% | Val Acc: {val_acc}% | Loss: {avg_loss}"
                print(f"[INFO] {msg}", flush=True)
                cl_manager.update_logs(msg)
                final_acc = val_acc 
            
            torch.save(model.state_dict(), MODEL_PATH)
            cl_manager.update_logs("Pelatihan selesai. Model berhasil disimpan.")
            
            # 5. Ambil data energi jika CodeCarbon aktif
            energy_kwh = 0
            if tracker:
                try:
                    emissions_data = tracker.stop()
                    energy_kwh = tracker.final_emissions_data.energy_consumed
                except: pass

            return {
                "status": "success", 
                "duration_s": round(time.time() - start_time, 2),
                "compute_energy_kwh": energy_kwh,
                "accuracy": final_acc,
                "epoch_history": epoch_history
            }
        except Exception as e:
            if tracker: tracker.stop()
            return {"status": "error", "message": str(e)}

    def generate_reference_and_eval(self):
        # Tahap 4: Pembuatan Basis Data Referensi Wajah dan Evaluasi Transmisi
        try:
            print("[INFO] Menghasilkan basis data referensi dan menghitung volume transmisi...", flush=True)
            model = MobileFaceNet().to(DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            
            ref_db = {}
            with torch.no_grad():
                for nrp in os.listdir(PROCESSED_DATA):
                    p = os.path.join(PROCESSED_DATA, nrp)
                    if not os.path.isdir(p): continue
                    all_files = sorted(os.listdir(p))
                    # Gunakan 50 gambar terbaik untuk rata-rata yang lebih stabil (Hardened)
                    train_files = all_files[:50]
                    
                    embs = []
                    for f in train_files:
                        try:
                            img_p = Image.open(os.path.join(p, f)).convert('RGB')
                            emb = face_handler.get_embedding(model, img_p)
                            embs.append(emb)
                        except: continue
                        
                    if embs:
                        # Rata-rata 50 gambar dan RE-NORMALISASI ke unit vector (PENTING untuk ArcFace)
                        centroid = torch.mean(torch.stack(embs), dim=0)
                        centroid = torch.nn.functional.normalize(centroid, p=2, dim=1)
                        ref_db[nrp] = centroid.cpu()

                # --- SELF-TEST: Verifikasi integritas embedding pada server ---
                print("[INFO] Menjalankan self-test pada server...", flush=True)
                test_results = []
                for nrp, centroid in ref_db.items():
                    p = os.path.join(PROCESSED_DATA, nrp)
                    if not os.path.isdir(p): continue
                    
                    files = sorted(os.listdir(p))
                    if not files: continue
                    
                    # Test dengan gambar pertama (seharusnya similarity sangat tinggi, >0.9)
                    first_img = files[0]
                    img_p = Image.open(os.path.join(p, first_img)).convert('RGB')
                    test_emb = face_handler.get_embedding(model, img_p).cpu()
                    
                    # Cosine Similarity (Keduanya unit vector, dot product = cos_sim)
                    # Pastikan shape (1, 128) -> (128,) untuk dot product yang benar
                    sim = torch.sum(test_emb.view(-1) * centroid.view(-1)).item()
                    test_results.append(sim)
                    print(f"  > [SELF-TEST] nrp: {nrp} | Sim: {sim:.4f} | Embedding Norm: {test_emb.norm():.2f}")
                
                avg_self_sim = sum(test_results) / len(test_results) if test_results else 0
                print(f"[OK] Self-test selesai. Rata-rata similarity internal: {avg_self_sim:.4f}")
            
            torch.save(ref_db, f"{MODEL_DIR}/reference_embeddings.pth")
            
            # Hitung Ukuran Aset yang akan didownload client (Model + Registri)
            download_size = 0
            if os.path.exists(MODEL_PATH): download_size += os.path.getsize(MODEL_PATH)
            REF_PATH = f"{MODEL_DIR}/reference_embeddings.pth"
            if os.path.exists(REF_PATH): download_size += os.path.getsize(REF_PATH)

            # 5. Kembalikan volume transmisi
            return {
                "status": "success",
                "download_volume_mb": round(download_size / (1024 * 1024), 2)
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

training_controller = TrainingController()
