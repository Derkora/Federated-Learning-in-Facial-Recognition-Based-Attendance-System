import os
import datetime
import time
import shutil
import random
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F

from app.utils.face_utils import face_handler, DEVICE
from app.utils.freezing import set_model_freeze

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
    # Kontroler Utama untuk Pelatihan Terpusat (Centralized Training).
    # Menangani alur data dari penerimaan hingga evaluasi model.
    
    def __init__(self):
        # Inisialisasi direktori penyimpanan data dan model
        os.makedirs(PROCESSED_DATA, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)


    def fetch_data(self, wait_timeout=3600, expected_clients=None):
        # Tahap 1: Sinkronisasi dan Menunggu Unggahan Data dari Terminal
        print(f"[INIT] Memulai pemantauan unggahan data di {UPLOAD_DIR} (Timeout: 1 Jam)...", flush=True)
        if expected_clients:
            print(f"[OK] Menunggu data dari minimal {expected_clients} terminal.", flush=True)
            
        start_time = time.time()
        last_img_count = -1
        stable_since = None
        last_log_time = 0
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
                            print(f"[SUCCESS] {msg}", flush=True)
                            cl_manager.update_logs(msg)
                            break
                        else:
                            if time.time() - last_log_time > 60:
                                print(f"[INFO] Menunggu kestabilan data... ({int(STABILIZATION_TIME - elapsed_stable)} detik lagi)", flush=True)
                                last_log_time = time.time()
                else:
                    if time.time() - last_log_time > 60:
                        print(f"[INFO] Menunggu unggahan pertama... (Durasi: {int(time.time() - start_time)} detik)", flush=True)
                        last_log_time = time.time()
            except Exception as e:
                print(f"[ERROR] Kesalahan saat pengecekan data: {e}", flush=True)
            
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
        # Tahap 2: Menyeleksi wajah terbaik dan melakukan pemotongan (Face Cropping)
        try:
            print("[INIT] Memulai proses seleksi ketajaman dan pemotongan wajah...", flush=True)
            if os.path.exists(PROCESSED_DATA): 
                shutil.rmtree(PROCESSED_DATA)
            os.makedirs(PROCESSED_DATA, exist_ok=True)
            
            # Pindai folder NRP yang diunggah
            folders = [f for f in os.listdir(UPLOAD_DIR) if os.path.isdir(os.path.join(UPLOAD_DIR, f))]
            total_folders = len(folders)
            for i, nrp_folder in enumerate(folders):
                src = os.path.join(UPLOAD_DIR, nrp_folder)
                dst = os.path.join(PROCESSED_DATA, nrp_folder)
                os.makedirs(dst, exist_ok=True)
                
                # 1. Seleksi Laplacian: Ambil Top 50 foto tertajam
                top_images = face_handler.select_best_faces(src, n=50)
                
                msg = f"Memproses {len(top_images)} foto terbaik untuk {nrp_folder} ({i+1}/{total_folders})"
                print(f"[OK] {msg}", flush=True)
                cl_manager.update_logs(msg)
                
                for img_name in top_images:
                    # 2. Deteksi & Alignment: Simpan hasil crop 112x96 dengan landmark alignment
                    face_handler.detect_and_save(os.path.join(src, img_name), os.path.join(dst, img_name))
            
            print("[SUCCESS] Preprocessing selesai. Seluruh wajah telah disejajarkan (Aligned).")
            return {"status": "success", "message": "Seleksi Laplacian dan Landmark Alignment selesai."}
        except Exception as e:
            print(f"[ERROR] Gagal melakukan preprocessing: {e}")
            return {"status": "error", "message": str(e)}

    def _get_lr(self, epoch):
        # LR Schedule Cosine Annealing (Smooth Decay)
        initial_lr = TRAINING_PARAMS.get("initial_lr", 0.1)
        min_lr = TRAINING_PARAMS.get("min_lr", 1e-4)
        total_epochs = TRAINING_PARAMS.get("total_epochs", 20)
        
        # Formula: min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * epoch / total_epochs))
        lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))
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
            print(f"[INIT] Memulai pelatihan dengan augmentasi dinamis ({epochs} epoch)...", flush=True)
            transform = transforms.Compose([
                transforms.Resize((112, 96)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Diperkuat
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196]),
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
            
            print(f"[DATASET] Ditemukan {len(full_dataset)} gambar dalam {num_classes} kelas.", flush=True)
            print(f"[DATASET] Split: {len(train_dataset)} latih, {len(val_dataset)} validasi.", flush=True)
            cl_manager.update_logs(f"Dataset: {len(full_dataset)} gambar, {num_classes} kelas. Batch Size: {batch_size}")
            
            model = MobileFaceNet().to(DEVICE)
            if os.path.exists(PRETRAINED_PATH):
                model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
            
            # --- PENERAPAN PARTIAL FREEZING ---
            # Opsi: "none", "early", "backbone"
            set_model_freeze(model, freeze_mode="early")
            
            metric_fc = ArcMarginProduct(128, num_classes).to(DEVICE)
            # Head selalu aktif
            for param in metric_fc.parameters():
                param.requires_grad = True

            
            # 3. SGD Optimizer dengan Per-Layer Weight Decay (Creator Standard)
            # - PReLU: weight_decay = 0
            # - Linear/Head: weight_decay = 4e-4
            # - Lainnya/Backbone: weight_decay = 4e-5
            criterion = nn.CrossEntropyLoss() # Gunakan Pure CrossEntropy (Tanpa Label Smoothing)
            
            # Pengelompokan Parameter
            ignored_params = list(map(id, model.linear1.parameters()))
            ignored_params += list(map(id, metric_fc.parameters()))
            
            prelu_params = []
            for m in model.modules():
                if isinstance(m, nn.PReLU):
                    ignored_params += list(map(id, m.parameters()))
                    prelu_params += [p for p in m.parameters() if p.requires_grad]

            
            base_params = [p for p in model.parameters() if id(p) not in ignored_params and p.requires_grad]

            
            optimizer = optim.SGD([
                {'params': base_params, 'weight_decay': 4e-5},
                {'params': [p for p in model.linear1.parameters() if p.requires_grad], 'weight_decay': 4e-4},

                {'params': [p for p in metric_fc.parameters() if p.requires_grad], 'weight_decay': 4e-4},

                {'params': prelu_params, 'weight_decay': 0.0}
            ], lr=self._get_lr(0), momentum=0.9, nesterov=True)
            
            epoch_history = []
            swa_snapshots = []
            swa_start = TRAINING_PARAMS.get("swa_start_epoch", 15)
            final_acc = 0
            start_time = time.time()
            
            for epoch in range(epochs):
                # 4. Perbarui Learning Rate sesuai Jadwal
                current_lr = self._get_lr(epoch)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                model.train()
                correct, total, total_loss = 0, 0, 0.0
                for b, (img, label) in enumerate(train_loader):
                    img, label = img.to(DEVICE), label.to(DEVICE)
                    optimizer.zero_grad()
                    
                    # Generate features once per batch
                    features = model(img)
                    
                    output = metric_fc(features, label)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # METRIK: Hitung Akurasi Sebenarnya (Tanpa Margin Pelatihan) - Functional Parity with FL
                    with torch.no_grad():
                        # Gunakan output murni (cosine similarity * s) untuk metrik akurasi dashboard
                        # Ini konsisten dengan cara FL menghitung akurasi agar tidak terlihat 0% di awal.
                        logits_for_acc = F.linear(F.normalize(features), F.normalize(metric_fc.weight)) * metric_fc.s
                        _, pred = torch.max(logits_for_acc.data, 1)
                        total += label.size(0)
                        correct += (pred == label).sum().item()
                    
                    # Log progres setiap 10 batch
                    if (b + 1) % 10 == 0 or (b + 1) == len(train_loader):
                        batch_acc = round(100 * correct / total, 2)
                        print(f"  [TRAIN] Epoch {epoch+1}: Batch {b+1}/{len(train_loader)} - Loss: {loss.item():.4f} - Akurasi: {batch_acc}%", flush=True)
                
                # 5. Validation Loop (Menyelaraskan dengan metrik FL)
                model.eval()
                val_correct, val_total, val_loss = 0, 0, 0.0
                with torch.no_grad():
                    for img, label in val_loader:
                        img, label = img.to(DEVICE), label.to(DEVICE)
                        
                        # Flip Trick: Ambil rata-rata embedding asli dan mirror
                        features_orig = model(img)
                        features_flip = model(torch.flip(img, [3]))
                        
                        features = torch.nn.functional.normalize(features_orig + features_flip, p=2, dim=1)
                        
                        # Hitung Logits Murni (Cosine Similarity * Scale) untuk Akurasi
                        logits = F.linear(features, F.normalize(metric_fc.weight)) * metric_fc.s
                        
                        # Hitung Loss menggunakan ArcFace Margin (Hanya pada fitur asli untuk validasi loss)
                        output = metric_fc(features_orig, label)
                        loss = criterion(output, label)
                        
                        val_loss += loss.item()
                        _, pred = torch.max(logits.data, 1)
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
                
                msg = f"Epoch {epoch+1}/{epochs} | Akurasi: {epoch_acc}% | Validasi: {val_acc}% | Loss: {avg_loss}"
                print(f"[OK] {msg}", flush=True)
                cl_manager.update_logs(msg)
                final_acc = val_acc 
                
                # SWA Snapshot Collection (Epoch 15-20)
                if epoch >= swa_start:
                    print(f"[SWA] Menyimpan snapshot dari epoch {epoch+1}...", flush=True)
                    swa_snapshots.append(copy.deepcopy(model.state_dict()))
            
            # --- STOCHASTIC WEIGHT AVERAGING (SWA) EXECUTION ---
            if len(swa_snapshots) > 0:
                print(f"[SWA] Melakukan rata-rata bobot dari {len(swa_snapshots)} snapshot...", flush=True)
                swa_state_dict = copy.deepcopy(swa_snapshots[0])
                for key in swa_state_dict:
                    # Rata-rata per-layer bobot dari semua snapshot
                    for i in range(1, len(swa_snapshots)):
                        swa_state_dict[key] += swa_snapshots[i][key]
                    swa_state_dict[key] = torch.div(swa_state_dict[key], len(swa_snapshots))
                
                model.load_state_dict(swa_state_dict)
                print("[SWA] Bobot model berhasil dirata-ratakan. Stabilitas & Generalisasi meningkat.", flush=True)
                cl_manager.update_logs("Snapshot Averaging (SWA) berhasil diterapkan.")
            
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
            print(f"[ERROR] Pelatihan gagal: {e}")
            return {"status": "error", "message": str(e)}

    def generate_reference_and_eval(self, dbs=None):
        # Tahap 4: Pembuatan Basis Data Referensi Wajah dan Evaluasi Transmisi
        try:
            print("[INIT] Membuat basis data referensi identitas...", flush=True)
            
            # --- PERSISTENSI VERSI (Simpan ke DB sebelum generate registry) ---
            if dbs:
                try:
                    new_v = models.ModelVersion(notes=f"Dibuat pada {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    dbs.add(new_v)
                    dbs.commit()
                    dbs.refresh(new_v)
                    print(f"[SUCCESS] Versi model v{new_v.version_id} disimpan.")
                except Exception as e:
                    print(f"[ERROR] Gagal menyimpan versi: {e}")
                    dbs.rollback()
            
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
                print("[INIT] Menjalankan uji mandiri integritas model...", flush=True)
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
                    print(f"  > [TEST] nrp: {nrp} | Sim: {sim:.4f}")
                
                avg_self_sim = sum(test_results) / len(test_results) if test_results else 0
                print(f"[SUCCESS] Uji mandiri selesai. Rata-rata Similarity: {avg_self_sim:.4f}")
            
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
            print(f"[ERROR] Gagal membuat registri: {e}")
            return {"status": "error", "message": str(e)}

training_controller = TrainingController()
