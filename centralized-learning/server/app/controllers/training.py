import os
from datetime import datetime, timedelta, timezone
import time
import shutil
import random
import math
import copy
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F

from app.utils.face_utils import face_handler, DEVICE
from app.utils.freezing import set_model_freeze

from app.utils.mobilefacenet import MobileFaceNet, ArcMarginProduct
from app.db import models
from app.db.db import SessionLocal
from app.db.models import UserGlobal
from app.server_manager_instance import cl_manager
from app.config import (
    UPLOAD_DIR, PROCESSED_DATA, MODEL_DIR, MODEL_PATH, REF_PATH,
    PRETRAINED_PATH, EMISSIONS_DIR, TRAINING_PARAMS, CODECARBON_AVAILABLE
)
from app.utils.logging import get_logger

OfflineEmissionsTracker = None
if CODECARBON_AVAILABLE:
    try:
        from codecarbon import OfflineEmissionsTracker
    except ImportError:
        CODECARBON_AVAILABLE = False

class TrainingController:
    # Kontroler Utama untuk Pelatihan Terpusat (Centralized Training).
    # Menangani alur data dari penerimaan hingga evaluasi model.
    
    def __init__(self):
        # Inisialisasi direktori penyimpanan data dan model
        os.makedirs(PROCESSED_DATA, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.logger = get_logger()


    def _wait_for_stable_data(self, wait_timeout=3600, expected_clients=None):
        # Helper untuk menunggu hingga data yang diunggah stabil (tidak bertambah lagi).
        start_time = time.time()
        last_img_count = -1
        stable_since = None
        last_log_time = 0
        STABILIZATION_TIME = 60 
        
        while (time.time() - start_time) < wait_timeout:
            try:
                if not os.path.exists(UPLOAD_DIR):
                    os.makedirs(UPLOAD_DIR, exist_ok=True)
                    
                subdirs = [d for d in os.listdir(UPLOAD_DIR) if os.path.isdir(os.path.join(UPLOAD_DIR, d))]
                img_count = 0
                if len(subdirs) > 0:
                    img_count = sum([len(os.listdir(os.path.join(UPLOAD_DIR, d))) for d in subdirs])
                
                unique_uploaders = set(cl_manager.uploader_map.values())
                uploader_count = len(unique_uploaders)
                
                status_str = f"[{uploader_count}/{expected_clients if expected_clients else '?'}]"
                
                if img_count > 0:
                    if img_count != last_img_count:
                        last_img_count = img_count
                        stable_since = time.time()
                        msg = f"Progres {status_str}: {len(subdirs)} kelas, {img_count} gambar terdeteksi."
                        self.logger.info(msg)
                        cl_manager.update_received_data(UPLOAD_DIR)
                    else:
                        elapsed_stable = time.time() - stable_since
                        ready_to_break = (elapsed_stable >= STABILIZATION_TIME)
                        if expected_clients:
                            ready_to_break = ready_to_break and (uploader_count >= expected_clients)
                        
                        if ready_to_break:
                            msg = f"Data {status_str} telah stabil! Total akhir: {len(subdirs)} kelas, {img_count} gambar."
                            self.logger.success(msg)
                            return True, last_img_count
                else:
                    if time.time() - last_log_time > 60:
                        self.logger.info(f"Menunggu unggahan data pertama... {status_str}")
                        last_log_time = time.time()
            except Exception as e:
                self.logger.error(f"Kesalahan saat pengecekan data masuk: {e}")
            
            time.sleep(5)
        return False, last_img_count

    def fetch_data(self, wait_timeout=3600, expected_clients=None):
        # Tahap Sinkronisasi dan Menunggu Unggahan Data dari Terminal
        self.logger.info(f"Memulai pemantauan unggahan data di {UPLOAD_DIR} (Batas Waktu: 1 Jam)...")
        if expected_clients:
            self.logger.info(f"Menunggu data masuk dari minimal {expected_clients} terminal klien.")
            
        success, last_img_count = self._wait_for_stable_data(wait_timeout, expected_clients)
        
        if not success and last_img_count <= 0:
            return {"status": "error", "message": "Tidak ada data yang diterima hingga batas waktu."}

        # Hitung Ukuran Payload dan Jumlah Kelas terdaftar
        total_size = 0
        subdirs = [d for d in os.listdir(UPLOAD_DIR) if os.path.isdir(os.path.join(UPLOAD_DIR, d))]
        for dirpath, _, filenames in os.walk(UPLOAD_DIR):
            for f in filenames: total_size += os.path.getsize(os.path.join(dirpath, f))
        
        return {
            "status": "success", 
            "upload_volume_mb": round(total_size / (1024 * 1024), 2),
            "classes": len(subdirs),
            "images": last_img_count
        }

    def workflow_preprocess(self, wait_timeout=3600, expected_clients=None, dataset="students"):
        # Tahap Seleksi Laplacian (Ketajaman) dan Penyelarasan Landmark Wajah
        tracker = None
        energy_kwh = 0.0
        if CODECARBON_AVAILABLE and OfflineEmissionsTracker is not None:
            try:
                tracker = OfflineEmissionsTracker(country_iso_code="IDN", measure_power_secs=15, log_level="error", save_to_file=False)
                tracker.start()
            except: pass

        try:
            upload_dir = UPLOAD_DIR
            processed_dir = PROCESSED_DATA
            
            if dataset and dataset != "students":
                opt_datasets_path = f"/app/datasets/{dataset}/students"
                if os.path.exists(opt_datasets_path):
                    upload_dir = opt_datasets_path
                else:
                    alt_path = os.path.join(os.path.dirname(UPLOAD_DIR), f"students_{dataset}")
                    if os.path.exists(alt_path):
                        upload_dir = alt_path
                    else:
                        upload_dir = os.path.join(UPLOAD_DIR, dataset)
                
                processed_dir = os.path.join(os.path.dirname(PROCESSED_DATA), f"datasets_processed_{dataset}")

            if not os.path.exists(upload_dir):
                return {"status": "error", "message": f"Dataset {dataset} tidak ditemukan di {upload_dir}."}
            
            folders = [f for f in os.listdir(upload_dir) if os.path.isdir(os.path.join(upload_dir, f))]
            total_folders = len(folders)
            
            if total_folders == 0:
                return {"status": "error", "message": f"Tidak ada data mahasiswa di {upload_dir} untuk diproses."}

            self.logger.info(f"Memulai tahap preprocessing wajah untuk {total_folders} mahasiswa di {dataset}...")
            
            for i, nrp_folder in enumerate(folders):
                self.logger.info(f"[{i+1}/{total_folders}] Memproses: {nrp_folder}...")
                src = os.path.join(upload_dir, nrp_folder)
                dst = os.path.join(processed_dir, nrp_folder)
                
                # Paksa proses ulang agar pengguna dapat memantau log Laplace Variance
                tmp_dst = dst + ".tmp"
                if os.path.exists(tmp_dst): shutil.rmtree(tmp_dst)
                os.makedirs(tmp_dst, exist_ok=True)
                
                top_images = face_handler.select_best_faces(src, n=50)
                if not top_images:
                    self.logger.warn(f"  ! Skip: Tidak ada citra wajah valid ditemukan di {nrp_folder}")
                    continue
                
                for img_name in top_images:
                    face_handler.detect_and_save(os.path.join(src, img_name), os.path.join(tmp_dst, img_name))
                
                if os.path.exists(dst): shutil.rmtree(dst)
                os.rename(tmp_dst, dst)
                
                # Pengelolaan penggunaan memori RAM perangkat
                if (i+1) % 5 == 0:
                    gc.collect()

            if tracker:
                try:
                    energy_kwh = tracker.stop()
                    if energy_kwh is None: energy_kwh = 0.0
                    cl_manager.update_metrics({"compute_energy_kwh": cl_manager.metrics.get("compute_energy_kwh", 0) + energy_kwh})
                except: pass

            self.logger.success(f"Preprocessing {dataset} selesai. Seluruh wajah berhasil diselaraskan (Aligned).")
            return {"status": "success", "message": "Seleksi Laplacian dan Landmark Alignment selesai."}
        except Exception as e:
            if tracker: tracker.stop()
            self.logger.error(f"Gagal melakukan preprocessing data wajah {dataset}: {e}")
            return {"status": "error", "message": str(e)}

    def sync_nrp_from_processed(self, dbs, dataset="students"):
        # Sinkronisasi ulang tabel UserGlobal dari processed data untuk keteraturan NRP.
        processed_dir = PROCESSED_DATA
        if dataset and dataset != "students":
            processed_dir = os.path.join(os.path.dirname(PROCESSED_DATA), f"datasets_processed_{dataset}")

        if not os.path.exists(processed_dir): return
        folders = sorted([f for f in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, f))])
        for folder in folders:
            parts = folder.split("_", 1)
            nrp = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else "Unknown"
            
            existing = dbs.query(models.UserGlobal).filter(models.UserGlobal.nrp == nrp).first()
            if not existing:
                dbs.add(models.UserGlobal(name=name, nrp=nrp))
        dbs.commit()
        self.logger.success(f"Sinkronisasi {len(folders)} identitas NRP dari folder processed {dataset} berhasil.")


    def _get_lr(self, epoch, total_epochs=None):
        # Penyesuaian Learning Rate Menggunakan Cosine Annealing (Smooth Decay)
        initial_lr = TRAINING_PARAMS.get("initial_lr", 0.05)
        min_lr = TRAINING_PARAMS.get("min_lr", 1e-4)
        if total_epochs is None:
            total_epochs = getattr(cl_manager, 'default_epochs', 10)
        
        # Rumus: min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * epoch / total_epochs))
        lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))
        return lr

    def train_model(self, epochs=None, dataset="students", dbs=None, model_version_id=None):
        if epochs is None:
            epochs = cl_manager.default_epochs
            
        # Tahap Pelatihan Model Terpusat MobileFaceNet
        tracker = None
        if CODECARBON_AVAILABLE and OfflineEmissionsTracker is not None:
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
            processed_dir = PROCESSED_DATA
            if dataset and dataset != "students":
                processed_dir = os.path.join(os.path.dirname(PROCESSED_DATA), f"datasets_processed_{dataset}")

            if not os.path.exists(processed_dir):
                return {"status": "error", "message": f"Dataset terproses {dataset} tidak ditemukan di {processed_dir}. Harap pra-proses terlebih dahulu."}

            train_transform = transforms.Compose([
                transforms.Resize((112, 96), interpolation=InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=20),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ColorJitter(brightness=(0.2, 1.5), contrast=(0.2, 1.5), saturation=0.4, hue=0.1),
                transforms.RandomGrayscale(p=0.3),
                transforms.RandomAutocontrast(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.4),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196]),
                transforms.RandomErasing(p=0.1)
            ])
            
            val_transform = transforms.Compose([
                transforms.Resize((112, 96), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196])
            ])
            
            # Buat dua instance ImageFolder agar transformasinya terdekopel
            train_dataset_full = datasets.ImageFolder(processed_dir, transform=train_transform)
            val_dataset_full = datasets.ImageFolder(processed_dir, transform=val_transform)
            num_classes = len(train_dataset_full.classes)
            
            # Lakukan pemisahan indeks secara deterministik
            total_samples = len(train_dataset_full)
            train_size = int(0.8 * total_samples)
            val_size = total_samples - train_size
            
            # Gunakan seed deterministik agar train/val split persis sama
            indices = list(range(total_samples))
            random.seed(42)
            random.shuffle(indices)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
            val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
            
            batch_size = cl_manager.default_batch_size
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            self.logger.info(f"Dataset: {total_samples} gambar dalam {num_classes} kelas. Split terdekopel: {len(train_dataset)} latih, {len(val_dataset)} validasi (clean).")
            self.logger.info(f"Ukuran Batch: {batch_size}")
            
            model = MobileFaceNet().to(DEVICE)
            pretrained_ref = None
            if os.path.exists(PRETRAINED_PATH):
                pretrained_ref = torch.load(PRETRAINED_PATH, map_location=DEVICE)
                model.load_state_dict(pretrained_ref)
                pretrained_ref = {k: v.clone().detach().to(DEVICE) for k, v in pretrained_ref.items()}
            
            ema_model = MobileFaceNet().to(DEVICE)
            ema_model.load_state_dict(model.state_dict())
            ema_model.eval()
            for param in ema_model.parameters():
                param.requires_grad = False

            # Pembekuan Parsial Layer Awal Backbone
            set_model_freeze(model, freeze_mode="early")
            
            metric_fc = ArcMarginProduct(128, num_classes, k=3).to(DEVICE)
            for param in metric_fc.parameters():
                param.requires_grad = True

            # Anchored Head Embedding: Inisialisasi bobot classifier head dengan embedding referensi terdaftar jika ada
            if os.path.exists(REF_PATH):
                try:
                    refs = torch.load(REF_PATH, map_location=DEVICE)
                    self.logger.info(f"Menginisialisasi classifier head dari file referensi: {REF_PATH} ({len(refs)} identitas)...")
                    k = metric_fc.k
                    anchored_count = 0
                    with torch.no_grad():
                        for idx, class_name in enumerate(train_dataset_full.classes):
                            # Ambil NRP dari nama folder kelas
                            nrp = class_name.split("_")[0] if "_" in class_name else class_name
                            if nrp in refs:
                                ref_emb = refs[nrp].to(DEVICE)
                                # L2 normalisasi untuk memastikan kebersihan unit vector
                                ref_emb = torch.nn.functional.normalize(ref_emb.view(-1), p=2, dim=0)
                                # Copy ke semua sub-centers untuk kelas ini
                                for sub_idx in range(k):
                                    metric_fc.weight.data[idx * k + sub_idx] = ref_emb
                                anchored_count += 1
                    self.logger.success(f"Berhasil mengaitkan (anchor) {anchored_count}/{num_classes} kelas pada classifier head.")
                except Exception as ref_err:
                    self.logger.warn(f"Gagal memuat Anchored Head Embedding dari reference: {ref_err}")
            else:
                self.logger.info("File reference_embeddings.pth tidak ditemukan. Head diinisialisasi secara acak.")

            # Inisialisasi Kriteria Loss Fungsi Utama CrossEntropy
            criterion = nn.CrossEntropyLoss()
            
            # Pengelompokan parameter untuk weight decay spesifik SGD
            backbone_decay = []
            backbone_no_decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if 'prelu' in name.lower() or 'bias' in name.lower():
                    backbone_no_decay.append(param)
                else:
                    backbone_decay.append(param)

            head_decay = []
            head_no_decay = []
            for name, param in metric_fc.named_parameters():
                if not param.requires_grad:
                    continue
                if 'bias' in name.lower():
                    head_no_decay.append(param)
                else:
                    head_decay.append(param)

            # Inisialisasi optimizer SGD dengan momentum Nesterov
            # Menggunakan parameter laju latih yang sama persis antara backbone dan head (agar setara dengan FL)
            optimizer = optim.SGD([
                {'params': backbone_decay, 'weight_decay': 4e-5},
                {'params': backbone_no_decay, 'weight_decay': 0.0},
                {'params': head_decay, 'weight_decay': 4e-4},
                {'params': head_no_decay, 'weight_decay': 0.0}
            ], lr=self._get_lr(0), momentum=0.9, nesterov=True)
            
            # Warm up dataloader/CPU dengan melakukan pre-fetch batch pertama sebelum timer mulai berjalan
            try:
                self.logger.info("Melakukan pre-fetching batch pertama untuk warm-up CPU...")
                _ = next(iter(train_loader))
            except Exception as warm_err:
                self.logger.debug(f"Pre-fetching batch pertama dilewati: {warm_err}")
            
            epoch_history = []
            final_acc = 0
            start_time = time.time()
            
            # Pemulihan dari Checkpoint Pelatihan Sebelumnya
            start_epoch = 0
            checkpoint_path = os.path.join(MODEL_DIR, "cl_training_checkpoint.pth")
            if os.path.exists(checkpoint_path):
                try:
                    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
                    if ckpt.get('dataset') != dataset or ckpt.get('total_epochs') != epochs:
                        self.logger.info("Konfigurasi pelatihan CL berubah. Menghapus checkpoint lama...")
                        try:
                            os.remove(checkpoint_path)
                        except: pass
                    else:
                        model.load_state_dict(ckpt['model_state_dict'])
                        metric_fc.load_state_dict(ckpt['metric_fc_state_dict'])
                        if 'ema_model_state_dict' in ckpt:
                            ema_model.load_state_dict(ckpt['ema_model_state_dict'])
                        start_epoch = ckpt.get('epoch', -1) + 1
                        epoch_history = ckpt.get('history', [])
                        if start_epoch < epochs:
                            self.logger.info(f"Melanjutkan pelatihan CL dari Epoch {start_epoch+1}")
                        else:
                            self.logger.info("Checkpoint menunjukkan pelatihan terpusat telah selesai.")
                except Exception as e:
                    self.logger.warn(f"Gagal memuat checkpoint CL: {e}")

            for epoch in range(start_epoch, epochs):
                epoch_start = time.time()
                
                # Perbarui nilai Learning Rate per epoch secara seragam (sama persis dengan FL)
                current_lr = self._get_lr(epoch, total_epochs=epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                model.train()
                correct, total, total_loss = 0, 0, 0.0
                for b, (img, label) in enumerate(train_loader):
                    img, label = img.to(DEVICE), label.to(DEVICE)
                    optimizer.zero_grad()
                    
                    features = model(img)
                    output = metric_fc(features, label)
                    ce_loss = criterion(output, label)
                    
                    # Hitung L2-SP (L2-Regularization terhadap bobot awal/pre-trained)
                    # Ini adalah metode regulasi terpusat standar (ICML 2018) untuk mencegah overfit transfer learning.
                    l2_sp_loss = torch.tensor(0.0, device=DEVICE)
                    if pretrained_ref is not None:
                        mu = 0.05 # Nilai parameter penalti disamakan persis dengan FL
                        for name, param in model.named_parameters():
                            if name in pretrained_ref:
                                l2_sp_loss += (mu / 2) * torch.sum((param - pretrained_ref[name])**2)
                                
                    loss = ce_loss + l2_sp_loss
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += ce_loss.item()
                    
                    # Pembaruan Rerata Konsensus Bobot secara Temporal (EMA)
                    # Meniru proses consensus averaging di FL untuk menstabilkan konvergensi domain
                    with torch.no_grad():
                        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                            ema_param.data.mul_(0.99).add_(param.data, alpha=0.01)
                        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
                            ema_buffer.copy_(buffer)

                    # Hitung metrik akurasi tanpa ArcFace Margin untuk visualisasi
                    with torch.no_grad():
                        logits_for_acc = metric_fc.get_logits(features)
                        _, pred = torch.max(logits_for_acc.data, 1)
                        total += label.size(0)
                        correct += (pred == label).sum().item()
                    

                
                # Jalankan Evaluasi Validasi Internal menggunakan model EMA (model averaged yang stabil)
                ema_model.eval()
                val_correct, val_total, val_loss = 0, 0, 0.0
                with torch.no_grad():
                    for img, label in val_loader:
                        img, label = img.to(DEVICE), label.to(DEVICE)
                        
                        features_orig = ema_model(img)
                        features_flip = ema_model(torch.flip(img, [3]))
                        
                        features = torch.nn.functional.normalize(features_orig + features_flip, p=2, dim=1)
                        logits = metric_fc.get_logits(features)
                        
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
                
                now_wib = datetime.now(timezone(timedelta(hours=7)))
                epoch_duration = round(time.time() - epoch_start, 2)
                current_epoch_data = {
                    "epoch": epoch + 1, 
                    "loss": avg_loss, 
                    "accuracy": epoch_acc,
                    "val_loss": val_avg_loss,
                    "val_accuracy": val_acc,
                    "duration_s": epoch_duration,
                    "timestamp": now_wib.strftime("%H:%M:%S"),
                    "timestamp_raw": now_wib
                }
                epoch_history.append(current_epoch_data)
                
                # Sinkronkan perkembangan real-time ke dasbor server
                cl_manager.update_metrics({
                    "accuracy": val_acc,
                    "loss": val_avg_loss,
                    "epoch_history": [current_epoch_data]
                })

                # Simpan ronde pelatihan langsung ke database secara real-time
                if dbs is not None:
                    try:
                        cl_manager.save_training_round(
                            dbs,
                            epoch + 1,
                            avg_loss,
                            epoch_acc,
                            val_loss=val_avg_loss,
                            val_accuracy=val_acc,
                            duration=epoch_duration,
                            energy=0.0,
                            upload=0.0,
                            download=0.0,
                            start_time=now_wib,
                            model_version_id=model_version_id
                        )
                    except Exception as db_err:
                        self.logger.error(f"Gagal menyimpan TrainingRound real-time: {db_err}")

                msg = f"Epoch {epoch+1}/{epochs} | Akurasi: {epoch_acc/100:.4f} | Loss: {avg_loss:.4f} | Val Akurasi (EMA): {val_acc/100:.4f} | Val Loss (EMA): {val_avg_loss:.4f}"
                self.logger.success(msg)

                final_acc = val_acc 
                
                # Tulis file checkpoint lokal per epoch secara atomik
                try:
                    tmp_ckpt_path = checkpoint_path + ".tmp"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'ema_model_state_dict': ema_model.state_dict(),
                        'metric_fc_state_dict': metric_fc.state_dict(),
                        'history': epoch_history,
                        'dataset': dataset,
                        'total_epochs': epochs
                    }, tmp_ckpt_path)
                    os.replace(tmp_ckpt_path, checkpoint_path)
                except: pass
            
            # Simpan model EMA yang stabil sebagai hasil pelatihan akhir secara atomik
            tmp_model_path = MODEL_PATH + ".tmp"
            torch.save(ema_model.state_dict(), tmp_model_path)
            os.replace(tmp_model_path, MODEL_PATH)
            
            if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
            self.logger.success("Pelatihan selesai. Model EMA berhasil disimpan.")
            
            # Catat emisi energi dari pelacak CodeCarbon
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
            self.logger.error(f"Pelatihan model terpusat gagal: {e}")
            return {"status": "error", "message": str(e)}

    def generate_reference_and_eval(self, dbs=None, dataset="students"):
        # Tahap Pembuatan Basis Data Referensi Wajah dan Evaluasi Transmisi
        try:
            processed_dir = PROCESSED_DATA
            if dataset and dataset != "students":
                processed_dir = os.path.join(os.path.dirname(PROCESSED_DATA), f"datasets_processed_{dataset}")

            self.logger.info(f"Membuat basis data referensi identitas mahasiswa untuk dataset {dataset}...")
            
            # Instansiasi model murni evaluasi
            model = MobileFaceNet().to(DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            
            # Lakukan kalibrasi model (AdaBN) secara stabil menggunakan data latih processed 
            try:
                # BN Calibration dilewati agar model tetap menggunakan statistik BN yang stabil dari pelatihan (EMA).
                # Ini mencegah bias statistik BN dari batch terakhir calib_loader, yang menurunkan similarity score di client.
                self.logger.info("Menjaga statistik BN asli hasil training (EMA) demi stabilitas jarak jauh & low-light...")
            except Exception as calib_err:
                self.logger.warn(f"Gagal melewati kalibrasi BN: {calib_err}")
                
            model.eval()
            
            ref_db = {}
            if os.path.exists(processed_dir):
                with torch.no_grad():
                    for nrp in os.listdir(processed_dir):
                        p = os.path.join(processed_dir, nrp)
                        if not os.path.isdir(p): continue
                        all_files = sorted(os.listdir(p))
                        train_files = all_files[:50]
                        
                        tensors = []
                        for f in train_files:
                            try:
                                img_p = Image.open(os.path.join(p, f)).convert('RGB')
                                if img_p.size != (96, 112):
                                    img_p = img_p.resize((96, 112), Image.BILINEAR)
                                tensors.append(face_handler.transform(img_p))
                            except: continue
                            
                        if tensors:
                            # Batch forward pass for optimal performance
                            batch_tensor = torch.stack(tensors).to(DEVICE)
                            batch_flipped = torch.flip(batch_tensor, [3])
                            with torch.no_grad():
                                emb_orig = model(batch_tensor)
                                emb_flip = model(batch_flipped)
                                combined = F.normalize(emb_orig + emb_flip, p=2, dim=1)
                                centroid = torch.mean(combined, dim=0)
                                centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1)
                                ref_db[nrp] = centroid.cpu()

                    # Jalankan uji mandiri konsistensi model server
                    self.logger.info("Menjalankan uji mandiri integritas embedding...")
                    test_results = []
                    for nrp, centroid in ref_db.items():
                        p = os.path.join(processed_dir, nrp)
                        if not os.path.isdir(p): continue
                        
                        files = sorted(os.listdir(p))
                        if not files: continue
                        
                        first_img = files[0]
                        img_p = Image.open(os.path.join(p, first_img)).convert('RGB')
                        test_emb = face_handler.get_embedding(model, img_p).cpu()
                        
                        sim = torch.sum(test_emb.view(-1) * centroid.view(-1)).item()
                        test_results.append(sim)
                        self.logger.info(f"  > [UJI] nrp: {nrp} | Similarity: {sim:.4f}")
                    
                    avg_self_sim = sum(test_results) / len(test_results) if test_results else 0
                    self.logger.success(f"Uji mandiri selesai. Rata-rata Similarity Wajah: {avg_self_sim:.4f}")
            
            self.logger.success(f"Registri identitas ({len(ref_db)} identitas) berhasil disimpan.")
            ref_path = f"{MODEL_DIR}/reference_embeddings.pth"
            tmp_ref_path = ref_path + ".tmp"
            torch.save(ref_db, tmp_ref_path)
            os.replace(tmp_ref_path, ref_path)

            # Catat dan simpan versi model baru ke basis data server jika belum dibuat saat training
            if dbs:
                try:
                    version_id = getattr(cl_manager, "current_db_version_id", None)
                    if version_id is None:
                        now_wib = datetime.now(timezone(timedelta(hours=7)))
                        new_v = models.ModelVersion(notes=f"Dataset: {dataset} | Dibuat pada {now_wib.strftime('%Y-%m-%d %H:%M:%S')}")
                        dbs.add(new_v)
                        dbs.commit()
                        dbs.refresh(new_v)
                        cl_manager.current_db_version_id = new_v.version_id
                    else:
                        pass
                except Exception as e:
                    self.logger.error(f"Gagal menyimpan/memperbarui versi model di database: {e}")
                    dbs.rollback()
            
            # Hitung total volume transmisi download klien
            download_size = 0
            if os.path.exists(MODEL_PATH): download_size += os.path.getsize(MODEL_PATH)
            REF_PATH = f"{MODEL_DIR}/reference_embeddings.pth"
            if os.path.exists(REF_PATH): download_size += os.path.getsize(REF_PATH)

            return {
                "status": "success",
                "download_volume_mb": round(download_size / (1024 * 1024), 2)
            }
        except Exception as e:
            self.logger.error(f"Gagal membuat basis data referensi wajah: {e}")
            return {"status": "error", "message": str(e)}

training_controller = TrainingController()
