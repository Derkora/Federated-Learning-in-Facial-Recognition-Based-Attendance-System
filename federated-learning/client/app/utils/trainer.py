import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import glob
import random
import copy
import gc
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from collections import OrderedDict
from torchvision.transforms import InterpolationMode

from torch.utils.data import DataLoader, Dataset

from app.utils.mobilefacenet import MobileFaceNet, ArcMarginProduct
from app.utils.freezing import set_model_freeze
from app.utils.preprocessing import image_processor
from app.utils.logging import get_logger

class TrainingNaNError(Exception):
    """Pengecualian khusus saat loss atau bobot pelatihan menjadi NaN."""
    pass

def hybrid_collate(batch):
    imgs = []
    embs = []
    labels = []
    is_embedding = []
    
    for data, label, is_emb in batch:
        labels.append(label)
        is_embedding.append(is_emb)
        if is_emb:
            embs.append(data.clone())
            imgs.append(torch.zeros((3, 112, 96)))
        else:
            imgs.append(data)
            embs.append(torch.zeros(128))

    return torch.stack(imgs), torch.stack(embs), torch.tensor(labels), torch.tensor(is_embedding)

class FaceDataset(Dataset):
    def __init__(self, data_root, global_embeddings=None, transform=None, mode="train", label_map=None, seed=42):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        self.class_counts = {}
        self.logger = get_logger()
        
        self.logger.info(f"Inisialisasi Dataset | Root: {data_root} | Mode: {mode}")
        
        local_nrps = []
        if os.path.exists(data_root):
            local_nrps = sorted([f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))])
            self.logger.info(f"Menemukan {len(local_nrps)} direktori di {data_root}")
            
        global_nrps = []
        if global_embeddings:
            global_nrps = [item['nrp'] for item in global_embeddings]
            
        if label_map:
            all_unique_nrps = label_map
            self.logger.info(f"Menggunakan Peta Label Global dengan {len(all_unique_nrps)} identitas.")
        else:
            all_unique_nrps = sorted(list(set(local_nrps + global_nrps)))
            self.logger.info(f"Membangun Peta Label Lokal dengan {len(all_unique_nrps)} identitas.")

        self.nrp_to_idx = {nrp: idx for idx, nrp in enumerate(all_unique_nrps)}
        self.id_map = {idx: nrp for nrp, idx in self.nrp_to_idx.items()}
        self.num_classes = len(all_unique_nrps)    
        self.class_counts = {idx: 0 for idx in range(self.num_classes)}

        self.logger.info(f"Memproses folder lokal untuk {len(local_nrps)} pengguna...")
        
        
        for nrp in local_nrps:
            if label_map and nrp not in self.nrp_to_idx:
                self.logger.warn(f"  [LEWATI] {nrp}: Tidak ada di peta label global.")
                continue
                
            folder_path = os.path.join(data_root, nrp)
            idx = self.nrp_to_idx[nrp]
            
            # Ambil semua file gambar di folder processed secara langsung
            # Tidak perlu memanggil select_best_faces karena data processed sudah disortir & dicrop di tahap preprocess
            all_images = sorted([
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]) if os.path.exists(folder_path) else []
            
            if mode == "train":
                split_idx = int(0.8 * len(all_images))
                selected = all_images[:split_idx]
            else:
                split_idx = int(0.8 * len(all_images))
                selected = all_images[split_idx:]
            
            if len(selected) == 0:
                self.logger.warn(f"  [PERINGATAN] {nrp}: Tidak ditemukan gambar wajah yang valid.")
                continue
            
            for p in selected:
                self.samples.append({"type": "image", "path": p, "label": idx})
                self.class_counts[idx] += 1
                        
            gc.collect()
            time.sleep(0.1)

        if global_embeddings:
            for item in global_embeddings:
                idx = self.nrp_to_idx[item['nrp']]
                emb_tensor = item['embedding']
                if emb_tensor.dim() == 1:
                    emb_tensor = emb_tensor.unsqueeze(0)
                emb_normalized = torch.nn.functional.normalize(emb_tensor, p=2, dim=1).squeeze(0)
                
                self.samples.append({"type": "embedding", "data": emb_normalized, "label": idx})
                self.class_counts[idx] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Ambil sampel dan label kelas mahasiswa berdasarkan indeks
        sample = self.samples[idx]
        label = sample['label']
        
        # Proses citra mentah lokal jika tipe datanya berupa gambar
        if sample['type'] == "image":
            image = Image.open(sample['path']).convert('RGB')
            # Jalankan augmentasi gambar acak secara on-the-fly
            if self.transform:
                image = self.transform(image)
            return image, label, False
        else:
            # Kembalikan data embedding memori global lintas-klien
            return sample['data'], label, True

class LocalTrainer:
    def __init__(self, backbone, head, device="cpu", data_path="/app/data"):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.backbone = backbone.to(self.device) if backbone is not None else None
        self.head = head.to(self.device) if head is not None else None
        self.data_path = data_path
        self.nrp_to_idx = {} 
        self.logger = get_logger()
        
        self.transform = transforms.Compose([
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

        self.val_transform = transforms.Compose([
            transforms.Resize((112, 96), interpolation=InterpolationMode.BILINEAR), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196])
        ])

    def save_checkpoint(self, round_num, epoch, history):
        try:
            checkpoint_dir = os.path.join(os.path.dirname(self.data_path), "models")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "training_checkpoint.pth")
            torch.save({
                'round_num': round_num,
                'epoch': epoch,
                'backbone_state_dict': self.backbone.state_dict(),
                'head_state_dict': self.head.state_dict(),
                'history': history
            }, checkpoint_path)
            self.logger.info(f"Progress disimpan (Ronde {round_num}, Epoch {epoch+1})")
        except Exception as e:
            self.logger.error(f"Gagal menyimpan checkpoint: {e}")

    # Loop Pelatihan PyTorch (Local Training Loop)
    def train(self, epochs=5, lr=0.0001, round_num=0, global_embeddings=None, label_map=None, mu=0.05, lam=0.1, status_callback=None):
        if self.backbone is None or self.head is None:
             self.logger.warn("Model belum terinisialisasi..")
             return 0.0, 0.0, 0, [], 0.0

        epoch_history = []
        
        # Pemuatan checkpoint lokal jika terputus di tengah jalan
        start_epoch = 0
        checkpoint_path = os.path.join(os.path.dirname(self.data_path), "models", "training_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                if ckpt.get('round_num') == round_num:
                    self.backbone.load_state_dict(ckpt['backbone_state_dict'])
                    self.head.load_state_dict(ckpt['head_state_dict'])
                    start_epoch = ckpt.get('epoch', -1) + 1
                    epoch_history = ckpt.get('history', [])
                    if start_epoch < epochs:
                        self.logger.info(f"Melanjutkan dari checkpoint (Ronde {round_num}, Mulai Epoch {start_epoch+1})")
                    else:
                        self.logger.info(f"Checkpoint menunjukkan ronde {round_num} sudah selesai.")
                        return 0.0, 0.0, 0, epoch_history, 0.0
            except Exception as e:
                self.logger.error(f"Gagal memuat checkpoint: {e}")

        # Konstruksi dataset gabungan citra lokal & memori global
        dataset = FaceDataset(self.data_path, global_embeddings=global_embeddings, transform=self.transform, mode="train", label_map=label_map)
        if len(dataset) < 2:
            self.logger.warn(f"Data terlalu kecil ({len(dataset)}) untuk pelatihan. Melewati ronde.")
            return 0.0, 0.0, len(dataset), [], 0.0
            
        # Penyesuaian dimensi output layer classification head
        # PENTING: head.weight.shape[0] = num_classes * k (sub-centers), bukan num_classes!
        # Bandingkan dengan num_classes asli untuk menghindari false mismatch.
        head_k = getattr(self.head, 'k', 1)
        head_num_classes = self.head.weight.shape[0] // head_k
        if dataset.num_classes > 0 and dataset.num_classes != head_num_classes:
            self.update_head(dataset.num_classes, dataset.nrp_to_idx)
        else:
            self.nrp_to_idx = dataset.nrp_to_idx
        # Selalu anchor bobot head setiap awal training agar sinkron dengan backbone global baru
        # (Sama persis dengan perilaku Centralized Learning yang selalu seed metric_fc)
        self.anchor_head_weights()

        # Inisialisasi data loader pembagi batch
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=hybrid_collate, drop_last=True)
        
        # Terapkan pembekuan layer awal backbone demi hemat daya CPU
        set_model_freeze(self.backbone, freeze_mode="early")
    
        for param in self.head.parameters():
            param.requires_grad = True

        # Duplikat snapshot model global/awal untuk proteksi NaN
        backbone_snapshot = copy.deepcopy(self.backbone.state_dict())
        head_snapshot = copy.deepcopy(self.head.state_dict())
        global_ref = copy.deepcopy(self.backbone.state_dict())

        # Pisahkan parameter untuk pengaturan weight decay SGD
        backbone_decay = []
        backbone_no_decay = []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad: continue
            if 'prelu' in name.lower() or 'bias' in name.lower():
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

        head_decay = []
        head_no_decay = []
        for name, param in self.head.named_parameters():
            if not param.requires_grad: continue
            if 'bias' in name.lower():
                head_no_decay.append(param)
            else:
                head_decay.append(param)

        # Konfigurasi optimizer SGD dengan momentum Nesterov
        optimizer = torch.optim.SGD([
            {'params': backbone_decay, 'weight_decay': 4e-5},
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': head_decay, 'weight_decay': 4e-4},
            {'params': head_no_decay, 'weight_decay': 0.0}
        ], lr=lr, momentum=0.9, nesterov=True)
        
        # Definisikan kriteria perhitungan loss fungsi
        criterion = nn.CrossEntropyLoss()

        # Warm up dataloader/CPU dengan melakukan pre-fetch batch pertama sebelum timer mulai berjalan
        try:
            self.logger.info("Melakukan pre-fetching batch pertama untuk warm-up CPU...")
            _ = next(iter(dataloader))
        except Exception as warm_err:
            self.logger.debug(f"Pre-fetching batch pertama dilewati: {warm_err}")

        # Mulai pencatatan waktu latihan murni (exclude dataset init, head anchoring, dan warm-up)
        start_train_time = time.time()
        self.logger.info(f"Ronde {round_num}: Melatih {len(dataset)} sampel data untuk {epochs} epoch")
        total_loss, correct, total = 0.0, 0, 0
        epoch_history = []
        
        # Loop iterasi epochs pelatihan
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            self.backbone.train()
            self.head.train()
            
            # Loop pembagian batch data latih
            for imgs, embs, labels, is_embedding in dataloader:
                imgs, embs, labels = imgs.to(self.device), embs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                features_local = torch.zeros((labels.size(0), self.head.weight.shape[1]), device=self.device)
                img_mask, emb_mask = ~is_embedding, is_embedding
                
                # Forward pass citra wajah lokal ke backbone
                if img_mask.any():
                    img_input = imgs[img_mask]
                    if img_input.size(0) == 1:
                        features_fixed = self.backbone(torch.cat([img_input, img_input], dim=0))
                        features_local[img_mask] = features_fixed[0:1]
                    else:
                        features_local[img_mask] = self.backbone(img_input)
                        
                # Forward pass data memori global lintas-klien
                if emb_mask.any():
                    features_local[emb_mask] = embs[emb_mask]
                
                # Hitung loss klasifikasi ArcFace pada data lokal
                outputs = self.head(features_local, labels)
                ce_loss = criterion(outputs, labels)
                
                # Hitung penalti regulasi jarak L2 terhadap model global (pFedFace)
                prox_loss = torch.tensor(0.0, device=self.device)
                for name, param in self.backbone.named_parameters():
                    if name in global_ref:
                        prox_loss += (mu / 2) * torch.norm(param - global_ref[name])**2
                
                # Gabungkan nilai total loss klasifikasi dan penalti regulasi
                loss = ce_loss + prox_loss
                
                # Validasi pengaman agar program tidak crash apabila loss NaN
                if torch.isnan(loss):
                    self.logger.error("Nilai loss NaN terdeteksi saat pelatihan. Mengembalikan bobot sebelumnya.")
                    self.backbone.load_state_dict(backbone_snapshot)
                    self.head.load_state_dict(head_snapshot)
                    raise TrainingNaNError(f"NaN loss di Round {round_num}")

                # Perhitungan backward pass dan update parameter model
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                
                # Hitung performa akurasi prediksi batch latih
                with torch.no_grad():
                    logits_for_acc = self.head.get_logits(features_local)
                    _, predicted = torch.max(logits_for_acc.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = correct / total if total > 0 else 0.0
            avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
            self.logger.info(f"  > Epoch {epoch+1}/{epochs} | Loss: {avg_epoch_loss:.4f} | Acc: {acc:.4f}")
            if status_callback:
                status_callback(epoch + 1, epochs, avg_epoch_loss, acc)
            epoch_history.append({"epoch": epoch + 1, "loss": avg_epoch_loss, "accuracy": acc})
            
            self.save_checkpoint(round_num, epoch, epoch_history)
                
        avg_loss = total_loss / (len(dataloader) * epochs) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        train_duration = time.time() - start_train_time
        return avg_loss, accuracy, total, epoch_history, train_duration

    def evaluate(self, global_embeddings=None, label_map=None):
        if self.backbone is None or self.head is None:
             return 0.0, 0.0, 0
             
        dataset = FaceDataset(self.data_path, global_embeddings=global_embeddings, transform=self.val_transform, mode="val", label_map=label_map)
        if len(dataset) < 2: return 0.0, 0.0, len(dataset)
            
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=hybrid_collate)
        criterion = nn.CrossEntropyLoss()
        self.backbone.eval()
        self.head.eval()
        
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, embs, labels, is_embedding in dataloader:
                imgs, embs, labels = imgs.to(self.device), embs.to(self.device), labels.to(self.device)
                
                max_label = labels.max().item()
                if max_label >= self.head.weight.shape[0]:
                    self.logger.warn(f"Label evaluasi ({max_label}) di luar batas head ({self.head.weight.shape[0]}). Melewati batch.")
                    continue

                features = torch.zeros((labels.size(0), self.head.weight.shape[1]), device=self.device)
                img_mask, emb_mask = ~is_embedding, is_embedding
                
                if img_mask.any():
                    img_input = imgs[img_mask]
                    
                    if img_input.size(0) == 1:
                        f_orig = self.backbone(torch.cat([img_input, img_input], dim=0))[0:1]
                        f_flip = self.backbone(torch.cat([torch.flip(img_input, [3]), torch.flip(img_input, [3])], dim=0))[0:1]
                    else:
                        f_orig = self.backbone(img_input)
                        f_flip = self.backbone(torch.flip(img_input, [3]))
                    
                    features[img_mask] = torch.nn.functional.normalize(f_orig + f_flip, p=2, dim=1)
                        
                if emb_mask.any():
                    features[emb_mask] = embs[emb_mask]
                
                logits = self.head.get_logits(features)
                outputs = self.head(features, labels)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy, total

    def update_head(self, new_num_classes, new_nrp_to_idx):
        """Mengekspansi head klasifikasi sambil mempertahankan bobot yang sudah terlatih dan mengaitkan (anchor) kelas baru dari database."""
        old_head = self.head
        old_nrp_to_idx = self.nrp_to_idx
        
        embedding_size = 128 
        if old_head is not None:
            embedding_size = old_head.weight.shape[1]
            
        new_head = ArcMarginProduct(embedding_size, new_num_classes, k=3).to(self.device)
        
        # Penanganan penyalinan bobot terlatih dari head lama (jika ada)
        k = new_head.k
        copied_count = 0
        if old_head is not None:
            with torch.no_grad():
                for nrp, new_idx in new_nrp_to_idx.items():
                    if nrp in old_nrp_to_idx:
                        old_idx = old_nrp_to_idx[nrp]
                        new_head.weight[new_idx*k : (new_idx+1)*k] = old_head.weight[old_idx*k : (old_idx+1)*k]
                        copied_count += 1
            self.logger.info(f"Classification Head Diperluas: {len(old_nrp_to_idx)} -> {new_num_classes} kelas ({copied_count} identitas dipertahankan)")
        else:
            self.logger.info(f"Classification Head Baru Dibuat: {new_num_classes} kelas (inisialisasi awal)")

        # Anchored Head Embedding: Inisialisasi kelas baru (atau seluruh kelas pada inisialisasi awal)
        # dengan menggunakan embedding pendaftaran (reference) dari database SQLite lokal
        anchored_count = 0
        try:
            from app.db.db import SessionLocal
            from app.db.models import EmbeddingLocal
            from app.utils.security import encryptor
        except ImportError:
            SessionLocal = None
            EmbeddingLocal = None
            encryptor = None

        if SessionLocal is not None and EmbeddingLocal is not None:
            db = SessionLocal()
            try:
                with torch.no_grad():
                    for nrp, new_idx in new_nrp_to_idx.items():
                        # Anchor SEMUA kelas baru (yang belum ada di head lama)
                        # Kelas yang disalin dari head lama tidak perlu di-anchor lagi di sini
                        # karena anchor_head_weights() akan menanganinya setelah update_head selesai
                        if old_head is not None and nrp in old_nrp_to_idx:
                            continue
                        
                        emb_rec = db.query(EmbeddingLocal).filter_by(user_id=nrp).first()
                        if emb_rec is not None:
                            try:
                                if emb_rec.is_global:
                                    # Data embedding global (synced dari server) disimpan tanpa enkripsi
                                    emb_np = np.frombuffer(emb_rec.embedding_data, dtype=np.float32).copy()
                                else:
                                    # Data embedding lokal dienkripsi
                                    if encryptor is not None and emb_rec.iv:
                                        emb_np = encryptor.decrypt_embedding(emb_rec.embedding_data, emb_rec.iv).copy()
                                    else:
                                        emb_np = np.frombuffer(emb_rec.embedding_data, dtype=np.float32).copy()
                                
                                ref_emb = torch.from_numpy(emb_np).to(self.device)
                                # L2 normalisasi untuk memastikan unit vector
                                ref_emb = torch.nn.functional.normalize(ref_emb.view(-1), p=2, dim=0)
                                # Salin ke seluruh sub-centers (k=3) untuk kelas/identitas ini
                                for sub_idx in range(k):
                                    new_head.weight.data[new_idx * k + sub_idx] = ref_emb
                                anchored_count += 1
                            except Exception as e:
                                self.logger.error(f"Gagal memuat/mendekripsi embedding untuk {nrp}: {e}")
            except Exception as db_err:
                self.logger.error(f"Gagal melakukan Anchored Head Embedding: {db_err}")
            finally:
                db.close()
                
        if anchored_count > 0:
            self.logger.success(f"Berhasil mengaitkan (anchor) {anchored_count} kelas baru pada classifier head dari database.")

        self.head = new_head
        self.nrp_to_idx = new_nrp_to_idx
        return new_head

    def anchor_head_weights(self):
        if self.head is None or len(self.nrp_to_idx) == 0:
            return
            
        k = getattr(self.head, 'k', 3)
        total_classes = self.head.weight.shape[0] // k
        anchored_from_db = 0
        anchored_from_images = 0
        
        # Kumpulkan semua embedding yang berhasil di-anchor untuk fallback mean
        anchored_embeddings = []
        
        try:
            from app.db.db import SessionLocal
            from app.db.models import EmbeddingLocal
            from app.utils.security import encryptor
        except ImportError:
            SessionLocal = None
            EmbeddingLocal = None
            encryptor = None

        # Anchor dari DB (local + global embeddings)
        db_embs = {}  # nrp -> tensor
        if SessionLocal is not None and EmbeddingLocal is not None:
            db = SessionLocal()
            try:
                for nrp, idx in self.nrp_to_idx.items():
                    emb_rec = db.query(EmbeddingLocal).filter_by(user_id=nrp).first()
                    if emb_rec is not None:
                        try:
                            if emb_rec.is_global:
                                emb_np = np.frombuffer(emb_rec.embedding_data, dtype=np.float32).copy()
                            else:
                                if encryptor is not None and emb_rec.iv:
                                    emb_np = encryptor.decrypt_embedding(emb_rec.embedding_data, emb_rec.iv).copy()
                                else:
                                    emb_np = np.frombuffer(emb_rec.embedding_data, dtype=np.float32).copy()
                            
                            ref_emb = torch.from_numpy(emb_np).to(self.device)
                            ref_emb = torch.nn.functional.normalize(ref_emb.view(-1), p=2, dim=0)
                            db_embs[nrp] = ref_emb
                        except Exception as e:
                            self.logger.error(f"Gagal membaca embedding DB untuk {nrp}: {e}")
            except Exception as db_err:
                self.logger.error(f"Gagal query EmbeddingLocal: {db_err}")
            finally:
                db.close()

        # Hitung on-the-fly dari gambar lokal untuk NRP yang tidak ada di DB
        # (Identik dengan cara CL membuat reference_embeddings.pth dari processed/ sebelum training)
        missing_nrps = [nrp for nrp in self.nrp_to_idx if nrp not in db_embs]
        if missing_nrps and self.backbone is not None:
            processed_root = self.data_path  # data_path sudah menunjuk ke folder processed/
            self.backbone.eval()
            with torch.no_grad():
                for nrp in missing_nrps:
                    nrp_folder = os.path.join(processed_root, nrp)
                    if not os.path.exists(nrp_folder):
                        continue
                    img_files = sorted([
                        f for f in os.listdir(nrp_folder)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ])[:50]
                    if not img_files:
                        continue
                    
                    tensors = []
                    for fname in img_files:
                        try:
                            img_pil = Image.open(os.path.join(nrp_folder, fname)).convert('RGB')
                            if self.val_transform:
                                t = self.val_transform(img_pil).unsqueeze(0)
                            else:
                                import torchvision.transforms as T
                                t = T.Compose([
                                    T.Resize((112, 96)),
                                    T.ToTensor(),
                                    T.Normalize([0.5, 0.5, 0.5], [0.50196, 0.50196, 0.50196])
                                ])(img_pil).unsqueeze(0)
                            tensors.append(t)
                        except:
                            continue
                    
                    if not tensors:
                        continue
                    
                    try:
                        batch = torch.cat(tensors, dim=0).to(self.device)
                        # Flip trick seperti CL
                        emb_orig = self.backbone(batch)
                        emb_flip = self.backbone(torch.flip(batch, [3]))
                        combined = F.normalize(emb_orig + emb_flip, p=2, dim=1)
                        centroid = torch.mean(combined, dim=0)
                        ref_emb = F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)
                        db_embs[nrp] = ref_emb
                        anchored_from_images += 1
                    except Exception as e:
                        self.logger.error(f"Gagal hitung embedding on-the-fly untuk {nrp}: {e}")
                        continue

        # Terapkan ke head weight + hitung mean fallback
        with torch.no_grad():
            for nrp, idx in self.nrp_to_idx.items():
                if nrp in db_embs:
                    ref_emb = db_embs[nrp]
                    for sub_idx in range(k):
                        self.head.weight.data[idx * k + sub_idx] = ref_emb
                    anchored_embeddings.append(ref_emb)
                    if nrp not in [n for n in self.nrp_to_idx if n in db_embs and db_embs.get(n) is not None]:
                        anchored_from_db += 1
            
            # Hitung lebih akurat
            anchored_from_db = len(db_embs) - anchored_from_images
            
            # Fallback mean untuk kelas yang benar-benar tidak ada data
            # Daripada Xavier random (berbahaya), gunakan rata-rata embedding yang ada
            if anchored_embeddings:
                mean_emb = F.normalize(torch.stack(anchored_embeddings).mean(dim=0).unsqueeze(0), p=2, dim=1).squeeze(0)
                fallback_count = 0
                for nrp, idx in self.nrp_to_idx.items():
                    if nrp not in db_embs:
                        for sub_idx in range(k):
                            self.head.weight.data[idx * k + sub_idx] = mean_emb
                        fallback_count += 1
                if fallback_count > 0:
                    self.logger.info(f"Fallback mean-anchor untuk {fallback_count} kelas tanpa data lokal/global.")
        
        total_anchored = len(db_embs)
        self.logger.success(
            f"Head Anchoring selesai: {anchored_from_db} dari DB, "
            f"{anchored_from_images} dari gambar lokal, "
            f"{total_classes - total_anchored} fallback mean "
            f"(total {total_classes} kelas)."
        )


    def _is_shared_param(self, name):
        name = name.lower()
        return any(x in name for x in ['weight', 'bias', 'bn', 'running_', 'num_batches_tracked'])

    def get_backbone_parameters(self, personalized=True):
        if self.backbone is None: return []
        state_dict = self.backbone.state_dict()
        shared_keys = [k for k in state_dict.keys() if not personalized or self._is_shared_param(k)]
        
        mode_str = "pFedFace (Local BN/Head)" if personalized else "Standard (Full Sync)"
        self.logger.info(f"Mengekstraksi {len(shared_keys)} parameter ({mode_str}).")
        
        params = [state_dict[k].cpu().numpy().copy() for k in shared_keys]
        
        if not personalized and self.head is not None:
            head_params = [p.detach().cpu().numpy().copy() for p in self.head.parameters()]
            self.logger.info(f"Termasuk {len(head_params)} parameter classification head.")
            params.extend(head_params)
            
        return params

    def get_bn_parameters(self):
        if self.backbone is None: return {}
        state_dict = self.backbone.state_dict()
        bn_keys = [k for k in state_dict.keys() if 
                   any(x in k.lower() for x in ['.bn.', '.conv.1.', '.conv.4.', '.conv.7.', 'running_', 'num_batches_tracked'])]
        
        self.logger.info(f"Mengumpulkan {len(bn_keys)} parameter BN ke dalam state_dict.")
        return {k: state_dict[k].cpu().numpy().copy() for k in bn_keys}

    def set_backbone_parameters(self, parameters, personalized=True):
        if self.backbone is None:
            self.logger.info("Backbone bernilai None, menginisialisasi MobileFaceNet baru...")
            self.backbone = MobileFaceNet().to(self.device)
            self.backbone.eval()
            
        state_dict = self.backbone.state_dict()
        shared_keys = [k for k in state_dict.keys() if not personalized or self._is_shared_param(k)]
        num_backbone = len(shared_keys)

        self.logger.info(f"Menginjeksi {num_backbone} parameter backbone.")
        new_state_dict = OrderedDict(state_dict)
        for i, k in enumerate(shared_keys):
            try:
                if i < len(parameters):
                    tensor_v = torch.from_numpy(parameters[i].copy())
                    if tensor_v.shape == state_dict[k].shape:
                        new_state_dict[k] = tensor_v
            except: continue
        self.backbone.load_state_dict(new_state_dict, strict=False)

        if not personalized and len(parameters) > num_backbone and self.head is not None:
            head_params = parameters[num_backbone:]
            self.logger.info(f"Menginjeksi {len(head_params)} parameter classification head.")
            for i, p in enumerate(self.head.parameters()):
                if i < len(head_params):
                    try:
                        p.data = torch.from_numpy(head_params[i].copy()).to(self.device)
                    except: continue

    def calculate_centroids(self, label_map=None):
        if self.backbone is None: return {}
        dataset = FaceDataset(self.data_path, transform=self.val_transform, mode="train", label_map=label_map)
        if len(dataset) == 0: return {}
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=hybrid_collate)
        self.backbone.eval()
        temp_embeddings = {nrp: [] for nrp in dataset.nrp_to_idx.keys()}
        self.logger.info("Menghitung Centroid...")
        with torch.no_grad():
            for imgs, embs, labels, is_embedding in dataloader:
                img_mask = ~is_embedding
                if not img_mask.any(): continue
                imgs_batch = imgs[img_mask].to(self.device)
                if imgs_batch.size(0) == 1:
                    f_orig = self.backbone(torch.cat([imgs_batch, imgs_batch]))[0:1]
                    f_flip = self.backbone(torch.cat([torch.flip(imgs_batch, [3]), torch.flip(imgs_batch, [3])]))[0:1]
                else:
                    f_orig = self.backbone(imgs_batch)
                    f_flip = self.backbone(torch.flip(imgs_batch, [3]))
                
                features = F.normalize(f_orig + f_flip, p=2, dim=1)
                
                batch_labels = labels[img_mask]
                for i in range(len(batch_labels)):
                    nrp = dataset.id_map[batch_labels[i].item()]
                    temp_embeddings[nrp].append(features[i].unsqueeze(0))
        centroids = {}
        for nrp, embs in temp_embeddings.items():
            if embs:
                subset = embs[:50]
                stack = torch.cat(subset, dim=0)
                centroid = torch.mean(stack, dim=0)
                centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1)
                centroids[nrp] = centroid.cpu().numpy()[0]
        return centroids
