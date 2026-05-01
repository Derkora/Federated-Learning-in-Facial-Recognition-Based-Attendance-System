import torch
import torch.nn as nn
import torch.nn.functional as F
from .mobilefacenet import MobileFaceNet, ArcMarginProduct
from .preprocessing import image_processor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import random
import copy
import gc
from collections import OrderedDict
from torchvision.transforms import InterpolationMode
import numpy as np

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
        
        print(f"[DATASET] Root: {data_root} | Mode: {mode}")
        
        local_nrps = []
        if os.path.exists(data_root):
            local_nrps = sorted([f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))])
            print(f"[DATASET] Found {len(local_nrps)} directories in {data_root}")
            
        global_nrps = []
        if global_embeddings:
            global_nrps = [item['nrp'] for item in global_embeddings]
            
        if label_map:
            all_unique_nrps = label_map
            print(f"[DATASET] Using Global Label Map with {len(all_unique_nrps)} identities.")
        else:
            all_unique_nrps = sorted(list(set(local_nrps + global_nrps)))
            print(f"[DATASET] Building Local Label Map with {len(all_unique_nrps)} identities.")

        self.nrp_to_idx = {nrp: idx for idx, nrp in enumerate(all_unique_nrps)}
        self.id_map = {idx: nrp for nrp, idx in self.nrp_to_idx.items()}
        self.num_classes = len(all_unique_nrps)
        
        # Inisialisasi hitungan jumlah sampel per kelas
        self.class_counts = {idx: 0 for idx in range(self.num_classes)}

        # Penyelarasan Data: Ikuti label_map global secara ketat jika tersedia
        print(f"[DATASET] Processing local folders for {len(local_nrps)} users...")
        
        
        for nrp in local_nrps:
            if label_map and nrp not in self.nrp_to_idx:
                print(f"  [SKIP] {nrp}: Not in global label map.")
                continue
                
            folder_path = os.path.join(data_root, nrp)
            idx = self.nrp_to_idx[nrp]
            
            # FUNCTIONAL PARITY WITH PRE: Gunakan seleksi wajah tertajam (Top 50)
            if mode == "train":
                # Pilih wajah terbaik untuk training
                # Jika folder berisi banyak gambar, kita ambil Top 50 saja
                selected_filenames = image_processor.select_best_faces(folder_path, n=50)
                selected = [os.path.join(folder_path, f) for f in selected_filenames]
                
                # Split 80/20 dari selected paths
                split_idx = int(0.8 * len(selected))
                selected = selected[:split_idx]
            else:
                # Untuk validasi, kita tetap ambil semua gambar yang tersisa setelah training
                # Tapi untuk kesederhanaan, kita ambil 20% terakhir dari Top 50 atau sisa gambar
                all_paths = sorted([
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]) if os.path.exists(folder_path) else []
                
                selected_train_filenames = image_processor.select_best_faces(folder_path, n=50)
                selected_train = [os.path.join(folder_path, f) for f in selected_train_filenames]
                selected = [p for p in all_paths if p not in selected_train]
                
                # Jika tidak ada sisa, ambil 20% terakhir dari selected_train
                if not selected:
                    split_idx = int(0.8 * len(selected_train))
                    selected = selected_train[split_idx:]
            
            if len(selected) == 0:
                print(f"  [WARN] {nrp}: No viable images found.")
                continue
            
            for p in selected:
                self.samples.append({"type": "image", "path": p, "label": idx})
                self.class_counts[idx] += 1
            
            print(f"  [OK] {nrp}: {len(selected)} samples selected ({mode}).")
            print(f"  [OK] {nrp}: Loaded {len(selected)} {mode} images.")

        # Tambahkan sampel embedding global (Berbagi Pengetahuan / Knowledge Sharing)
        if global_embeddings:
            for item in global_embeddings:
                idx = self.nrp_to_idx[item['nrp']]
                # Pastikan L2 Normalized (Penting agar tidak mengotor gradien PyTorch murni)
                emb_tensor = item['embedding']
                if emb_tensor.dim() == 1:
                    emb_tensor = emb_tensor.unsqueeze(0)
                emb_normalized = torch.nn.functional.normalize(emb_tensor, p=2, dim=1).squeeze(0)
                
                self.samples.append({"type": "embedding", "data": emb_normalized, "label": idx})
                self.class_counts[idx] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        
        if sample['type'] == "image":
            image = Image.open(sample['path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, False
        else:
            return sample['data'], label, True

class LocalTrainer:
    def __init__(self, backbone, head, device="cpu", data_path="/app/data"):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.backbone = backbone.to(self.device) if backbone is not None else None
        self.head = head.to(self.device) if head is not None else None
        self.data_path = data_path
        self.nrp_to_idx = {} 
        
        self.transform = transforms.Compose([
            transforms.Resize((112, 96), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196]),
            transforms.RandomErasing(p=0.1)
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((112, 96), interpolation=InterpolationMode.BILINEAR), 
            transforms.ToTensor(),
            # Normalisasi MobileFaceNet (True Standard Creator Alignment): std=128/255 = 0.50196
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196])
        ])

    def train(self, epochs=5, lr=0.0001, round_num=0, global_embeddings=None, label_map=None, mu=0.05, lam=0.1):
        if self.backbone is None or self.head is None:
             print("[TRAINER] Model belum terinisialisasi..")
             return 0.0, 0.0, 0, []

        dataset = FaceDataset(self.data_path, global_embeddings=global_embeddings, transform=self.transform, mode="train", label_map=label_map)
        if len(dataset) < 2:
            print(f"[TRAINER] Data terlalu kecil ({len(dataset)}) untuk training. Melewati round.")
            return 0.0, 0.0, len(dataset), []
            
        if dataset.num_classes > 0 and dataset.num_classes != self.head.weight.shape[0]:
            self.update_head(dataset.num_classes, dataset.nrp_to_idx)
        else:
            self.nrp_to_idx = dataset.nrp_to_idx

        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=hybrid_collate, drop_last=True)
        
        backbone_snapshot = copy.deepcopy(self.backbone.state_dict())
        head_snapshot = copy.deepcopy(self.head.state_dict())
        global_ref = copy.deepcopy(self.backbone.state_dict())

        # --- SELEKSI PARAMETER UNTUK PER-LAYER WEIGHT DECAY ---
        # 1. Backbone Params (Decay: 4e-5)
        backbone_decay = []
        backbone_no_decay = []
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad: continue
            # PReLU dan Bias tidak boleh kena Weight Decay (Stabilitas)
            if 'prelu' in name.lower() or 'bias' in name.lower():
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

        # 2. Head Params (Decay: 4e-4 - Lebih kuat untuk klasifikasi)
        head_decay = []
        head_no_decay = []
        for name, param in self.head.named_parameters():
            if 'bias' in name.lower():
                head_no_decay.append(param)
            else:
                head_decay.append(param)

        # OPTIMIZER: SGD + Nesterov Momentum (Standar Creator MobileFaceNet)
        optimizer = torch.optim.SGD([
            {'params': backbone_decay, 'weight_decay': 4e-5},
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': head_decay, 'weight_decay': 4e-4},
            {'params': head_no_decay, 'weight_decay': 0.0}
        ], lr=lr, momentum=0.9, nesterov=True)
        
        # LOSS: Pure CrossEntropy (Tanpa Label Smoothing agar margin tegas)
        criterion = nn.CrossEntropyLoss()

        print(f"[TRAINER] Round {round_num}: Training {len(dataset)} data untuk {epochs} epochs")
        total_loss, correct, total = 0.0, 0, 0
        epoch_history = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.backbone.train()
            self.head.train()
            
            for imgs, embs, labels, is_embedding in dataloader:
                imgs, embs, labels = imgs.to(self.device), embs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                features_local = torch.zeros((labels.size(0), self.head.weight.shape[1]), device=self.device)
                img_mask, emb_mask = ~is_embedding, is_embedding
                
                if img_mask.any():
                    img_input = imgs[img_mask]
                    if img_input.size(0) == 1:
                        features_fixed = self.backbone(torch.cat([img_input, img_input], dim=0))
                        features_local[img_mask] = features_fixed[0:1]
                    else:
                        features_local[img_mask] = self.backbone(img_input)
                        
                if emb_mask.any():
                    features_local[emb_mask] = embs[emb_mask]
                
                outputs = self.head(features_local, labels)
                ce_loss = criterion(outputs, labels)
                
                prox_loss = torch.tensor(0.0, device=self.device)
                for name, param in self.backbone.named_parameters():
                    if name in global_ref:
                        prox_loss += (mu / 2) * torch.norm(param - global_ref[name])**2
                
                loss = ce_loss + prox_loss
                
                if torch.isnan(loss):
                    print(f"[ERROR] NaN loss terdeteksi...")
                    self.backbone.load_state_dict(backbone_snapshot)
                    self.head.load_state_dict(head_snapshot)
                    raise TrainingNaNError(f"NaN loss di Round {round_num}")

                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                
                # METRIK: Hitung Akurasi Sebenarnya (Tanpa Margin Pelatihan)
                with torch.no_grad():
                    logits_for_acc = F.linear(F.normalize(features_local), F.normalize(self.head.weight)) * self.head.s
                    _, predicted = torch.max(logits_for_acc.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = correct / total if total > 0 else 0.0
            avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
            print(f"  > Epoch {epoch+1}/{epochs} | Loss: {avg_epoch_loss:.4f} | Acc: {acc:.4f}")
            epoch_history.append({"epoch": epoch + 1, "loss": avg_epoch_loss, "accuracy": acc})
                
        avg_loss = total_loss / (len(dataloader) * epochs) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy, total, epoch_history

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
                
                # Defensive check: skip if labels are out of bounds for head weight
                max_label = labels.max().item()
                if max_label >= self.head.weight.shape[0]:
                    print(f"[TRAINER WARNING] Eval labels ({max_label}) out of head bounds ({self.head.weight.shape[0]}). Skipping batch.")
                    continue

                features = torch.zeros((labels.size(0), self.head.weight.shape[1]), device=self.device)
                img_mask, emb_mask = ~is_embedding, is_embedding
                
                if img_mask.any():
                    img_input = imgs[img_mask]
                    
                    # --- FLIP TRICK (Alignment dengan Registry & Live Inference) ---
                    # Menghitung rata-rata embedding asli dan mirror untuk validasi yang lebih akurat
                    if img_input.size(0) == 1:
                        # Case single image (MTCNN quirks)
                        f_orig = self.backbone(torch.cat([img_input, img_input], dim=0))[0:1]
                        f_flip = self.backbone(torch.cat([torch.flip(img_input, [3]), torch.flip(img_input, [3])], dim=0))[0:1]
                    else:
                        f_orig = self.backbone(img_input)
                        f_flip = self.backbone(torch.flip(img_input, [3]))
                    
                    features[img_mask] = torch.nn.functional.normalize(f_orig + f_flip, p=2, dim=1)
                        
                if emb_mask.any():
                    features[emb_mask] = embs[emb_mask]
                
                # Akurasi Sebenarnya untuk Pelaporan
                logits = F.linear(F.normalize(features), F.normalize(self.head.weight)) * self.head.s
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
        """Mengekspansi head klasifikasi sambil mempertahankan bobot yang sudah terlatih."""
        old_head = self.head
        old_nrp_to_idx = self.nrp_to_idx
        
        embedding_size = 128 # Default for MobileFaceNet
        if old_head is not None:
            embedding_size = old_head.weight.shape[1]
            
        new_head = ArcMarginProduct(embedding_size, new_num_classes).to(self.device)
        
        if old_head is None:
            print(f"[TRAINER] New Head Created: {new_num_classes} classes (initial/lazy)")
            self.head = new_head
            self.nrp_to_idx = new_nrp_to_idx
            return new_head

        copied_count = 0
        with torch.no_grad():
            for nrp, new_idx in new_nrp_to_idx.items():
                if nrp in old_nrp_to_idx:
                    old_idx = old_nrp_to_idx[nrp]
                    if old_idx < old_head.weight.shape[0] and new_idx < new_num_classes: 
                         new_head.weight[new_idx] = old_head.weight[old_idx]
                         copied_count += 1
        
        print(f"[TRAINER] Head Expanded: {old_head.weight.shape[0]} -> {new_num_classes} ({copied_count} weights preserved)")
        self.head = new_head
        self.nrp_to_idx = new_nrp_to_idx
        return new_head


    def _is_shared_param(self, name):
        """Filter ketat untuk pFedFace: Kecualikan BatchNorm agar tetap lokal."""
        name = name.lower()
        if any(x in name for x in ['bn', 'running_', 'num_batches_tracked']):
            return False
        return any(x in name for x in ['weight', 'bias'])

    def get_backbone_parameters(self, personalized=True):
        if self.backbone is None: return []
        state_dict = self.backbone.state_dict()
        shared_keys = [k for k in state_dict.keys() if not personalized or self._is_shared_param(k)]
        
        mode_str = "pFedFace (Local BN/Head)" if personalized else "Standard (Full Sync)"
        print(f"[TRAINER] Extracting {len(shared_keys)} parameters ({mode_str}).")
        
        params = [state_dict[k].cpu().numpy().copy() for k in shared_keys]
        
        # Sertakan Head hanya jika bukan Personalized (pFedFace)
        if not personalized and self.head is not None:
            head_params = [p.detach().cpu().numpy().copy() for p in self.head.parameters()]
            print(f"[TRAINER] Including {len(head_params)} head parameters.")
            params.extend(head_params)
            
        return params

    def get_bn_parameters(self):
        """
        Mengambil seluruh parameter BatchNorm dalam bentuk state_dict (Orderly).
        Digunakan untuk finalisasi di fase Registry.
        """
        if self.backbone is None: return {}
        state_dict = self.backbone.state_dict()
        # Filter semua key yang berkaitan dengan BN
        bn_keys = [k for k in state_dict.keys() if 
                   any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked'])]
        
        # Kembalikan sebagai dictionary of numpy arrays (Serialized State Dict)
        print(f"[TRAINER] Collecting {len(bn_keys)} BN parameters into orderly state_dict.")
        return {k: state_dict[k].cpu().numpy().copy() for k in bn_keys}

    def set_backbone_parameters(self, parameters, personalized=True):
        if self.backbone is None:
            print("[TRAINER] Backbone is None, initializing new MobileFaceNet...")
            self.backbone = MobileFaceNet().to(self.device)
            self.backbone.eval()
            
        state_dict = self.backbone.state_dict()
        shared_keys = [k for k in state_dict.keys() if not personalized or self._is_shared_param(k)]
        num_backbone = len(shared_keys)

        print(f"[TRAINER] Injecting {num_backbone} backbone parameters.")
        new_state_dict = OrderedDict(state_dict)
        for i, k in enumerate(shared_keys):
            try:
                if i < len(parameters):
                    tensor_v = torch.from_numpy(parameters[i].copy())
                    if tensor_v.shape == state_dict[k].shape:
                        new_state_dict[k] = tensor_v
            except: continue
        self.backbone.load_state_dict(new_state_dict, strict=False)

        # Pasang Head Parameters (Hanya jika Full Sync dan ada dalam payload)
        if not personalized and len(parameters) > num_backbone and self.head is not None:
            head_params = parameters[num_backbone:]
            print(f"[TRAINER] Injecting {len(head_params)} head parameters.")
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
        print(f"[TRAINER] Calculating Centroids...")
        with torch.no_grad():
            for imgs, embs, labels, is_embedding in dataloader:
                img_mask = ~is_embedding
                if not img_mask.any(): continue
                imgs_batch = imgs[img_mask].to(self.device)
                # --- FLIP TRICK (Mandatory for Centroid Stability) ---
                if imgs_batch.size(0) == 1:
                    f_orig = self.backbone(torch.cat([imgs_batch, imgs_batch]))[0:1]
                    f_flip = self.backbone(torch.cat([torch.flip(imgs_batch, [3]), torch.flip(imgs_batch, [3])]))[0:1]
                else:
                    f_orig = self.backbone(imgs_batch)
                    f_flip = self.backbone(torch.flip(imgs_batch, [3]))
                
                # Rata-rata dan normalisasi unit vector
                features = F.normalize(f_orig + f_flip, p=2, dim=1)
                
                batch_labels = labels[img_mask]
                for i in range(len(batch_labels)):
                    nrp = dataset.id_map[batch_labels[i].item()]
                    temp_embeddings[nrp].append(features[i].unsqueeze(0))
        centroids = {}
        for nrp, embs in temp_embeddings.items():
            if embs:
                # Gunakan subset (maks 50) untuk rata-rata yang stabil
                subset = embs[:50]
                stack = torch.cat(subset, dim=0)
                centroid = torch.mean(stack, dim=0)
                # Normalisasi L2 eksplisit agar konsisten dengan hypersphere ArcFace
                centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1)
                centroids[nrp] = centroid.cpu().numpy()[0]
        return centroids
