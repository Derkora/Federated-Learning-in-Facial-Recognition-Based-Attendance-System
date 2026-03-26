import torch
import torch.nn as nn
import torch.nn.functional as F
from .mobilefacenet import ArcMarginProduct
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
import random
import copy
import cv2

from collections import OrderedDict
from torchvision.transforms import InterpolationMode
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class TrainingNaNError(Exception):
    """Custom exception when training loss or weights become NaN."""
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
            imgs.append(torch.zeros((3, 112, 112)))
        else:
            imgs.append(data)
            embs.append(torch.zeros(128))

    return torch.stack(imgs), torch.stack(embs), torch.tensor(labels), torch.tensor(is_embedding)

def estimate_blur(img_pil):
    """Estimate blur using Laplacian variance (requires cv2)."""
    try:
        img_np = np.array(img_pil)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except:
        return 100.0 # Fallback

class FaceDataset(Dataset):
    def __init__(self, data_root, global_embeddings=None, transform=None, val_split=0.2, mode="train", seed=42, min_blur_score=50.0):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        self.class_counts = {}
        
        local_nrps = []
        if os.path.exists(data_root):
            local_nrps = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
            
        global_nrps = []
        if global_embeddings:
            global_nrps = [item['nrp'] for item in global_embeddings]
            
        all_unique_nrps = sorted(list(set(local_nrps + global_nrps)))
        self.nrp_to_idx = {nrp: idx for idx, nrp in enumerate(all_unique_nrps)}
        self.id_map = {idx: nrp for nrp, idx in self.nrp_to_idx.items()}
        self.num_classes = len(all_unique_nrps)

        # Initialize counts
        for idx in range(self.num_classes):
            self.class_counts[idx] = 0

        # Add local image samples with Quality Filtering
        local_samples = []
        print(f"[DATASET] Loading local samples from {data_root}...")
        for nrp in local_nrps:
            folder_path = os.path.join(data_root, nrp)
            paths = glob.glob(os.path.join(folder_path, "*.*"))
            idx = self.nrp_to_idx[nrp]
            
            nrp_count = 0
            for p in paths:
                try:
                    with Image.open(p) as img:
                        if estimate_blur(img) < min_blur_score:
                            continue
                    local_samples.append({"type": "image", "path": p, "label": idx})
                    nrp_count += 1
                except: continue
            
            self.class_counts[idx] += nrp_count
        
        self.samples.extend(local_samples)

        # Add global embedding samples with BALANCED OVERSAMPLING (1:1 Target)
        if global_embeddings:
            global_samples_base = []
            for item in global_embeddings:
                idx = self.nrp_to_idx[item['nrp']]
                global_samples_base.append({"type": "embedding", "data": item['embedding'], "label": idx})
                self.class_counts[idx] += 1
            
            if len(local_samples) > 0 and len(global_samples_base) > 0:
                # Calculate required multiplier to reach 1:1 ratio
                multiplier = max(1, len(local_samples) // len(global_samples_base))
                print(f"[DATASET] Balanced Oversampling: x{multiplier} for {len(global_samples_base)} global embs to match {len(local_samples)} local images.")
                for _ in range(multiplier):
                    for s in global_samples_base:
                        self.samples.append(s)
            else:
                self.samples.extend(global_samples_base)

        # Split into train/validation
        if val_split > 0:
            random.seed(seed)
            random.shuffle(self.samples)
            val_size = int(len(self.samples) * val_split)
            if mode == "train":
                self.samples = self.samples[val_size:]
            else:
                self.samples = self.samples[:val_size]

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
        self.backbone = backbone.to(self.device)
        self.head = head.to(self.device)
        self.data_path = data_path
        self.nrp_to_idx = {} 
        
        self.transform = transforms.Compose([
            transforms.Resize((112, 112), interpolation=InterpolationMode.BILINEAR), 
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            transforms.RandomResizedCrop((112, 112), scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            transforms.RandomErasing(p=0.5) 
        ])

    def train(self, epochs=5, lr=0.0001, round_num=0, global_embeddings=None, mu=0.01, lam=0.1):
        dataset = FaceDataset(self.data_path, global_embeddings=global_embeddings, transform=self.transform, mode="train")
        if len(dataset) < 2:
            print(f"[TRAINER] Data too small ({len(dataset)}) for training. Skipping round.")
            return 0.0, 0.0, len(dataset)
            
        if dataset.num_classes > 0 and dataset.num_classes != self.head.out_features:
            self._update_head(dataset.num_classes, dataset.nrp_to_idx)
        else:
            self.nrp_to_idx = dataset.nrp_to_idx
            
        m_base = getattr(self.head, 'm_base', 0.5)
        margins = []
        for i in range(dataset.num_classes):
            n_i = dataset.class_counts.get(i, 1)
            # Clipped Adaptive Margin: 0.3 - 0.5
            m_i = np.clip(m_base / np.sqrt(max(1, n_i)), 0.3, 0.5)
            margins.append(m_i)
        self.head.update_margin(torch.tensor(margins))

        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=hybrid_collate, drop_last=True)
        
        generic_backbone = copy.deepcopy(self.backbone)
        generic_backbone.eval() # Generic remains static during round
        
        backbone_snapshot = copy.deepcopy(self.backbone.state_dict())
        head_snapshot = copy.deepcopy(self.head.state_dict())

        params_old = [p.detach().clone() for p in self.backbone.parameters() if p.requires_grad and self._is_shared_param("dummy." + "dummy")]
        shared_params_old = {}
        for name, param in self.backbone.named_parameters():
            if self._is_shared_param(name):
                shared_params_old[name] = param.detach().clone()

        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        optimizer = torch.optim.SGD([
            {'params': backbone_params, 'lr': lr},
            {'params': head_params, 'lr': lr * 10}
        ], momentum=0.9, weight_decay=1e-3)
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        warmup_epochs = 1
        print(f"[TRAINER] Round {round_num}: BN Warm-up for {warmup_epochs} epochs...")
        for epoch in range(warmup_epochs):
            self.backbone.train()
            # Freeze everything except BN
            for name, m in self.backbone.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters(): p.requires_grad = True
                else:
                    for p in m.parameters(): p.requires_grad = False
            
            # Head is also frozen during BN Warm-up to focus on stats
            for p in self.head.parameters(): p.requires_grad = False
            
            for imgs, embs, labels, is_embedding in dataloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                img_mask = ~is_embedding
                if img_mask.any():
                    img_input = imgs[img_mask]
                    if img_input.size(0) == 1:
                        # BatchNorm2d needs > 1 sample per batch. Duplicate if only one image.
                        img_input = torch.cat([img_input, img_input], dim=0)
                    _ = self.backbone(img_input)
                # No backward/step here, just updating running stats? 
                # Actually, some papers do 1-step update, but "warmup" usually means running stats.
                # If we want to update BN weights/biases too:
                # optimizer.step() - but let's keep it simple as stats warmup.
            print(f"  > BN Warm-up Epoch {epoch+1} complete.")

        # Full Training
        print(f"[TRAINER] Round {round_num}: Full Training {len(dataset)} samples for {epochs} epochs")
        total_loss, correct, total = 0.0, 0, 0
        global_step = 0
        warmup_steps = 50
        last_batch_loss = None
        for p in self.backbone.parameters(): p.requires_grad = True
        for p in self.head.parameters(): p.requires_grad = True
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.backbone.train()
            self.head.train()
            
            for imgs, embs, labels, is_embedding in dataloader:
                if global_step < warmup_steps:
                    curr_lr = 1e-6 + (lr - 1e-6) * (global_step / warmup_steps)
                    optimizer.param_groups[0]['lr'] = curr_lr      # Backbone
                    optimizer.param_groups[1]['lr'] = curr_lr * 10 # Head
                
                imgs, embs, labels = imgs.to(self.device), embs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                features_local = torch.zeros((labels.size(0), self.head.in_features), device=self.device)
                img_mask, emb_mask = ~is_embedding, is_embedding
                
                if img_mask.any():
                    img_input = imgs[img_mask]
                    if img_input.size(0) == 1:
                        img_input_fixed = torch.cat([img_input, img_input], dim=0)
                        features_fixed = self.backbone(img_input_fixed)
                        features_local[img_mask] = features_fixed[0:1]
                    else:
                        features_local[img_mask] = self.backbone(img_input)
                        
                if emb_mask.any():
                    features_local[emb_mask] = embs[emb_mask]
                
                outputs = self.head(features_local, labels)
                ce_loss = criterion(outputs, labels)
                
                pd_loss = torch.tensor(0.0, device=self.device)
                if img_mask.any():
                    img_input = imgs[img_mask]
                    with torch.no_grad():
                        if img_input.size(0) == 1:
                            img_input_fixed = torch.cat([img_input, img_input], dim=0)
                            features_generic_fixed = generic_backbone(img_input_fixed)
                            features_generic = features_generic_fixed[0:1]
                        else:
                            features_generic = generic_backbone(img_input)
                    
                    s_local = outputs[img_mask]
                    with torch.no_grad():
                        s_generic = self.head(features_generic, labels[img_mask])
                    
                    pd_loss = F.mse_loss(F.softmax(s_local / self.head.s, dim=1), 
                                         F.softmax(s_generic / self.head.s, dim=1))
                
                prox_loss = torch.tensor(0.0, device=self.device)
                for name, param in self.backbone.named_parameters():
                    if name in shared_params_old:
                        prox_loss += (mu / 2) * torch.norm(param - shared_params_old[name])**2
                
                # Total Loss
                loss = ce_loss + (lam * pd_loss) + prox_loss
                
                curr_loss_val = loss.item()
                if last_batch_loss is not None and not torch.isnan(loss) and curr_loss_val > 3 * last_batch_loss:
                    loss = loss * (last_batch_loss / curr_loss_val)
                last_batch_loss = curr_loss_val
                global_step += 1
                
                if torch.isnan(loss):
                    print(f"[ERROR] NaN loss detected at Round {round_num}, Epoch {epoch+1}. Rolling back weights...")
                    self.backbone.load_state_dict(backbone_snapshot)
                    self.head.load_state_dict(head_snapshot)
                    raise TrainingNaNError(f"NaN loss at Round {round_num}, Epoch {epoch+1}")

                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), 5.0)
                optimizer.step()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            scheduler.step()
            print(f"  > Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                
        avg_loss = total_loss / (len(dataloader) * epochs) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy, total

    def evaluate(self, global_embeddings=None):
        dataset = FaceDataset(self.data_path, global_embeddings=global_embeddings, transform=self.transform, mode="val")
        if len(dataset) < 2:
            return 0.0, 0.0, len(dataset)
            
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=hybrid_collate)
        criterion = nn.CrossEntropyLoss()
        self.backbone.eval()
        self.head.eval()
        
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, embs, labels, is_embedding in dataloader:
                imgs, embs, labels = imgs.to(self.device), embs.to(self.device), labels.to(self.device)
                features = torch.zeros((labels.size(0), self.head.in_features), device=self.device)
                img_mask, emb_mask = ~is_embedding, is_embedding
                
                if img_mask.any():
                    img_input = imgs[img_mask]
                    if img_input.size(0) == 1:
                        features_fixed = self.backbone(torch.cat([img_input, img_input], dim=0))
                        features[img_mask] = features_fixed[0:1]
                    else:
                        features[img_mask] = self.backbone(img_input)
                        
                if emb_mask.any():
                    features[emb_mask] = embs[emb_mask]
                
                outputs = self.head(features, labels)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy, total

    def _update_head(self, new_num_classes, new_nrp_to_idx):
        old_head = self.head
        old_nrp_to_idx = self.nrp_to_idx
        new_head = ArcMarginProduct(old_head.in_features, new_num_classes).to(self.device)
        
        copied_count = 0
        with torch.no_grad():
            for nrp, new_idx in new_nrp_to_idx.items():
                if nrp in old_nrp_to_idx:
                    old_idx = old_nrp_to_idx[nrp]
                    if old_idx < old_head.weight.shape[0]: 
                         new_head.weight[new_idx] = old_head.weight[old_idx]
                         copied_count += 1
        
        print(f"[TRAINER] Dynamic Head Update: {old_head.out_features} -> {new_num_classes} (Warm-start: {copied_count} copied)")
        self.head = new_head
        self.nrp_to_idx = new_nrp_to_idx

    def _is_shared_param(self, name):
        """
        Strict filter for pFedFace:
        - Include: conv.weight, conv.bias, linear.weight, linear.bias
        - Exclude: everything related to BatchNorm (bn, running, tracked)
        """
        name = name.lower()
        if any(x in name for x in ['bn', 'running_', 'num_batches_tracked']):
            return False
        return any(x in name for x in ['weight', 'bias'])

    def get_backbone_parameters(self, personalized=True):
        """
        Extracts backbone parameters with strict naming-based filtering.
        """
        state_dict = self.backbone.state_dict()
        if personalized:
            # pFedFace: Only share Conv2d and Linear weights/biases
            shared_keys = [k for k in state_dict.keys() if self._is_shared_param(k)]
            mode_str = "pFedFace (Conv/Linear Only)"
        else:
            shared_keys = list(state_dict.keys())
            mode_str = "Standard (Full Sync)"
            
        print(f"[TRAINER] Extracting {len(shared_keys)} parameters ({mode_str}).")
        
        return [state_dict[k].cpu().numpy().copy() for k in shared_keys]

    def set_backbone_parameters(self, parameters, personalized=True):
        """
        Injects parameters into the backbone using Named Mapping.
        Protects against shape mismatches and logs missing/unexpected keys.
        """
        state_dict = self.backbone.state_dict()
        
        if personalized:
            shared_keys = [k for k in state_dict.keys() if self._is_shared_param(k)]
            mode_str = "pFedFace Named Mapping"
        else:
            shared_keys = list(state_dict.keys())
            mode_str = "Standard Mapping"

        # Robustness: Check if count matches shared_keys
        if len(parameters) != len(shared_keys):
            print(f"[WARNING] Parameter count mismatch! Model expects {len(shared_keys)}, but received {len(parameters)}.")
            # Fallback check: if received is full model
            if len(parameters) == len(state_dict):
                print("[INFO] Fallback: Received FULL parameters. Updating all shared keys.")
                shared_keys = list(state_dict.keys())
            else:
                print("[ERROR] Fatal Mismatch: Cannot map received parameters to expected layers. Sync aborted.")
                return

        print(f"[TRAINER] Injecting {len(parameters)} parameters ({mode_str}).")
        
        new_state_dict = OrderedDict(state_dict)
        for k, v in zip(shared_keys, parameters):
            try:
                tensor_v = torch.from_numpy(v.copy())
                
                # SHAPE VALIDATION: Protect against 4D-to-1D leaks (e.g. Bias receiving Weights)
                if tensor_v.shape != state_dict[k].shape:
                    print(f"[WARNING] Size mismatch for layer '{k}': Model expects {state_dict[k].shape}, received {tensor_v.shape}. Skipping.")
                    continue
                
                new_state_dict[k] = tensor_v
            except Exception as e:
                print(f"[DEBUG] Error processing layer '{k}': {e}")
                continue
        
        # Load with strict=False to allow local personal layers (BN) to persist
        load_result = self.backbone.load_state_dict(new_state_dict, strict=False)
        
        if load_result.missing_keys:
            # Expected: BN layers will be missing from the update (intended)
            bn_missing = [k for k in load_result.missing_keys if any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked'])]
            other_missing = [k for k in load_result.missing_keys if not any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked'])]
            if other_missing:
                print(f"[DEBUG] Unexpected Missing Keys: {other_missing}")
            else:
                print(f"[INFO] {len(bn_missing)} local BN layers preserved.")
        
        if load_result.unexpected_keys:
            print(f"[WARNING] Unexpected keys in update: {load_result.unexpected_keys}")
