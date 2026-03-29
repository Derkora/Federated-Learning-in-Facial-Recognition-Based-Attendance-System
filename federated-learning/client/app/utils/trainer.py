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
            imgs.append(torch.zeros((3, 112, 96)))
        else:
            imgs.append(data)
            embs.append(torch.zeros(128))

    return torch.stack(imgs), torch.stack(embs), torch.tensor(labels), torch.tensor(is_embedding)

class FaceDataset(Dataset):
    def __init__(self, data_root, global_embeddings=None, transform=None, mode="train", seed=42):
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
        self.class_counts = {idx: 0 for idx in range(self.num_classes)}

        # Data Alignment with uji-fl
        print(f"[DATASET] Loading local folders for {len(local_nrps)} users...")
        for nrp in local_nrps:
            folder_path = os.path.join(data_root, nrp)
            paths = sorted(glob.glob(os.path.join(folder_path, "*.*")))
            idx = self.nrp_to_idx[nrp]
            
            if len(paths) == 0: continue
            
            # Simple 80/20 Split as per user preference (Normal Split)
            split_idx = int(0.8 * len(paths))
            if mode == "train":
                selected = paths[:split_idx]
            else:
                selected = paths[split_idx:]
            
            for p in selected:
                self.samples.append({"type": "image", "path": p, "label": idx})
                self.class_counts[idx] += 1
            
            print(f"  [OK] {nrp}: Loaded {len(selected)} {mode} images.")

        # Add global embedding samples (Knowledge Sharing)
        if global_embeddings:
            for item in global_embeddings:
                idx = self.nrp_to_idx[item['nrp']]
                self.samples.append({"type": "embedding", "data": item['embedding'], "label": idx})
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
        self.backbone = backbone.to(self.device)
        self.head = head.to(self.device)
        self.data_path = data_path
        self.nrp_to_idx = {} 
        
        self.transform = transforms.Compose([
            transforms.Resize((112, 96), interpolation=InterpolationMode.BILINEAR), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((112, 96), interpolation=InterpolationMode.BILINEAR), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def train(self, epochs=5, lr=0.0001, round_num=0, global_embeddings=None, mu=0.05, lam=0.1):
        dataset = FaceDataset(self.data_path, global_embeddings=global_embeddings, transform=self.transform, mode="train")
        if len(dataset) < 2:
            print(f"[TRAINER] Data too small ({len(dataset)}) for training. Skipping round.")
            return 0.0, 0.0, len(dataset)
            
        if dataset.num_classes > 0 and dataset.num_classes != self.head.weight.shape[0]:
            self._update_head(dataset.num_classes, dataset.nrp_to_idx)
        else:
            self.nrp_to_idx = dataset.nrp_to_idx

        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=hybrid_collate, drop_last=True)
        
        backbone_snapshot = copy.deepcopy(self.backbone.state_dict())
        head_snapshot = copy.deepcopy(self.head.state_dict())
        global_ref = copy.deepcopy(self.backbone.state_dict())

        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        optimizer = torch.optim.Adam([
            {'params': backbone_params},
            {'params': head_params}
        ], lr=lr)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        print(f"[TRAINER] Round {round_num}: Training {len(dataset)} samples for {epochs} epochs")
        total_loss, correct, total = 0.0, 0, 0
        
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
                    print(f"[ERROR] NaN loss detected. Rolling back weights...")
                    self.backbone.load_state_dict(backbone_snapshot)
                    self.head.load_state_dict(head_snapshot)
                    raise TrainingNaNError(f"NaN loss at Round {round_num}")

                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                
                # METRICS: Calculate True Accuracy (Without Training Margin)
                with torch.no_grad():
                    logits_for_acc = F.linear(F.normalize(features_local), F.normalize(self.head.weight)) * self.head.s
                    _, predicted = torch.max(logits_for_acc.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            acc = correct / total if total > 0 else 0.0
            print(f"  > Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f} | Acc: {acc:.4f}")
                
        avg_loss = total_loss / (len(dataloader) * epochs) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy, total

    def evaluate(self, global_embeddings=None):
        dataset = FaceDataset(self.data_path, global_embeddings=global_embeddings, transform=self.val_transform, mode="val")
        if len(dataset) < 2: return 0.0, 0.0, len(dataset)
            
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=hybrid_collate)
        criterion = nn.CrossEntropyLoss()
        self.backbone.eval()
        self.head.eval()
        
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, embs, labels, is_embedding in dataloader:
                imgs, embs, labels = imgs.to(self.device), embs.to(self.device), labels.to(self.device)
                features = torch.zeros((labels.size(0), self.head.weight.shape[1]), device=self.device)
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
                
                # True Accuracy for Reporting
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

    def _update_head(self, new_num_classes, new_nrp_to_idx):
        old_head = self.head
        old_nrp_to_idx = self.nrp_to_idx
        new_head = ArcMarginProduct(old_head.weight.shape[1], new_num_classes).to(self.device)
        
        copied_count = 0
        with torch.no_grad():
            for nrp, new_idx in new_nrp_to_idx.items():
                if nrp in old_nrp_to_idx:
                    old_idx = old_nrp_to_idx[nrp]
                    if old_idx < old_head.weight.shape[0]: 
                         new_head.weight[new_idx] = old_head.weight[old_idx]
                         copied_count += 1
        
        print(f"[TRAINER] Dynamic Head Update: {old_head.weight.shape[0]} -> {new_num_classes} ({copied_count} copied)")
        self.head = new_head
        self.nrp_to_idx = new_nrp_to_idx

    def _is_shared_param(self, name):
        """Strict filter for pFedFace: Exclude BatchNorm."""
        name = name.lower()
        if any(x in name for x in ['bn', 'running_', 'num_batches_tracked']):
            return False
        return any(x in name for x in ['weight', 'bias'])

    def get_backbone_parameters(self, personalized=True):
        state_dict = self.backbone.state_dict()
        if personalized:
            shared_keys = [k for k in state_dict.keys() if self._is_shared_param(k)]
            mode_str = "pFedFace (Conv/Linear Only)"
        else:
            shared_keys = list(state_dict.keys())
            mode_str = "Standard (Full Sync)"
        print(f"[TRAINER] Extracting {len(shared_keys)} parameters ({mode_str}).")
        return [state_dict[k].cpu().numpy().copy() for k in shared_keys]

    def get_bn_parameters(self):
        state_dict = self.backbone.state_dict()
        bn_keys = [k for k in state_dict.keys() if any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked'])]
        print(f"[TRAINER] Extracting {len(bn_keys)} BN parameters.")
        return {k: state_dict[k].cpu().numpy().copy() for k in bn_keys}

    def set_backbone_parameters(self, parameters, personalized=True):
        state_dict = self.backbone.state_dict()
        if personalized:
            shared_keys = [k for k in state_dict.keys() if self._is_shared_param(k)]
        else:
            shared_keys = list(state_dict.keys())

        if len(parameters) != len(shared_keys):
            if len(parameters) == len(state_dict):
                shared_keys = list(state_dict.keys())
            else: return

        print(f"[TRAINER] Injecting {len(parameters)} parameters.")
        new_state_dict = OrderedDict(state_dict)
        for k, v in zip(shared_keys, parameters):
            try:
                tensor_v = torch.from_numpy(v.copy())
                if tensor_v.shape == state_dict[k].shape:
                    new_state_dict[k] = tensor_v
            except: continue
        self.backbone.load_state_dict(new_state_dict, strict=False)

    def calculate_centroids(self):
        dataset = FaceDataset(self.data_path, transform=self.val_transform, mode="train")
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
                if imgs_batch.size(0) == 1:
                    features = self.backbone(torch.cat([imgs_batch, imgs_batch]))[0:1]
                else:
                    features = self.backbone(imgs_batch)
                features = F.normalize(features, p=2, dim=1)
                batch_labels = labels[img_mask]
                for i in range(len(batch_labels)):
                    nrp = dataset.id_map[batch_labels[i].item()]
                    temp_embeddings[nrp].append(features[i].unsqueeze(0))
        centroids = {}
        for nrp, embs in temp_embeddings.items():
            if embs:
                stack = torch.cat(embs, dim=0)
                centroid = torch.mean(stack, dim=0)
                centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1)
                centroids[nrp] = centroid.cpu().numpy()[0]
        return centroids
