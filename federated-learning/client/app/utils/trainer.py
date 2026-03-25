import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        if label.size(0) == 0:
            return torch.zeros(cosine.size(), device=input.device)
            
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

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

class FaceDataset(Dataset):
    def __init__(self, data_root, global_embeddings=None, transform=None, val_split=0.2, mode="train", seed=42):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        
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

        # Add local image samples
        local_samples = []
        for nrp in local_nrps:
            folder_path = os.path.join(data_root, nrp)
            paths = glob.glob(os.path.join(folder_path, "*.*"))
            idx = self.nrp_to_idx[nrp]
            for p in paths:
                local_samples.append({"type": "image", "path": p, "label": idx})
        
        self.samples.extend(local_samples)

        # Add global embedding samples with OVERSAMPLING for balance
        if global_embeddings:
            global_samples_base = []
            for item in global_embeddings:
                idx = self.nrp_to_idx[item['nrp']]
                global_samples_base.append({"type": "embedding", "data": item['embedding'], "label": idx})
            
            if len(local_samples) > 0 and len(global_samples_base) > 0:
                multiplier = max(1, len(local_samples) // len(global_samples_base))
                print(f"[DATASET] Oversampling global embeddings x{multiplier} to balance with {len(local_samples)} local images.")
                for _ in range(multiplier):
                    self.samples.extend(global_samples_base)
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
        
        from torchvision.transforms import InterpolationMode
        self.transform = transforms.Compose([
            transforms.Resize((112, 112), interpolation=InterpolationMode.BILINEAR), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((112, 112), scale=(0.8, 1.0), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def train(self, epochs=1, lr=0.0001, round_num=0, global_embeddings=None):
        dataset = FaceDataset(self.data_path, global_embeddings=global_embeddings, transform=self.transform, mode="train")
        if len(dataset) < 2:
            print(f"[TRAINER] Data too small ({len(dataset)}) for training. Skipping round.")
            return 0.0, 0.0, len(dataset)
            
        if dataset.num_classes > 0 and dataset.num_classes != self.head.out_features:
            self._update_head(dataset.num_classes, dataset.nrp_to_idx)
        else:
            self.nrp_to_idx = dataset.nrp_to_idx

        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=hybrid_collate, drop_last=True)
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        criterion = nn.CrossEntropyLoss()
        trainable_params = [p for p in self.backbone.parameters() if p.requires_grad]
        trainable_params += list(self.head.parameters())
        
        optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=1e-3)
        
        self.backbone.train()
        self.head.train()
        
        total_loss, correct, total = 0.0, 0, 0
        for epoch in range(epochs):
            for imgs, embs, labels, is_embedding in dataloader:
                imgs, embs, labels = imgs.to(self.device), embs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                features = torch.zeros((labels.size(0), self.head.in_features), device=self.device)
                img_mask, emb_mask = ~is_embedding, is_embedding
                
                if img_mask.any():
                    img_input = imgs[img_mask]
                    if img_input.size(0) == 1:
                        img_input_fixed = torch.cat([img_input, img_input], dim=0)
                        features_fixed = self.backbone(img_input_fixed)
                        features[img_mask] = features_fixed[0:1]
                    else:
                        features[img_mask] = self.backbone(img_input)
                        
                if emb_mask.any():
                    features[emb_mask] = embs[emb_mask]
                
                outputs = self.head(features, labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
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

    def _is_bn_param(self, name):
        name = name.lower()
        return 'bn' in name or 'running_' in name or 'num_batches_tracked' in name

    def get_backbone_parameters(self, round_num=1):
        # Round-based filtering for pFedFace: Sync BN only for first 3 rounds
        include_bn = (round_num <= 3)
        state_dict = self.backbone.state_dict()
        shared_keys = [k for k in state_dict.keys() if include_bn or not self._is_bn_param(k)]
        
        shared_params = []
        for k in shared_keys:
            shared_params.append(state_dict[k].cpu().numpy().copy())
        return shared_params

    def set_backbone_parameters(self, parameters, round_num=1):
        include_bn = (round_num <= 3)
        state_dict = self.backbone.state_dict()
        shared_keys = [k for k in state_dict.keys() if include_bn or not self._is_bn_param(k)]
        
        if len(shared_keys) != len(parameters):
            print(f"[WARNING] Parameter count mismatch! Local filter expects {len(shared_keys)}, but server sent {len(parameters)} (Round: {round_num})")
            if len(parameters) == len(state_dict):
                print("[INFO] Correcting to FULL state_dict keys.")
                shared_keys = list(state_dict.keys())
            else:
                non_bn_keys = [k for k in state_dict.keys() if not self._is_bn_param(k)]
                if len(parameters) == len(non_bn_keys):
                    print("[INFO] Correcting to NON-BN keys only.")
                    shared_keys = non_bn_keys
                else:
                    print("[ERROR] Fatal Mismatch: Cannot map parameters to model layers. Skipping sync.")
                    return

        new_state_dict = OrderedDict(state_dict)
        for k, v in zip(shared_keys, parameters):
            try:
                # v is a numpy array from Flower
                tensor_v = torch.from_numpy(v.copy())
                
                if tensor_v.shape != state_dict[k].shape:
                    # Provide informative error as requested
                    print(f"[ERROR] Size mismatch for layer '{k}': Model expects {state_dict[k].shape}, Server sent {tensor_v.shape}")
                    continue # Do not update this specific layer
                
                new_state_dict[k] = tensor_v
            except Exception as e:
                print(f"[DEBUG] Error processing layer '{k}': {e}")
                raise e
        
        # Load with strict=True to ensure all buffers and params are accounted for
        self.backbone.load_state_dict(new_state_dict, strict=True)
