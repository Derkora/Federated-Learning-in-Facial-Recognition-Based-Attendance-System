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

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
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

class FaceDataset(Dataset):
    def __init__(self, data_root, global_embeddings=None, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        self.id_map = {}
        
        # 1. Map all identities (Folders + Global Embeddings)
        local_nrps = []
        if os.path.exists(data_root):
            local_nrps = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
            
        global_nrps = []
        if global_embeddings:
            global_nrps = [item['nrp'] for item in global_embeddings]
            
        all_unique_nrps = sorted(list(set(local_nrps + global_nrps)))
        nrp_to_idx = {nrp: idx for idx, nrp in enumerate(all_unique_nrps)}
        self.id_map = {idx: nrp for nrp, idx in nrp_to_idx.items()}
        self.num_classes = len(all_unique_nrps)

        # 2. Add local image samples
        for nrp in local_nrps:
            folder_path = os.path.join(data_root, nrp)
            paths = glob.glob(os.path.join(folder_path, "*.*"))
            idx = nrp_to_idx[nrp]
            for p in paths:
                self.samples.append({"type": "image", "path": p, "label": idx})

        # 3. Add global embedding samples
        if global_embeddings:
            for item in global_embeddings:
                idx = nrp_to_idx[item['nrp']]
                # Duplicate embedding a bit to balance with image counts? 
                # (Optional, but let's add at least one)
                self.samples.append({"type": "embedding", "data": item['embedding'], "label": idx})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        
        if sample['type'] == "image":
            image = Image.open(sample['path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label, False # is_embedding = False
        else:
            # It's a pre-computed embedding tensor (already 128-d)
            return sample['data'], label, True # is_embedding = True

class LocalTrainer:
    def __init__(self, backbone, head, device="cpu", data_path="/app/data"):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.backbone = backbone.to(self.device)
        self.head = head.to(self.device)
        self.data_path = data_path
        
        # Training Augmentation: Focus on Scale & Translation (as per expert review)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((112, 96), scale=(0.8, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1), # Very subtle
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def save_confusion_matrix(self, y_true, y_pred, round_num=0):
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Round {round_num}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            os.makedirs('app/static/metrics', exist_ok=True)
            plt.savefig(f'app/static/metrics/cm_round_{round_num}.png')
            plt.close()
            print(f"[METRICS] Saved confusion matrix for round {round_num}")
        except Exception as e:
            print(f"[METRICS ERROR] Failed to save CM: {e}")

    def train(self, epochs=1, lr=0.0001, round_num=0, global_embeddings=None):
        dataset = FaceDataset(self.data_path, global_embeddings=global_embeddings, transform=self.transform)
        if len(dataset) == 0:
            print("No data found for training.")
            return 0.0, 0.0, 0
            
        # Re-initialize head if class count changed (Sync case)
        if dataset.num_classes > 0 and dataset.num_classes != self.head.out_features:
            print(f"[TRAINER] Dynamic Head Update: {self.head.out_features} -> {dataset.num_classes}")
            self.head = ArcMarginProduct(self.head.in_features, dataset.num_classes).to(self.device)

        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Train all layers for better adaptability with limited data
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        criterion = nn.CrossEntropyLoss()
        trainable_params = [p for p in self.backbone.parameters() if p.requires_grad]
        trainable_params += list(self.head.parameters())
        
        optimizer = torch.optim.SGD(
            trainable_params, 
            lr=lr, momentum=0.9, weight_decay=1e-3 # Stronger regularization
        )
        
        self.backbone.train()
        self.head.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            for data, labels, is_embedding in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Hybrid Logic: Image -> Backbone, Embedding -> Direct
                features = torch.zeros((data.size(0), self.head.in_features), device=self.device)
                
                # Process images in batch
                img_mask = ~is_embedding
                emb_mask = is_embedding
                
                if img_mask.any():
                    features[img_mask] = self.backbone(data[img_mask])
                
                if emb_mask.any():
                    features[emb_mask] = data[emb_mask]
                
                outputs = self.head(features, labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        # Calculate metrics
        avg_loss = total_loss / (len(dataloader) * epochs) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        if round_num > 0 and total > 0:
            self.save_confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy(), round_num)

        # Reset to eval mode for inference
        self.backbone.eval()
        self.head.eval()
        
        return avg_loss, accuracy, total

    def get_backbone_parameters(self):
        """Returns only backbone parameters for FL sync."""
        return [val.cpu().numpy() for _, val in self.backbone.state_dict().items()]

    def set_backbone_parameters(self, parameters):
        """Sets backbone parameters from global model."""
        state_dict = self.backbone.state_dict()
        keys = list(state_dict.keys())
        
        if len(keys) != len(parameters):
            print(f"WARNING: Parameter mismatch! Model has {len(keys)} layers, but received {len(parameters)}.")
            
        new_state_dict = OrderedDict()
        for i, (k, v) in enumerate(zip(keys, parameters)):
            try:
                # Use from_numpy for efficiency and better type handling
                tensor_v = torch.from_numpy(np.array(v))
                
                # Check for shape mismatch before loading
                if tensor_v.shape != state_dict[k].shape:
                    print(f"[DEBUG] Shape Mismatch at {k}: Expected {state_dict[k].shape}, got {tensor_v.shape}")
                    # If it's just a 0-d vs 1-d scalar issue (like num_batches_tracked), reshape it
                    if tensor_v.numel() == state_dict[k].numel():
                        tensor_v = tensor_v.view(state_dict[k].shape)
                
                new_state_dict[k] = tensor_v
            except Exception as e:
                print(f"[DEBUG] Error processing {k} (index {i}): {e}")
                raise e
        
        try:
            self.backbone.load_state_dict(new_state_dict, strict=True)
            print("[TRAINER] Backbone parameters updated successfully.")
        except Exception as e:
            print(f"[DEBUG] load_state_dict failed: {e}")
            raise e
