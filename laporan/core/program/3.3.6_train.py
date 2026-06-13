import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train(
    self, epochs, lr, round_num, 
    global_embeddings=None, label_map=None, mu=0.05
):
    # Inisialisasi dataset dan loader
    dataset = FaceDataset(
        self.data_path, 
        global_embeddings=global_embeddings, 
        transform=self.transform
    )
    dataloader = DataLoader(
        dataset, batch_size=32, 
        shuffle=True, collate_fn=hybrid_collate
    )
    
    # Bekukan lapisan awal model lokal
    set_model_freeze(self.backbone, freeze_mode="early")
    global_ref = copy.deepcopy(self.backbone.state_dict())
 
    # Pengaturan optimizer SGD dan kriteria loss
    optimizer = torch.optim.SGD(
        self.backbone.parameters(), 
        lr=lr, momentum=0.9, nesterov=True
    )
    criterion = nn.CrossEntropyLoss()
 
    for epoch in range(epochs):
        self.backbone.train()
        self.head.train()
        
        for imgs, embs, labels, is_embedding in dataloader:
            imgs = imgs.to(self.device)
            embs = embs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            
            features_local = torch.zeros(
                (labels.size(0), 128), device=self.device
            )
            img_mask, emb_mask = ~is_embedding, is_embedding
            
            # Forward pass citra wajah ke backbone
            if img_mask.any():
                features_local[img_mask] = (
                    self.backbone(imgs[img_mask])
                )
            if emb_mask.any():
                features_local[emb_mask] = embs[emb_mask]
            
            # Hitung loss klasifikasi ArcFace
            outputs = self.head(features_local, labels)
            ce_loss = criterion(outputs, labels)
            
            # Hitung penalti regulasi jarak L2 (FedProx)
            prox_loss = torch.tensor(0.0, device=self.device)
            for name, param in self.backbone.named_parameters():
                if name in global_ref:
                    diff_norm = (
                        torch.norm(param - global_ref[name])**2
                    )
                    prox_loss += (mu / 2) * diff_norm
            
            # Total Loss = ArcFace Loss + Proximity Loss
            loss = ce_loss + prox_loss
            
            # Backward pass & update bobot
            loss.backward()
            optimizer.step()
