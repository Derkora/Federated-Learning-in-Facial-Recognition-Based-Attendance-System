import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def evaluate(self, global_embeddings=None, label_map=None):
    # Memuat dataset validasi lokal
    dataset = FaceDataset(
        self.data_path, 
        global_embeddings=global_embeddings, 
        transform=self.val_transform, 
        mode="val", label_map=label_map
    )
    if len(dataset) < 2: 
        return 0.0, 0.0, len(dataset)
        
    dataloader = DataLoader(
        dataset, batch_size=16, 
        shuffle=False, collate_fn=hybrid_collate
    )
    criterion = nn.CrossEntropyLoss()
    self.backbone.eval()
    self.head.eval()
    
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, embs, labels, is_embedding in dataloader:
            imgs = imgs.to(self.device)
            embs = embs.to(self.device)
            labels = labels.to(self.device)
            
            features = torch.zeros(
                (labels.size(0), 128), device=self.device
            )
            img_mask, emb_mask = ~is_embedding, is_embedding
            
            if img_mask.any():
                img_input = imgs[img_mask]
                
                # Ekstraksi embedding asli & mirror (Flip Trick)
                f_orig = self.backbone(img_input)
                f_flip = self.backbone(torch.flip(img_input, [3]))
                
                # Rata-rata dan normalisasi L2
                features[img_mask] = (
                    torch.nn.functional.normalize(
                        f_orig + f_flip, p=2, dim=1
                    )
                )
                    
            if emb_mask.any():
                features[emb_mask] = embs[emb_mask]
            
            # Hitung logits kelas dan loss function
            logits = self.head.get_logits(features)
            outputs = self.head(features, labels)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = (
        total_loss / len(dataloader) 
        if len(dataloader) > 0 else 0.0
    )
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy, total
