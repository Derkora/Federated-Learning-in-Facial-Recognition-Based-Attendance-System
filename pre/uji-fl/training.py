import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy
import os
import time

# Import arsitektur MobileFaceNet (112x96)
from mobilefacenet import MobileFaceNet, ArcMarginProduct

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROUNDS = 15
LOCAL_EPOCHS = 3
MU_PROX = 0.05
BATCH_SIZE = 16

# Transformasi Augmentasi (Meningkatkan Generalisasi)
train_transform = transforms.Compose([
    transforms.Resize((112, 96)),               
    transforms.RandomRotation(degrees=15),      
    transforms.RandomHorizontalFlip(p=0.5),     
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),                      
    transforms.Normalize([0.5]*3, [0.5]*3)      
])

val_transform = transforms.Compose([
    transforms.Resize((112, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class FederatedClient:
    def __init__(self, client_id, data_dir, global_model_path):
        self.client_id = client_id
        self.data_dir = data_dir
        
        # Inisialisasi Backbone
        self.backbone = MobileFaceNet().to(DEVICE)
        if os.path.exists(global_model_path):
            self.backbone.load_state_dict(torch.load(global_model_path, map_location=DEVICE), strict=False)
        
        # Inisialisasi Local Head (Kunci Identitas Lokal)
        train_path = os.path.join(data_dir, "train")
        val_path = os.path.join(data_dir, "val")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"[ERROR] Folder {train_path} tidak ditemukan. Jalankan preprocessing.py dulu!")
            
        self.num_classes = len(os.listdir(train_path))
        print(f"[INIT] {client_id} terdeteksi memiliki {self.num_classes} Mahasiswa.")
        self.head = ArcMarginProduct(128, self.num_classes).to(DEVICE)
        
        # Data Loaders
        self.train_loader = DataLoader(datasets.ImageFolder(train_path, train_transform), 
                                       batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(datasets.ImageFolder(val_path, val_transform), 
                                     batch_size=BATCH_SIZE)

    def train_locally(self, server_weights, current_lr):
        # Sync Backbone saja (Personalized BN tetap lokal)
        self.backbone.load_state_dict(server_weights, strict=False)
        global_ref = copy.deepcopy(self.backbone.state_dict())
        
        optimizer = optim.Adam([
            {'params': self.backbone.parameters()},
            {'params': self.head.parameters()}
        ], lr=current_lr)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.backbone.train(); self.head.train()
        for epoch in range(LOCAL_EPOCHS):
            running_loss = 0.0
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                features = self.backbone(imgs)
                outputs = self.head(features, labels)
                
                loss_cls = criterion(outputs, labels)
                
                # FedProx Penalty
                prox_loss = 0
                for name, param in self.backbone.named_parameters():
                    prox_loss += (MU_PROX / 2) * torch.norm(param - global_ref[name])**2
                
                total_loss = loss_cls + prox_loss
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()
            
            acc = self.evaluate()
            print(f"   [{self.client_id}] Epoch {epoch+1} Loss: {running_loss/len(self.train_loader):.4f} | Val Acc: {acc:.2f}%")
        
        # Federated Update (Exclude BN layers for pFedFace compliance)
        return {k: v for k, v in self.backbone.state_dict().items() if 'bn' not in k}
        
    def evaluate(self):
        self.backbone.eval(); self.head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                features = self.backbone(imgs)
                outputs = self.head(features, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / (total if total > 0 else 1)

def aggregate_weights(client_updates):
    server_weights = copy.deepcopy(client_updates[0])
    for k in server_weights.keys():
        for i in range(1, len(client_updates)):
            server_weights[k] += client_updates[i][k]
        server_weights[k] = torch.div(server_weights[k], len(client_updates))
    return server_weights

def generate_global_assets(clients):
    """Membangun BN Gabungan dan Registry Centroid untuk app.py"""
    print("\n[FINISH] Membangun Aset Global...")
    
    # Global BN (Averaging)
    all_bn_states = [c.backbone.state_dict() for c in clients]
    global_bn = {}
    bn_keys = [k for k in all_bn_states[0].keys() if 'bn' in k]
    for key in bn_keys:
        if all_bn_states[0][key].dtype == torch.long:
            global_bn[key] = all_bn_states[0][key]
        else:
            global_bn[key] = torch.stack([state[key] for state in all_bn_states]).mean(0)
    
    torch.save(global_bn, "global_bn_combined.pth")
    print("✅ Generated: global_bn_combined.pth")

    # Centroid Registry (Kunci Identifikasi Universal)
    all_centroids = []
    for client in clients:
        client.backbone.eval()
        client.backbone.load_state_dict(global_bn, strict=False)
        temp_embeddings = {i: [] for i in range(len(client.train_loader.dataset.classes))}
        
        with torch.no_grad():
            for imgs, labels in client.train_loader:
                imgs = imgs.to(DEVICE)
                embeddings = client.backbone(imgs)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                for i in range(len(labels)):
                    temp_embeddings[labels[i].item()].append(embeddings[i].unsqueeze(0))
        
        for label_idx in sorted(temp_embeddings.keys()):
            if temp_embeddings[label_idx]:
                stack = torch.cat(temp_embeddings[label_idx], dim=0)
                centroid = torch.mean(stack, dim=0, keepdim=True)
                centroid = torch.nn.functional.normalize(centroid, p=2, dim=1)
                all_centroids.append(centroid)

    global_registry = torch.cat(all_centroids, dim=0)
    torch.save(global_registry, "global_embedding_registry.pth")
    print(f"✅ Generated: global_embedding_registry.pth ({len(all_centroids)} identitas)")

if __name__ == "__main__":
    # --- FLOW TRAINING (Path dinamis agar bisa di-run dari mana saja) ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Path relatif ke folder parent untuk baseline
    BASE_MODEL = os.path.join(BASE_DIR, "..", "global_model_v0.pth") 
    
    c1_path = os.path.join(BASE_DIR, "datasets_processed", "client1")
    c2_path = os.path.join(BASE_DIR, "datasets_processed", "client2")
    
    if not os.path.exists(c1_path):
        print(f"[ABORT] Folder {c1_path} tidak ada. Jalankan 'python preprocessing.py' dulu!")
        exit()

    client1 = FederatedClient("client1", c1_path, BASE_MODEL) 
    client2 = FederatedClient("client2", c2_path, BASE_MODEL)

    server_weights = torch.load(BASE_MODEL, map_location=DEVICE)
    
    print(f"\n[START] Memulai {ROUNDS} Ronde Federated Learning...")
    for r in range(ROUNDS):
        if r < 5:
            current_lr = 1e-4
        elif 5 <= r < 10:
            current_lr = 5e-5
        else:
            current_lr = 1e-5
        
        print(f"\n--- ROUND {r+1}/{ROUNDS} (LR: {current_lr}) ---")
        u1 = client1.train_locally(server_weights, current_lr)
        u2 = client2.train_locally(server_weights, current_lr)
        server_weights = aggregate_weights([u1, u2])
    
    # Simpan Backbone Global Final
    torch.save(server_weights, "global_model_final_fl.pth")
    
    # SIMPAN LOCAL HEAD
    torch.save(client1.head.state_dict(), "local_head_client1.pth")
    torch.save(client2.head.state_dict(), "local_head_client2.pth")
    
    # Buat Aset Pendukung (BN & Registry)
    generate_global_assets([client1, client2])
