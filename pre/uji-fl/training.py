import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy
import os

# Import model unik xiaoccer
from mobilefacenet import MobileFaceNet, ArcMarginProduct

WEIGHTS_LOCAL_DIR = "weights_local"
if not os.path.exists(WEIGHTS_LOCAL_DIR):
    os.makedirs(WEIGHTS_LOCAL_DIR)

# --- Konfigurasi ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROUNDS = 20
LOCAL_EPOCHS = 3 
MU_PROX = 0.05   
BATCH_SIZE = 16

# Transformasi On-the-Fly (Augmentasi)
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
        
        # 1. Inisialisasi Backbone (MobileFaceNet)
        self.backbone = MobileFaceNet().to(DEVICE)
        self.backbone.load_state_dict(torch.load(global_model_path, map_location=DEVICE))
        
        # 2. Inisialisasi Local Head (ArcMargin) - Terisolasi per Client
        # Menghitung jumlah NRP unik di folder client tersebut
        train_path = os.path.join(data_dir, "train")
        num_classes = len(os.listdir(train_path))
        self.head = ArcMarginProduct(128, num_classes).to(DEVICE)
        
        # Data Loader
        self.train_loader = DataLoader(datasets.ImageFolder(train_path, train_transform), 
                                       batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(datasets.ImageFolder(os.path.join(data_dir, "val"), val_transform), 
                                     batch_size=BATCH_SIZE)

    def train_locally(self, server_weights, current_lr):
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
                
                # Loss Utama
                loss_cls = criterion(outputs, labels)
                
                # Loss FedProx (Penalty jika terlalu jauh dari model server)
                prox_loss = 0
                for name, param in self.backbone.named_parameters():
                    prox_loss += (MU_PROX / 2) * torch.norm(param - global_ref[name])**2
                
                total_loss = loss_cls + prox_loss
                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()
            
            # Evaluasi Lokal
            acc = self.evaluate()
            print(f"   [{self.client_id}] Epoch {epoch+1} Loss: {running_loss/len(self.train_loader):.4f} | Val Acc: {acc:.2f}%")
        
        # Kembalikan hanya Backbone ke server (BN tetap lokal sesuai pFedFace)
        return {k: v for k, v in self.backbone.state_dict().items() if 'bn' not in k}

    def save_local_state(self):
        # Simpan State BN Lokal (pFedFace style) 
        bn_state = {k: v for k, v in self.backbone.state_dict().items() if 'bn' in k}
        torch.save(bn_state, f"{WEIGHTS_LOCAL_DIR}/{self.client_id}_bn.pth")
        
        # Simpan Head Lokal (Identitas NRP) 
        torch.save(self.head.state_dict(), f"{WEIGHTS_LOCAL_DIR}/{self.client_id}_head.pth")
        print(f"[SUCCESS] Personalized weights saved for {self.client_id}")
        
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
        return 100 * correct / total

# --- SERVER AGGREGATION ---
def aggregate_weights(client_updates):
    server_weights = copy.deepcopy(client_updates[0])
    for k in server_weights.keys():
        for i in range(1, len(client_updates)):
            server_weights[k] += client_updates[i][k]
        server_weights[k] = torch.div(server_weights[k], len(client_updates))
    return server_weights

def generate_global_embedding_registry():
    #Load BN (Statistik Lokal)
    bn1 = torch.load("weights_local/client1_bn.pth")
    bn2 = torch.load("weights_local/client2_bn.pth")
    
    # Rata-ratakan BN (Simple Averaging untuk Global Style)
    global_bn = {}
    for key in bn1.keys():
        global_bn[key] = (bn1[key] + bn2[key]) / 2
    
    # 3. Gabungkan Head (Identitas Lokal)
    head1 = torch.load("weights_local/client1_head.pth")['weight']
    head2 = torch.load("weights_local/client2_head.pth")['weight']
    global_registry = torch.cat([head1, head2], dim=0)
    
    # 4. Simpan Paket "Sakti" ini
    torch.save(global_bn, "global_bn_combined.pth")
    torch.save(global_registry, "global_embedding_registry.pth")
    
if __name__ == "__main__":
    # Path awal dari baseline
    BASE_MODEL = "../global_model_v0.pth" 
    
    # Inisialisasi Client (Hilangkan tanda koma di akhir baris client1!)
    client1 = FederatedClient("client1", "datasets_processed/client1", BASE_MODEL) # Koma dihapus
    client2 = FederatedClient("client2", "datasets_processed/client2", BASE_MODEL)

    # Load bobot awal server untuk Round 1
    server_weights = torch.load(BASE_MODEL, map_location=DEVICE)
    
    print("[START] Memulai Federated Learning...")
    for r in range(ROUNDS):
        
        if r < 15:
            current_lr = 1e-4    
        elif 15 <= r < 25:
            current_lr = 5e-5    
        else:
            current_lr = 1e-5
        
        print(f"\n--- ROUND {r+1}/{ROUNDS} ---")
        
        u1 = client1.train_locally(server_weights, current_lr)
        u2 = client2.train_locally(server_weights, current_lr)
        
        # Server melakukan agregasi (FedAvg)
        server_weights = aggregate_weights([u1, u2])
        
    # Simpan backbone global yang sudah kolaboratif
    torch.save(server_weights, "global_model_final_fl.pth")
    
    client1.save_local_state()
    client2.save_local_state()
    
    generate_global_embedding_registry()
    
