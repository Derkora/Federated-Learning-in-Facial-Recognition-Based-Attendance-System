import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from mobilefacenet import MobileFaceNet, ArcMarginProduct

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Hyperparameters Terkalibrasi ---
ROUNDS = 12           # Dinaikkan sedikit agar konvergensi lebih mantap
EPOCHS = 1            # 1 Epoch per round untuk stabilitas pFedFace
LR_BACKBONE = 1e-4
LR_HEAD = 1e-2
MU_PROX = 0.5         # Menjaga model client tidak melenceng terlalu jauh
LABEL_SMOOTHING = 0.1 # Melawan skor halu 93%

# Load Mapping Identitas 34 Mahasiswa
if not os.path.exists("global_labels.pth"):
    print("Error: Jalankan preprocessing.py dulu, Ndan!")
    exit()

GLOBAL_LABELS = torch.load("global_labels.pth")
LABEL_MAP = {nrp: i for i, nrp in enumerate(GLOBAL_LABELS)}
NUM_CLASSES = len(GLOBAL_LABELS)

# [Metode pFedFace] Fungsi Filter untuk Adaptive BN
def is_shared_param(name):
    # Parameter BN, Running Mean, dan Var TETAP LOKAL (Tidak diagregasi)
    return not any(x in name.lower() for x in ['bn', 'running_', 'tracked'])

class FaceDataset(Dataset):
    def __init__(self, dir):
        self.samples = []
        for nrp in os.listdir(dir):
            cp = os.path.join(dir, nrp)
            for img in os.listdir(cp):
                self.samples.append((os.path.join(cp, img), LABEL_MAP[nrp]))
        self.tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3,[0.5]*3)])
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, l = self.samples[i]
        return self.tf(Image.open(p)), l

def evaluate(model, head, loader):
    model.eval(); head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            output = head(model(imgs), labels)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train_client(name, g_backbone_sd, g_head_sd, round_idx, local_sd=None):
    train_loader = DataLoader(FaceDataset(f"data_{name}/train"), batch_size=16, shuffle=True)
    val_loader = DataLoader(FaceDataset(f"data_{name}/val"), batch_size=8, shuffle=False)
    
    model = MobileFaceNet().to(DEVICE)
    head = ArcMarginProduct(128, NUM_CLASSES).to(DEVICE)
    
    # 1. LOAD WEIGHTS DENGAN LOGIKA ADAPTIVE BN
    if local_sd:
        # Gunakan status BN lokal milik client ronde sebelumnya
        model.load_state_dict(local_sd)
    
    if g_backbone_sd:
        # Timpa parameter konvolusi/linear dengan bobot global terbaru
        current_sd = model.state_dict()
        for k, v in g_backbone_sd.items():
            if is_shared_param(k): current_sd[k] = v
        model.load_state_dict(current_sd)

    if g_head_sd: head.load_state_dict(g_head_sd)
    
    # [Metode FedProx] Simpan snapshot untuk penalty
    backbone_fixed = {n: p.clone().detach() for n, p in model.named_parameters()}
    
    # Learning Rate Decay per Round
    decay = 0.95 ** (round_idx - 1)
    opt = optim.SGD([
        {'params': model.parameters(), 'lr': LR_BACKBONE * decay},
        {'params': head.parameters(), 'lr': LR_HEAD * decay}
    ], momentum=0.9)

    total_loss = 0.0
    model.train(); head.train()
    for _ in range(EPOCHS):
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            emb = model(imgs)
            output = head(emb, labels)
            
            # Loss dengan Label Smoothing
            loss_cls = nn.functional.cross_entropy(output, labels, label_smoothing=LABEL_SMOOTHING)
            # Penalty FedProx
            prox = sum((MU_PROX/2) * torch.norm(p - backbone_fixed[n])**2 for n, p in model.named_parameters())
            
            loss = loss_cls + prox
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        total_loss = epoch_loss / len(train_loader) 

    acc = evaluate(model, head, val_loader)
    print(f"    R{round_idx} | {name} | Loss: {total_loss:.4f} | Val Acc: {acc:.2f}%")
    
    return model.state_dict(), head.state_dict()

if __name__ == "__main__":
    g_backbone, g_head = None, None
    client_local_sds = {'client1': None, 'client2': None} # Penyimpanan BN lokal
    clients = ['client1', 'client2']
    
    for r in range(1, ROUNDS + 1):
        print(f"\n[SERVER] Round {r}/{ROUNDS}")
        b_updates, h_updates = [], []
        
        for c in clients:
            b_sd, h_sd = train_client(c, g_backbone, g_head, r, local_sd=client_local_sds[c])
            b_updates.append(b_sd)
            h_updates.append(h_sd)
            client_local_sds[c] = b_sd # Simpan statistik BN untuk ronde berikutnya
        
        # AGREGASI GLOBAL (Hanya Shared Parameters)
        g_backbone = {}
        for k in b_updates[0].keys():
            if is_shared_param(k) and b_updates[0][k].is_floating_point():
                g_backbone[k] = torch.stack([u[k] for u in b_updates]).mean(0)
            else:
                # BN tidak diagregasi, ambil satu saja sebagai placeholder untuk Server
                g_backbone[k] = b_updates[0][k]

        g_head = {k: torch.stack([u[k] for u in h_updates]).mean(0) 
                  if h_updates[0][k].is_floating_point() else h_updates[0][k] 
                  for k in h_updates[0].keys()}

    # Simpan Model Global Akhir
    torch.save(g_backbone, "global_model_final.pth")
    torch.save(g_head, "global_head_final.pth")
    print("\n[FINISH] Training Selesai, Ndan. Model Global (Backbone + Head) tersimpan.")