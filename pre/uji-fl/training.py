import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import random
from mobilefacenet import MobileFaceNet, ArcMarginProduct # Pastikan file ini satu folder
from facenet_pytorch import MTCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_MODEL_PATH = "../global_model_v0.pth" 

# Hyperparameters pFedFace (Miniature FL)
ROUNDS = 5            # Number of FL rounds
LR_BACKBONE = 0.0001
LR_HEAD = 0.001       
LAMBDA_PD = 0.1       
MU_PROX = 0.01        # FedProx regularization
TAU_MARGIN = 0.2      
EPOCHS_WARMUP = 1     
EPOCHS_FULL = 5       

# List Client yang akan di-uji
CLIENTS = [
    {"name": "client1", "path": "../../datasets/client1_data/students"},
    {"name": "client2", "path": "../../datasets/client2_data/students"}
]

def is_shared_param(name):
    """pFedFace: Shared (Backbone Conv/Linear), Local (BN)."""
    name = name.lower()
    if any(x in name for x in ['bn', 'running_', 'num_batches_tracked']):
        return False
    return any(x in name for x in ['weight', 'bias'])

def preprocess_all_clients():
    print(f"\n{'='*10} STAGE 1: PREPROCESSING & BALANCING {'='*10}")
    
    # Setup MTCNN once
    mtcnn = MTCNN(image_size=112, margin=20, device=DEVICE, post_process=True)
    aug_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(p=0.5)
    ])

    for client in CLIENTS:
        print(f"\n[PREPROCESS] Processing {client['name']}...")
        source_dir = client['path']
        output_dir = f"processed_{client['name']}"
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(source_dir):
            print(f"[SKIP] Source {source_dir} not found.")
            continue

        for nrp in os.listdir(source_dir):
            nrp_path = os.path.join(source_dir, nrp)
            if not os.path.isdir(nrp_path): continue
            
            save_path = os.path.join(output_dir, nrp)
            os.makedirs(save_path, exist_ok=True)

            print(f"  - NRP {nrp}: Preprocessing & Balancing to 100...")
            
            # 1. MTCNN Crop & Collect Originals
            clean_faces = []
            img_list = os.listdir(nrp_path)
            total_processed = 0
            for img_name in img_list:
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                total_processed += 1
                try:
                    img = Image.open(os.path.join(nrp_path, img_name)).convert('RGB')
                    face = mtcnn(img)
                    if face is not None:
                        # Convert to PIL [0, 1]
                        face_pil = transforms.ToPILImage()(face * 0.5 + 0.5)
                        clean_faces.append(face_pil)
                except: pass
                if len(clean_faces) >= 40: break 
            
            if not clean_faces:
                print(f"    [!] NRP {nrp}: Gagal mendeteksi wajah sama sekali.")
                continue

            # 2. Save Originals (max 40)
            for i, face in enumerate(clean_faces):
                face.save(os.path.join(save_path, f"orig_{i:03d}.jpg"))
            
            # 3. Augment to 100 (Oversampling)
            needed = 100 - len(clean_faces)
            for i in range(needed):
                base_face = random.choice(clean_faces)
                aug_face = aug_transform(base_face)
                aug_face.save(os.path.join(save_path, f"aug_{i:03d}.jpg"))
                
            print(f"    [+] OK: {len(clean_faces)} Ori (dari {total_processed} foto) + {needed} Aug = 100 total.")

class SimpleFaceDataset(Dataset):
    def __init__(self, processed_dir):
        self.processed_dir = processed_dir
        self.transform = transforms.Compose([
            transforms.Resize((112, 96)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.classes = sorted([f for f in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, f))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(processed_dir, cls)
            for img_name in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, img_name), self.class_to_idx[cls]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

class PDLoss(nn.Module):
    def __init__(self, tau=0.2):
        super(PDLoss, self).__init__()
        self.tau = tau

    def forward(self, f_local, f_generic, labels):
        f_l = nn.functional.normalize(f_local, p=2, dim=1)
        f_g = nn.functional.normalize(f_generic, p=2, dim=1)
        S_l = torch.mm(f_l, f_l.t())
        S_g = torch.mm(f_g, f_g.t())
        mask = labels.expand(len(labels), len(labels)).eq(labels.expand(len(labels), len(labels)).t()).float()
        pos_loss = torch.min((S_l - S_g) * mask, torch.zeros_like(S_l)).sum()
        neg_loss = torch.max((S_l - S_g) * (1 - mask), torch.tensor(self.tau).to(DEVICE)).sum()
        return -(pos_loss - neg_loss) / len(labels)

def train_client(name, processed_path, global_weights, local_head=None):
    print(f"\n      [CLIENT] {name.upper()} training on balanced data...")
    
    if not os.path.exists(processed_path): return None, 0, local_head, 0
    dataset = SimpleFaceDataset(processed_path)
    if len(dataset) == 0: return None, 0, local_head, 0

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    num_classes = len(dataset.classes)

    model_local = MobileFaceNet().to(DEVICE)
    model_generic = MobileFaceNet().to(DEVICE)
    
    if global_weights:
        model_local.load_state_dict(global_weights, strict=False)
        model_generic.load_state_dict(global_weights, strict=False)
    model_generic.eval()

    backbone_old = {n: p.clone().detach() for n, p in model_local.named_parameters() if is_shared_param(n)}

    if local_head is None:
        local_head = ArcMarginProduct(128, num_classes).to(DEVICE)
        class_counts = [0] * num_classes
        for _, lbl in dataset.samples: class_counts[lbl] += 1
        margins = torch.tensor([min(0.5, 0.5 / (count**0.5 + 1e-6)) for count in class_counts])
        local_head.update_margin(margins.clamp(0.3, 0.5))
    
    criterion_pd = PDLoss(tau=TAU_MARGIN)

    for name_p, param in model_local.named_parameters():
        param.requires_grad = "bn" in name_p.lower()
    
    opt_warmup = optim.Adam(filter(lambda p: p.requires_grad, model_local.parameters()), lr=LR_BACKBONE)
    for _ in range(EPOCHS_WARMUP):
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt_warmup.zero_grad()
            loss = nn.functional.cross_entropy(local_head(model_local(imgs), labels), labels)
            loss.backward()
            opt_warmup.step()

    # FULL TRAINING
    for param in model_local.parameters(): param.requires_grad = True
    optimizer = optim.Adam([
        {'params': model_local.parameters(), 'lr': LR_BACKBONE},
        {'params': local_head.parameters(), 'lr': LR_HEAD}
    ])

    for epoch in range(EPOCHS_FULL):
        t_loss = 0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            f_l = model_local(imgs)
            with torch.no_grad(): f_g = model_generic(imgs)
            
            loss_cls = nn.functional.cross_entropy(local_head(f_l, labels), labels)
            loss_pd = criterion_pd(f_l, f_g, labels)
            
            prox_loss = 0
            for n, p in model_local.named_parameters():
                if n in backbone_old:
                    prox_loss += (MU_PROX / 2) * torch.norm(p - backbone_old[n])**2
            
            total_loss = loss_cls + (LAMBDA_PD * loss_pd) + prox_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model_local.parameters(), 5.0)
            optimizer.step()
            t_loss += total_loss.item()
        
        print(f"        Epoch {epoch+1}/{EPOCHS_FULL} | Loss: {t_loss/len(dataloader):.4f}")
            
    shared_dict = {n: p.cpu().detach() for n, p in model_local.state_dict().items() if is_shared_param(n)}
    return shared_dict, len(dataset), local_head, (t_loss/len(dataloader))

def federated_averaging(client_updates):
    print("\n[SERVER] Aggregating Client Updates (FedAvg)...")
    total_samples = sum(u['samples'] for u in client_updates)
    if total_samples == 0: return None
    
    global_dict = {}
    first_update = client_updates[0]['params']
    for key in first_update.keys():
        weighted_sum = sum(u['params'][key] * (u['samples'] / total_samples) for u in client_updates)
        global_dict[key] = weighted_sum
    return global_dict

if __name__ == "__main__":
    preprocess_all_clients()

    print(f"\n{'='*20} STAGE 2: MINIATURE FL TRAINING {'='*20}")
    
    global_weights = None
    if os.path.exists(GLOBAL_MODEL_PATH):
        global_weights = torch.load(GLOBAL_MODEL_PATH, map_location='cpu')
        print(f"[SERVER] Loaded initial global model from {GLOBAL_MODEL_PATH}")
    
    client_heads = {c['name']: None for c in CLIENTS}
    
    for rnd in range(1, ROUNDS + 1):
        print(f"\n{'-'*10} FL ROUND {rnd} / {ROUNDS} {'-'*10}")
        client_updates = []
        
        for c_config in CLIENTS:
            params, samples, head, final_loss = train_client(
                c_config['name'], f"processed_{c_config['name']}", 
                global_weights, client_heads[c_config['name']]
            )
            if params:
                client_updates.append({'params': params, 'samples': samples, 'loss': final_loss})
                client_heads[c_config['name']] = head
        
        if client_updates:
            global_weights_update = federated_averaging(client_updates)
            if global_weights is None:
                global_weights = global_weights_update
            else:
                for k, v in global_weights_update.items():
                    global_weights[k] = v
            
            avg_round_loss = sum(u['loss'] for u in client_updates) / len(client_updates)
            print(f"[SERVER] Round {rnd} aggregation complete. Avg Loss: {avg_round_loss:.4f}")
        
    print(f"\n{'='*20} MINIATURE FL COMPLETED {'='*20}")
    torch.save(global_weights, "global_model_final.pth")
    print("[SERVER] Final global model saved as 'global_model_final.pth'")

    # Simpan local heads (Personalized)
    for name, head in client_heads.items():
        if head:
            torch.save(head.state_dict(), f"head_{name}.pth")
            print(f"[CLIENT] Saved local head for {name} as head_{name}.pth")