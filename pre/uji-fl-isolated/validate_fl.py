import os, sys, copy, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# ─── Path relatif ke folder parent ───────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.join(BASE_DIR, "..", "..")
C1_DATA_DIR = os.path.join(REPO_ROOT, "datasets", "client1_data", "students")
C2_DATA_DIR = os.path.join(REPO_ROOT, "datasets", "client2_data", "students")
MODEL_V0    = os.path.join(REPO_ROOT, "federated-learning", "server", "app", "model", "global_model_v0.pth")
sys.path.insert(0, os.path.join(BASE_DIR, "..", "uji-fl"))  # reuse mobilefacenet.py

from mobilefacenet import MobileFaceNet, ArcMarginProduct

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROUNDS       = 5       # quick validation
LOCAL_EPOCHS = 3
LR           = 1e-4
BATCH_SIZE   = 8

print(f"[VALIDATE] Device: {DEVICE}")
print(f"[VALIDATE] Client-1 Data: {C1_DATA_DIR}")
print(f"[VALIDATE] Client-2 Data: {C2_DATA_DIR}")
print(f"[VALIDATE] Global Model v0: {MODEL_V0}")
print("=" * 60)

# ─── Transforms (sama persis dengan sistem utama) ────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((112, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
val_transform = transforms.Compose([
    transforms.Resize((112, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ─── Dataset ─────────────────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, root, label_map, transform):
        self.transform = transform
        self.samples   = []
        for folder in sorted(os.listdir(root)):
            nrp = folder.split("_")[0]
            if nrp not in label_map:
                continue
            label = label_map[nrp]
            fdir  = os.path.join(root, folder)
            for img_name in os.listdir(fdir):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(fdir, img_name), label, nrp))
    
    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx):
        path, label, nrp = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, nrp

# ─── Build Global Label Map from both clients ──────────────────────────────
def list_nrps(data_dir):
    return [f.split("_")[0] for f in sorted(os.listdir(data_dir))
            if os.path.isdir(os.path.join(data_dir, f))]

client1_nrps = list_nrps(C1_DATA_DIR)
client2_nrps = list_nrps(C2_DATA_DIR)

# Global label map: shared across both clients (same as our server logic)
global_nrps = sorted(set(client1_nrps + client2_nrps))
global_label_map = {nrp: i for i, nrp in enumerate(global_nrps)}

print(f"[SPLIT] Client-1: {len(client1_nrps)} students | Client-2: {len(client2_nrps)} students")
print(f"[SPLIT] Total global classes: {len(global_nrps)}")

# ─── pFedFace Filter (same as trainer._is_shared_param) ─────────────────────
def is_shared_param(name):
    n = name.lower()
    if any(x in n for x in ["bn", "running_", "num_batches_tracked"]):
        return False
    return any(x in n for x in ["weight", "bias"])

def get_shared_weights(state_dict):
    return {k: v for k, v in state_dict.items() if is_shared_param(k)}

def apply_shared_weights(backbone, shared_weights):
    sd = backbone.state_dict()
    sd.update(shared_weights)
    backbone.load_state_dict(sd, strict=False)

# ─── Load Initial Global Model ───────────────────────────────────────────────
def load_initial_backbone():
    m = MobileFaceNet().to(DEVICE)
    if os.path.exists(MODEL_V0):
        loaded = torch.load(MODEL_V0, map_location=DEVICE)
        if isinstance(loaded, dict):
            m.load_state_dict(loaded, strict=False)
            print(f"[INIT] Loaded global_model_v0 (state_dict)")
        else:
            print(f"[WARN] global_model_v0 is a list — using random init")
    else:
        print(f"[WARN] global_model_v0 not found at {MODEL_V0}, using random init")
    return m

# ─── Simulated Clients ───────────────────────────────────────────────────────
class SimulatedClient:
    def __init__(self, cid, nrp_list, data_dir):
        self.cid       = cid
        self.nrp_list  = nrp_list
        self.data_dir  = data_dir
        self.backbone  = load_initial_backbone()
        n_classes      = len(global_nrps)  # Use GLOBAL label space
        self.head      = ArcMarginProduct(128, n_classes).to(DEVICE)
        
        dataset = FaceDataset(data_dir, global_label_map, train_transform)
        
        self.dataset    = dataset
        self.loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        print(f"[{cid}] {len(nrp_list)} students, {len(dataset)} images")

    def train_local(self, server_shared, lr=LR):
        apply_shared_weights(self.backbone, server_shared)
        global_ref  = copy.deepcopy(self.backbone.state_dict())
        
        optimizer = optim.Adam([
            {"params": self.backbone.parameters()},
            {"params": self.head.parameters()},
        ], lr=lr)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.backbone.train(); self.head.train()
        for ep in range(LOCAL_EPOCHS):
            total_loss, correct, total = 0, 0, 0
            for imgs, labels, _ in self.loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                feats   = self.backbone(imgs)
                outputs = self.head(feats, labels)
                loss_cls = criterion(outputs, labels)
                # FedProx
                prox = sum((0.05/2)*torch.norm(p - global_ref[n])**2
                           for n, p in self.backbone.named_parameters())
                loss = loss_cls + prox
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
                total   += labels.size(0)
            acc = 100 * correct / max(total, 1)
            print(f"   [{self.cid}] Ep{ep+1} Loss={total_loss/max(len(self.loader),1):.4f} Acc={acc:.1f}%")
        
        return get_shared_weights(self.backbone.state_dict())

def fedavg(updates):
    agg = copy.deepcopy(updates[0])
    for k in agg:
        for i in range(1, len(updates)):
            agg[k] += updates[i][k]
        agg[k] = torch.div(agg[k], len(updates))
    return agg

# ─── Training Phase ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE TRAINING")
print("="*60)

c1 = SimulatedClient("Client-1", client1_nrps, C1_DATA_DIR)
c2 = SimulatedClient("Client-2", client2_nrps, C2_DATA_DIR)

# Initial shared weights from Server's blank backbone
server_shared = get_shared_weights(load_initial_backbone().state_dict())

for rnd in range(ROUNDS):
    lr = LR if rnd < 3 else LR/2
    print(f"\n--- ROUND {rnd+1}/{ROUNDS} (LR={lr}) ---")
    u1 = c1.train_local(server_shared, lr)
    u2 = c2.train_local(server_shared, lr)
    server_shared = fedavg([u1, u2])

print("\n[TRAINING] Complete!")

# ─── Global BN Phase ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE BN COMBINATION")
print("="*60)

all_bn_keys = [k for k in c1.backbone.state_dict().keys() if "bn" in k.lower()]
global_bn = {}
for k in all_bn_keys:
    v1 = c1.backbone.state_dict()[k]
    v2 = c2.backbone.state_dict()[k]
    if "num_batches" in k:
        global_bn[k] = v1
    else:
        global_bn[k] = torch.stack([v1, v2]).mean(0)

# Apply Global Model (shared conv + combined BN) into a reference backbone
global_backbone = load_initial_backbone()
apply_shared_weights(global_backbone, server_shared)
global_backbone.load_state_dict(global_bn, strict=False)
global_backbone.eval()
print(f"[BN] Combined {len(global_bn)} BN parameters from both clients.")

# ─── Registry Phase ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE REGISTRY GENERATION")
print("="*60)

registry = {}  # nrp -> centroid tensor (L2 normalized)

def extract_centroids(client, backbone):
    backbone.eval()
    emb_dict = {}
    val_tf = transforms.Compose([
        transforms.Resize((112, 96)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    for folder in sorted(os.listdir(client.data_dir)):
        nrp = folder.split("_")[0]
        fdir = os.path.join(client.data_dir, folder)
        if not os.path.isdir(fdir): continue
        imgs_tensors = []
        for img_name in os.listdir(fdir):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            try:
                img = Image.open(os.path.join(fdir, img_name)).convert("RGB")
                imgs_tensors.append(val_tf(img).unsqueeze(0))
            except: continue
        if not imgs_tensors: continue
        batch = torch.cat(imgs_tensors, dim=0).to(DEVICE)
        with torch.no_grad():
            embs = backbone(batch)
            embs = F.normalize(embs, p=2, dim=1)
            centroid = embs.mean(dim=0)
            centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)
        emb_dict[nrp] = centroid
        print(f"   [{client.cid}] {nrp}: {len(imgs_tensors)} images → centroid computed")
    return emb_dict

c1_centroids = extract_centroids(c1, global_backbone)
c2_centroids = extract_centroids(c2, global_backbone)

# Merge centroids from both clients
all_nrps_in_registry = set(c1_centroids.keys()) | set(c2_centroids.keys())
for nrp in sorted(all_nrps_in_registry):
    vecs = []
    if nrp in c1_centroids: vecs.append(c1_centroids[nrp])
    if nrp in c2_centroids: vecs.append(c2_centroids[nrp])
    avg = torch.stack(vecs).mean(dim=0)
    registry[nrp] = F.normalize(avg.unsqueeze(0), p=2, dim=1).squeeze(0)

print(f"\n[REGISTRY] {len(registry)} identities registered.")

# ─── Validation / Inference Simulation ───────────────────────────────────────
print("\n" + "="*60)
print("PHASE VALIDATION (Self-Similarity Test)")
print("="*60)
print("Each student's own photos → should score HIGH against their centroid.")
print("-"*60)

val_tf = transforms.Compose([
    transforms.Resize((112, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

results = []
import random
for data_dir, client_nrps in [(C1_DATA_DIR, client1_nrps), (C2_DATA_DIR, client2_nrps)]:
    for folder in sorted(os.listdir(data_dir)):
        nrp  = folder.split("_")[0]
        name = folder.split("_")[1] if "_" in folder else nrp
        if nrp not in registry:
            continue
        fdir = os.path.join(data_dir, folder)
        if not os.path.isdir(fdir): continue

        all_imgs = [f for f in os.listdir(fdir) if f.lower().endswith((".jpg",".png",".jpeg"))]
        if not all_imgs: continue
        
        sample_imgs = random.sample(all_imgs, min(5, len(all_imgs)))
        
        scores = []
        for img_name in sample_imgs:
            img = Image.open(os.path.join(fdir, img_name)).convert("RGB")
            t   = val_tf(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = global_backbone(t)
                emb = F.normalize(emb, p=2, dim=1)
            sim = F.linear(emb, registry[nrp].unsqueeze(0)).item()
            scores.append(sim)
        
        avg_sim = np.mean(scores)
        results.append((nrp, name, avg_sim))
        status = "✅" if avg_sim > 0.35 else ("⚠️" if avg_sim > 0.2 else "❌")
        print(f"  {status} {nrp} ({name}): {avg_sim:.4f} (samples: {len(scores)})")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
above_50 = sum(1 for _, _, s in results if s > 0.50)
above_35 = sum(1 for _, _, s in results if s > 0.35)
above_20 = sum(1 for _, _, s in results if s > 0.20)
avg_all  = np.mean([s for _, _, s in results])

print(f"Total Identities : {len(results)}")
print(f"Avg Similarity   : {avg_all:.4f}")
print(f"Score > 0.50     : {above_50}/{len(results)} ({100*above_50//max(len(results),1)}%)")
print(f"Score > 0.35     : {above_35}/{len(results)} ({100*above_35//max(len(results),1)}%)")
print(f"Score > 0.20     : {above_20}/{len(results)} ({100*above_20//max(len(results),1)}%)")

if avg_all > 0.40:
    print("\n🎯 ARCHITECTURE: VALID — Scores acceptable for FL Privacy Tax range")
elif avg_all > 0.20:
    print("\n⚠️  ARCHITECTURE: MARGINAL — Scores low, may need more rounds or data")
else:
    print("\n❌ ARCHITECTURE: BROKEN — Something fundamentally wrong (normalization/backbone)")

# Save artifacts
OUT = BASE_DIR
torch.save(registry, os.path.join(OUT, "test_registry.pth"))
torch.save(global_backbone.state_dict(), os.path.join(OUT, "test_backbone.pth"))
print(f"\n[SAVED] test_registry.pth, test_backbone.pth → {OUT}")
