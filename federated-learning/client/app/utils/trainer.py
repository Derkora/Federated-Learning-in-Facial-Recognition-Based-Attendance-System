import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sqlalchemy.orm import Session
from app.db.models import UserLocal, Embedding
from app.utils.security import EmbeddingEncryptor
from app.utils.classifier import load_backbone, save_local_model, LocalClassifierHead 
import requests
from app.config import config

MAX_USERS_CAPACITY = 100

class LocalTrainer:
    def __init__(self, db: Session):
        self.db = db
        self.encryptor = EmbeddingEncryptor()
        self.device = torch.device('cpu')

    def _fetch_global_label(self, nrp): 
        try:
            resp = requests.post(f"{config.get_server_url()}/api/training/get_label", json={"nrp": nrp}, timeout=2)
            if resp.status_code == 200:
                return resp.json()["label"]
        except:
            return None
        return None
    
    def train_local(self, epochs=20, lr=0.01):
        users = self.db.query(UserLocal).all()
        if not users:
            return {"status": "error", "reason": "No users found"}

        user_map = {}
        for u in users:
            # Update: Prioritize existing global_label from registration/sync
            if u.global_label is not None:
                user_map[u.user_id] = u.global_label
            else:
                # Fallback to server sync
                lbl = self._fetch_global_label(u.nrp)
                if lbl is not None:
                    user_map[u.user_id] = lbl
                    u.global_label = lbl # Save for next time
                else:
                    print(f"[TRAIN LOCAL] Gagal sync label untuk {u.name}, skip.")
        if users: self.db.commit()
        
        X_train = [] # Data Embedding
        y_train = [] # Label (0, 1, 2...)

        embeddings = self.db.query(Embedding).all()
        
        count_used = 0
        for emb in embeddings:
            if emb.user_id not in user_map: 
                continue 
            
            try:
                vec_numpy = self.encryptor.decrypt_embedding(emb.encrypted_embedding, emb.iv)
                
                X_train.append(vec_numpy)
                y_train.append(user_map[emb.user_id])
                count_used += 1
            except Exception as e:
                print(f"[TRAIN] Gagal decrypt embedding {emb.embedding_id}: {e}")
                continue

        if not X_train:
            return {"status": "error", "reason": "No valid embeddings found"}

        X_tensor = torch.tensor(np.array(X_train), dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)

        # Setup Model: Load Backbone dan buat Head Lokal
        backbone = load_backbone(path="local_backbone.pth", embedding_size=128).to(self.device)
        head = LocalClassifierHead(num_classes=MAX_USERS_CAPACITY).to(self.device)
        
        # FREEZE & THAW STRATEGY:
        # Bekukan lapisan fitur universal (conv1 s/d blocks) agar tidak rusak oleh data lokal.
        # Thaw (Cairkan) hanya lapisan terakhir untuk penajaman embedding lokal.
        for name, param in backbone.named_parameters():
            if "linear7" in name or "linear1" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        backbone.train()
        head.train()
        
        # Hanya ambil parameter yang requires_grad untuk optimizer
        trainable_params = [p for p in backbone.parameters() if p.requires_grad]
        trainable_params += list(head.parameters())
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(trainable_params, lr=0.001)

        # Training Loop
        print(f"[TRAIN] Start training on {count_used} images for {len(user_map)} users...")
        final_loss = 0.0
        for epoch in range(epochs):
            optimizer.zero_grad()
            # Ekstraksi embedding menggunakan backbone
            embeddings = backbone(X_tensor) 
            # Klasifikasi menggunakan head lokal
            outputs = head(embeddings) 
            
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            final_loss = loss.item()
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss {final_loss:.4f}")

        # Save Model Backbone + Head Lokal 
        save_local_model(backbone, head)
        
        return {
            "status": "success", 
            "users_count": len(user_map), 
            "samples": count_used,
            "final_loss": final_loss,
            "saved_model": "local_backbone.pth"
        }