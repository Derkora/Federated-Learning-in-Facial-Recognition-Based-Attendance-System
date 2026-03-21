import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sqlalchemy.orm import Session
from app.db.models import UserLocal, Embedding
from app.utils.security import EmbeddingEncryptor
from app.utils.classifier import load_backbone, save_backbone, LocalClassifierHead # Import yang diubah
import requests
from app.config import config

MAX_USERS_CAPACITY = 100

class LocalTrainer:
    def __init__(self, db: Session):
        self.db = db
        self.encryptor = EmbeddingEncryptor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            lbl = self._fetch_global_label(u.nrp)
            if lbl is not None:
                user_map[u.user_id] = lbl
            else:
                print(f"[TRAIN LOCAL] Gagal sync label untuk {u.name}, skip.")
        
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
        
        backbone.train()
        head.train()
        
        # Gabungkan parameter untuk dioptimasi
        all_params = list(backbone.parameters()) + list(head.parameters())
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(all_params, lr=0.001)

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

        # Save Model Backbone Lokal 
        save_backbone(backbone, path="local_backbone.pth")
        
        return {
            "status": "success", 
            "users_count": len(user_map), 
            "samples": count_used,
            "final_loss": final_loss,
            "saved_model": "local_backbone.pth"
        }