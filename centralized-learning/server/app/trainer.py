import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from sqlalchemy.orm import Session
from .models import User, FaceEmbedding

# Classifier Sederhana (Head)
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=10):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)

def train_central_model(db: Session):
    # 1. Ambil Data
    users = db.query(User).all()
    if not users:
        return {"status": "error", "message": "Belum ada user terdaftar"}
    
    embeddings_data = db.query(FaceEmbedding).all()
    if len(embeddings_data) < 2:
        return {"status": "error", "message": "Data wajah terlalu sedikit"}

    # 2. Siapkan Data
    # Mapping ID Database -> Index Class (0, 1, 2...)
    user_id_to_idx = {u.id: i for i, u in enumerate(users)}
    idx_to_name = {i: u.name for i, u in enumerate(users)}
    
    X_train = []
    y_train = []
    
    for emb in embeddings_data:
        if emb.user_id in user_id_to_idx:
            X_train.append(emb.embedding)
            y_train.append(user_id_to_idx[emb.user_id])
            
    X_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    
    # 3. Setup Model Head
    num_classes = len(users)
    model = SimpleClassifier(input_dim=128, num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 4. Training Loop (Cepat karena hanya melatih head)
    epochs = 100
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        
    # 5. Simpan Model & Metadata
    torch.save(model.state_dict(), "models/classifier_head.pth")
    with open("models/class_mapping.pkl", "wb") as f:
        pickle.dump(idx_to_name, f)
        
    return {
        "status": "success", 
        "message": f"Training selesai. Loss: {loss.item():.4f}", 
        "users": num_classes
    }