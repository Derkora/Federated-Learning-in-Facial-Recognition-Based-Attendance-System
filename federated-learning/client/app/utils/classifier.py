import torch
import torch.nn as nn
import os
from .mobilefacenet import MobileFaceNet

class LocalClassifierHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=10):
        super(LocalClassifierHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)
  
def save_backbone(model, path="local_backbone.pth"):
    torch.save(model.state_dict(), path)  
    
def load_backbone(path="local_backbone.pth", embedding_size=128, use_quantized=False):
    # Cek apakah versi quantized tersedia jika diminta
    quant_path = path.replace(".pth", "_quant.pt")
    if use_quantized and os.path.exists(quant_path):
        try:
            print(f"[MODEL] Loading Quantized TorchScript from {quant_path}...")
            model = torch.jit.load(quant_path)
            model.eval()
            return model
        except Exception as e:
            print(f"[MODEL ERROR] Failed to load quantized model: {e}")
            print("[MODEL] Fallback to standard .pth model.")

    # Model MobileFaceNet 
    model = MobileFaceNet(embedding_size=embedding_size)
    try:
        # Load bobot yang sudah di-trained atau di-agregasi FL
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"[MODEL] Loaded MobileFaceNet backbone from {path}")
        else:
             print("[MODEL] No existing backbone found. Creating a new randomly initialized one.")
    except Exception as e:
        print(f"[MODEL ERROR] Failed to load state dict: {e}")
        print("[MODEL] Using randomly initialized model instead.")
        
    return model

def build_local_model(num_classes, embedding_size=128, backbone_path="local_backbone.pth", use_quantized=False):
    # Load Backbone
    backbone = load_backbone(path=backbone_path, embedding_size=embedding_size, use_quantized=use_quantized)
    # Inisialisasi Head Lokal
    head = LocalClassifierHead(input_dim=embedding_size, num_classes=num_classes)
    
    return backbone, head