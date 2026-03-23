import torch
import torch.nn as nn
import os
from .mobilefacenet import MobileFaceNet

class LocalClassifierHead(nn.Module):
    def __init__(self, input_dim=128, num_classes=10, s=16.0, dropout=0.5):
        super(LocalClassifierHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes, bias=False) # ArcMargin has no bias
        self.s = s
        self.dropout = nn.Dropout(p=dropout) # Penting untuk cegah overfit saat training
        
    def forward(self, x):
        # Align with ArcMargin logic for inference:
        if x.dim() == 1: x = x.unsqueeze(0)
        
        # Apply Dropout (aktif saat mode training)
        x = self.dropout(x)
        
        # Normalize features & weights
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        w_norm = torch.nn.functional.normalize(self.fc.weight, p=2, dim=1)
        
        # Logits = s * cos(theta)
        logits = torch.matmul(x_norm, w_norm.t())
        return logits * self.s

def save_local_model(backbone, head, backbone_path="local_backbone.pth", head_path="local_head.pth"):
    """Simpan backbone dan head secara terpisah."""
    torch.save(backbone.state_dict(), backbone_path)
    torch.save(head.state_dict(), head_path)
    print(f"[MODEL] Saved models to {backbone_path} and {head_path}")

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
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"[MODEL] Loaded backbone: {path}")
        else:
            print("[MODEL] No existing backbone found.")
    except Exception as e:
        print(f"[MODEL ERROR] Backbone load fail: {e}")
        
    return model

def build_local_model(num_classes, embedding_size=128, backbone_path="local_backbone.pth", head_path="local_head.pth", use_quantized=False):
    # 1. Load Backbone
    backbone = load_backbone(path=backbone_path, embedding_size=embedding_size, use_quantized=use_quantized)
    
    # 2. Inisialisasi Head (Linear)
    head = LocalClassifierHead(input_dim=embedding_size, num_classes=num_classes)
    
    # 3. Load Head Weights if exists
    try:
        if os.path.exists(head_path):
            state_dict = torch.load(head_path, map_location="cpu")
            # Handle possible ArcMarginProduct -> Linear weight mapping
            if 'weight' in state_dict:
                head.fc.weight.data.copy_(state_dict['weight'].data)
                # If bias exists in state_dict but not on head, or vice versa
                if 'bias' in state_dict and head.fc.bias is not None:
                    head.fc.bias.data.copy_(state_dict['bias'].data)
                print(f"[MODEL] Loaded head weights from {head_path}")
            else:
                head.load_state_dict(state_dict)
                print(f"[MODEL] Loaded head: {head_path}")
    except Exception as e:
        print(f"[MODEL ERROR] Head load fail: {e}")
    
    return backbone, head