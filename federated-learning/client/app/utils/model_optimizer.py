import torch
import torch.quantization
from app.utils.mobilefacenet import MobileFaceNet
import os

def quantize_and_save(input_path="local_backbone.pth", output_path="local_backbone_quant.pt"):
    """
    Mengonversi model MobileFaceNet (.pth) ke TorchScript Quantized INT8 (.pt)
    untuk inferensi CPU yang lebih cepat.
    """
    if not os.path.exists(input_path):
        print(f"[OPTIMIZER] Input model {input_path} not found.")
        return False

    print(f"[OPTIMIZER] Loading model from {input_path}...")
    # Load Original Model
    model = MobileFaceNet(embedding_size=128) # Asumsi embedding size 128
    
    # Load state dict
    try:
        state_dict = torch.load(input_path, map_location="cpu")
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[OPTIMIZER] Error loading state dict: {e}")
        return False
        
    model.eval()
    
    print("[OPTIMIZER] Quantizing (Dynamic)...")
    dummy_input = torch.randn(1, 3, 112, 112)
    traced_model = torch.jit.trace(model, dummy_input)
    
    optimized_model = torch.jit.optimize_for_inference(traced_model)
    
    print(f"[OPTIMIZER] Saving optimized TorchScript to {output_path}...")
    torch.jit.save(optimized_model, output_path)
    print("[OPTIMIZER] Done.")
    return True

if __name__ == "__main__":
    quantize_and_save()
