import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import numpy as np

def export_backbone_to_onnx(model, save_dir, device="cpu"):
    """
    Ekspor model MobileFaceNet ke format ONNX dan lakukan kuantisasi QUInt8.
    Dapat menerima objek model langsung atau string path ke file .pth.
    """
    os.makedirs(save_dir, exist_ok=True)
    onnx_path = os.path.join(save_dir, "backbone.onnx")
    quantized_path = os.path.join(save_dir, "backbone_quantized.onnx")

    # Optimasi RAM: Jika model adalah path, muat ke CPU, ekspor, lalu hapus.
    from app.utils.mobilefacenet import MobileFaceNet
    
    if isinstance(model, str):
        print(f"[ONNX] Loading model from disk for export: {model}")
        torch_path = model
        model = MobileFaceNet()
        if os.path.exists(torch_path):
            try:
                model.load_state_dict(torch.load(torch_path, map_location="cpu"), strict=False)
            except Exception as e:
                print(f"[ONNX ERROR] Failed to load {torch_path}: {e}")
                return None
    
    model.eval()
    
    dummy_input = torch.randn(1, 3, 112, 96).to(device)

    # Ekspor Standar
    print(f"[ONNX] Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,  
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # Kuantisasi Dinamis (Sangat efektif untuk CPU Edge)
    print(f"[ONNX] Quantizing to {quantized_path}...")
    try:
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QUInt8 
        )
        print("[ONNX] Quantization complete.")
        result = quantized_path
    except Exception as e:
        print(f"[ONNX ERROR] Quantization failed: {e}")
        result = onnx_path
    
    # Bebaskan model PyTorch dari RAM setelah ekspor selesai
    import gc
    del model
    gc.collect()
    
    return result
