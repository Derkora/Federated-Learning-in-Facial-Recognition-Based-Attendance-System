import os
import torch
import torch.onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from app.utils.mobilefacenet import MobileFaceNet

def export_backbone_to_onnx(torch_path, onnx_path, quantized_path=None):
    """
    Ekspor model MobileFaceNet ke format ONNX dan lakukan kuantisasi.
    Memastikan inferensi di perangkat edge lebih efisien dalam RAM dan penyimpanan.
    """
    print(f"[*] Mengekspor model CL dari {torch_path} ke {onnx_path}...")
    
    device = torch.device('cpu')
    model = MobileFaceNet()
    
    if os.path.exists(torch_path):
        try:
            model.load_state_dict(torch.load(torch_path, map_location=device), strict=False)
        except Exception as e:
            print(f"[!] Gagal membuat model dari {torch_path}: {e}")
            return
    
    model.eval()
    
   # Input dummy untuk penentuan shape (Batches, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 112, 96, device=device)
    
    # Ekspor ke ONNX (Gunakan Opset 11 agar lebih stabil untuk kuantisasi dinamis)
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
    print(f"[OK] Model ONNX CL berhasil disimpan.")
    
    if quantized_path:
        print(f"[*] Melakukan kuantisasi dinamis (uint8) ke {quantized_path}...")
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QUInt8 
        )
        print(f"[OK] Model ONNX CL terkuantisasi berhasil disimpan.")

if __name__ == "__main__":
    # Path default untuk data model di CL Client
    base_dir = "app/data"
    torch_model = os.path.join(base_dir, "models", "backbone.pth")
    onnx_model = os.path.join(base_dir, "models", "backbone.onnx")
    q_model = os.path.join(base_dir, "models", "backbone_quantized.onnx")
    
    if os.path.exists(torch_model):
        export_backbone_to_onnx(torch_model, onnx_model, q_model)
    else:
        print("[!] File backbone.pth tidak ditemukan.")
