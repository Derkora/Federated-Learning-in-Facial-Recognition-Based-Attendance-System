import torch
import os
import random
from PIL import Image
from app.utils.preprocessing import image_processor, DEVICE

def set_model_freeze(model, freeze_mode="early"):
    """
    Mengontrol pembekuan lapisan (Partial Freezing) pada MobileFaceNet.
    """
    for param in model.parameters():
        param.requires_grad = True

    if freeze_mode == "none":
        return

    if freeze_mode == "early":
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.dw_conv1.parameters():
            param.requires_grad = False
        for i in range(12):
            for param in model.blocks[i].parameters():
                param.requires_grad = False
                
    elif freeze_mode == "backbone":
        for param in model.parameters():
            param.requires_grad = False

from app.utils.logging import get_logger

def calibrate_bn(model, raw_data_path, num_samples=100):
    """
    Melakukan BN Adaptation (Kalibrasi Statistik BN) pada data lokal.
    Ini membuat model Centralized memiliki kemampuan adaptasi lingkungan yang setara dengan pFedFace.
    """
    logger = get_logger()
    logger.info(f"Memulai kalibrasi BN menggunakan data lokal di {raw_data_path}...")
    
    # 1. Kumpulkan sampel gambar lokal
    sample_paths = []
    root_dir = os.path.join(raw_data_path, "students")
    if os.path.exists(root_dir):
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_paths.append(os.path.join(root, f))
    
    if not sample_paths:
        logger.warn("Tidak ada data lokal untuk kalibrasi BN.")
        return

    # Batasi jumlah sampel agar tidak terlalu lama
    random.shuffle(sample_paths)
    sample_paths = sample_paths[:num_samples]

    # 2. Jalankan Forward Pass dalam mode .train()
    count = 0
    with torch.no_grad():
        for img_path in sample_paths:
            try:
                img_pil = Image.open(img_path).convert('RGB')
                face_img, _, _ = image_processor.detect_face(img_pil)
                if face_img:
                    input_tensor = image_processor.prepare_for_model(face_img)
                    if input_tensor is not None:
                        model.train() # Pastikan mode train aktif untuk update stats
                        _ = model(input_tensor.to(DEVICE))
                        count += 1
            except Exception as e:
                continue
    
    model.eval() # Kembalikan ke mode evaluasi permanen
    logger.success(f"Kalibrasi BN selesai menggunakan {count} wajah lokal.")

