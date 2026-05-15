import torch.nn as nn
import torch
import os
import random
from PIL import Image
from .preprocessing import image_processor

from .logging import get_logger

def set_model_freeze(model, freeze_mode="early"):
    """
    Mengontrol pembekuan lapisan (Partial Freezing) pada MobileFaceNet.
    
    Args:
        model (nn.Module): Instance dari MobileFaceNet.
        freeze_mode (str): Opsi mode pembekuan:
            - "none": Tidak ada yang dibekukan (semua parameter dilatih).
            - "early": Membekukan Conv1 sampai bottleneck stage 2 (conv_3).
            - "backbone": Membekukan seluruh backbone MobileFaceNet.
    """
    logger = get_logger()
    # 1. Aktifkan semua parameter terlebih dahulu (Reset)
    for param in model.parameters():
        param.requires_grad = True

    if freeze_mode == "none":
        logger.info("Mode Freezing: 'none'. Seluruh backbone akan dilatih.")
        return

    if freeze_mode == "early":
        logger.info("Mode Freezing: 'early'. Membekukan Conv1 hingga Stage 2 (conv_3).")
        # Bekukan Conv1 dan dw_conv1 (Awal)
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.dw_conv1.parameters():
            param.requires_grad = False
            
        # Bekukan blocks (Bottlenecks)
        # Berdasarkan Mobilefacenet_bottleneck_setting:
        # Index 0-4: Stage 1 (conv_23)
        # Index 5-11: Stage 2 (conv_34) -> Akhir dari "early layers" (conv_3)
        # Index 12-14: Stage 3 (conv_45) -> Ini tetap AKTIF (Late layers)
        
        for i in range(12): # Bekukan 12 bottleneck pertama (0 s/d 11)
            for param in model.blocks[i].parameters():
                param.requires_grad = False
                
    elif freeze_mode == "backbone":
        logger.info("Mode Freezing: 'backbone'. Seluruh backbone MobileFaceNet dibekukan.")
        for param in model.parameters():
            param.requires_grad = False

def calibrate_bn(model, raw_data_path, device="cpu", num_samples=100):
    """
    Melakukan BN Adaptation (Kalibrasi Statistik BN) pada data lokal.
    Sangat krusial untuk pFedFace agar bobot global baru sinkron dengan statistik lokal.
    """
    logger = get_logger()
    
    logger.info(f"Memulai kalibrasi BN menggunakan data lokal di {raw_data_path}...")
    
    # 1. Kumpulkan sampel gambar lokal (Pencarian Bertingkat)
    sample_paths = []
    
    # Prioritas 1: Folder Students (Raw)
    root_dir = os.path.join(raw_data_path, "students")
    
    # Prioritas 2: Root Raw Data
    if not os.path.exists(root_dir):
        root_dir = raw_data_path
        
    # Prioritas 3: Processed Data (Paling Mungkin Ada di Docker Client)
    if not os.path.exists(root_dir) or len(os.listdir(root_dir) if os.path.exists(root_dir) else []) == 0:
        processed_dir = os.path.join(os.path.dirname(raw_data_path), "data", "processed")
        if os.path.exists(processed_dir):
            root_dir = processed_dir
            logger.info(f"Menggunakan fallback data processed di: {root_dir}")
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    sample_paths.append(os.path.join(root, f))
    
    if not sample_paths:
        logger.info("[SKIP] Tidak ada data lokal untuk kalibrasi BN.")
        return

    # Batasi jumlah sampel agar tidak terlalu lama (Edge device performance)
    random.shuffle(sample_paths)
    sample_paths = sample_paths[:num_samples]

    # 2. Jalankan Forward Pass dalam mode .train()
    count = 0
    model.train() # Pastikan mode train aktif untuk update stats
    
    with torch.no_grad():
        for img_path in sample_paths:
            try:
                img_pil = Image.open(img_path).convert('RGB')
                face_img, _, _ = image_processor.detect_face(img_pil)
                if face_img:
                    input_tensor = image_processor.prepare_for_model(face_img)
                    if input_tensor is not None:
                        _ = model(input_tensor.to(device))
                        count += 1
            except Exception as e:
                continue
    
    model.eval() # Kembalikan ke mode evaluasi untuk inferensi
    logger.success(f"Kalibrasi BN selesai menggunakan {count} wajah lokal.")
