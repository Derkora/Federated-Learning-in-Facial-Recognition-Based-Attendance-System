import torch.nn as nn
import torch
import os
import random
from PIL import Image
from app.utils.preprocessing import image_processor
from app.utils.logging import get_logger

def set_model_freeze(model, freeze_mode="early"):
    # Inisialisasi logger aktivitas pembekuan parameter
    logger = get_logger()
    
    # Aktifkan seluruh parameter model untuk dilatih kembali
    for param in model.parameters():
        param.requires_grad = True

    # Biarkan seluruh parameter aktif tanpa ada lapisan yang dibekukan
    if freeze_mode == "none":
        logger.info("Mode Freezing: 'none'. Seluruh backbone akan dilatih.")
        return

    # Bekukan parameter awal backbone untuk menghemat komputasi edge
    if freeze_mode == "early":
        logger.info("Mode Freezing: 'early'. Membekukan Conv1 hingga Stage 2 (conv_3).")
        
        # Nonaktifkan gradien untuk lapisan input awal MobileFaceNet
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.dw_conv1.parameters():
            param.requires_grad = False
        
        # Bekukan dua belas blok bottleneck awal pada backbone
        for i in range(12): 
            for param in model.blocks[i].parameters():
                param.requires_grad = False
                
    # Bekukan seluruh parameter backbone jika menggunakan model beku penuh
    elif freeze_mode == "backbone":
        logger.info("Mode Freezing: 'backbone'. Seluruh backbone MobileFaceNet dibekukan.")
        for param in model.parameters():
            param.requires_grad = False

def calibrate_bn(model, raw_data_path, device="cpu", num_samples=100, batch_size=16):
    """
    Melakukan BN Adaptation (Kalibrasi Statistik BN) pada data lokal dengan benar.
    Sangat krusial untuk pFedFace agar bobot global baru sinkron dengan statistik lokal.
    Menghindari error ValueError akibat batch_size=1 pada mode training.
    """
    logger = get_logger()
    
    logger.info(f"Memulai kalibrasi BN menggunakan data lokal di {raw_data_path}...")
    
    # 1. Kumpulkan sampel gambar lokal (Pencarian Bertingkat)
    sample_paths = []
    
    # Coba cari data terproses (aligned faces) terlebih dahulu untuk menghindari pemrosesan MTCNN
    processed_dir = os.path.join(os.path.dirname(raw_data_path), "data", "processed")
    if os.path.exists(processed_dir) and len(os.listdir(processed_dir)) > 0:
        root_dir = processed_dir
        logger.info(f"Menggunakan data terproses (aligned faces) di: {root_dir}")
    else:
        # Fallback ke raw data jika belum ada data terproses
        root_dir = os.path.join(raw_data_path, "students")
        if not os.path.exists(root_dir) or len(os.listdir(root_dir) if os.path.exists(root_dir) else []) == 0:
            root_dir = raw_data_path
            logger.info(f"Menggunakan data mentah di: {root_dir}")
        else:
            logger.info(f"Menggunakan data mentah siswa di: {root_dir}")

    if os.path.exists(root_dir):
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

    # 2. Jalankan Forward Pass dalam mode .train() secara berkelompok (Batched)
    model.train() # Pastikan mode train aktif untuk update stats
    
    tensors_batch = []
    count = 0
    
    with torch.no_grad():
        for img_path in sample_paths:
            try:
                img_pil = Image.open(img_path).convert('RGB')
                # Jika gambar sudah memiliki ukuran portrait 96x112 terproses, lewati proses deteksi wajah MTCNN
                if img_pil.size == (96, 112):
                    face_img = img_pil
                else:
                    face_img, _, _ = image_processor.detect_face(img_pil)
                    
                if face_img:
                    input_tensor = image_processor.prepare_for_model(face_img)
                    if input_tensor is not None:
                        tensors_batch.append(input_tensor.squeeze(0))
                        
                    if len(tensors_batch) == batch_size:
                        batch_tensor = torch.stack(tensors_batch).to(device)
                        _ = model(batch_tensor)
                        count += len(tensors_batch)
                        tensors_batch = []
            except Exception as e:
                logger.error(f"Error processing {img_path} during BN adaptation: {e}")
                continue
                
        # Sisa batch terakhir
        if len(tensors_batch) > 0:
            try:
                # Jika sisa batch hanya 1, duplikasi agar ukurannya 2 (mencegah error BN batch size 1)
                if len(tensors_batch) == 1:
                    tensors_batch.append(tensors_batch[0])
                batch_tensor = torch.stack(tensors_batch).to(device)
                _ = model(batch_tensor)
                count += len(tensors_batch)
            except Exception:
                pass
    
    model.eval() # Kembalikan ke mode evaluasi untuk inferensi
    logger.success(f"Kalibrasi BN selesai menggunakan {count} wajah lokal.")

