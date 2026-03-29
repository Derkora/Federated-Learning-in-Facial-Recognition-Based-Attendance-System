import os
import cv2
import torch
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN
from torchvision import transforms

# --- Konfigurasi Tetap ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LIMIT_PER_STUDENT = 50 

# Inisialisasi Detektor Wajah Presisi (Keep_all=False untuk fokus 1 wajah utama)
mtcnn = MTCNN(image_size=112, margin=20, device=DEVICE, post_process=True)

CLIENTS = [
    {"name": "client1", "path": os.path.join(BASE_DIR, "..", "..", "datasets", "client1_data", "students")},
    {"name": "client2", "path": os.path.join(BASE_DIR, "..", "..", "datasets", "client2_data", "students")}
]

OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "datasets_processed")

def get_blur_score(image_path):
    """Menghitung skor ketajaman menggunakan Laplacian Variance."""
    img = cv2.imread(image_path)
    if img is None: return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def run_unified_preprocessing():
    print(f"[START] Memulai Preprocessing & Equalization (Top {LIMIT_PER_STUDENT} Laplace Variance)...")
    
    if not os.path.exists(OUTPUT_BASE_DIR):
        os.makedirs(OUTPUT_BASE_DIR)

    for client in CLIENTS:
        client_name = client["name"]
        raw_path = client["path"]
        
        if not os.path.exists(raw_path):
            print(f"[SKIP] Folder mentah {client_name} tidak ditemukan di: {raw_path}")
            continue

        print(f"\n[CLIENT] Mengolah {client_name} dari {raw_path}...")
        
        folders = sorted([f for f in os.listdir(raw_path) if os.path.isdir(os.path.join(raw_path, f))])
        
        for nrp_folder in tqdm(folders, desc=f"Processing {client_name}"):
            path_nrp = os.path.join(raw_path, nrp_folder)
            images = [f for f in os.listdir(path_nrp) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if not images: continue
            
            # --- SELEKSI KUALITAS (Laplace Variance) ---
            scored_images = []
            for img_name in images:
                path = os.path.join(path_nrp, img_name)
                score = get_blur_score(path)
                scored_images.append((img_name, score))
            
            # Urutkan dari yang paling tajam (score tertinggi)
            scored_images.sort(key=lambda x: x[1], reverse=True)
            top_images = [img[0] for img in scored_images[:LIMIT_PER_STUDENT]]
            
            # --- SPLIT DATA 80/20 ---
            # Dari 50 foto terbaik: 40 Train, 10 Val
            split_idx = int(len(top_images) * 0.8)
            train_list = top_images[:split_idx]
            val_list = top_images[split_idx:]
            
            # Buat folder output
            for split in ['train', 'val']:
                os.makedirs(os.path.join(OUTPUT_BASE_DIR, client_name, split, nrp_folder), exist_ok=True)

            # --- CROP & SAVE ---
            for img_name in top_images:
                img_path = os.path.join(path_nrp, img_name)
                split = 'train' if img_name in train_list else 'val'
                save_path = os.path.join(OUTPUT_BASE_DIR, client_name, split, nrp_folder, img_name)
                
                try:
                    # Load & Crop
                    img_pil = Image.open(img_path).convert('RGB')
                    face = mtcnn(img_pil)
                    
                    if face is not None:
                        # Convert Tensor ke PIL
                        face_img = transforms.ToPILImage()(face * 0.5 + 0.5)
                        # Resize paksa ke 112x96 (H x W)
                        face_img = face_img.resize((96, 112), Image.BILINEAR)
                        face_img.save(save_path)
                    else:
                        # Jika wajah tidak terdeteksi MTCNN, gunakan center crop sebagai fallback
                        w, h = img_pil.size
                        s = min(w, h)
                        img_pil = img_pil.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
                        img_pil = img_pil.resize((96, 112), Image.BILINEAR)
                        img_pil.save(save_path)
                        
                except Exception as e:
                    print(f"    [!] Error {img_name}: {e}")

    print("\n[DONE] Preprocessing selesai. Data tersimpan di: ", os.path.abspath(OUTPUT_BASE_DIR))

if __name__ == "__main__":
    run_unified_preprocessing()