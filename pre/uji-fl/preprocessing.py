import os
import cv2
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

# --- Konfigurasi Tetap ---
# MobileFaceNet xiaoccer menggunakan input 112x96 (Height x Width)
IMG_SIZE = (112, 96) 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inisialisasi Detektor Wajah
mtcnn = MTCNN(image_size=112, margin=20, device=DEVICE, post_process=True)

CLIENTS = [
    {"name": "client1", "path": "../../datasets/client1_data/students"},
    {"name": "client2", "path": "../../datasets/client2_data/students"}
]

OUTPUT_BASE_DIR = "datasets_processed"

def run_unified_preprocessing():
    print(f"[START] Memulai Preprocessing untuk {len(CLIENTS)} Client...")
    
    for client in CLIENTS:
        client_name = client["name"]
        raw_path = client["path"]
        
        # Folder output per client: datasets_processed/client1/train/NRP
        client_output_dir = os.path.join(OUTPUT_BASE_DIR, client_name)
        
        if not os.path.exists(raw_path):
            print(f"[SKIP] Folder data mentah {client_name} tidak ditemukan di: {raw_path}")
            continue

        print(f"\n[CLIENT] Mengolah {client_name}...")
        
        # Iterasi setiap folder NRP mahasiswa
        for nrp_folder in os.listdir(raw_path):
            path_nrp = os.path.join(raw_path, nrp_folder)
            if not os.path.isdir(path_nrp): continue
            
            # Buat folder train dan val untuk Local Validation
            for split in ['train', 'val']:
                os.makedirs(os.path.join(client_output_dir, split, nrp_folder), exist_ok=True)
            
            images = [f for f in os.listdir(path_nrp) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            # Pembagian data 80% Train, 20% Val secara lokal
            split_idx = int(len(images) * 0.8)
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            print(f"  > NRP {nrp_folder}: {len(images)} foto (Split: {len(train_images)} Train, {len(val_images)} Val)")

            # Proses setiap gambar
            for idx, img_name in enumerate(images):
                img_path = os.path.join(path_nrp, img_name)
                split = 'train' if img_name in train_images else 'val'
                save_path = os.path.join(client_output_dir, split, nrp_folder, img_name)
                
                try:
                    # Load gambar
                    img = Image.open(img_path).convert('RGB')
                    
                    # Deteksi dan Crop Wajah
                    face = mtcnn(img)
                    
                    if face is not None:
                        face_img = transforms.ToPILImage()(face * 0.5 + 0.5)
                        
                        face_img = face_img.resize((96, 112), Image.BILINEAR)
                        
                        # Simpan hasil
                        face_img.save(save_path)
                    else:
                        pass
                        
                except Exception as e:
                    print(f"    [!] Gagal memproses {img_name}: {e}")

    print("\n[DONE] Preprocessing selesai. Data siap di: ", os.path.abspath(OUTPUT_BASE_DIR))

if __name__ == "__main__":
    run_unified_preprocessing()