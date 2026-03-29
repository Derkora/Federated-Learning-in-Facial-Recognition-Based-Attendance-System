import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
from PIL import Image
from facenet_pytorch import MTCNN
from mobilefacenet import MobileFaceNet
from collections import deque

# --- KONFIGURASI SISTEM (Path dinamis agar bisa di-run dari mana saja) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_GLOBAL_PATH = os.path.join(BASE_DIR, "global_model_final_fl.pth")
BN_COMBINED_PATH = os.path.join(BASE_DIR, "global_bn_combined.pth")
REGISTRY_PATH = os.path.join(BASE_DIR, "global_embedding_registry.pth")
THRESHOLD = 0.35  
VOTE_SIZE = 10    

# 1. Inisialisasi Detektor (Presisi Tinggi)
mtcnn = MTCNN(keep_all=False, device=DEVICE, post_process=True)

def load_universal_system():
    print("[SYSTEM] Memuat arsitektur MobileFaceNet (112x96)...")
    backbone = MobileFaceNet().to(DEVICE)
    
    # Load Backbone Global
    if os.path.exists(MODEL_GLOBAL_PATH):
        backbone.load_state_dict(torch.load(MODEL_GLOBAL_PATH, map_location=DEVICE), strict=False)
        print(f"[OK] Backbone loaded from {MODEL_GLOBAL_PATH}")
    
    # Load Combined BN (Penting untuk Akurasi Global)
    if os.path.exists(BN_COMBINED_PATH):
        backbone.load_state_dict(torch.load(BN_COMBINED_PATH, map_location=DEVICE), strict=False)
        print(f"[OK] Combined BN loaded from {BN_COMBINED_PATH}")
    
    # Load Centroid Registry (Kumpulan Identitas Mahasiswa)
    if os.path.exists(REGISTRY_PATH):
        global_centroids = torch.load(REGISTRY_PATH, map_location=DEVICE)
        # Normalisasi L2 untuk memastikan akurasi Cosine Similarity
        global_centroids = nn.functional.normalize(global_centroids, p=2, dim=1)
        print(f"[OK] Registry Centroids loaded: {global_centroids.shape[0]} identities")
    else:
        print("[ERROR] File Registry Global tidak ditemukan!")
        return None, None

    backbone.eval()
    return backbone, global_centroids

def get_label_map():
    full_nrp_list = []
    # Otomatis deteksi dari folder datasets_processed (path absolut relatif ke script)
    for cid in ["client1", "client2"]:
        path = os.path.join(BASE_DIR, "datasets_processed", cid, "train")
        if os.path.exists(path):
            names = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
            full_nrp_list.extend(names)
    
    label_map = {i: nrp for i, nrp in enumerate(full_nrp_list)}
    return label_map

def main():
    backbone, global_centroids = load_universal_system()
    if backbone is None: return
    
    label_map = get_label_map()
    print(f"[INFO] Monitoring {len(label_map)} Mahasiswa secara Live.")
    
    # Buffer untuk Temporal Voting
    embedding_buffer = deque(maxlen=VOTE_SIZE)
    current_label = "Unknown"
    current_score = 0.0
    
    cap = cv2.VideoCapture(0)
    print("\n[START] Kamera Aktif. Fokuskan wajah ke kamera.")
    print("[HINT] Tekan 'q' untuk keluar.\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Simpan frame asli untuk visualisasi
        display_frame = frame.copy()
        
        # Deteksi Wajah (MTCNN)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Deteksi Box
        boxes, _ = mtcnn.detect(img_pil)
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                
                # Crop dan Preprocess (Normalisasi L2)
                face = mtcnn(img_pil)
                if face is not None:
                    # Convert Tensor [-1, 1] ke PIL Image
                    face_img = (face.permute(1, 2, 0).numpy() * 128.0 + 127.5).astype(np.uint8)
                    face_resize = cv2.resize(face_img, (96, 112))
                    
                    # Convert ke Tensor Backbone (112x96)
                    face_tensor = (face_resize.astype(np.float32) - 127.5) / 128.0
                    face_tensor = torch.from_numpy(face_tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        # Extract Embedding Lokal
                        emb = backbone(face_tensor)
                        emb = nn.functional.normalize(emb, p=2, dim=1)
                        
                        # --- TEMPORAL VOTING ---
                        embedding_buffer.append(emb)
                        
                        # Hanya hitung jika buffer sudah cukup (mengurangi jitter)
                        if len(embedding_buffer) >= VOTE_SIZE:
                            # 1. Rata-ratakan embedding (Temporal Centroiding)
                            voted_emb = torch.stack(list(embedding_buffer)).mean(0)
                            voted_emb = nn.functional.normalize(voted_emb, p=2, dim=1) # Norm ulang setelah rata-rata
                            
                            # 2. Match ke Centroid Registry (Global)
                            similarities = torch.matmul(voted_emb, global_centroids.t())
                            max_similarity, predicted_idx = torch.max(similarities, 1)
                            
                            current_score = max_similarity.item()
                            if current_score > THRESHOLD:
                                current_label = label_map.get(predicted_idx.item(), "ID Unmapped")
                            else:
                                current_label = "Unknown"
                        
                        # Visualisasi
                        color = (0, 255, 0) if current_label != "Unknown" else (0, 0, 255)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Header Progress Buffer
                        prog = int((len(embedding_buffer) / VOTE_SIZE) * 100)
                        cv2.putText(display_frame, f"Stability: {prog}%", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Label Identitas
                        label_text = f"{current_label} ({current_score:.2f})"
                        cv2.putText(display_frame, label_text, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            # Jika wajah hilang, kosongkan buffer agar tidak salah deteksi (forgetting)
            embedding_buffer.clear()
            current_label = "Unknown"
            current_score = 0.0

        cv2.imshow('Universal FL Attendance - Temporal Voting', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[EXIT] Program dihentikan.")

if __name__ == "__main__":
    main()