import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from PIL import Image
from facenet_pytorch import MTCNN

# Import model unik xiaoccer (112x96)
from mobilefacenet import MobileFaceNet, ArcMarginProduct

# --- KONFIGURASI ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_GLOBAL_PATH = "global_model_final_fl.pth"
BN_COMBINED_PATH = "global_bn_combined.pth"
REGISTRY_PATH = "global_embedding_registry.pth"

# Detektor Wajah Presisi
mtcnn = MTCNN(keep_all=False, device=DEVICE)

@st.cache_resource
def load_global_embedding_system():
    # 1. Load Backbone Global (Hasil Kolaborasi FL)
    backbone = MobileFaceNet().to(DEVICE)
    if os.path.exists(MODEL_GLOBAL_PATH):
        backbone.load_state_dict(torch.load(MODEL_GLOBAL_PATH, map_location=DEVICE), strict=False)
    
    # 2. Load Combined BN (Statistik Rata-rata Lingkungan Client)
    if os.path.exists(BN_COMBINED_PATH):
        # Mengisi lapisan BN di backbone dengan statistik gabungan agar adaptif
        backbone.load_state_dict(torch.load(BN_COMBINED_PATH, map_location=DEVICE), strict=False)
    
    # 3. Load Global Embedding Registry (Database Semua Mahasiswa)
    if os.path.exists(REGISTRY_PATH):
        # Bobot identitas dari semua client yang sudah digabung
        global_embeddings = torch.load(REGISTRY_PATH, map_location=DEVICE)
        # Normalisasi agar siap dihitung jarak Cosine-nya
        global_embeddings = nn.functional.normalize(global_embeddings, p=2, dim=1)
    else:
        st.error("File Registry Global tidak ditemukan!")
        return None, None

    backbone.eval()
    return backbone, global_embeddings

def recognize_snapshot_universal(img_pil, backbone, global_embeddings, label_map, threshold):
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # 1. Deteksi Wajah
    faces_coords, _ = mtcnn.detect(img_pil)
    
    if faces_coords is not None:
        for (x1, y1, x2, y2) in faces_coords:
            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
            face_roi = img_cv2[y:y+h, x:x+w]
            if face_roi.size == 0: continue
            
            # 2. Preprocess 112x96 (xiaoccer format)
            face_resize = cv2.resize(face_roi, (96, 112))
            face_tensor = (face_resize.astype(np.float32) - 127.5) / 128.0
            face_tensor = torch.from_numpy(face_tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            # 3. Extract Global Feature Embedding
            with torch.no_grad():
                current_embedding = backbone(face_tensor)
                current_embedding = nn.functional.normalize(current_embedding, p=2, dim=1)
                
                # 4. Universal Matching (Cosine Similarity)
                # Membandingkan dengan SEMUA identitas dari Client 1 & Client 2 sekaligus
                similarities = torch.matmul(current_embedding, global_embeddings.t())
                max_similarity, predicted_idx = torch.max(similarities, 1)
                
            score = max_similarity.item()
            
            # 5. Keputusan Absensi
            if score > threshold:
                name = label_map.get(predicted_idx.item(), "ID Not Mapped")
                color = (0, 255, 0) # Hijau jika dikenal
            else:
                name = "Unknown"
                color = (0, 0, 255) # Merah jika asing
            
            cv2.rectangle(img_cv2, (x, y), (x+w, y+h), color, 3)
            cv2.putText(img_cv2, f"{name} Sim:{score:.2f}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
    return cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

def main():
    st.set_page_config(page_title="Universal Absensi FL", page_icon="🌎")
    st.title("🌎 Sistem Absensi Universal (Global Embedding)")
    st.write("Menggunakan gabungan statistik BN dan database identitas dari seluruh node Client.")

    st.sidebar.header("⚙️ Konfigurasi Global")
    threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.50, 0.05)
    
    # --- GABUNGKAN LABEL MAP ---
    # Karena kita pakai Registry Global, kita harus gabung daftar nama dari semua client
    full_nrp_list = []
    for cid in ["client1", "client2"]:
        path = f"datasets_processed/{cid}/train"
        if os.path.exists(path):
            names = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
            full_nrp_list.extend(names)
    
    label_map = {i: nrp for i, nrp in enumerate(full_nrp_list)}
    
    # Load Sistem Global
    backbone, global_embeddings = load_global_embedding_system()
    
    if backbone is not None:
        st.sidebar.success(f"✅ Sistem Global Aktif")
        st.sidebar.info(f"👥 Total Terdaftar: {len(full_nrp_list)} Mahasiswa")
        
        img_file = st.camera_input("Ambil foto untuk verifikasi universal")

        if img_file:
            img_pil = Image.open(img_file)
            with st.spinner('Mencocokkan dengan database global...'):
                result_img = recognize_snapshot_universal(img_pil, backbone, global_embeddings, label_map, threshold)
                # Gunakan parameter lebar terbaru Streamlit agar tidak ada warning
                st.image(result_img, caption="Hasil Verifikasi Global", use_container_width=True)
                
                st.info("💡 Mahasiswa dari Client 1 maupun Client 2 kini bisa absen di sini.")

if __name__ == "__main__":
    main()