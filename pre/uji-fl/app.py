import streamlit as st
import cv2, torch, os
import numpy as np
from PIL import Image
from mobilefacenet import MobileFaceNet, ArcMarginProduct
from facenet_pytorch import MTCNN
import torchvision.transforms as T

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Kalibrasi Inferensi Final ---
# Kita gunakan Temperature 1.1 agar distribusi probabilitas lebih tajam
TEMPERATURE = 1.1 
# Threshold dinaikkan sedikit ke 0.55 karena model sudah lebih akurat
THRESHOLD = 0.55  

st.set_page_config(page_title="FL Face Attendance", page_icon="📸", layout="centered")
st.title("📸 Sistem Absensi Federated Learning")
st.info(f"Device: {DEVICE} | Mode: Global Consensus (Backbone + Head)")

# Path model hasil training terbaru
backbone_path = "global_model_final.pth"
head_path = "global_head_final.pth"
labels_path = "global_labels.pth"

if not os.path.exists(backbone_path) or not os.path.exists(head_path) or not os.path.exists(labels_path):
    st.error("Model Global atau Labels tidak ditemukan, Ndan!")
    st.stop()

# 1. LOAD GLOBAL LABELS
GLOBAL_LABELS = torch.load(labels_path)
NUM_CLASSES = len(GLOBAL_LABELS)

# 2. LOAD MODELS
@st.cache_resource
def load_models():
    mtcnn = MTCNN(keep_all=True, device=DEVICE)
    model = MobileFaceNet().to(DEVICE)
    model.load_state_dict(torch.load(backbone_path, map_location=DEVICE))
    model.eval()
    
    head = ArcMarginProduct(128, NUM_CLASSES).to(DEVICE)
    head.load_state_dict(torch.load(head_path, map_location=DEVICE))
    head.eval()
    return mtcnn, model, head

mtcnn, model, head = load_models()

img_file = st.camera_input("Ambil Foto Absen")

if img_file:
    img = Image.open(img_file).convert('RGB')
    boxes, _ = mtcnn.detect(img)
    
    if boxes is not None:
        img_draw = np.array(img)
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            # Preprocessing 96x112 sesuai standar training
            face = img.crop((x1, y1, x2, y2)).resize((96, 112))
            face_tensor = T.Compose([
                T.ToTensor(), 
                T.Normalize([0.5]*3, [0.5]*3)
            ])(face).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                emb = model(face_tensor)
                # Prediksi menggunakan Global Head
                logits = head(emb, torch.tensor([0]).to(DEVICE))
                
                # Softmax dengan Temperature Scaling
                prob = torch.softmax(logits / TEMPERATURE, dim=1)
                score, idx = torch.max(prob, 1)
                
                score_val = score.item()
                predicted_name = GLOBAL_LABELS[idx.item()]
                
                # Penentuan Hasil
                if score_val > THRESHOLD:
                    display_text = f"{predicted_name} ({score_val:.1%})"
                    color = (0, 255, 0) # Hijau
                else:
                    display_text = f"Unknown ({score_val:.1%})"
                    color = (255, 0, 0) # Merah
                
                # Drawing
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img_draw, display_text, (x1, y1-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        st.image(img_draw, caption="Hasil Analisis Wajah")
    else:
        st.warning("Wajah tidak terdeteksi")