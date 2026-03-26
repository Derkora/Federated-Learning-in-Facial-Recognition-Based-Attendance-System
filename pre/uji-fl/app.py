import streamlit as st
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from mobilefacenet import MobileFaceNet, ArcMarginProduct
from facenet_pytorch import MTCNN
import os

# --- 1. CONFIGURATION & MODELS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_mtcnn():
    return MTCNN(keep_all=True, device=DEVICE)

@st.cache_resource
def load_backbone(model_path):
    model = MobileFaceNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model.to(DEVICE)
    model.eval()
    return model

def get_client_info(client_name):
    """Ambil class names (NRP) dari folder processed."""
    processed_dir = f"processed_{client_name}"
    if not os.path.exists(processed_dir):
        return [], None
    
    classes = sorted([f for f in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, f))])
    num_classes = len(classes)
    
    head_path = f"head_{client_name}.pth"
    if not os.path.exists(head_path):
        return classes, None
        
    head = ArcMarginProduct(128, num_classes).to(DEVICE)
    head.load_state_dict(torch.load(head_path, map_location=DEVICE), strict=False)
    head.eval()
    return classes, head

# --- 2. STREAMLIT UI ---
st.set_page_config(page_title="Miniature FL Face Recognition", layout="wide")
st.title("📸 Real-time Face Recognition (Miniature FL)")

st.sidebar.header("Settings")
client_choice = st.sidebar.selectbox("Pilih Model Client:", ["client1", "client2"])

# Load Resources
mtcnn = load_mtcnn()
backbone = load_backbone("global_model_final.pth")
class_names, head = get_client_info(client_choice)

if not head:
    st.error(f"Penting: File head_{client_choice}.pth tidak ditemukan. Pastikan sudah jalankan training.py!")
else:
    st.sidebar.success(f"Model {client_choice} Loaded! ({len(class_names)} Classes)")

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((112, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- 3. CAMERA INPUT ---
img_file_buffer = st.camera_input("Ambil Foto untuk Recognize")

if img_file_buffer is not None and head is not None:
    # Convert buffer to PIL
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    
    # Detect Faces
    boxes, _ = mtcnn.detect(image)
    
    if boxes is not None:
        for box in boxes:
            # Draw Bounding Box (OpenCV style)
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Crop Face for Recognition
            face = image.crop((x1, y1, x2, y2))
            face_tensor = transform(face).unsqueeze(0).to(DEVICE)
            
            # Recognition
            with torch.no_grad():
                embedding = backbone(face_tensor)
                # Gunakan labels dummy (0) untuk ArcMarginProduct forward 
                # (biasanya kita butuh kosinus kemiripan, tapi head(embedding, labels) merata-rata)
                # Untuk inference murni, kita bisa hitung cosine similarity manual 
                # atau ambil argmax dari head output
                output = head(embedding, torch.tensor([0]).to(DEVICE)) # Label dummy
                prob = torch.softmax(output, dim=1)
                score, idx = torch.max(prob, 1)
                
                name = class_names[idx.item()]
                confidence = score.item() * 100
                
                # Label text
                label_text = f"{name} ({confidence:.1f}%)"
                cv2.putText(img_array, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        st.image(img_array, caption="Hasil Deteksi", use_container_width=True)
    else:
        st.warning("Wajah tidak terdeteksi. Coba lagi Ndan!")

st.info("💡 Tips: Pastikan pencahayaan bagus dan wajah menghadap kamera.")
