import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from mobilefacenet import MobileFaceNet
import os

device = torch.device('cpu')
MODEL_PATH = 'test_model.pth'
DATASET_DIR = 'datasets_balanced' 

mtcnn = MTCNN(image_size=112, margin=20, keep_all=False, device=device, post_process=True)

model = MobileFaceNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize((112, 96)), # Menyesuaikan kernel (7, 6) 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print("[INFO] Membangun basis data referensi dari folder NRP...")
reference_embeddings = {}
for nrp in os.listdir(DATASET_DIR):
    nrp_path = os.path.join(DATASET_DIR, nrp)
    if os.path.isdir(nrp_path):
        img_name = os.listdir(nrp_path)[0] # Ambil 1 foto sebagai contoh
        img = Image.open(os.path.join(nrp_path, img_name)).convert('RGB')
        face = mtcnn(img)
        if face is not None:
            face_img = transforms.ToPILImage()(face * 0.5 + 0.5)
            face_tensor = preprocess(face_img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(face_tensor)
                reference_embeddings[nrp] = F.normalize(emb)

cap = cv2.VideoCapture(0) 

print("[START] Kamera aktif. Tekan 'q' untuk berhenti.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Deteksi Wajah dengan MTCNN
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Deteksi box saja untuk visualisasi
    boxes, _ = mtcnn.detect(img_pil)
    
    if boxes is not None:
        for box in boxes:
            # Crop & Preprocess Wajah
            face = mtcnn(img_pil)
            
            if face is not None:
                face_img = transforms.ToPILImage()(face * 0.5 + 0.5)
                face_tensor = preprocess(face_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    test_emb = F.normalize(model(face_tensor))
                
                # Cari NRP terdekat dengan Cosine Similarity 
                best_match = "Unknown"
                max_sim = -1
                
                for nrp, ref_emb in reference_embeddings.items():
                    sim = F.cosine_similarity(test_emb, ref_emb).item()
                    if sim > max_sim:
                        max_sim = sim
                        best_match = nrp
                
                # Threshold Pengenalan (0.65 - 0.7)
                label = f"{best_match} ({max_sim:.2f})" if max_sim > 0.65 else "Unknown"
                color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
                
                # Gambar Bounding Box
                x, y, w, h = box.astype(int)
                cv2.rectangle(frame, (x, y), (w, h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Live Face Attendance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()