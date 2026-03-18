import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from .mobilefacenet import MobileFaceNet
import io
import os
import torch.nn.functional as F 

class FaceEngine:
    def __init__(self):
        self.device = torch.device('cpu')
        print(f"[ENGINE] Running on {self.device}")

        # MTCNN tetap output 112x112 untuk deteksi awal
        self.mtcnn = MTCNN(
            image_size=112, 
            margin=0, 
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )

        # Inisialisasi Backbone MobileFaceNet
        self.backbone = MobileFaceNet().to(self.device)
        
        # Load Bobot
        model_path = "/app/global_model_v0.pth"
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                
                # Bersihkan prefix 'module.' jika ada (sisa training parallel)
                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                
                # Load dengan strict=True untuk memastikan struktur benar-benar cocok
                self.backbone.load_state_dict(new_state_dict, strict=True)
                print(f"[ENGINE] Sukses memuat bobot pretrained dari {model_path}")
            except Exception as e:
                print(f"[ENGINE] ERROR memuat bobot: {e}")
        else:
            print(f"[ENGINE] WARNING: File {model_path} tidak ditemukan!")

        self.backbone.eval()

    def get_embedding(self, image_bytes):
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            face_tensor, prob = self.mtcnn(img, return_prob=True)
            
            if face_tensor is None or prob < 0.90:
                return None, "Wajah tidak terdeteksi"

            face_batch = face_tensor.unsqueeze(0).to(self.device)
            face_resized = F.interpolate(
                face_batch, 
                size=(112, 96), 
                mode='bilinear', 
                align_corners=False
            )

            with torch.no_grad():
                embedding = self.backbone(face_resized)

            emb_numpy = embedding.cpu().numpy()[0]
            norm = np.linalg.norm(emb_numpy)
            if norm != 0:
                emb_numpy = emb_numpy / norm
                
            return emb_numpy, "Success"

        except Exception as e:
            return None, str(e)

engine = FaceEngine()