import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from facenet_pytorch import MTCNN
from .mobilefacenet import MobileFaceNet

from .logging import get_logger

DEVICE = torch.device('cpu')

# Landmark kanonik 96x112 Portrait (standar InsightFace/MobileFaceNet)
CANONICAL_LANDMARKS = np.array([
    [30.2946, 51.6963],  # mata kiri
    [65.5318, 51.5014],  # mata kanan
    [48.0252, 71.7366],  # hidung
    [33.5493, 92.3655],  # mulut kiri
    [62.7299, 92.2041],  # mulut kanan
], dtype=np.float32)

class FaceHandler:
    def __init__(self):
        # post_process=False agar data yang disimpan di disk adalah RAW pixel [0, 255]
        self.mtcnn = MTCNN(
            image_size=112, 
            margin=20, 
            device=DEVICE, 
            post_process=False
        )
        self.transform = T.Compose([
            T.Resize((112, 96)),
            T.ToTensor(),
            # Normalisasi MobileFaceNet (Standard): (x - 127.5) / 128.0
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196])
        ])
        self.logger = get_logger()

    def detect_and_save(self, img_path, dst_path):
        """
        Deteksi wajah dan simpan hasil crop Portrait 96x112 dengan alignment.
        """
        try:
            img = Image.open(img_path).convert('RGB')
            boxes, _, landmarks = self.mtcnn.detect(img, landmarks=True)

            if boxes is not None and len(boxes) > 0:
                face_img = None

                # 1. Landmark Alignment (Prioritas Utama)
                if landmarks is not None and landmarks[0] is not None:
                    try:
                        src = np.array(landmarks[0], dtype=np.float32)
                        M, _ = cv2.estimateAffinePartial2D(src, CANONICAL_LANDMARKS, method=cv2.LMEDS)
                        if M is not None:
                            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                            aligned = cv2.warpAffine(img_cv, M, (96, 112),
                                                     flags=cv2.INTER_LINEAR,
                                                     borderMode=cv2.BORDER_REPLICATE)
                            face_img = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
                    # except Exception as e:
                    #     self.logger.info(f"Alignment gagal pada {os.path.basename(img_path)}: {e}")
                    except: pass


                # 2. Fallback: Bbox Crop
                if face_img is None:
                    box = boxes[0]
                    margin = 20
                    x1 = max(0, int(box[0] - margin/2))
                    y1 = max(0, int(box[1] - margin/2))
                    x2 = min(img.width, int(box[2] + margin/2))
                    y2 = min(img.height, int(box[3] + margin/2))
                    face_img = img.crop((x1, y1, x2, y2)).resize((96, 112), Image.BILINEAR)

                face_img.save(dst_path)
                return True
            # else:
            #     self.logger.info(f"Skip: Wajah tidak terdeteksi pada {os.path.basename(img_path)}")

        except Exception as e:
            self.logger.error(f"ERROR deteksi/simpan pada {os.path.basename(img_path)}: {e}")
        return False

    def get_blur_score(self, image_path):
        """Hitung skor Laplacian Variance."""
        try:
            img = cv2.imread(image_path)
            if img is None: return 0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except: return 0

    def select_best_faces(self, folder_path, n=50):
        """Pilih N gambar terbaik berdasarkan ketajaman dengan dukungan Cache."""
        if not os.path.exists(folder_path): return []
        
        # --- FITUR CACHE: Cek apakah sudah pernah dihitung sebelumnya ---
        cache_path = os.path.join(folder_path, ".selection_cache.json")
        import json
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)
                    if cache_data.get("n") == n:
                        return cache_data.get("filenames", [])
            except: pass

        imgs = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not imgs: return []
        
        scored = []
        import time, gc
        for i, p in enumerate(imgs):
            score = self.get_blur_score(p)
            scored.append((p, score))
            # Optimasi RAM: Bersihkan setiap 10 foto
            if i % 10 == 0:
                gc.collect()
                time.sleep(0.01)

        scored.sort(key=lambda x: x[1], reverse=True)
        selected_scores = [s[1] for s in scored[:n]]
        avg_sharpness = sum(selected_scores) / len(selected_scores) if selected_scores else 0
        
        nrp_name = os.path.basename(folder_path)
        self.logger.info(f"  > [Sharpness] User {nrp_name}: Terpilih {len(selected_scores)} foto | Rata-rata Skor: {avg_sharpness:.2f}")
        
        selected = [os.path.basename(s[0]) for s in scored[:n]]

        # --- SIMPAN CACHE ---
        try:
            with open(cache_path, "w") as f:
                json.dump({"n": n, "filenames": selected}, f)
        except: pass

        gc.collect()
        return selected

    def get_embedding(self, model, img_pil):
        """Generate embedding menggunakan Flip Trick (Avg Orig + Mirror)."""
        # Pastikan img_pil sudah berukuran 112x96 (Portrait)
        if img_pil.size != (96, 112):
            img_pil = img_pil.resize((96, 112), Image.BILINEAR)
            
        img_tensor = self.transform(img_pil).unsqueeze(0).to(DEVICE)
        img_flip = torch.flip(img_tensor, [3])
        
        with torch.no_grad():
            emb_orig = model(img_tensor)
            emb_flip = model(img_flip)
            # Rata-rata dan normalisasi ulang unit vector
            combined = torch.nn.functional.normalize(emb_orig + emb_flip, p=2, dim=1)
        return combined

face_handler = FaceHandler()

