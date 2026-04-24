import os
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from facenet_pytorch import MTCNN
from .mobilefacenet import MobileFaceNet

DEVICE = torch.device('cpu')

class FaceHandler:
    def __init__(self):
        # post_process=False agar data yang disimpan di disk adalah RAW pixel [0, 255]
        # Ini mencegah double normalization saat training loader memroses gambar.
        self.mtcnn = MTCNN(
            image_size=112, 
            margin=20, 
            device=DEVICE, 
            post_process=False
        )
        self.transform = T.Compose([
            T.Resize((112, 96)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_and_save(self, img_path, dst_path):
        """
        Deteksi wajah dan simpan hasil crop Portrait (96x112) - Architectural Alignment.
        Urutan: Detect -> PIL Crop (with Margin) -> Resize (96x112).
        """
        try:
            img = Image.open(img_path).convert('RGB')
            boxes, _ = self.mtcnn.detect(img)
            
            if boxes is not None and len(boxes) > 0:
                box = boxes[0] # [x, y, x2, y2]
                
                # Tambahkan margin manual agar konsisten dengan behavior MTCNN
                margin = 20
                w = box[2] - box[0]
                h = box[3] - box[1]
                box[0] = max(0, box[0] - margin/2)
                box[1] = max(0, box[1] - margin/2)
                box[2] = min(img.width, box[2] + margin/2)
                box[3] = min(img.height, box[3] + margin/2)

                # 1. PIL Crop
                face_img = img.crop((box[0], box[1], box[2], box[3]))
                
                # 2. Squash Resize ke Portrait 96x112
                # PIL menggunakan (Width, Height)
                face_img = face_img.resize((96, 112), Image.BILINEAR)
                
                # 3. Simpan
                face_img.save(dst_path)
                return True
            else:
                print(f"[FaceHandler] Skip: Tidak ditemukan wajah pada {os.path.basename(img_path)}")
        except Exception as e:
            print(f"[FaceHandler] ERROR pada {os.path.basename(img_path)}: {e}")
        return False

    def get_blur_score(self, image_path):
        try:
            
            img = cv2.imread(image_path)
            if img is None: return 0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except: return 0

    def select_best_faces(self, folder_path, n=50):
        if not os.path.exists(folder_path): return []
        imgs = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not imgs: return []
        scored = sorted([(p, self.get_blur_score(p)) for p in imgs], key=lambda x: x[1], reverse=True)
        return [os.path.basename(s[0]) for s in scored[:n]]

    def get_embedding(self, model, img_pil):
        """Generate embedding from a PIL image."""
        img_tensor = self.transform(img_pil).unsqueeze(0).to(DEVICE)
        return torch.nn.functional.normalize(model(img_tensor))

face_handler = FaceHandler()
