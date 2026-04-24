import os
import cv2
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN
from PIL import Image

DEVICE = torch.device('cpu')

class ImageProcessor:
    def __init__(self):
        self.mtcnn = MTCNN(
            image_size=112, 
            margin=20, 
            keep_all=False, 
            device=DEVICE, 
            post_process=False
        )
        
        # Normalisasi MobileFaceNet
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def get_detector(self):
        return self.mtcnn

    def unload_detector(self):
        if self._mtcnn is not None:
            print("[INFO] Unloading MTCNN from RAM...")
            del self._mtcnn
            self._mtcnn = None
            import gc
            gc.collect()

    def detect_face(self, img, save_path=None):
        """
        Deteksi wajah dan kembalikan (PIL_image_96x112, box, probabilitas).
        """
        try:
            boxes, probs = self.mtcnn.detect(img)
            if boxes is None or len(boxes) == 0:
                return None, None, 0.0
            
            # URUTAN: Detect -> Manual Crop -> Resize 96x112
            face_img = self.get_face_crop(img, boxes[0])
            
            if save_path and face_img:
                face_img.save(save_path)
                
            return face_img, boxes[0], probs[0]
        except Exception as e:
            print(f"Face detection error: {e}")
            return None, None, 0.0

    def get_face_crop(self, img_pil, box=None):
        """
        Crop wajah manual dari koordinat MTCNN untuk menghindari distorsi Square-to-Portrait.
        Output: PIL Image 96x112
        """
        try:
            if box is None:
                boxes, _ = self.mtcnn.detect(img_pil)
                if boxes is None or len(boxes) == 0:
                    return None
                box = boxes[0] 
            
            # 1. MTCNN Standard Margin (20px)
            margin = 20
            x1, y1 = max(0, int(box[0] - margin/2)), max(0, int(box[1] - margin/2))
            x2, y2 = min(img_pil.width, int(box[2] + margin/2)), min(img_pil.height, int(box[3] + margin/2))
            
            img_crop = img_pil.crop((x1, y1, x2, y2))
            
            # 2. Resizing ke Portrait 96x112 menggunakan metode BILINEAR
            face_img = img_crop.resize((96, 112), Image.BILINEAR)
            print(f"[PREPROCESS] Wajah telah di-crop dan di-resize ke 96x112.") 
            
            return face_img
        except Exception as e:
            print(f"[PREPROCESS ERROR] {e}")
            return None

    def prepare_for_model(self, face_data):
        """
        Mengonversi output MTCNN atau PIL menjadi tensor 112x96 dengan normalisasi [-1, 1].
        """
        if face_data is None: return None

        if isinstance(face_data, torch.Tensor):
            # Jika face_data adalah tensor dari MTCNN (post_process=False), nilainya [0, 255]
            if face_data.max() > 2.0: # Check if raw pixels [0, 255]
                 face_data = face_data / 255.0
            face_img = TF.to_pil_image((face_data).clamp(0, 1))
        else:
            # Asumsikan Input PIL
            face_img = face_data

        # Resize/Squash ke Portrait 96x112
        img_resized = face_img.resize((96, 112), Image.BILINEAR)
        face_tensor = TF.to_tensor(img_resized) # ToTensor() handles [0, 1] scaling

        # Normalisasi akhir ke rentang [-1, 1]
        normalized_tensor = self.normalize(face_tensor)
        
        return normalized_tensor.unsqueeze(0).to(DEVICE)

    def get_blur_score(self, image_path):
        """
        Menghitung skor ketajaman gambar menggunakan Laplacian Variance.
        Semakin tinggi nilai, semakin tajam gambarnya.
        """
        try:
            img = cv2.imread(image_path)
            if img is None: return 0
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 0

    def select_best_faces(self, folder_path, n=50):
        """
        Memilih N wajah terbaik (paling tajam) dari sebuah folder NRP.
        Mengembalikan list path file yang terpilih.
        """
        if not os.path.exists(folder_path):
            return []
            
        all_imgs = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        if not all_imgs:
            return []

        # Hitung skor blur untuk setiap gambar
        scored = []
        for img_path in all_imgs:
            score = self.get_blur_score(img_path)
            scored.append((img_path, score))
            
        # Urutkan berdasarkan skor tertinggi (paling tajam)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Ambil Top N
        selected = [os.path.basename(s[0]) for s in scored[:n]]
        return selected

image_processor = ImageProcessor()
