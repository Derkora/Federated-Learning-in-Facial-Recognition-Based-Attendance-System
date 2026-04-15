import os
import torch
import torchvision.transforms as T
from PIL import Image
from facenet_pytorch import MTCNN
from .mobilefacenet import MobileFaceNet

DEVICE = torch.device('cpu')

class FaceHandler:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=112, margin=20, device=DEVICE, post_process=True)
        self.final_resize = T.Resize((112, 96))
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_and_save(self, img_path, dst_path):
        """
        Deteksi wajah dan simpan hasil crop Portrait (96x112) tanpa distorsi.
        """
        try:
            img = Image.open(img_path).convert('RGB')
            boxes, _ = self.mtcnn.detect(img)
            
            if boxes is not None and len(boxes) > 0:
                box = boxes[0] # [x1, y1, x2, y2]
                
                # Manual Crop (Mencegah Squash 112x112 ke 96x112)
                x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
                x2, y2 = min(img.width, int(box[2])), min(img.height, int(box[3]))
                
                face_img = img.crop((x1, y1, x2, y2))
                # Resize Portrait 96x112 (Architectural consistency)
                face_img = face_img.resize((96, 112), Image.BILINEAR)
                face_img.save(dst_path)
                return True
            else:
                print(f"[FaceHandler] Skip: Tidak ditemukan wajah pada {os.path.basename(img_path)}")
        except Exception as e:
            print(f"[FaceHandler] ERROR pada {os.path.basename(img_path)}: {e}")
        return False

    def get_embedding(self, model, img_pil):
        """Generate embedding from a PIL image."""
        img_tensor = self.transform(img_pil).unsqueeze(0).to(DEVICE)
        return torch.nn.functional.normalize(model(img_tensor))

face_handler = FaceHandler()
