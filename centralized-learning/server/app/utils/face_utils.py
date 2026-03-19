import os
import torch
import torchvision.transforms as T
from PIL import Image
from facenet_pytorch import MTCNN
from .mobilefacenet import MobileFaceNet

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FaceHandler:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=112, margin=20, device=DEVICE, post_process=True)
        self.final_resize = T.Resize((112, 96))
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_and_save(self, img_path, dst_path):
        """Detect face and save processed image."""
        try:
            img = Image.open(img_path).convert('RGB')
            face = self.mtcnn(img)
            if face is not None:
                face_img = T.ToPILImage()(face * 0.5 + 0.5)
                face_img = self.final_resize(face_img)
                face_img.save(dst_path)
                return True
        except Exception as e:
            print(f"[FaceHandler] Error processing {img_path}: {e}")
        return False

    def get_embedding(self, model, img_pil):
        """Generate embedding from a PIL image."""
        img_tensor = self.transform(img_pil).unsqueeze(0).to(DEVICE)
        return torch.nn.functional.normalize(model(img_tensor))

face_handler = FaceHandler()
