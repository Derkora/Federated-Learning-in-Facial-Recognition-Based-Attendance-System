import torch
import torchvision.transforms as T
from PIL import Image
from facenet_pytorch import MTCNN
from .mobilefacenet import MobileFaceNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageProcessor:
    def __init__(self):
        self.mtcnn = MTCNN(image_size=112, margin=20, keep_all=False, device=DEVICE, post_process=True)
        self.preprocess = T.Compose([
            T.Resize((112, 96)), 
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_face(self, img_pil):
        return self.mtcnn(img_pil)

    def prepare_for_model(self, face_tensor):
        face_img = T.ToPILImage()(face_tensor * 0.5 + 0.5)
        return self.preprocess(face_img).unsqueeze(0).to(DEVICE)

image_processor = ImageProcessor()
