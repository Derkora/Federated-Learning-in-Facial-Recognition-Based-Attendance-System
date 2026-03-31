import torch
import torchvision.transforms as T
from PIL import Image
from facenet_pytorch import MTCNN
from .mobilefacenet import MobileFaceNet

# Konfigurasi Perangkat (CPU dipaksakan untuk terminal)
DEVICE = torch.device('cpu')

class ImageProcessor:
    # Utilitas Pemrosesan Gambar (Deteksi & Normalisasi Wajah)
    
    def __init__(self):
        # Inisialisasi MTCNN untuk deteksi wajah 112px
        self.mtcnn = MTCNN(image_size=112, margin=20, keep_all=False, device=DEVICE, post_process=True)
        # Transformasi normalisasi sesuai arsitektur MobileFaceNet
        self.preprocess = T.Compose([
            T.Resize((112, 96)), 
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_face(self, img_pil):
        # Mendeteksi dan melakukan cropping wajah dari gambar asli
        return self.mtcnn(img_pil)

    def prepare_for_model(self, face_tensor):
        # Menyiapkan tensor wajah hasil deteksi ke format input model
        face_img = T.ToPILImage()(face_tensor * 0.5 + 0.5)
        return self.preprocess(face_img).unsqueeze(0).to(DEVICE)

# Instansi Utilitas Global
image_processor = ImageProcessor()
