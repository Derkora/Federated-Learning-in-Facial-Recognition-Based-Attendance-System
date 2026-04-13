import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN
from PIL import Image

DEVICE = torch.device('cpu')

class ImageProcessor:
    def __init__(self):
        # MTCNN akan dimuat secara malas (Lazy Load) untuk menghemat RAM saat standby
        self._mtcnn = None
        
        # Normalisasi Standar MobileFaceNet: rentang [0, 1] digeser menjadi [-1, 1]
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    @property
    def mtcnn(self):
        if self._mtcnn is None:
            print("[INFO] Loading MTCNN detector to RAM...")
            self._mtcnn = MTCNN(
                image_size=112, 
                margin=20, 
                keep_all=False, 
                device=DEVICE, 
                post_process=False 
            )
        return self._mtcnn

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
        Deteksi wajah dan kembalikan (face_tensor, box, probabilitas).
        Format kotak: [x, y, x2, y2]
        """
        try:
            boxes, probs = self.mtcnn.detect(img)
            if boxes is None or len(boxes) == 0:
                return None, None, 0.0
            
            face = self.mtcnn(img, save_path=save_path)
            return face, boxes[0], probs[0]
        except Exception as e:
            print(f"Face detection error: {e}")
            return None, None, 0.0

    def prepare_for_model(self, face_data):
        """
        Mengonversi wajah PIL atau Tensor menjadi tensor 112x96 dengan normalisasi.
        Bentuk Keluaran: (1, 3, 112, 96)
        """
        # Keamanan: Jika face_data adalah tuple (hasil detect_face yang belum di-unpack), ambil elemen pertama
        if isinstance(face_data, tuple):
            face_data = face_data[0]
            
        if face_data is None: return None

        if isinstance(face_data, torch.Tensor):
            # MTCNN dengan post_process=False mengembalikan float [0, 255]
            # Skala ke [0, 1] sebelum konversi PIL atau normalisasi apa pun
            if face_data.max() > 1.0:
                face_tensor = face_data.float() / 255.0
            else:
                face_tensor = face_data.float()

            # Pastikan bentuk 112x96 (H, W)
            if face_tensor.shape[1:] != (112, 96):
                img_pil = TF.to_pil_image(face_tensor)
                img_resized = img_pil.resize((96, 112), Image.BILINEAR)
                face_tensor = TF.to_tensor(img_resized)
        else:
            # Asumsikan Input PIL
            img_resized = face_data.resize((96, 112), Image.BILINEAR)
            face_tensor = TF.to_tensor(img_resized)

        # Normalisasi Akhir ke [-1, 1]
        normalized_tensor = self.normalize(face_tensor)
        
        return normalized_tensor.unsqueeze(0).to(DEVICE)

image_processor = ImageProcessor()
