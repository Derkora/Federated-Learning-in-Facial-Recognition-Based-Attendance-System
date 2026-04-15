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
                post_process=True 
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
        """
        try:
            boxes, probs = self.mtcnn.detect(img)
            if boxes is None or len(boxes) == 0:
                return None, None, 0.0
            
            # Kita tetap kembalikan face_tensor (SQUARE 112x112) untuk compat, 
            # tapi sarankan pakai get_face_crop untuk akurasi.
            face = self.mtcnn(img, save_path=save_path)
            return face, boxes[0], probs[0]
        except Exception as e:
            print(f"Face detection error: {e}")
            return None, None, 0.0

    def get_face_crop(self, img_pil):
        """
        Crop wajah manual dari koordinat MTCNN untuk menghindari distorsi Square-to-Portrait.
        Output: PIL Image 96x112
        """
        try:
            boxes, _ = self.mtcnn.detect(img_pil)
            if boxes is None or len(boxes) == 0:
                return None
            
            box = boxes[0] # [x1, y1, x2, y2]
            # Menghindari koordinat negatif
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(img_pil.width, int(box[2])), min(img_pil.height, int(box[3]))
            
            img_crop = img_pil.crop((x1, y1, x2, y2))
            # Resize ke 96x112 (Portrait) - distorsi terkontrol sesuai training
            return img_crop.resize((96, 112), Image.BILINEAR)
        except Exception as e:
            print(f"[PREPROCESS ERROR] {e}")
            return None

    def prepare_for_model(self, face_data):
        """
        Normalisasi data wajah untuk input MobileFaceNet.
        Input: Tensor [-1, 1] atau PIL Image.
        """
        if face_data is None: return None

        if isinstance(face_data, torch.Tensor):
            # Jika masukan adalah tensor MTCNN (112x112 square), resize ke portrait
            # Tapi sarankan pakai get_face_crop(pil) untuk hasil terbaik.
            face_img = TF.to_pil_image((face_data * 0.5 + 0.5).clamp(0, 1))
            img_resized = face_img.resize((96, 112), Image.BILINEAR)
            face_tensor = TF.to_tensor(img_resized)
        else:
            # Input PIL (dari get_face_crop)
            face_tensor = TF.to_tensor(face_data)

        # Normalisasi ke [-1, 1] - Satukan standar di sini
        normalized_tensor = self.normalize(face_tensor)
        return normalized_tensor.unsqueeze(0).to(DEVICE)

image_processor = ImageProcessor()
