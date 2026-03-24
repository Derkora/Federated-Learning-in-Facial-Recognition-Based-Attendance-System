import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN

DEVICE = torch.device('cpu')

class ImageProcessor:
    def __init__(self):
        # MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=112, 
            margin=20, 
            keep_all=False, 
            device=DEVICE, 
            post_process=False 
        )
        
        # Preprocessing matching the trainer and model.ipynb (Tensors)
        self.preprocess = T.Compose([
            T.Resize((112, 96)), 
            T.ConvertImageDtype(torch.float32), 
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_face(self, img):
        """Detect and crop face from image (PIL or Tensor)."""
        try:
            # MTCNN handles PIL or Tensors
            face = self.mtcnn(img)
            return face
        except Exception as e:
            print(f"Face detection error: {e}")
            return None

    def prepare_for_model(self, face_tensor):
        """Normalize face tensor for the backbone."""
        # face_tensor is (3, 112, 112) uint8 or float from MTCNN
        if not isinstance(face_tensor, torch.Tensor):
            face_tensor = TF.to_tensor(face_tensor)
            
        # Ensure it is uint8 if coming from post_process=False MTCNN
        if face_tensor.dtype == torch.float32 and face_tensor.max() > 1.0:
            face_tensor = face_tensor / 255.0

        return self.preprocess(face_tensor).unsqueeze(0).to(DEVICE)

image_processor = ImageProcessor()
