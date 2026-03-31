import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN
from PIL import Image

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
        
        # Standard MobileFaceNet Normalization: [0, 1] range shifted to [-1, 1]
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def detect_face(self, img, save_path=None):
        """
        Detect face and return (face_tensor, box, probability).
        Box format: [x, y, x2, y2]
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
        Converts PIL face or Tensor into 112x96 tensor with normalization.
        Output Shape: (1, 3, 112, 96)
        """
        if isinstance(face_data, torch.Tensor):
            # MTCNN with post_process=False returns [0, 255] float
            # Scale to [0, 1] before any PIL conversion or normalization
            if face_data.max() > 1.0:
                face_tensor = face_data.float() / 255.0
            else:
                face_tensor = face_data.float()

            # Ensure 112x96 shape (H, W)
            if face_tensor.shape[1:] != (112, 96):
                img_pil = TF.to_pil_image(face_tensor)
                img_resized = img_pil.resize((96, 112), Image.BILINEAR)
                face_tensor = TF.to_tensor(img_resized)
        else:
            # Assume PIL Input
            img_resized = face_data.resize((96, 112), Image.BILINEAR)
            face_tensor = TF.to_tensor(img_resized)

        # Final Normalization to [-1, 1]
        normalized_tensor = self.normalize(face_tensor)
        
        return normalized_tensor.unsqueeze(0).to(DEVICE)

image_processor = ImageProcessor()
