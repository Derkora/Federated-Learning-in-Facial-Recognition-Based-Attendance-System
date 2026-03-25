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
            margin=40, 
            keep_all=False, 
            device=DEVICE, 
            post_process=False 
        )
        
        # Standard MobileFaceNet Normalization: [0, 1] range shifted to [-1, 1]
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def detect_face(self, img):
        """Detect and crop face from image (PIL)."""
        try:
            face = self.mtcnn(img)
            return face
        except Exception as e:
            print(f"Face detection error: {e}")
            return None

    def resize_and_pad(self, img_pil, target_size=(112, 112)):
        """
        Resize image to target size with symmetric padding to maintain aspect ratio,
        specifically targeting the 112x96 (HxW) to 112x112 research requirement.
        """
        w, h = img_pil.size
        t_h, t_w = target_size
        
        # Calculate aspect ratio
        ratio = min(t_w / w, t_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
        
        # Symmetric Padding
        pad_w = t_w - new_w
        pad_h = t_h - new_h
        
        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top
        
        return TF.pad(img_resized, [left, top, right, bottom], fill=0)

    def prepare_for_model(self, face_data):
        """
        Converts PIL face or Tensor into 112x112 tensor with symmetric padding and normalization.
        Output Shape: (1, 3, 112, 112)
        """
        # If face_data is from MTCNN with post_process=False, it's (C, H, W) Tensor [0, 255]
        if isinstance(face_data, torch.Tensor):
            # Check for MTCNN output format (3, 112, 112)
            if face_data.shape[1:] == (112, 112):
                face_tensor = face_data.float() / 255.0
            else:
                # Fallback conversion for raw tensor
                img_pil = TF.to_pil_image(face_data)
                img_padded = self.resize_and_pad(img_pil)
                face_tensor = TF.to_tensor(img_padded)
        else:
            # Assume PIL Input
            # ALIGNMENT CHECK: Symmetric padding ensures anchor points (eyes/nose) stay centered
            # relative to the 112x96 facial structure defined in research.
            img_padded = self.resize_and_pad(face_data)
            face_tensor = TF.to_tensor(img_padded)

        # Normalize to [-1, 1] range
        normalized_tensor = self.normalize(face_tensor)
        
        return normalized_tensor.unsqueeze(0).to(DEVICE)

image_processor = ImageProcessor()
