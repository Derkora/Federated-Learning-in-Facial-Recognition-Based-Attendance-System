import torch
import torchvision.transforms.functional as TF
from PIL import Image

def prepare_for_model(self, face_data):
    if face_data is None: 
        return None
  
    # Konversi ke format PIL Image
    if isinstance(face_data, torch.Tensor):
        if face_data.max() > 2.0:
            face_data = face_data / 255.0
        face_img = TF.to_pil_image(face_data.clamp(0, 1))
    else:
        face_img = face_data
  
    # Resize ke dimensi masukan MobileFaceNet (96x112)
    if face_img.size != (96, 112):
        face_img = face_img.resize((96, 112), Image.BILINEAR)
  
    face_tensor = TF.to_tensor(face_img)
    normalized_tensor = self.normalize(face_tensor)
    
    # Tambah dimensi batch (1, C, H, W)
    return normalized_tensor.unsqueeze(0).to(DEVICE)
