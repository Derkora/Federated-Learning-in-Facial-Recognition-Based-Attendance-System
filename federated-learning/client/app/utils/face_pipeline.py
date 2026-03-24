import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import cv2
import os

class FaceAnalysisPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu') # Force CPU if needed
        
        print(f"[PIPELINE] Loading MTCNN on {self.device}...")
        self.mtcnn = MTCNN(
            image_size=112, 
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True, 
            device=self.device
        )
        
    def _check_quality(self, img_tensor):
        """Check for blur, too dark, or too bright using tensor."""
        try:
            # Simple brightness check on tensor
            mean_brightness = torch.mean(img_tensor.float())
            if mean_brightness < 20: return False, "Too Dark"
            if mean_brightness > 240: return False, "Too Bright"
            
            # For blur, we still use CV2 for now as it's efficient
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 50: 
                return False, f"Blurry ({blur_score:.1f})"
            return True, "OK"
        except:
            return True, "Quality Check Skip"

    def detect_and_crop(self, img_pil):
        """Detect and crop face into 112x96 size (Hybrid PIL/MTCNN)."""
        try:
            # Resize if too large to speed up detection
            if img_pil.width > 800:
                scale = 800 / img_pil.width
                img_pil = img_pil.resize((800, int(img_pil.height * scale)), Image.LANCZOS)

            is_good, _ = self._check_quality_pil(img_pil)
            if not is_good: return None
            
            boxes, probs = self.mtcnn.detect(img_pil)
            if boxes is None or len(boxes) == 0: return None
            
            best_idx = np.argmax(probs)
            if probs[best_idx] < 0.70: return None
            
            box = boxes[best_idx].astype(int)
            face_img = img_pil.crop(box)
            # Resize to 112x96 (H x W) -> PIL uses (W, H)
            face_img = face_img.resize((96, 112), Image.LANCZOS)
            return face_img
        except Exception as e:
            print(f"[PIPELINE ERROR] {e}")
            return None

    def _check_quality_pil(self, img_pil):
        """Check for blur, too dark, or too bright using PIL/CV2."""
        try:
            img_np = np.array(img_pil)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 50: 
                return False, f"Blurry ({blur_score:.1f})"
            mean_brightness = np.mean(gray)
            if mean_brightness < 20: return False, "Too Dark"
            if mean_brightness > 240: return False, "Too Bright"
            return True, "OK"
        except:
            return True, "Quality Check Skip"

face_pipeline = FaceAnalysisPipeline()
