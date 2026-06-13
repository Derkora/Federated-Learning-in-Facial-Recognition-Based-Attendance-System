import cv2
import numpy as np
from PIL import Image

def detect_face(self, img, save_path=None):
    try:
        orig_w, orig_h = img.size
        max_size = 640
        # Downscaling gambar agar proses MTCNN lebih cepat
        if orig_w > max_size or orig_h > max_size:
            scale = max_size / max(orig_w, orig_h)
            new_size = (
                int(orig_w * scale), 
                int(orig_h * scale)
            )
            img_detect = img.resize(new_size, Image.BILINEAR)
        else:
            img_detect = img
            scale = 1.0
  
        # Inferensi deteksi wajah menggunakan MTCNN
        boxes, probs, landmarks = self.mtcnn.detect(
            img_detect, landmarks=True
        )
        if boxes is None or len(boxes) == 0:
            return None, None, 0.0
            
        if scale != 1.0:
            boxes = boxes / scale
            if landmarks is not None:
                landmarks = landmarks / scale
  
        face_img = None
  
        # Affine Alignment berbasis koordinat 5 titik landmark
        if landmarks is not None and landmarks[0] is not None:
            try:
                src = np.array(landmarks[0], dtype=np.float32)
                M, _ = cv2.estimateAffinePartial2D(
                    src, CANONICAL_LANDMARKS, method=cv2.LMEDS
                )
                if M is not None:
                    img_cv = cv2.cvtColor(
                        np.array(img), cv2.COLOR_RGB2BGR
                    )
                    aligned = cv2.warpAffine(
                        img_cv, M, (96, 112), 
                        flags=cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_REPLICATE
                    )
                    face_img = Image.fromarray(
                        cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                    )
            except Exception:
                pass
  
        # Fallback Bounding Box Crop jika alignment gagal
        if face_img is None:
            box = boxes[0]
            margin = 20
            x1 = max(0, int(box[0] - margin / 2))
            y1 = max(0, int(box[1] - margin / 2))
            x2 = min(img.width, int(box[2] + margin / 2))
            y2 = min(img.height, int(box[3] + margin / 2))
            
            face_img = img.crop((x1, y1, x2, y2)).resize(
                (96, 112), Image.BILINEAR
            )
  
        if save_path and face_img:
            face_img.save(save_path)
  
        return face_img, boxes[0], probs[0]
    except Exception:
        return None, None, 0.0
