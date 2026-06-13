import io
import os
import time
import base64
import torch
from PIL import Image

async def process_inference(self, image_b64: str, db: Session):
    # Dekode citra base64 ke PIL Image
    img_bytes = base64.b64decode(image_b64)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # Deteksi dan crop wajah menggunakan MTCNN
    face_tensor, box, prob = image_processor.detect_face(img_pil)
    if face_tensor is None:
        img_pil.close()
        return {"matched": "Unknown", "confidence": 0}
    
    threshold = self.fl_manager.inference_threshold
    input_tensor = (
        image_processor
        .prepare_for_model(face_tensor)
        .to(self.fl_manager.device)
    )

    # Ekstraksi Embedding Wajah dengan Flip Trick
    with torch.no_grad():
        self.fl_manager.backbone.eval()
        emb_orig = self.fl_manager.backbone(input_tensor)
        face_flipped = torch.flip(input_tensor, dims=[3])
        emb_mirror = self.fl_manager.backbone(face_flipped)
        query_emb = (
            torch.nn.functional.normalize(
                (emb_orig + emb_mirror) / 2, p=2, dim=1
            )
        )
        
    # Logika buffer Temporal Voting (maksimal 3 detik)
    now = time.time()
    if now - self.fl_manager.last_face_time > 3.0:
        self.fl_manager.prediction_buffer.clear()
        
    self.fl_manager.prediction_buffer.append(query_emb)
    self.fl_manager.last_face_time = now
    
    # Rata-rata embedding temporal
    mean_emb_tensor = (
        torch.stack(
            list(self.fl_manager.prediction_buffer)
        ).mean(0)
    )
    mean_emb_tensor = (
        torch.nn.functional.normalize(
            mean_emb_tensor, p=2, dim=1
        )
    )
    mean_emb = mean_emb_tensor.cpu().numpy()[0]
    
    # Pencocokan identitas berbasis Cosine Similarity
    local_refs = self._get_cached_identities(db)
    best_match, confidence = (
        identify_user_globally(
            mean_emb, local_refs, threshold=threshold
        )
    )
    
    # Validasi ambang batas kemiripan
    user_id = best_match if confidence >= threshold else "Unknown"
    
    # Simpan hasil absensi jika wajah dikenali
    if user_id != "Unknown":
        try:
            new_attendance = AttendanceLocal(
                user_id=user_id,
                confidence=confidence,
                device_id=os.getenv("HOSTNAME", "terminal-1")
            )
            db.add(new_attendance)
            db.commit()
        except Exception:
            db.rollback()

    img_pil.close()
    return {
        "matched": user_id, 
        "confidence": float(confidence), 
        "box": box.tolist() if box is not None else None
    }
