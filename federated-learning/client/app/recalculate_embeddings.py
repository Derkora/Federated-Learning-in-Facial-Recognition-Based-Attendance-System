import os
import torch
import numpy as np
from PIL import Image
from sqlalchemy.orm import Session
from .db.db import SessionLocal
from .db.models import UserLocal, EmbeddingLocal
from .utils.image_processing import image_processor
from .utils.security import encryptor
from .utils.mobilefacenet import get_model

def recalculate():
    db = SessionLocal()
    device = torch.device("cpu")
    
    # 1. Load Model
    model = get_model().to(device)
    model_path = os.getenv("ARTIFACTS_PATH", "/app/artifacts") + "/models/backbone.pth"
    
    if os.path.exists(model_path):
        print(f"[RECALC] Loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("[RECALC] WARNING: No trained model found. Using random initialized model!")
    
    model.eval()
    
    # 2. Iterate Users
    users = db.query(UserLocal).all()
    data_path = os.getenv("DATA_PATH", "/app/data")
    
    print(f"[RECALC] Processing {len(users)} users with Quality Filters...")
    
    for user in users:
        user_dir = os.path.join(data_path, user.name)
        if not os.path.exists(user_dir):
            continue
            
        image_files = [f for f in os.listdir(user_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            continue
            
        embeddings = []
        weights = []
        
        for img_name in image_files:
            try:
                img_path = os.path.join(user_dir, img_name)
                img_pil = Image.open(img_path).convert('RGB')
                
                # Quality Check (Similarity to face_pipeline._check_quality_pil)
                import cv2
                img_np = np.array(img_pil)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
                brightness = np.mean(gray)
                
                if blur_score < 40 or brightness < 20 or brightness > 240:
                    # print(f"[RECALC] Skipping low quality image: {img_name} (Blur: {blur_score:.1f}, Bright: {brightness:.1f})")
                    continue
                
                # Detect face and get confidence
                face_tensor, box, prob = image_processor.detect_face(img_pil)
                
                if face_tensor is not None:
                    input_tensor = image_processor.prepare_for_model(face_tensor).to(device)
                    with torch.no_grad():
                        emb = model(input_tensor)
                        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                        embeddings.append(emb.cpu().numpy()[0])
                        # Use MTCNN probability combined with blur score for weight
                        weight = float(prob) * min(blur_score / 100.0, 2.0)
                        weights.append(weight)
            except Exception as e:
                print(f"[RECALC] Error processing {img_name} for {user.name}: {e}")
                
        if embeddings:
            # Calculate Weighted Centroid
            embeddings_np = np.array(embeddings)
            weights_np = np.array(weights).reshape(-1, 1)
            
            weighted_centroid = np.sum(embeddings_np * weights_np, axis=0) / np.sum(weights_np)
            # Re-normalize centroid
            final_centroid = weighted_centroid / np.linalg.norm(weighted_centroid)
            
            # Encrypt and Update DB
            encrypted_data, iv = encryptor.encrypt_embedding(final_centroid)
            
            existing_emb = db.query(EmbeddingLocal).filter_by(user_id=user.user_id, is_global=False).first()
            if existing_emb:
                existing_emb.embedding_data = encrypted_data
                existing_emb.iv = iv
                print(f"[RECALC] Updated centroid for {user.name} ({len(embeddings)}/{len(image_files)} images used)")
            else:
                new_emb = EmbeddingLocal(user_id=user.user_id, embedding_data=encrypted_data, iv=iv, is_global=False)
                db.add(new_emb)
                print(f"[RECALC] Created new centroid for {user.name} ({len(embeddings)} sources)")
                
            db.commit()
            
    print("[RECALC] DONE.")
    db.close()

if __name__ == "__main__":
    recalculate()
