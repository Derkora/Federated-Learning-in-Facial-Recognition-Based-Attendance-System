from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

import os
import time
import base64
import io
from PIL import Image
import uvicorn
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as io_torch
import torchvision.transforms.functional as TF

from .db.db import engine, get_db, Base
from .db.models import UserLocal, EmbeddingLocal, AttendanceLocal
from .utils.image_processing import image_processor
from .utils.classifier import identify_user_globally
from .utils.security import encryptor
from .client_manager_instance import fl_manager

# Initialize Database
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Federated Face Recognition Terminal")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
templates.env.globals.update(os=os)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    users_count = db.query(UserLocal).count()
    attendance_count = db.query(AttendanceLocal).count()
    return templates.TemplateResponse("attendance.html", {
        "request": request,
        "users_count": users_count,
        "attendance_count": attendance_count,
        "is_training": fl_manager.is_training,
        "client_id": fl_manager.client_id,
        "model_version": fl_manager.model_version
    })

@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

@app.post("/api/inference")
async def api_inference(data: dict, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    start_time = time.time()
    try:
        img_bytes = base64.b64decode(data['image'])
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        face_tensor = image_processor.detect_face(img_pil)
        if face_tensor is None:
            return JSONResponse({"matched": "Unknown", "confidence": 0, "message": "No face detected"})
        
        input_tensor = image_processor.prepare_for_model(face_tensor).to(fl_manager.device)
        
        with torch.no_grad():
            fl_manager.backbone.eval()
            query_embedding_tensor = fl_manager.backbone(input_tensor)
            # L2 NORMALIZATION (Critical for similarity)
            query_embedding_tensor = torch.nn.functional.normalize(query_embedding_tensor, p=2, dim=1)
            query_embedding = query_embedding_tensor.cpu().numpy()[0]
        
        embeddings = db.query(EmbeddingLocal).all()
        local_refs = {}
        for emb in embeddings:
            try:
                if emb.is_global:
                    dec_emb = np.frombuffer(emb.embedding_data, dtype=np.float32)
                else:
                    dec_emb = encryptor.decrypt_embedding(emb.embedding_data, emb.iv)
                local_refs[emb.user_id] = torch.from_numpy(dec_emb).to(fl_manager.device)
            except: continue
        
        # Identify
        user_id, confidence = identify_user_globally(query_embedding, local_refs)
        
        if user_id != "Unknown":
            user = db.query(UserLocal).filter_by(user_id=user_id).first()
            user_name = user.name if user else "Unknown"
            
            new_attendance = AttendanceLocal(
                user_id=user_id,
                confidence=confidence,
                device_id=os.getenv("HOSTNAME", "terminal-1")
            )
            db.add(new_attendance)
            db.commit()
            
            background_tasks.add_task(
                sync_record_to_server, 
                user_id, user_name, float(confidence), os.getenv("HOSTNAME", "terminal-1")
            )
            
        latency = int((time.time() - start_time) * 1000)
        print(f"[INFERENCE] Matched: {user_id} ({confidence:.2f}) in {latency}ms")
        return {"matched": user_id, "confidence": float(confidence), "latency_ms": latency}
        
    except Exception as e:
        import traceback
        print(f"[INFERENCE ERROR] {e}")
        traceback.print_exc()
        return JSONResponse({"matched": "Error", "error": str(e)}, status_code=400)

async def sync_record_to_server(user_id, name, confidence, client_id):
    server_url = os.getenv("SERVER_API_URL", "http://server-fl:8080")
    payload = [{
        "user_id": user_id,
        "name": name,
        "client_id": client_id,
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "confidence": confidence
    }]
    try:
        requests.post(f"{server_url}/api/attendance/sync", json=payload, timeout=5)
        print(f"Synced attendance for {user_id} to server.")
    except Exception as e:
        print(f"Sync failed: {e}")

@app.post("/api/register")
async def register_user(user_id: str, name: str, image_base64: str, db: Session = Depends(get_db)):
    try:
        new_user = UserLocal(user_id=user_id, name=name)
        db.add(new_user)
        db.commit()

        img_bytes = base64.b64decode(image_base64)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        user_dir = os.path.join(fl_manager.data_path, name)
        os.makedirs(user_dir, exist_ok=True)
        target_path = os.path.join(user_dir, f"{int(time.time())}.jpg")
        img_pil.save(target_path)
        
        face_tensor = image_processor.detect_face(img_pil)
        if face_tensor is not None:
            input_tensor = image_processor.prepare_for_model(face_tensor)
            with torch.no_grad():
                fl_manager.backbone.eval()
                embedding = fl_manager.backbone(input_tensor)
                # L2 NORMALIZATION (Expert requirement)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                embedding_np = embedding.cpu().numpy()[0]
                
                # Encrypt and save locally
                encrypted_data, iv = encryptor.encrypt_embedding(embedding_np)
                new_emb = EmbeddingLocal(user_id=user_id, embedding_data=encrypted_data, iv=iv, is_global=False)
                db.add(new_emb)
                db.commit()
            
            server_url = os.getenv("SERVER_API_URL", "http://server-fl:8080")
            try:
                embedding_b64 = base64.b64encode(embedding_np.tobytes()).decode('utf-8')
                requests.post(f"{server_url}/api/training/get_label", json={
                    "nrp": user_id,
                    "name": name,
                    "client_id": os.getenv("HOSTNAME", "terminal-1"),
                    "embedding": embedding_b64
                }, timeout=5)
            except Exception as e:
                print(f"[REGISTRATION] Gagal kirim embedding ke server: {e}")

            new_emb = EmbeddingLocal(
                user_id=user_id, 
                embedding_data=encrypted_data,
                iv=iv,
                is_global=False
            )
            db.add(new_emb)
            db.commit()
            
        return {"status": "success", "user_id": user_id}
    except Exception as e:
        import traceback
        print(f"[INFERENCE ERROR] {e}")
        traceback.print_exc()
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/users")
async def get_users(db: Session = Depends(get_db)):
    users = db.query(UserLocal).all()
    return [u.name for u in users]

@app.on_event("startup")
def startup_event():
    print(f"[STARTUP] Personalized FL Client Initialized", flush=True)

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
