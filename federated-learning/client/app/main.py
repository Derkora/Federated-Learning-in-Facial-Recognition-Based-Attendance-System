from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

import os
import time
import requests
import base64
import io
from PIL import Image
import uvicorn
import traceback
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as io_torch
import torchvision.transforms.functional as TF

from .db.db import engine, get_db, Base
from .db.models import UserLocal, EmbeddingLocal, AttendanceLocal
from .utils.preprocessing import image_processor
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
async def index(request: Request):
    return templates.TemplateResponse("attendance.html", {
        "request": request,
        "client_id": fl_manager.client_id,
        "model_version": fl_manager.model_version,
        "title": "Edge Terminal"
    })

@app.post("/api/inference")
async def api_inference(data: dict, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    start_time = time.time()
    try:
        img_bytes = base64.b64decode(data['image'])
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        face_tensor, box, prob = image_processor.detect_face(img_pil)
        if face_tensor is None:
            print("[INFERENCE] No face detected.")
            return JSONResponse({"matched": "Unknown", "confidence": 0, "message": "No face detected"})
        
        print(f"[INFERENCE] Face Detected (Prob: {prob:.2f})")
        
        # Convert box to list for JSON serialization
        face_box = box.tolist() if box is not None else None
        
        input_tensor = image_processor.prepare_for_model(face_tensor).to(fl_manager.device)
        print("[INFERENCE] Preprocessing Done.")
        
        with torch.no_grad():
            fl_manager.backbone.eval()
            query_embedding_tensor = fl_manager.backbone(input_tensor)
            # L2 NORMALIZATION (Critical for similarity)
            query_embedding_tensor = torch.nn.functional.normalize(query_embedding_tensor, p=2, dim=1)
            query_embedding = query_embedding_tensor.cpu().numpy()[0]
        
        if not hasattr(fl_manager, 'cached_refs') or time.time() - getattr(fl_manager, 'last_cache_update', 0) > 30:
            embeddings = db.query(EmbeddingLocal).all()
            local_refs = {}
            for emb in embeddings:
                try:
                    if emb.is_global:
                        dec_emb = np.frombuffer(emb.embedding_data, dtype=np.float32).copy()
                    else:
                        dec_emb = encryptor.decrypt_embedding(emb.embedding_data, emb.iv).copy()
                    local_refs[emb.user_id] = torch.from_numpy(dec_emb).to(fl_manager.device)
                except: continue

            registry_path = os.path.join(fl_manager.artifacts_path, "models", "global_embedding_registry.pth")
            if os.path.exists(registry_path):
                try:
                    registry = torch.load(registry_path, map_location="cpu")
                    if isinstance(registry, dict):
                        for nrp, vec in registry.items():
                            if isinstance(vec, torch.Tensor):
                                local_refs[nrp] = vec.to(fl_manager.device)
                            else:
                                local_refs[nrp] = torch.from_numpy(np.array(vec).copy()).to(fl_manager.device)
                    print(f"[CACHE] Loaded {len(registry)} universal identities.")
                except Exception as e:
                    pass
            
            fl_manager.cached_refs = local_refs
            fl_manager.last_cache_update = time.time()
        
        local_refs = fl_manager.cached_refs

        # --- TEMPORAL VOTING  ---
        now = time.time()
        # If face is lost for > 1.0 second, flush buffer
        if now - fl_manager.last_face_time > 1.0:
            fl_manager.prediction_buffer.clear()
        
        fl_manager.prediction_buffer.append(query_embedding_tensor)
        fl_manager.last_face_time = now
        
        # Calculate mean embedding from buffer
        mean_embedding_tensor = torch.stack(list(fl_manager.prediction_buffer)).mean(0)
        mean_embedding_tensor = torch.nn.functional.normalize(mean_embedding_tensor, p=2, dim=1)
        mean_embedding = mean_embedding_tensor.cpu().numpy()[0]
        
        # Identify using Centroid Matcher with Production Threshold (0.50)
        user_id, confidence = identify_user_globally(mean_embedding, local_refs, threshold=0.50)
        
        # Immediate Attendance (Voting Removed)
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
            print(f"[ATTENDANCE] Confirmed: {user_id} ({confidence:.2f})")
            
        latency = int((time.time() - start_time) * 1000)
        return {
            "matched": user_id, 
            "is_confirmed": True if user_id != "Unknown" else False,
            "confidence": float(confidence), 
            "box": face_box,
            "latency_ms": latency
        }
        
    except Exception as e:
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
        
        face_tensor, box, prob = image_processor.detect_face(img_pil)
        if face_tensor is not None:
            input_tensor = image_processor.prepare_for_model(face_tensor)
            with torch.no_grad():
                fl_manager.backbone.eval()
                embedding = fl_manager.backbone(input_tensor)
                # L2 NORMALIZATION (Expert & Similarity requirement)
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                embedding_np = embedding.cpu().numpy()[0]
                
                # Encrypt and save locally
                encrypted_data, iv = encryptor.encrypt_embedding(embedding_np)
                new_emb = EmbeddingLocal(user_id=user_id, embedding_data=encrypted_data, iv=iv, is_global=False)
                db.add(new_emb)
                db.commit()
            
            server_url = os.getenv("SERVER_API_URL", "http://server-fl:8080")
            try:
                # Send L2-normalized embedding to server
                embedding_b64 = base64.b64encode(embedding_np.tobytes()).decode('utf-8')
                requests.post(f"{server_url}/api/training/get_label", json={
                    "nrp": user_id,
                    "name": name,
                    "client_id": os.getenv("HOSTNAME", "terminal-1"),
                    "embedding": embedding_b64
                }, timeout=5)
            except Exception as e:
                print(f"[REGISTRATION] Gagal kirim embedding ke server: {e}")
            
        return {"status": "success", "user_id": user_id}
    except Exception as e:
        print(f"[INFERENCE ERROR] {e}")
        traceback.print_exc()
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/request-data")
async def request_data(background_tasks: BackgroundTasks):
    def do_preparation():
        print("[ORCHESTRATOR] Server requested data prep. Running Sync + Preprocess...")
        fl_manager.run_sync_phase()
        fl_manager.run_preprocess_phase()
        print("[ORCHESTRATOR] Remote Data Prep Complete.")
        
        # Signal READY to server
        try:
            import requests # Ensure it's available in this scope if needed, though usually global
            requests.post(f"{fl_manager.server_api_url}/api/clients/ready", json={"client_id": fl_manager.client_id}, timeout=5)
            print(f"[ORCHESTRATOR] Reported READY ({fl_manager.client_id}) to server.")
        except Exception as e:
            print(f"[ORCHESTRATOR WARNING] Could not report READY to server: {e}")
    
    background_tasks.add_task(do_preparation)
    return {"status": "success", "message": "Data preparation started"}

@app.on_event("startup")
def startup_event():
    fl_manager.start_background_tasks()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
