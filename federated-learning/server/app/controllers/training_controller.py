from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from app.server_manager_instance import fl_manager
from app.controllers.client_controller import registered_clients
from app.db.db import SessionLocal, get_db
from app.db import models
from app.db.models import ModelVersion, TrainingRound, TrainingUpdate
from app.utils.mobilefacenet import MobileFaceNet
import torch
import io
import asyncio
import json

class LabelRequest(BaseModel):
    nrp: str
    name: str = ""
    registered_edge_id: Optional[str] = None 

router = APIRouter()

@router.post("/get_label")
def get_or_create_label(req: LabelRequest, db: SessionLocal = Depends(get_db)):
    user = db.query(models.UserGlobal).filter(models.UserGlobal.nrp == req.nrp).first()
    
    if not user:
        # Check capacity
        count = db.query(models.UserGlobal).count()
        if count >= 100:
            raise HTTPException(status_code=400, detail="Full")
            
        user = models.UserGlobal(
            name=req.name,
            nrp=req.nrp,
            registered_edge_id=req.registered_edge_id # Auto-assign on first register
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"[REGISTRY] New: {req.name} ({req.nrp}) -> Label {user.user_id-1} on {user.registered_edge_id}")
    else:
        # If already assigned to someone else
        if user.registered_edge_id and user.registered_edge_id != req.registered_edge_id and req.registered_edge_id:
             # Option: allow update if admin hasn't locked it? 
             # For now, let's keep the existing assignment check
             print(f"[REGISTRY] User {req.nrp} already belongs to {user.registered_edge_id}")
    
    # Return label (we use user_id - 1 as label to match 0-indexed)
    return {
        "label": user.user_id - 1,
        "name": user.name,
        "nrp": user.nrp,
        "registered_edge_id": user.registered_edge_id
    }

@router.get("/global_users")
def get_all_users(db: SessionLocal = Depends(get_db)):
    users = db.query(models.UserGlobal).all()
    return [{"label": u.user_id-1, "name": u.name, "nrp": u.nrp, "registered_edge_id": u.registered_edge_id} for u in users]

@router.delete("/users/{nrp}")
def delete_user(nrp: str, db: SessionLocal = Depends(get_db)):
    user = db.query(models.UserGlobal).filter(models.UserGlobal.nrp == nrp).first()
    if user:
        db.delete(user)
        db.commit()
        print(f"[REGISTRY] User deleted: {nrp}")
        return {"status": "success", "message": f"User {nrp} deleted"}
    raise HTTPException(status_code=404, detail="User not found")

class AssignRequest(BaseModel):
    nrp: str
    edge_id: str

@router.post("/assign_client")
def assign_client(req: AssignRequest, db: SessionLocal = Depends(get_db)):
    user = db.query(models.UserGlobal).filter(models.UserGlobal.nrp == req.nrp).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.registered_edge_id = req.edge_id
    db.commit()
    print(f"[REGISTRY] Assigned {req.nrp} to {req.edge_id}")
    return {"status": "success", "nrp": req.nrp, "edge_id": req.edge_id}

@router.post("/start")
async def start_training(rounds: int = 10):
    fl_manager.run_lifecycle(rounds)
    return {"status": "started", "rounds": rounds}

@router.post("/reset")
async def reset_model():
    if fl_manager.running:
         raise HTTPException(status_code=400, detail="Cannot reset while training is running. Please stop training first.")
    
    # Reset FL Manager State
    fl_manager.metrics_history = []
    fl_manager.current_round = 0
    fl_manager.model_size_bytes = 0 
    fl_manager.reset_counter += 1
    
    # Reset Database (Re-init Model)
    db = SessionLocal()
    try:
        # Hapus semua versi model lama
        db.query(ModelVersion).delete()
        
        # Inisialisasi ulang MobileFaceNet (Backbone)
        model = MobileFaceNet(embedding_size=128) 
        weights = model.state_dict()
        buffer = io.BytesIO()
        torch.save(weights, buffer)
        blob = buffer.getvalue()
        
        new_version = ModelVersion(
            head_blob=blob, 
            notes="Reset MobileFaceNet Backbone (Version 0)"
        )
        db.add(new_version)
        db.commit()
        print("[SERVER] Model successfully reset to initial state.")
    except Exception as e:
        db.rollback()
        print(f"[SERVER RESET ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
        
    return {"status": "success", "message": "Model reset to initial state"}

@router.get("/status")
async def get_status():
    # Ambil status FL 
    fl_status = fl_manager.status()

    clients_list = list(registered_clients.values())
    
    return {
        **fl_status, 
        "clients": clients_list
    }

@router.get("/stream_status")
async def stream_status():
    async def event_generator():
        while True:
            # Ambil status terbaru
            fl_status = fl_manager.status()
            clients_list = list(registered_clients.values())
            data = {**fl_status, "clients": clients_list}
            
            # Format SSE format: "data: <json>\n\n"
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1) # Interval pengiriman event

    return StreamingResponse(event_generator(), media_type="text/event-stream")
