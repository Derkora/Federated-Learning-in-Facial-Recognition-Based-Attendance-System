from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from app.server_manager_instance import fl_manager
from app.controllers.client_controller import registered_clients
from app.db.db import SessionLocal
from app.db.models import ModelVersion
from app.utils.mobilefacenet import MobileFaceNet
import torch
import io
import asyncio
import json

router = APIRouter()

global_user_db = {}

class LabelRequest(BaseModel):
    nrp: str
    name: str = ""
    registered_edge_id: Optional[str] = None 

@router.post("/get_label")
def get_or_create_label(req: LabelRequest):
    global global_user_db
    
    if req.nrp not in global_user_db:
        new_label = len(global_user_db)
        if new_label >= 100: 
            raise HTTPException(status_code=400, detail="Full")
            
        global_user_db[req.nrp] = {
            "label": new_label,
            "name": req.name,
            "nrp": req.nrp,
            # Simpan Edge ID saat registrasi pertama
            "registered_edge_id": req.registered_edge_id 
        }
        print(f"[REGISTRY] New: {req.name} ({req.nrp}) -> Label {new_label} on {req.registered_edge_id}")
        
    else:
        print(f"[REGISTRY] Existing: {req.name} ({req.nrp}) -> Label {global_user_db[req.nrp]['label']} on {req.registered_edge_id}")
        existing_data = global_user_db[req.nrp]
        if existing_data.get("registered_edge_id") and existing_data.get("registered_edge_id") != req.registered_edge_id and req.registered_edge_id:
             raise HTTPException(status_code=403, detail=f"User already registered on {existing_data.get('registered_edge_id')}")

    return global_user_db[req.nrp]

@router.get("/global_users")
def get_all_users():
    return list(global_user_db.values())

@router.delete("/users/{nrp}")
def delete_user(nrp: str):
    global global_user_db
    if nrp not in global_user_db:
        raise HTTPException(status_code=404, detail="User found")
    
    del global_user_db[nrp]
    print(f"[REGISTRY] User deleted: {nrp}")
    return {"status": "success", "message": f"User {nrp} deleted"}

@router.post("/start")
async def start_training(rounds: int = 10):
    fl_manager.start_training(rounds)
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
