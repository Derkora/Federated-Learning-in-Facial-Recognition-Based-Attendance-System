from pydantic import BaseModel
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from app.db.db import SessionLocal, get_db
from app.db import models
from fastapi import APIRouter, Depends
from datetime import datetime

router = APIRouter()

class ClientStatus(BaseModel):
    id: str
    ip_address: str
    fl_status: str       
    last_seen: Any 
    metrics: Optional[Dict[str, Any]] = None    

# Simulasi database sementara
registered_clients = {}  

from app.server_manager_instance import fl_manager

@router.post("/register")
def register_client(client: ClientStatus, db: Session = Depends(get_db)):
    # 1. Update In-Memory Cache (for dashboard list speed)
    registered_clients[client.id] = client.dict()
    
    # 2. Persist to Database Table (for FK integrity)
    try:
        existing = db.query(models.Client).filter(models.Client.edge_id == client.id).first()
        if existing:
            existing.ip_address = client.ip_address
            existing.status = "online"
            existing.last_seen = datetime.utcnow()
        else:
            new_client = models.Client(
                edge_id=client.id,
                ip_address=client.ip_address,
                status="online",
                last_seen=datetime.utcnow()
            )
            db.add(new_client)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] Gagal simpan client {client.id}: {e}")

    # Menyertakan reset_counter dalam response
    return {
        "message": "Client registered", 
        "client": client, 
        "server_reset_counter": fl_manager.reset_counter
    }

@router.get("/")
def get_all_clients():
    return {"clients": list(registered_clients.values())}

@router.get("/{client_id}")
def get_client(client_id: str):
    if client_id not in registered_clients:
        return {"error": "Client not found"}
    return registered_clients[client_id]

@router.post("/{client_id}/update")
def update_client_status(client_id: str, status: dict):
    if client_id not in registered_clients:
        return {"error": "Client not registered"}
    
    registered_clients[client_id].update(status)
    return {"message": "Status updated", "client": registered_clients[client_id]}
