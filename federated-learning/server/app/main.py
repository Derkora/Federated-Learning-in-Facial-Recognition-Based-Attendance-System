from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import os
import time
import json
import requests
import uvicorn
from uuid import uuid4
from threading import Thread
from datetime import datetime
import numpy as np
import torch
import math

from .db.db import engine, get_db, Base, SessionLocal
from .db.models import FLSession, FLRound, GlobalModel, Client, UserGlobal, AttendanceRecap
from .server import start_flower_server

# Initialize Database
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Federated Learning Server Dashboard")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Global state
active_session_id = None
is_server_running = False
current_phase = "idle"
phase_logs = []
registered_clients = {}

def add_phase_log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    phase_logs.append(f"[{ts}] {msg}")
    print(f"PHASE LOG: {msg}")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    sessions = db.query(FLSession).order_by(FLSession.start_time.desc()).all()
    attendance_count = db.query(AttendanceRecap).count()
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "sessions": sessions,
        "is_running": is_server_running,
        "active_session_id": active_session_id,
        "attendance_count": attendance_count
    })

@app.get("/records", response_class=HTMLResponse)
async def records(request: Request, db: Session = Depends(get_db)):
    # Get all students (UserGlobal)
    all_users = db.query(UserGlobal).order_by(UserGlobal.name.asc()).all()
    
    # Get today's attendance records (AttendanceRecap)
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_recap = db.query(AttendanceRecap).filter(AttendanceRecap.timestamp >= today_start).all()
    
    # Map for easy lookup
    recap_map = {rec.user_id: rec for rec in today_recap}
    
    records_display = []
    for u in all_users:
        rec = recap_map.get(u.user_id)
        records_display.append({
            "nrp": u.nrp,
            "name": u.name,
            "status": "PRESENT" if rec else "ABSENT",
            "timestamp": rec.timestamp if rec else None,
            "confidence": rec.confidence if rec else 0.0,
            "edge_id": rec.edge_id if rec else u.registered_edge_id
        })
        
    return templates.TemplateResponse("records.html", {
        "request": request,
        "records": records_display
    })

@app.post("/api/fl/start")
async def start_fl_training(rounds: int = 10, min_clients: int = 2):
    global active_session_id, is_server_running, current_phase
    
    if is_server_running:
        return {"status": "already_running"}
    
    session_id = str(uuid4())[:8]
    active_session_id = session_id
    is_server_running = True
    phase_logs.clear()
    
    # Save session to DB
    db = SessionLocal()
    new_session = FLSession(session_id=session_id)
    db.add(new_session)
    db.commit()
    db.close()
    
    def run_lifecycle():
        global is_server_running, current_phase
        try:
            add_phase_log("🚀 Starting Unified FL Lifecycle...")
            
            # Helper to check client readiness
            def wait_for_clients(target_status, timeout=300):
                start_w = time.time()
                while time.time() - start_w < timeout:
                    ready = [c for c in registered_clients.values() if target_status.lower() in (c.get('fl_status', '')).lower()]
                    if len(ready) >= min_clients:
                        return True
                    time.sleep(2)
                return False

            # PREPROCESSING 
            current_phase = "preprocessing"
            add_phase_log("Phase 1: Instructing clients to perform face extraction and registration...")
            if not wait_for_clients("Siap Training", timeout=600):
                 add_phase_log("❌ Phase 1 Failed: Not all clients finished preprocessing. Aborting.")
                 return 

            # SYNCING 
            current_phase = "syncing"
            add_phase_log("Phase 2: Instructing clients to synchronize collective student data...")
            if not wait_for_clients("Siap Preprocess", timeout=300):
                add_phase_log("❌ Phase 2 Failed: Not all clients synced in time. Aborting.")
                return 

            # TRAINING
            current_phase = "training"
            add_phase_log(f"Phase 3: Starting Flower Server Training (Rounds: {rounds}, Min Clients: {min_clients})...")
            start_flower_server(session_id, rounds=rounds, min_clients=min_clients)
            
            current_phase = "completed"
            add_phase_log("✅ Federated Training Cycle Finished Successfully.")

        except Exception as e:
            add_phase_log(f"❌ Lifecycle Error: {e}")
        finally:
            is_server_running = False
            current_phase = "idle"
            db = SessionLocal()
            session = db.query(FLSession).filter_by(session_id=session_id).first()
            if session:
                session.status = "completed"
                db.commit()
            db.close()

    Thread(target=run_lifecycle, daemon=True).start()
    return {"status": "started", "session_id": session_id}

@app.post("/api/clients/register")
async def register_client(data: dict, db: Session = Depends(get_db)):
    cid = data.get("id")
    ip = data.get("ip_address", "0.0.0.0")
    
    if cid:
        registered_clients[cid] = data
        # Sync to DB
        client = db.query(Client).filter_by(edge_id=cid).first()
        if not client:
            client = Client(edge_id=cid, name=cid, ip_address=ip, status="online")
            db.add(client)
        else:
            client.ip_address = ip
            client.status = "online"
            client.last_seen = datetime.utcnow()
        db.commit()
        
    return {"status": "ok", "server_time": time.time()}

from fastapi.responses import Response

@app.get("/api/model/backbone")
async def get_backbone_model(db: Session = Depends(get_db)):
    global_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
    if not global_model or not global_model.weights:
        raise HTTPException(status_code=404, detail="Global model not found")
    return Response(content=global_model.weights, media_type="application/octet-stream")

@app.get("/api/training/status")
async def get_training_status():
    return {
        "current_phase": current_phase,
        "is_running": is_server_running,
        "active_session_id": active_session_id,
        "logs": phase_logs[-10:]
    }

@app.post("/api/training/get_label")
async def get_label(data: dict, db: Session = Depends(get_db)):
    nrp = data.get("nrp")
    name = data.get("name", "Unknown")
    edge_id = data.get("client_id", "edge-1")
    embedding_b64 = data.get("embedding")
    
    embedding_bytes = None
    if embedding_b64:
        import base64
        embedding_bytes = base64.b64decode(embedding_b64)

    user = db.query(UserGlobal).filter_by(nrp=nrp).first()
    if not user:
        user = UserGlobal(nrp=nrp, name=name, registered_edge_id=edge_id, embedding=embedding_bytes)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        user.name = name
        if embedding_bytes:
            user.embedding = embedding_bytes
        db.commit()
        
    export_reference_embeddings()

    return {"nrp": nrp, "label": user.user_id}

@app.get("/api/users/global")
async def get_global_users(db: Session = Depends(get_db)):
    users = db.query(UserGlobal).all()
    import base64
    return [{
        "user_id": u.user_id,
        "nrp": u.nrp,
        "name": u.name,
        "embedding": base64.b64encode(u.embedding).decode('utf-8') if u.embedding else None
    } for u in users]

@app.post("/api/attendance/sync")
async def sync_attendance(records: list, db: Session = Depends(get_db)):
    new_records = 0
    for rec in records:
        # Resolve user_id from nrp (if needed, but usually rec['user_id'] is the NRP or Global ID)
        # Assuming rec['user_id'] on the client side now matches UserGlobal.user_id or nrp
        nrp = rec['user_id']
        user = db.query(UserGlobal).filter_by(nrp=nrp).first()
        if not user: continue
        
        ts = datetime.fromisoformat(rec['timestamp'])
        exists = db.query(AttendanceRecap).filter_by(
            user_id=user.user_id, 
            timestamp=ts
        ).first()
        
        if not exists:
            new_item = AttendanceRecap(
                user_id=user.user_id,
                edge_id=rec['client_id'],
                timestamp=ts,
                confidence=rec['confidence']
            )
            db.add(new_item)
            new_records += 1
    
    db.commit()
    return {"status": "success", "new_records": new_records}

@app.get("/api/fl/status/{session_id}")
async def get_fl_status(session_id: str, db: Session = Depends(get_db)):
    rounds = db.query(FLRound).filter_by(session_id=session_id).order_by(FLRound.round_number.asc()).all()
    history = []
    for r in rounds:
        m = json.loads(r.metrics) if r.metrics else {}
        history.append({
            "round": r.round_number, 
            "loss": r.loss,
            "accuracy": m.get("accuracy", 0.0)
        })
    
    # Also fetch active students from clients for the dashboard
    clients = ["http://client1-fl:8080", "http://client2-fl:8080"]
    received_data = []
    for c_url in clients:
        try:
            res = requests.get(f"{c_url}/api/users", timeout=1)
            if res.status_code == 200:
                received_data.extend(res.json())
        except:
            pass
            
    # Sanitize history for JSON (NaN is not JSON compliant)
    sanitized_history = []
    for h in history:
        sanitized_h = h.copy()
        if isinstance(h["loss"], float) and (math.isnan(h["loss"]) or math.isinf(h["loss"])):
            sanitized_h["loss"] = 0.0
        if isinstance(h["accuracy"], float) and (math.isnan(h["accuracy"]) or math.isinf(h["accuracy"])):
            sanitized_h["accuracy"] = 0.0
        sanitized_history.append(sanitized_h)
            
    return {
        "session_id": session_id,
        "rounds_completed": len(rounds),
        "history": sanitized_history,
        "received_data": list(set(received_data)),
        "phase": current_phase,
        "phase_logs": phase_logs
    }

def export_latest_model_to_disk():
    """Exports the latest global model from DB to disk for visibility/backup."""
    db = SessionLocal()
    try:
        global_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
        if global_model and global_model.weights:
            import io
            import torch
            import numpy as np
            
            os.makedirs("data", exist_ok=True)
            # weights are saved as ndarrays in the buffer
            weights_np = torch.load(io.BytesIO(global_model.weights))
            torch.save(weights_np, "data/backbone.pth")
            print(f"[STARTUP] Exported global model (v{global_model.version}) to data/backbone.pth")
    except Exception as e:
        print(f"[STARTUP] Failed to export model to disk: {e}")
    finally:
        db.close()

def export_reference_embeddings():
    """Exports all UserGlobal embeddings to data/reference_embeddings.pth."""
    db = SessionLocal()
    try:
        users = db.query(UserGlobal).filter(UserGlobal.embedding.isnot(None)).all()
        ref_dict = {}
        for u in users:
            # Convert LargeBinary to numpy then to torch
            emb_np = np.frombuffer(u.embedding, dtype=np.float32)
            ref_dict[u.nrp] = torch.from_numpy(emb_np.copy())
        
        os.makedirs("data", exist_ok=True)
        torch.save(ref_dict, "data/reference_embeddings.pth")
        print(f"[EXPORT] Saved {len(users)} embeddings to data/reference_embeddings.pth")
    except Exception as e:
        print(f"[EXPORT ERROR] {e}")
    finally:
        db.close()

@app.on_event("startup")
def startup_event():
    print(f"[STARTUP] FL Server Dashboard Initialized", flush=True)
    export_latest_model_to_disk()
    export_reference_embeddings()

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
