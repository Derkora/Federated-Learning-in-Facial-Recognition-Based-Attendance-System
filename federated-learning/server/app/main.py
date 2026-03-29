from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, Response
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
import base64
import io

from .db.db import engine, get_db, Base, SessionLocal
from .db.models import FLSession, FLRound, GlobalModel, Client, UserGlobal, AttendanceRecap
from .server import start_flower_server
from .server_manager_instance import fl_manager
from collections import defaultdict

# Initialize Database
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Federated Learning Server Dashboard")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Helper to bridge to fl_manager
def add_phase_log(msg):
    fl_manager.update_logs(msg)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    status = fl_manager.get_status(db=db)
    return templates.TemplateResponse("index.html", {"request": request, "title": "Dashboard", "status": status})

@app.get("/api/status")
async def get_system_status():
    return fl_manager.get_status()
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
async def start_fl_training(rounds: int = None, min_clients: int = None, epochs: int = None):
    # Use centralized defaults from manager if not provided
    rounds = rounds or fl_manager.default_rounds
    min_clients = min_clients or fl_manager.default_min_clients
    if epochs: fl_manager.default_epochs = epochs # Temporary override for this session
    
    if fl_manager.is_busy or fl_manager.is_running:
        return {"status": "already_running"}
    
    session_id = f"session_{int(time.time())}"
    fl_manager.session_id = session_id
    fl_manager.current_logs = [] # Clear logs for new session
    fl_manager.received_data = [] # Clear student list for new session
    
    # Save session to DB
    db = SessionLocal()
    new_session = FLSession(session_id=session_id)
    db.add(new_session)
    db.commit()
    db.close()

    def orchestrate():
        try:
            db = SessionLocal()
            fl_manager.start_phase("Data Preparation")
            fl_manager.ensure_model_seeded(db)
            db.close()
            
            fl_manager.registry_submissions.clear()
            
            add_phase_log("🚀 Starting One-Button Full Lifecycle [Federated]...")
            add_phase_log("📡 Triggering Remote Data Prep (Sync & Preprocess)...")
            fl_manager.ready_clients.clear()
            trigger_all_clients_command("/api/request-data")
            
            # Wait for clients to signal READY (with 10-min timeout)
            add_phase_log("⏳ Waiting for Clients to complete Sync + MTCNN Preprocessing...")
            start_wait = time.time()
            max_wait = 600 
            
            while len(fl_manager.ready_clients) < min_clients:
                if time.time() - start_wait > max_wait:
                    add_phase_log("⚠️ Preparation Timeout (10m). Proceeding with available clients...")
                    break
                time.sleep(5)
            
            fl_manager.start_phase("Training")
            add_phase_log(f"🎓 Starting Flower Federated Training with {len(fl_manager.ready_clients)} ready clients...")
            fl_manager.start_training(session_id, rounds=rounds, min_clients=min_clients)
            
            fl_manager.start_phase("Completed")
            add_phase_log("✅ Full Lifecycle Finished.")

        except Exception as e:
            add_phase_log(f"❌ Lifecycle Error: {e}")
        finally:
            fl_manager.end_phase()
            db = SessionLocal()
            session = db.query(FLSession).filter_by(session_id=session_id).first()
            if session:
                session.status = "completed"
                db.commit()
            db.close()

    Thread(target=orchestrate, daemon=True).start()
    return {"status": "started", "session_id": session_id}

def trigger_all_clients_command(endpoint):
    """Broadcast a command to all registered clients via their API."""
    for cid, data in fl_manager.registered_clients.items():
        ip = data.get("ip_address")
        if ip:
            url = f"http://{ip}:8080{endpoint}" 
            try:
                requests.post(url, timeout=2)
                print(f"[ORCHESTRATOR] Sent {endpoint} to {cid} at {ip}")
            except Exception as e:
                print(f"[ORCHESTRATOR ERROR] Could not signal client {cid}: {e}")

@app.post("/api/clients/register")
async def register_client(data: dict, db: Session = Depends(get_db)):
    cid = data.get("id")
    ip = data.get("ip_address", "0.0.0.0")
    
    if cid:
        fl_manager.registered_clients[cid] = data
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

@app.post("/api/clients/ready")
async def report_client_ready(data: dict):
    cid = data.get("client_id")
    if cid:
        fl_manager.ready_clients.add(cid)
        print(f"[ORCHESTRATOR] Client {cid} reported READY.")
    return {"status": "ok"}

@app.get("/api/model/backbone")
async def get_backbone_model(db: Session = Depends(get_db)):
    global_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
    if not global_model or not global_model.weights:
        raise HTTPException(status_code=404, detail="Global model not found")
    return Response(content=global_model.weights, media_type="application/octet-stream")

@app.get("/api/model/bn")
async def get_bn_model():
    path = "data/global_bn_combined.pth"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return Response(content=f.read(), media_type="application/octet-stream")
    return {"error": "BN Assets not found"}

@app.get("/api/model/registry")
async def get_registry():
    path = "data/global_embedding_registry.pth"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return Response(content=f.read(), media_type="application/octet-stream")
    return {"error": "Registry not found"}

@app.get("/api/training/status")
async def get_training_status():
    return {
        "current_phase": fl_manager.current_phase,
        "is_running": fl_manager.is_running,
        "active_session_id": fl_manager.session_id,
        "logs": fl_manager.current_logs[-10:]
    }

@app.post("/api/training/get_label")
async def get_label(data: dict, db: Session = Depends(get_db)):
    nrp = data.get("nrp")
    name = data.get("name", "Unknown")
    edge_id = data.get("client_id", "edge-1")
    embedding_b64 = data.get("embedding")
    
    embedding_bytes = None
    if embedding_b64:
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
    
    # Add to Dashboard Student List
    if nrp not in fl_manager.received_data:
        fl_manager.received_data.append(nrp)
        
    return {"nrp": nrp, "label": user.user_id}

@app.post("/api/training/registry_assets")
async def receive_registry_assets(data: dict, db: Session = Depends(get_db)):
    """
    Registry Updates: Automated Asset Collection
    Receives BN parameters and Centroids from clients at Final Round.
    """
    client_id = data.get("client_id")
    rnd = data.get("round")
    serialized_bn = data.get("bn_params")
    centroids = data.get("centroids")
    
    # Decode BN
    bn_bytes = base64.b64decode(serialized_bn)
    bn_params = torch.load(io.BytesIO(bn_bytes), map_location="cpu")
    
    # Decode Centroids
    decoded_centroids = {}
    for nrp, b64_vec in centroids.items():
        vec_bytes = base64.b64decode(b64_vec)
        decoded_centroids[nrp] = np.frombuffer(vec_bytes, dtype=np.float32)
        
    # Store in global state for aggregation
    fl_manager.registry_submissions[client_id] = {
        "bn": bn_params,
        "centroids": decoded_centroids
    }
    
    add_phase_log(f"Received Registry assets from {client_id} (Round {rnd}, {len(decoded_centroids)} centroids)")
    
    # 3. Aggregation Trigger (when we have enough clients)
    # In a real scenario, we'd wait for all min_clients. For now, let's trigger it.
    if len(fl_manager.registry_submissions) >= 2: # Assuming 2 clients as per compose
        aggregate_and_save_registry_assets()
        
    return {"status": "received"}

def aggregate_and_save_registry_assets():
    """
    Registry Generation: Combined BN & Centroid Registry
    Averages BN stats and merges centroids into final global assets.
    """
    try:
        add_phase_log("⚙️ Starting Registry Generation: Asset Aggregation...")
        
        # Combine BN (Averaging)
        clients_bn = [sub['bn'] for sub in fl_manager.registry_submissions.values()]
        global_bn = {}
        bn_keys = list(clients_bn[0].keys())
        
        for key in bn_keys:
            # Stats like num_batches_tracked are long/int
            # We just take the value from the first client or average them
            if clients_bn[0][key].dtype == np.int64 or 'num_batches_tracked' in key:
                global_bn[key] = torch.from_numpy(clients_bn[0][key])
            else:
                # Stack and mean (CPU Only)
                tensors = [torch.from_numpy(bn[key]) for bn in clients_bn]
                global_bn[key] = torch.stack(tensors).mean(0)
        
        os.makedirs("data", exist_ok=True)
        torch.save(global_bn, "data/global_bn_combined.pth")
        add_phase_log("✅ Generated data/global_bn_combined.pth")
        
        # Centroid Registry (Merge all)
        all_centroids = {}
        for client_id, sub in fl_manager.registry_submissions.items():
            for nrp, vec in sub['centroids'].items():
                all_centroids[nrp] = torch.from_numpy(vec)
                
        torch.save(all_centroids, "data/global_embedding_registry.pth")
        add_phase_log(f"✅ Generated data/global_embedding_registry.pth ({len(all_centroids)} identities)")
        
        if len(all_centroids) == 0:
            add_phase_log("⚠️ WARNING: Generated registry is EMPTY. Check if clients have training data.")
        
    except Exception as e:
        add_phase_log(f"Aggregation Error: {e}")
        import traceback
        traceback.print_exc()

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
        nrp = rec['user_id']
        user = db.query(UserGlobal).filter_by(nrp=nrp).first()
        if not user: continue
        
        # Add to Dashboard Student List if not already there
        if nrp not in fl_manager.received_data:
            fl_manager.received_data.append(nrp)
        
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
        "phase": fl_manager.current_phase,
        "phase_logs": fl_manager.current_logs
    }

@app.on_event("startup")
def startup_event():
    print(f"[STARTUP] FL Server Dashboard Initialized", flush=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
