import os
import time
import json
import uvicorn
import base64
import torch
import io
import numpy as np
import math
from datetime import datetime
from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.db.db import engine, get_db, Base
from app.db.models import FLSession, FLRound, GlobalModel, Client, UserGlobal, AttendanceRecap
from app.server_manager_instance import fl_manager
from app.controllers.fl_controller import FLController
from app.config import REGISTRY_PATH, BN_PATH

# Inisialisasi Database Server
Base.metadata.create_all(bind=engine)

# Konfigurasi Aplikasi FastAPI
app = FastAPI(title="Federated Learning Server Dashboard")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Inisialisasi Controller FL
fl_controller = FLController(fl_manager)

# Dashboard Utama
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    status = fl_manager.get_status(db=db)
    return templates.TemplateResponse("index.html", {"request": request, "title": "Dashboard", "status": status})

# Status Sistem Global
@app.get("/api/status")
async def get_system_status():
    return fl_manager.get_status()

# API Hasil Pelatihan Global (Metrik)
@app.get("/api/results")
async def get_results_api(db: Session = Depends(get_db)):
    status = fl_manager.get_status(db=db)
    return status.get("metrics", {})

# Rekapitulasi Presensi Mahasiswa
@app.get("/records", response_class=HTMLResponse)
async def records(request: Request, db: Session = Depends(get_db)):
    all_users = db.query(UserGlobal).order_by(UserGlobal.name.asc()).all()
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    today_recap = db.query(AttendanceRecap).filter(AttendanceRecap.timestamp >= today_start).all()
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

# Halaman Hasil Evaluasi (Copy-Paste friendly)
@app.get("/results", response_class=HTMLResponse)
async def view_results(request: Request, db: Session = Depends(get_db)):
    status = fl_manager.get_status(db=db)
    return templates.TemplateResponse("results.html", {"request": request, "title": "Evaluation Results", "status": status})

# --- OPERASI FEDERATED LEARNING ---

# Memulai Siklus Pelatihan Federated Learning
@app.post("/api/fl/start")
async def start_fl_training(rounds: int = None, min_clients: int = None, epochs: int = None):
    # Logika orkestrasi didelegasikan ke fl_controller
    rounds = rounds or fl_manager.default_rounds
    min_clients = min_clients or fl_manager.default_min_clients
    return fl_controller.start_lifecycle(rounds, min_clients, epochs)

# Mengatur Ulang (Reset) Status Server
@app.post("/api/fl/reset")
async def reset_fl_state():
    fl_manager.is_running = False
    fl_manager.start_phase("Idle")
    fl_manager.current_logs = []
    fl_manager.registry_submissions.clear()
    fl_manager.ready_clients.clear()
    fl_manager.discovery_clients.clear()
    fl_manager.update_logs("[OK] Status server telah di-reset secara manual.")
    return {"status": "reset_ok"}

# Pendaftaran Terminal (Client) Baru
@app.post("/api/clients/register")
async def register_client(data: dict, db: Session = Depends(get_db)):
    cid = data.get("id")
    ip = data.get("ip_address", "0.0.0.0")
    if cid:
        fl_manager.registered_clients[cid] = data
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

# Laporan Penyelesaian Tahap Discovery dari Terminal
@app.post("/api/clients/discovery_done")
async def report_discovery_done(data: dict):
    cid = data.get("client_id")
    if cid:
        fl_manager.discovery_clients.add(cid)
    return {"status": "ok"}

# Laporan Kesiapan Terminal untuk Training (READY)
@app.post("/api/clients/ready")
async def report_client_ready(data: dict):
    cid = data.get("client_id")
    if cid:
        fl_manager.ready_clients.add(cid)
    return {"status": "ok"}

# --- AKSES MODEL DAN ASET ---

# Mengunduh Bobot Model Backbone (MobileFaceNet)
@app.get("/api/model/backbone")
async def get_backbone_model(db: Session = Depends(get_db)):
    global_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
    if not global_model or not global_model.weights:
        raise HTTPException(status_code=404, detail="Global model not found")
    return Response(content=global_model.weights, media_type="application/octet-stream")

# Mengunduh Parameter Batch Normalization (BN)
@app.get("/api/model/bn")
async def get_bn_model():
    if os.path.exists(BN_PATH):
        with open(BN_PATH, "rb") as f:
            return Response(content=f.read(), media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="BN Assets not found")

# Mengunduh Pustaka Identitas Global (Registry)
@app.get("/api/model/registry")
async def get_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "rb") as f:
            return Response(content=f.read(), media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Registry not found")

# Mendapatkan Status Terkini Pelatihan
@app.get("/api/training/status")
async def get_training_status():
    return {
        "current_phase": fl_manager.current_phase,
        "is_running": fl_manager.is_running,
        "active_session_id": fl_manager.session_id,
        "logs": fl_manager.current_logs[-10:]
    }

# Sinkronisasi Label Mahasiswa (ID Mapping)
@app.post("/api/training/get_label")
async def get_label(data: dict, db: Session = Depends(get_db)):
    # Digunakan untuk memberikan ID numerik unik ke setiap mahasiswa baru
    # yang didaftarkan di client mana pun.
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
    
    if nrp not in fl_manager.received_data:
        fl_manager.received_data.append(nrp)
        
    return {"nrp": nrp, "label": user.user_id}

@app.post("/api/training/registry_assets")
async def receive_registry_assets(data: dict, db: Session = Depends(get_db)):
    client_id = data.get("client_id")
    serialized_bn = data.get("bn_params")
    centroids = data.get("centroids")
    
    bn_bytes = base64.b64decode(serialized_bn)
    bn_params = torch.load(io.BytesIO(bn_bytes), map_location="cpu")
    
    decoded_centroids = {}
    for nrp, b64_vec in centroids.items():
        vec_bytes = base64.b64decode(b64_vec)
        decoded_centroids[nrp] = np.frombuffer(vec_bytes, dtype=np.float32)
        
    fl_manager.registry_submissions[client_id] = {
        "bn": bn_params,
        "centroids": decoded_centroids
    }
    
    submission_dir = "data/submissions"
    os.makedirs(submission_dir, exist_ok=True)
    with open(os.path.join(submission_dir, f"{client_id}_assets.pth"), "wb") as f:
        torch.save(fl_manager.registry_submissions[client_id], f)
    
    fl_manager.update_logs(f"[OK] Menerima aset fitur wajah (Centroids) dari {client_id}")
    return {"status": "received"}

# --- DASHBOARD DATA & SINKRONISASI ---

# Mendapatkan Daftar NRP Global untuk Label Mapping
@app.get("/api/training/label_map")
async def get_label_map(db: Session = Depends(get_db)):
    users = db.query(UserGlobal).order_by(UserGlobal.nrp).all()
    return [u.nrp for u in users]

# Sinkronisasi Hasil Presensi dari Terminal
@app.post("/api/attendance/sync")
async def sync_attendance(records: list, db: Session = Depends(get_db)):
    new_records = 0
    for rec in records:
        nrp = rec['user_id']
        user = db.query(UserGlobal).filter_by(nrp=nrp).first()
        if not user: continue
        
        ts = datetime.fromisoformat(rec['timestamp'])
        exists = db.query(AttendanceRecap).filter_by(user_id=user.user_id, timestamp=ts).first()
        if not exists:
            new_item = AttendanceRecap(user_id=user.user_id, edge_id=rec['client_id'], timestamp=ts, confidence=rec['confidence'])
            db.add(new_item)
            new_records += 1
    db.commit()
    return {"status": "success", "new_records": new_records}

# Mendapatkan Detail Status Sesi Pelatihan
@app.get("/api/fl/status/{session_id}")
async def get_fl_status(session_id: str, db: Session = Depends(get_db)):
    rounds = db.query(FLRound).filter_by(session_id=session_id).order_by(FLRound.round_number.asc()).all()
    history = []
    for r in rounds:
        m = json.loads(r.metrics) if r.metrics else {}
        history.append({"round": r.round_number, "loss": r.loss, "accuracy": m.get("accuracy", 0.0)})
            
    sanitized_history = []
    for h in history:
        sanitized_h = h.copy()
        if isinstance(h["loss"], float) and (math.isnan(h["loss"]) or math.isinf(h["loss"])): sanitized_h["loss"] = 0.0
        if isinstance(h["accuracy"], float) and (math.isnan(h["accuracy"]) or math.isinf(h["accuracy"])): sanitized_h["accuracy"] = 0.0
        sanitized_history.append(sanitized_h)
            
    return {
        "session_id": session_id,
        "rounds_completed": len(rounds),
        "history": sanitized_history,
        "phase": fl_manager.current_phase,
        "phase_logs": fl_manager.current_logs
    }

@app.on_event("startup")
def startup_event():
    print(f"[STARTUP] Federated Learning Server Initialized", flush=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
