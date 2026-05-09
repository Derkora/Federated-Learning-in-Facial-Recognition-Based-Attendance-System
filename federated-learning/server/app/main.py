import os
import time
import json
import requests
import uvicorn
import base64
import torch
import io
import numpy as np
import math
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks, Body
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from app.db.db import engine, get_db, Base
from app.db.models import FLSession, FLRound, GlobalModel, Client, UserGlobal, AttendanceRecap
Base.metadata.create_all(bind=engine)

from app.server_manager_instance import fl_manager
from app.controllers.fl_controller import FLController
from app.config import REGISTRY_PATH, BN_PATH

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
async def get_system_status(db: Session = Depends(get_db)):
    return fl_manager.get_status(db=db)

# API Hasil Pelatihan Global (Metrik)
@app.get("/api/results")
async def get_results_api(db: Session = Depends(get_db)):
    status = fl_manager.get_status(db=db)
    return status.get("metrics", {})

# Rekapitulasi Presensi Mahasiswa
@app.get("/records", response_class=HTMLResponse)
async def records(request: Request, db: Session = Depends(get_db)):
    all_users = db.query(UserGlobal).order_by(UserGlobal.name.asc()).all()
    tz_wib = timezone(timedelta(hours=7))
    today_start = datetime.now(tz_wib).replace(hour=0, minute=0, second=0, microsecond=0)
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

# Halaman Pengaturan (Settings)
@app.get("/settings", response_class=HTMLResponse)
async def view_settings(request: Request, db: Session = Depends(get_db)):
    status = fl_manager.get_status(db=db)
    return templates.TemplateResponse("settings.html", {"request": request, "title": "System Settings", "status": status})

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
    # Jangan hapus round_history agar tetap persisten di UI
    # fl_manager.metrics["round_history"] = [] 
    
    # Reset metrik sesi berjalan saja
    fl_manager.metrics["compute_energy_kwh"] = 0
    fl_manager.metrics["total_round_time_s"] = 0
    fl_manager.metrics["accuracy"] = 0
    fl_manager.metrics["loss"] = 0
    
    fl_manager.start_phase("Idle")
    fl_manager.discovery_clients.clear()
    fl_manager.update_logs("[INFO] Status server telah di-reset secara manual.")
    return {"status": "reset_ok"}

# Pendaftaran Terminal (Client) Baru
@app.post("/api/clients/register")
async def register_client(request: Request, data: dict, db: Session = Depends(get_db)):
    cid = data.get("id")
    # Gunakan IP pengirim request (request.client.host) agar bisa dihubungi balik oleh server
    # Jika berada di belakang proxy, bisa gunakan header X-Forwarded-For
    ip = request.client.host
    
    if cid:
        data["ip_address"] = ip # Update data dengan IP asli
        fl_manager.registered_clients[cid] = data
        client = db.query(Client).filter_by(edge_id=cid).first()
        if not client:
            client = Client(edge_id=cid, name=cid, ip_address=ip, status="online")
            db.add(client)
        else:
            client.ip_address = ip
            client.status = "online"
            client.last_seen = datetime.now(timezone(timedelta(hours=7)))
        db.commit()
# Endpoint Baru: Menerima Log Inferensi Real-time dari Client (untuk FAR/TAR)
@app.post("/api/logs/inference")
async def receive_inference_log(data: dict):
    """Menerima data hasil identifikasi wajah dari terminal untuk pemantauan terpusat."""
    client_id = data.get("client_id", "unknown")
    user_id = data.get("user_id", "Unknown")
    confidence = data.get("confidence", 0.0)
    latency = data.get("latency_ms", 0)
    status = data.get("status", "UNKNOWN")
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Simpan ke memori manager untuk dashboard
    log_entry = {
        "timestamp": timestamp,
        "client_id": client_id,
        "user_id": user_id,
        "confidence": f"{confidence:.4f}",
        "latency": f"{latency}ms",
        "status": status
    }
    
    if "inference_logs" not in fl_manager.metrics:
        fl_manager.metrics["inference_logs"] = []
        
    fl_manager.metrics["inference_logs"].insert(0, log_entry)
    # Simpan hingga 10.000 baris untuk riwayat riset yang panjang
    fl_manager.metrics["inference_logs"] = fl_manager.metrics["inference_logs"][:10000]
    
    # Simpan ke disk agar persisten
    fl_manager.save_inference_logs()
    
    return {"status": "logged"}

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

# Endpoint Baru: Ambil Log Jarak Jauh dari Client
@app.get("/api/clients/logs/{client_id}")
async def get_client_logs(client_id: str):
    """Mengambil log dari client tertentu via API client tersebut."""
    client_data = fl_manager.registered_clients.get(client_id)
    if not client_data:
        raise HTTPException(status_code=404, detail="Client tidak terdaftar atau offline.")
    
    ip = client_data.get("ip_address")
    try:
        # Panggil endpoint /api/logs di sisi client
        res = requests.get(f"http://{ip}:8080/api/logs", timeout=3)
        if res.status_code == 200:
            return res.json()
        return {"logs": f"Gagal mengambil log: HTTP {res.status_code}"}
    except Exception as e:
        return {"logs": f"Client tidak merespon: {str(e)}"}

# --- AKSES MODEL DAN ASET ---

@app.get("/api/model/backbone")
async def get_backbone_model(db: Session = Depends(get_db)):
    global_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
    if not global_model or not global_model.weights:
        raise HTTPException(status_code=404, detail="Global model weights not found in database.")
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
        "model_version": fl_manager.model_version,
        "inference_threshold": fl_manager.inference_threshold,
        "current_logs": fl_manager.current_logs[-10:]
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
    
    fl_manager.update_logs(f"[SUCCESS] Menerima aset fitur wajah (Centroids) dari {client_id}")
    return {"status": "received"}

# --- DASHBOARD DATA & SINKRONISASI ---

# Mendapatkan Daftar NRP Global untuk Label Mapping
@app.get("/api/training/label_map")
async def get_label_map(db: Session = Depends(get_db)):
    users = db.query(UserGlobal).order_by(UserGlobal.nrp).all()
    return [u.nrp for u in users]

# Mendapatkan Detail Identitas Mahasiswa Global (NRP + Nama + Embedding)
@app.get("/api/training/identities")
async def get_global_identities(db: Session = Depends(get_db)):
    users = db.query(UserGlobal).all()
    results = []
    for u in users:
        item = {"nrp": u.nrp, "name": u.name}
        if u.embedding:
            item["embedding"] = base64.b64encode(u.embedding).decode('utf-8')
        results.append(item)
    return results


# Sinkronisasi Hasil Presensi dari Terminal
@app.post("/api/attendance/sync")
async def sync_attendance(records: List[Dict[str, Any]] = Body(...), db: Session = Depends(get_db)):
    new_records = 0
    errors = []
    # fl_manager.logger.info(f"Menerima {len(records)} data presensi dari terminal.")
    
    for rec in records:
        try:
            nrp = rec.get('user_id')
            if not nrp:
                errors.append("Missing user_id")
                continue
                
            user = db.query(UserGlobal).filter_by(nrp=nrp).first()
            if not user:
                fl_manager.logger.warn(f"User dengan NRP {nrp} tidak ditemukan di database global.")
                errors.append(f"User {nrp} not found")
                continue
            
            # Parsing timestamp dengan lebih aman
            try:
                ts_str = rec['timestamp']
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except Exception as e:
                fl_manager.logger.error(f"Format timestamp salah: {rec.get('timestamp')} | Error: {e}")
                ts = datetime.utcnow()
                
            exists = db.query(AttendanceRecap).filter_by(user_id=user.user_id, timestamp=ts).first()
            if not exists:
                conf = rec.get('confidence', 0.0)
                client_id = rec.get('client_id', 'unknown-edge')
                
                # Pastikan client_id ada di tabel clients untuk menghindari FK error
                client_exists = db.query(Client).filter_by(edge_id=client_id).first()
                if not client_exists:
                    new_client = Client(edge_id=client_id, name=client_id, status="online")
                    db.add(new_client)
                    db.flush()


                new_item = AttendanceRecap(user_id=user.user_id, edge_id=client_id, timestamp=ts, confidence=conf)
                db.add(new_item)
                new_records += 1
        except Exception as e:
            fl_manager.logger.error(f"Gagal memproses record: {e}")
            errors.append(str(e))
            
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        fl_manager.logger.error(f"Gagal commit sinkronisasi: {e}")
        raise HTTPException(status_code=500, detail=f"Database commit failed: {str(e)}")

    return {
        "status": "success" if not errors else "partial_success", 
        "new_records": new_records,
        "errors": errors if errors else None
    }

# API Ketepatan/Threshold Inferensi (Dinamis dari Dashboard)
@app.post("/api/settings")
async def update_settings(data: dict):
    try:
        # Perbarui konfigurasi server secara real-time
        if 'rounds' in data: fl_manager.default_rounds = int(data['rounds'])
        if 'epochs' in data: fl_manager.default_epochs = int(data['epochs'])
        if 'min_clients' in data: fl_manager.default_min_clients = int(data['min_clients'])
        if 'threshold' in data: fl_manager.inference_threshold = float(data['threshold'])
        
        fl_manager.update_logs("[SUCCESS] Pengaturan sistem diperbarui secara dinamis.")
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# API Data Kehadiran Realtime (Untuk records.html Polling)
@app.get("/api/attendance")
async def get_attendance_json(db: Session = Depends(get_db)):
    all_users = db.query(UserGlobal).all()
    tz_wib = timezone(timedelta(hours=7))
    today_start = datetime.now(tz_wib).replace(hour=0, minute=0, second=0, microsecond=0)
    today_recap = db.query(AttendanceRecap).filter(AttendanceRecap.timestamp >= today_start).all()
    recap_map = {rec.user_id: rec for rec in today_recap}
    
    results = []
    for u in all_users:
        rec = recap_map.get(u.user_id)
        results.append({
            "nrp": u.nrp,
            "name": u.name,
            "status": "HADIR" if rec else "MENUNGGU",
            "time": rec.timestamp.strftime("%H:%M:%S") if rec else "-",
            "confidence": f"{rec.confidence:.2f}" if rec else "-",
            "registered_edge_id": rec.edge_id if rec else (u.registered_edge_id or "-")
        })
    return results

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
        "current_logs": fl_manager.current_logs
    }

@app.on_event("startup")
def startup_event():
    fl_manager.logger.success("Server Federated Learning telah siap.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
