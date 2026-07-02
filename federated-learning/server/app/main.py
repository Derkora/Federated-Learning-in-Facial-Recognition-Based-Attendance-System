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
import subprocess
import shutil
import gc
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks, Body
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from app.db.db import engine, get_db, Base
from app.db.models import FLSession, FLRound, GlobalModel, Client, UserGlobal, AttendanceRecap
Base.metadata.create_all(bind=engine)

# Jalankan migrasi kolom dataset secara aman
try:
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE users_global ADD COLUMN dataset VARCHAR DEFAULT 'students'"))
except Exception:
    pass

from app.server_manager_instance import fl_manager
from app.controllers.fl_controller import FLController
from app.config import REGISTRY_PATH, BN_PATH

# Konfigurasi endpoint API
api_status = "/api/status"
api_results = "/api/results"
api_fl_start = "/api/fl/start"
api_fl_reset = "/api/fl/reset"
api_clients_register = "/api/clients/register"
api_logs_inference = "/api/logs/inference"
api_clients_discovery_done = "/api/clients/discovery_done"
api_clients_ready = "/api/clients/ready"
api_clients_logs = "/api/clients/logs/{client_id}"
api_model_backbone = "/api/model/backbone"
api_model_bn = "/api/model/bn"
api_model_registry = "/api/model/registry"
api_training_status = "/api/training/status"
api_training_get_label = "/api/training/get_label"
api_training_registry_assets = "/api/training/registry_assets"
api_training_label_map = "/api/training/label_map"
api_training_identities = "/api/training/identities"
api_attendance_sync = "/api/attendance/sync"
api_settings = "/api/settings"
api_attendance = "/api/attendance"
api_logs = "/api/logs"

api_video = "/api/video"
api_video_upload = f"{api_video}/upload"
api_video_list = f"{api_video}/list"
api_video_stream = f"{api_video}/stream/{{video_name}}"
api_video_cache = f"{api_video}/cache/{{video_name}}"
api_video_delete = f"{api_video}/delete/{{video_name}}"

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
@app.get(api_status)
async def get_system_status(db: Session = Depends(get_db)):
    return fl_manager.get_status(db=db)

@app.get("/api/models/list")
def list_models():
    # Mengembalikan daftar model yang tersedia untuk absensi atau uji coba
    models_list = []
    models_list.append({"version": "v0", "rounds": 0, "epochs": 0, "file": "global_model_v0.pth", "label": "Model v0 (Base Model)"})
    from app.config import DATA_ROOT
    if os.path.exists(DATA_ROOT):
        for f in os.listdir(DATA_ROOT):
            if f.startswith("global_model_") and f.endswith(".pth") and f not in ("global_model_v0.pth", "global_model.pth"):
                version_str = f.replace("global_model_", "").replace(".pth", "")
                parts = version_str.split("_")
                learning_type = parts[0]
                dataset_str = parts[1] if len(parts) >= 2 else "unknown"
                
                if learning_type == "cl":
                    epochs_str = parts[2].replace("e", "") if len(parts) >= 3 else "0"
                    label = f"Model Centralized | Dataset: {dataset_str} | Epoch: {epochs_str}"
                    epochs_val = int(epochs_str) if epochs_str.isdigit() else 0
                    rounds_val = 0
                elif learning_type == "fl":
                    rounds_str = parts[2].replace("r", "") if len(parts) >= 3 else "0"
                    epochs_str = parts[3].replace("e", "") if len(parts) >= 4 else "0"
                    label = f"Model Federated | Dataset: {dataset_str} | Round: {rounds_str} | Epoch: {epochs_str}"
                    rounds_val = int(rounds_str) if rounds_str.isdigit() else 0
                    epochs_val = int(epochs_str) if epochs_str.isdigit() else 0
                else:
                    label = f"Model {version_str}"
                    epochs_val = 0
                    rounds_val = 0
                    
                models_list.append({
                    "version": version_str,
                    "dataset": dataset_str,
                    "rounds": rounds_val,
                    "epochs": epochs_val,
                    "file": f,
                    "label": label
                })
                
    # Custom sorting: versioned (descending), v0 (last)
    v0_models = [m for m in models_list if m["version"] == "v0"]
    other_models = [m for m in models_list if m["version"] != "v0"]
    other_models.sort(key=lambda x: x["version"], reverse=True)
    
    models_list = other_models + v0_models
    return {"status": "success", "models": models_list}

# API Hasil Pelatihan Global (Metrik)
@app.get(api_results)
async def get_results_api(version: str = None, db: Session = Depends(get_db)):
    if version:
        if version == "v0" or version == "0":
            return {"round_history": [], "accuracy": 0.0, "loss": 0.0, "unique_client_ids": []}
        elif version == "active":
            status = fl_manager.get_status(db=db)
            return status.get("metrics", {})
            
        version_metrics = fl_manager.load_version_metrics(version)
        if version_metrics:
            return version_metrics
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

# Halaman Upload & Manajemen Video Simulasi
@app.get("/video-simulasi", response_class=HTMLResponse)
async def view_video_simulasi(request: Request):
    """Merender halaman manajemen video simulasi"""
    return templates.TemplateResponse("video_simulasi.html", {"request": request, "title": "Video Simulasi"})

# Memulai Siklus Pelatihan Federated Learning
@app.post(api_fl_start)
async def start_fl_training(rounds: int = None, min_clients: int = None, epochs: int = None, dataset: str = "students"):
    # Logika orkestrasi didelegasikan ke fl_controller
    rounds = rounds or fl_manager.default_rounds
    min_clients = min_clients or fl_manager.default_min_clients
    epochs = epochs or fl_manager.default_epochs
    fl_manager.reset_training_metrics()
    return fl_controller.start_lifecycle(rounds, min_clients, epochs, dataset)

@app.post("/api/fl/preprocess")
async def start_fl_preprocess(min_clients: int = None, dataset: str = "students"):
    min_clients = min_clients or fl_manager.default_min_clients
    return fl_controller.start_preprocess(min_clients, dataset)

@app.post("/api/fl/train")
async def start_fl_train(rounds: int = None, min_clients: int = None, epochs: int = None, dataset: str = "students"):
    rounds = rounds or fl_manager.default_rounds
    min_clients = min_clients or fl_manager.default_min_clients
    epochs = epochs or fl_manager.default_epochs
    fl_manager.reset_training_metrics()
    return fl_controller.start_train(rounds, min_clients, epochs, dataset)

# Mengatur Ulang (Reset) Status Server
@app.post(api_fl_reset)
async def reset_fl_state():
    fl_manager.is_running = False
    # Jangan hapus round_history agar tetap persisten di UI
    # fl_manager.metrics["round_history"] = [] 
    
    # Reset metrik sesi berjalan saja
    fl_manager.metrics["compute_energy_kwh"] = 0
    fl_manager.metrics["total_round_time_s"] = 0
    fl_manager.metrics["accuracy"] = 0
    fl_manager.metrics["loss"] = 0
    
    fl_manager.end_phase()
    fl_manager.discovery_clients.clear()
    fl_manager.update_logs("[INFO] Status server telah di-reset secara manual.")
    return {"status": "reset_ok"}

# Pendaftaran Terminal (Client) Baru
@app.post(api_clients_register)
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
@app.post(api_logs_inference)
async def receive_inference_log(data: dict):
    """Menerima data hasil identifikasi wajah dari terminal untuk pemantauan terpusat."""
    client_id = data.get("client_id", "unknown")
    user_id = data.get("user_id", "Unknown")
    confidence = data.get("confidence", 0.0)
    latency = data.get("latency_ms", 0)
    status = data.get("status", "UNKNOWN")
    timestamp = datetime.now(timezone(timedelta(hours=7))).strftime("%H:%M:%S")
    
    # fl_manager.logger.info(f"[CIM] Log Inferensi Diterima: {user_id} ({confidence:.4f}) dari {client_id}")
    
    # Simpan ke memori manager untuk dashboard
    log_entry = {
        "timestamp": timestamp,
        "client_id": client_id,
        "user_id": user_id,
        "confidence": f"{confidence:.4f}",
        "latency": latency,
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
@app.post(api_clients_discovery_done)
async def report_discovery_done(data: dict):
    cid = data.get("client_id")
    if cid:
        fl_manager.discovery_clients.add(cid)
    return {"status": "ok"}

# Laporan Kesiapan Terminal untuk Training (READY)
@app.post(api_clients_ready)
async def report_client_ready(data: dict):
    cid = data.get("client_id")
    if cid:
        fl_manager.ready_clients.add(cid)
    return {"status": "ok"}

# Endpoint Baru: Ambil Log Jarak Jauh dari Client
@app.get(api_clients_logs)
async def get_client_logs(client_id: str):
    """Mengambil log dari client tertentu via API client tersebut."""
    client_data = fl_manager.registered_clients.get(client_id)
    if not client_data:
        raise HTTPException(status_code=404, detail="Client tidak terdaftar atau offline.")
    
    ip = client_data.get("ip_address")
    try:
        # Panggil endpoint /api/logs di sisi client
        res = requests.get(f"http://{ip}:8080{api_logs}", timeout=3)
        if res.status_code == 200:
            return res.json()
        return {"logs": f"Gagal mengambil log: HTTP {res.status_code}"}
    except Exception as e:
        return {"logs": f"Client tidak merespon: {str(e)}"}

@app.post("/api/clients/delete/{client_id}")
async def delete_client(client_id: str, db: Session = Depends(get_db)):
    client = db.query(Client).filter_by(edge_id=client_id).first()
    if client:
        db.delete(client)
        db.commit()
    if client_id in fl_manager.registered_clients:
        del fl_manager.registered_clients[client_id]
    return {"status": "success", "message": f"Client {client_id} berhasil dihapus"}


# Akses model dan aset

@app.post("/api/models/delete-specific")
async def delete_specific_model(version: str, db: Session = Depends(get_db)):
    try:
        if version == "v0" or version == "0" or version == "active":
            raise HTTPException(400, "Tidak dapat menghapus model dasar atau model aktif")
            
        # 1. Hapus file fisik
        from app.config import DATA_ROOT
        model_file = os.path.join(DATA_ROOT, f"global_model_{version}.pth")
        registry_file = os.path.join(DATA_ROOT, f"global_embedding_registry_{version}.pth")
        bn_file = os.path.join(DATA_ROOT, f"global_bn_combined_{version}.pth")
        metrics_file = os.path.join(DATA_ROOT, f"metrics_{version}.json")
        
        if os.path.exists(model_file):
            try: os.remove(model_file)
            except Exception as e: print(f"Error removing model file: {e}")
            
        if os.path.exists(registry_file):
            try: os.remove(registry_file)
            except Exception as e: print(f"Error removing registry file: {e}")
            
        if os.path.exists(bn_file):
            try: os.remove(bn_file)
            except Exception as e: print(f"Error removing bn file: {e}")
            
        if os.path.exists(metrics_file):
            try: os.remove(metrics_file)
            except Exception as e: print(f"Error removing metrics file: {e}")
            
        fl_manager.logger.info(f"Model versi {version} berhasil dihapus dari sistem.")
        return {"status": "success", "message": f"Model versi {version} berhasil dihapus."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/models/reset")
async def reset_models(db: Session = Depends(get_db)):
    try:
        # 1. Hapus semua record ronde dan sesi pelatihan dari DB
        db.query(FLRound).delete()
        db.query(FLSession).delete()
        db.query(GlobalModel).filter(GlobalModel.version > 0).delete()
        db.commit()
        
        # 2. Hapus file fisik model cl/fl dan file metrics
        from app.config import DATA_ROOT, MODEL_DIR, REGISTRY_PATH, BN_PATH
        if os.path.exists(DATA_ROOT):
            for f in os.listdir(DATA_ROOT):
                if f.startswith("global_model_") and f.endswith(".pth") and f not in ("global_model_v0.pth", "global_model.pth", "backbone.pth"):
                    try: os.remove(os.path.join(DATA_ROOT, f))
                    except: pass
                if f.startswith("global_embedding_registry_") and f.endswith(".pth") and f != "global_embedding_registry.pth":
                    try: os.remove(os.path.join(DATA_ROOT, f))
                    except: pass
                if f.startswith("global_bn_combined_") and f.endswith(".pth") and f != "global_bn_combined.pth":
                    try: os.remove(os.path.join(DATA_ROOT, f))
                    except: pass
                if f.startswith("metrics_") and f.endswith(".json"):
                    try: os.remove(os.path.join(DATA_ROOT, f))
                    except: pass
        
        # 3. Kembalikan model aktif (backbone.pth) ke v0
        v0_path = os.path.join(MODEL_DIR, "global_model_v0.pth")
        if not os.path.exists(v0_path):
            v0_path = os.path.join(DATA_ROOT, "global_model_v0.pth")
            
        active_path = "data/backbone.pth"
        if os.path.exists(v0_path):
            shutil.copy2(v0_path, active_path)
            
        # Hapus registry embeddings aktif dan BN params karena v0 tidak memiliki wajah terdaftar
        if os.path.exists(REGISTRY_PATH):
            try: os.remove(REGISTRY_PATH)
            except: pass
        if os.path.exists(BN_PATH):
            try: os.remove(BN_PATH)
            except: pass
            
        # 4. Reset state manager FL
        fl_manager.model_version = 0
        fl_manager.model_version_str = "v0"
        active_version_path = os.path.join(DATA_ROOT, "active_version.txt")
        if os.path.exists(active_version_path):
            try: os.remove(active_version_path)
            except: pass
        fl_manager.metrics = {
            "round_history": [],
            "accuracy": 0.0,
            "loss": 0.0,
            "total_round_time_s": 0,
            "compute_energy_kwh": 0,
            "upload_volume_mb": 0,
            "download_volume_mb": 0,
            "transmission_cost_idr": 0,
            "compute_cost_idr": 0
        }
        
        # Update logs
        fl_manager.logger.info("Server FL di-reset ke model dasar v0. Semua versi model terlatih telah dihapus.")
        
        return {"status": "success", "message": "Model FL berhasil di-reset ke v0."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get(api_model_backbone)
async def get_backbone_model(version: str = None, db: Session = Depends(get_db)):
    from app.config import DATA_ROOT
    if version == "active":
        active_path = "data/backbone.pth"
        if os.path.exists(active_path):
            with open(active_path, "rb") as f:
                return Response(content=f.read(), media_type="application/octet-stream")
        raise HTTPException(status_code=404, detail="Model aktif (backbone.pth) tidak ditemukan di disk.")
    elif version and version != "v0" and version != "0":
        file_path = os.path.join(DATA_ROOT, f"global_model_{version}.pth")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return Response(content=f.read(), media_type="application/octet-stream")
        raise HTTPException(status_code=404, detail=f"Model versi {version} tidak ditemukan di disk.")
    elif version == "v0" or version == "0":
        # Muat model baseline v0
        v0_path = os.path.join(DATA_ROOT, "global_model_v0.pth")
        if os.path.exists(v0_path):
            with open(v0_path, "rb") as f:
                return Response(content=f.read(), media_type="application/octet-stream")
        global_model = db.query(GlobalModel).filter(GlobalModel.version == 0).first()
        if global_model and global_model.weights:
            return Response(content=global_model.weights, media_type="application/octet-stream")
        from app.config import FALLBACK_MODEL_PATH
        if os.path.exists(FALLBACK_MODEL_PATH):
            with open(FALLBACK_MODEL_PATH, "rb") as f:
                return Response(content=f.read(), media_type="application/octet-stream")
        raise HTTPException(status_code=404, detail="Model v0 tidak ditemukan.")
    else:
        # Default fallback: get latest updated model
        global_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
        if not global_model or not global_model.weights:
            raise HTTPException(status_code=404, detail="Global model weights not found in database.")
        return Response(content=global_model.weights, media_type="application/octet-stream")

# Mengunduh Parameter Batch Normalization (BN)
@app.get(api_model_bn)
async def get_bn_model():
    if os.path.exists(BN_PATH):
        with open(BN_PATH, "rb") as f:
            return Response(content=f.read(), media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="BN Assets not found")

# Mengunduh Pustaka Identitas Global (Registry)
@app.get(api_model_registry)
async def get_registry(version: str = None):
    from app.config import DATA_ROOT
    path = REGISTRY_PATH
    if version == "active":
        pass  # path tetap REGISTRY_PATH
    elif version and version != "v0" and version != "0":
        versioned_path = os.path.join(DATA_ROOT, f"global_embedding_registry_{version}.pth")
        if os.path.exists(versioned_path):
            path = versioned_path
        else:
            raise HTTPException(status_code=404, detail=f"Registry versi {version} tidak ditemukan")
    elif version == "v0" or version == "0":
        import io
        import torch
        buf = io.BytesIO()
        torch.save({}, buf)
        return Response(content=buf.getvalue(), media_type='application/octet-stream')
        
    if os.path.exists(path):
        with open(path, "rb") as f:
            return Response(content=f.read(), media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Registry not found")

# Mendapatkan Status Terkini Pelatihan
@app.get(api_training_status)
async def get_training_status():
    return {
        "current_phase": fl_manager.current_phase,
        "is_running": fl_manager.is_running,
        "active_session_id": fl_manager.session_id,
        "model_version": fl_manager.model_version_str,
        "inference_threshold": fl_manager.inference_threshold,
        "current_logs": fl_manager.current_logs[-10:]
    }

# Sinkronisasi Label Mahasiswa (ID Mapping)
@app.post(api_training_get_label)
async def get_label(data: dict, db: Session = Depends(get_db)):
    nrp = data.get("nrp")
    name = data.get("name", "Unknown")
    edge_id = data.get("client_id", "edge-1")
    embedding_b64 = data.get("embedding")
    dataset = data.get("dataset", "students")
    
    # Dekode embedding dari Base64 jika disertakan
    embedding_bytes = None
    if embedding_b64:
        embedding_bytes = base64.b64decode(embedding_b64)

    # Periksa apakah NRP sudah terdaftar di database global
    user = db.query(UserGlobal).filter_by(nrp=nrp).first()
    if not user:
        # Jika belum terdaftar, buat entri baru secara serial (AUTOINCREMENT menjamin keunikan label ID)
        user = UserGlobal(nrp=nrp, name=name, registered_edge_id=edge_id, embedding=embedding_bytes, dataset=dataset)
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # Jika sudah terdaftar, lakukan pembaruan data nama atau embedding
        user.name = name
        user.dataset = dataset
        if embedding_bytes:
            user.embedding = embedding_bytes
        db.commit()
    
    # Masukkan NRP ke daftar data yang diterima untuk ronde ini
    if nrp not in fl_manager.received_data:
        fl_manager.received_data.append(nrp)
        
    return {"nrp": nrp, "label": user.user_id}

# Konsolidasi Registry Centroid Wajah Global
@app.post(api_training_registry_assets)
async def receive_registry_assets(data: dict, db: Session = Depends(get_db)):
    client_id = data.get("client_id")
    serialized_bn = data.get("bn_params")
    centroids = data.get("centroids")
    
    # Dekode dan rekonstruksi parameter Batch Normalization (BN)
    bn_bytes = base64.b64decode(serialized_bn)
    bn_params = torch.load(io.BytesIO(bn_bytes), map_location="cpu")
    
    # Dekode dan konversi centroid wajah dari Base64 ke array NumPy
    decoded_centroids = {}
    for nrp, b64_vec in centroids.items():
        vec_bytes = base64.b64decode(b64_vec)
        decoded_centroids[nrp] = np.frombuffer(vec_bytes, dtype=np.float32)
        
    # Simpan aset fitur wajah ke penampung memori server
    fl_manager.registry_submissions[client_id] = {
        "bn": bn_params,
        "centroids": decoded_centroids
    }
    
    # Tulis file submission lokal klien ke folder data untuk backup
    submission_dir = "data/submissions"
    submission_path = os.path.join(submission_dir, f"{client_id}_assets.pth")
    tmp_submission_path = submission_path + ".tmp"
    with open(tmp_submission_path, "wb") as f:
        torch.save(fl_manager.registry_submissions[client_id], f)
    os.replace(tmp_submission_path, submission_path)
    
    fl_manager.update_logs(f"[SUCCESS] Menerima aset fitur wajah (Centroids) dari {client_id}")
    return {"status": "received"}

# Mendapatkan Daftar NRP Global untuk Label Mapping
@app.get(api_training_label_map)
async def get_label_map(db: Session = Depends(get_db)):
    # Ambil semua identitas terdaftar, urutkan berdasarkan NRP
    users = db.query(UserGlobal).order_by(UserGlobal.nrp).all()
    # Mengembalikan array NRP terurut sebagai indeks/label pemetaan kelas terpadu
    return [u.nrp for u in users]

# Mendapatkan Detail Identitas Mahasiswa Global (NRP + Nama + Embedding)
@app.get(api_training_identities)
async def get_global_identities(db: Session = Depends(get_db)):
    users = db.query(UserGlobal).all()
    results = []
    for u in users:
        item = {"nrp": u.nrp, "name": u.name}
        if u.embedding:
            item["embedding"] = base64.b64encode(u.embedding).decode('utf-8')
        results.append(item)
    return results


# API status dataset dan status preprocessing
@app.get("/api/datasets/status")
def get_datasets_status(dbs: Session = Depends(get_db)):
    status_list = []
    
    # Ambil data client aktif
    status_data = fl_manager.get_status(db=dbs)
    active_clients = status_data.get("active_clients", [])
    available_datasets = status_data.get("available_datasets", ["students"])
    
    for ds in available_datasets:
        online_clients_with_ds = [c for c in active_clients if c["status"] == "ONLINE" and ds in c.get("available_datasets", {})]
        
        if not online_clients_with_ds:
            clients_to_check = active_clients
        else:
            clients_to_check = online_clients_with_ds
            
        if not clients_to_check:
            is_preprocessed = False
            status_label = "Belum Diproses"
            info = "Tidak ada client terhubung"
        else:
            preprocessed_count = 0
            total_count = 0
            for c in clients_to_check:
                avail = c.get("available_datasets", {})
                if ds in avail:
                    total_count += 1
                    if avail[ds]:
                        preprocessed_count += 1
                elif c.get("dataset") == ds:
                    total_count += 1
                    if c.get("is_preprocessed"):
                        preprocessed_count += 1
            
            if total_count > 0 and preprocessed_count == total_count:
                is_preprocessed = True
                status_label = "Sudah Diproses"
                info = f"Semua client ({preprocessed_count}/{total_count}) telah melakukan pra-pemrosesan"
            else:
                is_preprocessed = False
                status_label = "Belum Diproses"
                info = f"Hanya {preprocessed_count}/{total_count} client yang siap"
                
        status_list.append({
            "dataset": ds,
            "raw_exists": True,
            "raw_classes": 0,
            "raw_images": 0,
            "processed_exists": is_preprocessed,
            "processed_classes": 0,
            "processed_images": 0,
            "is_preprocessed": is_preprocessed,
            "status_label": status_label,
            "info": info
        })
        
    return {"status": "success", "datasets": status_list}



# Sinkronisasi Hasil Presensi dari Terminal
@app.post(api_attendance_sync)
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
                ts = datetime.now(timezone(timedelta(hours=7)))
                
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
@app.post(api_settings)
async def update_settings(data: dict):
    try:
        # Perbarui konfigurasi server secara real-time
        if 'rounds' in data: fl_manager.default_rounds = int(data['rounds'])
        if 'epochs' in data: fl_manager.default_epochs = int(data['epochs'])
        if 'min_clients' in data: fl_manager.default_min_clients = int(data['min_clients'])
        if 'threshold' in data: fl_manager.inference_threshold = float(data['threshold'])
        
        fl_manager.save_settings()
        fl_manager.update_logs("[SUCCESS] Pengaturan sistem diperbarui secara dinamis.")
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get(api_attendance)
async def get_attendance_json(dataset: str = None, db: Session = Depends(get_db)):
    query = db.query(UserGlobal)
    if dataset:
        query = query.filter(UserGlobal.dataset == dataset)
    all_users = query.all()
    tz_wib = timezone(timedelta(hours=7))
    today_start = datetime.now(tz_wib).replace(hour=0, minute=0, second=0, microsecond=0)
    today_recap = db.query(AttendanceRecap).filter(AttendanceRecap.timestamp >= today_start).all()
    recap_map = {rec.user_id: rec for rec in today_recap}
    
    results = []
    for u in all_users:
        rec = recap_map.get(u.user_id)
        
        time_str = "-"
        if rec:
            dt = rec.timestamp
            if dt.tzinfo is None:
                wib_dt = dt.replace(tzinfo=timezone.utc).astimezone(tz_wib)
            else:
                wib_dt = dt.astimezone(tz_wib)
            time_str = wib_dt.strftime("%H:%M:%S")
            
        results.append({
            "nrp": u.nrp,
            "name": u.name,
            "status": "HADIR" if rec else "MENUNGGU",
            "time": time_str,
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

# Endpoints streaming video dan cache server
STORED_VIDEOS_DIR = "stored_videos"
VIDEO_CACHES_DIR = "video_caches"
os.makedirs(STORED_VIDEOS_DIR, exist_ok=True)
os.makedirs(VIDEO_CACHES_DIR, exist_ok=True)

def strip_audio_from_video(file_path: str, logger):
    temp_output = file_path + ".noaudio.mp4"
    try:
        cmd = ["ffmpeg", "-y", "-i", file_path, "-an", "-vcodec", "copy", temp_output]
        logger.info(f"Mencoba menghapus audio dari video: {file_path}")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=15)
        
        if os.path.exists(temp_output):
            os.remove(file_path)
            shutil.move(temp_output, file_path)
            logger.info(f"Sukses menghapus audio dari video. Ukuran baru: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            return True
        return False
    except FileNotFoundError:
        logger.warning("Alat ffmpeg tidak terdeteksi di server. Melanjutkan dengan video orisinal (audio tetap ada).")
        return False
    except Exception as e:
        logger.error(f"Gagal menghapus audio menggunakan ffmpeg: {e}")
        if os.path.exists(temp_output):
            try: os.remove(temp_output)
            except: pass
        return False

# Handlers streaming video dan cache server

@app.post(api_video_upload)
async def upload_video(request: Request):
    """Mengunggah video ke server pusat."""
    try:
        filename = request.headers.get("X-File-Name", f"video_{int(time.time())}.mp4")
        filename = "".join(c for c in filename if c.isalnum() or c in "._-").strip()
        if not filename.lower().endswith(".mp4"):
            filename += ".mp4"
            
        file_path = os.path.join(STORED_VIDEOS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            
        with open(file_path, "wb") as buffer:
            async for chunk in request.stream():
                buffer.write(chunk)
                
        # Hilangkan audio untuk meringankan ukuran berkas dan beban pemutaran streaming
        strip_audio_from_video(file_path, fl_manager.logger)
                
        fl_manager.logger.info(f"Video {filename} berhasil diunggah ke server.")
        return {"status": "success", "message": f"Video {filename} uploaded to server"}
    except Exception as e:
        fl_manager.logger.error(f"Gagal mengunggah video ke server: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get(api_video_list)
async def list_videos():
    """Mengembalikan daftar video yang tersimpan di server."""
    files = [f for f in os.listdir(STORED_VIDEOS_DIR) if f.lower().endswith(".mp4")]
    files = sorted(files, key=lambda s: s.lower())
    return {"status": "success", "videos": files}

@app.get(api_video_stream)
async def stream_video(video_name: str, request: Request):
    """Streaming video dengan Byte-Range Request untuk mendukung seeking/scrubbing."""
    video_name = "".join(c for c in video_name if c.isalnum() or c in "._-").strip()
    file_path = os.path.join(STORED_VIDEOS_DIR, video_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video tidak ditemukan")
        
    range_header = request.headers.get("Range", None)
    return send_bytes_range(file_path, range_header)

def send_bytes_range(file_path: str, range_header: str):
    file_size = os.path.getsize(file_path)
    chunk_size = 1024 * 1024 * 4 # 4MB chunk size

    # Tanpa Range header: kembalikan seluruh file dengan status 200
    if not range_header:
        def full_file_iterator():
            with open(file_path, "rb") as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    yield data
        return StreamingResponse(full_file_iterator(), status_code=200, headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": "video/mp4",
        })

    # Dengan Range header: kembalikan partial content (206)
    start = 0
    end = file_size - 1
    try:
        range_val = range_header.replace("bytes=", "").split("-")
        if range_val[0]:
            start = int(range_val[0])
        if len(range_val) > 1 and range_val[1]:
            end = int(range_val[1])
        else:
            end = min(start + chunk_size, file_size - 1)
    except Exception:
        pass

    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    content_length = end - start + 1

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": "video/mp4",
    }

    def file_iterator():
        with open(file_path, "rb") as f:
            f.seek(start)
            bytes_left = content_length
            while bytes_left > 0:
                chunk_to_read = min(1024 * 64, bytes_left)
                data = f.read(chunk_to_read)
                if not data:
                    break
                bytes_left -= len(data)
                yield data

    return StreamingResponse(file_iterator(), status_code=206, headers=headers)

@app.get(api_video_cache)
async def get_video_cache(video_name: str):
    """Mendapatkan metadata koordinat wajah (bounding boxes) dari server cache."""
    video_name = "".join(c for c in video_name if c.isalnum() or c in "._-").strip()
    cache_path = os.path.join(VIDEO_CACHES_DIR, f"{video_name}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                data = {
                    "metadata": {
                        "processing_duration_s": 0.0,
                        "processed_by": "unknown",
                        "is_complete": True,
                        "processed_at": ""
                    },
                    "detections": data
                }
            return {"status": "success", "cached": True, "data": data}
        except Exception as e:
            return {"status": "error", "message": f"Gagal membaca cache: {e}"}
    return {"status": "success", "cached": False, "data": {"metadata": {"is_complete": False}, "detections": []}}

@app.post(api_video_cache)
async def save_video_cache(video_name: str, cache_data: dict = Body(...)):
    """Menyimpan metadata koordinat wajah (bounding boxes) ke server cache."""
    video_name = "".join(c for c in video_name if c.isalnum() or c in "._-").strip()
    cache_path = os.path.join(VIDEO_CACHES_DIR, f"{video_name}.json")
    try:
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=4)
        return {"status": "success", "message": "Cache saved successfully"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.delete(api_video_cache)
async def delete_video_cache(video_name: str):
    """Menghapus file cache koordinat wajah dari disk server."""
    video_name = "".join(c for c in video_name if c.isalnum() or c in "._-").strip()
    cache_path = os.path.join(VIDEO_CACHES_DIR, f"{video_name}.json")
    try:
        if os.path.exists(cache_path):
            os.remove(cache_path)
            return {"status": "success", "message": "Cache deleted successfully"}
        return {"status": "success", "message": "Cache file did not exist"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/video/caches")
async def list_video_caches():
    """Mengembalikan daftar semua berkas cache video yang tersedia beserta metadata singkat."""
    caches = []
    if os.path.exists(VIDEO_CACHES_DIR):
        for f in os.listdir(VIDEO_CACHES_DIR):
            if f.endswith(".json"):
                cache_path = os.path.join(VIDEO_CACHES_DIR, f)
                try:
                    with open(cache_path, "r") as file:
                        data = json.load(file)
                    
                    if isinstance(data, list):
                        caches.append({
                            "cache_name": f[:-5],
                            "processed_by": "unknown",
                            "processing_duration_s": 0.0,
                            "is_complete": True,
                            "last_frame": 0
                        })
                    else:
                        metadata = data.get("metadata", {})
                        caches.append({
                            "cache_name": f[:-5],
                            "processed_by": metadata.get("processed_by", "unknown"),
                            "processing_duration_s": metadata.get("processing_duration_s", 0.0),
                            "is_complete": metadata.get("is_complete", False),
                            "last_frame": metadata.get("last_frame", 0)
                        })
                except Exception:
                    pass
    return {"status": "success", "caches": caches}

@app.delete(api_video_delete)
async def delete_video(video_name: str):
    """Menghapus file video dan file cache koordinat wajah dari disk server."""
    video_name = "".join(c for c in video_name if c.isalnum() or c in "._-").strip()
    
    # Path video (.mp4)
    video_path = os.path.join(STORED_VIDEOS_DIR, video_name)
    
    deleted_files = []
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
            deleted_files.append(video_path)
        
        # Hapus semua file cache yang berhubungan dengan video ini (misal: client-1_video.mp4.json atau video.mp4.json)
        if os.path.exists(VIDEO_CACHES_DIR):
            for f in os.listdir(VIDEO_CACHES_DIR):
                if f.endswith(".json"):
                    if f == f"{video_name}.json" or f.endswith(f"_{video_name}.json"):
                        cache_path = os.path.join(VIDEO_CACHES_DIR, f)
                        os.remove(cache_path)
                        deleted_files.append(cache_path)
            
        gc.collect()
        
        fl_manager.logger.success(f"Video {video_name} dan cache pendeteksian dihapus dari disk server.")
        return {
            "status": "success", 
            "message": f"Berhasil menghapus video {video_name} dan cache-nya.",
            "deleted": deleted_files
        }
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Gagal menghapus file: {e}"}, status_code=500)

@app.on_event("startup")
def startup_event():
    fl_manager.logger.success("Server Federated Learning telah siap.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, access_log=False)
