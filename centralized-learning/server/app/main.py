# Pustaka standar
import os
import shutil
import time
import json
import zipfile
import subprocess
import gc
from datetime import datetime, time as dt_time, timedelta, timezone

# Pustaka pihak ketiga
import torch
import requests
import uvicorn
from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import text

# Modul aplikasi
from app.db import models, db, schemas, crud
models.Base.metadata.create_all(bind=db.engine)

# Auto-fix: Tambahkan kolom baru jika belum ada (Migration Lite)
try:
    with db.engine.connect() as conn:
        conn.execute(text("ALTER TABLE training_rounds ADD COLUMN IF NOT EXISTS val_loss FLOAT;"))
        conn.execute(text("ALTER TABLE training_rounds ADD COLUMN IF NOT EXISTS val_accuracy FLOAT;"))
        try:
            conn.execute(text("ALTER TABLE users_global ADD COLUMN dataset VARCHAR;"))
        except Exception:
            pass
        conn.commit()
        # cl_manager.logger.info("Database schema updated (val_loss/val_accuracy added).")
except Exception as e:
    print(f"Migration check failed: {e}")
from app.utils.mobilefacenet import MobileFaceNet
from app.config import (
    MODEL_PATH, REF_PATH, UPLOAD_DIR, TRAINING_PARAMS, PROCESSED_DATA, MODEL_DIR, PRETRAINED_PATH
)
from app.server_manager_instance import cl_manager
from app.controllers.student import student_controller
from app.controllers.training import training_controller
from app.controllers.inference import inference_controller

# Konfigurasi endpoint API
api_settings = "/api/settings"
api_clients_ready = "/api/clients/ready"
api_clients_logs = "/api/clients/logs/{client_id}"
api_status = "/api/status"
api_results = "/api/results"
api_attendance = "/api/attendance"
api_logs_inference = "/api/logs/inference"
api_logs_energy = "/api/logs/energy"
api_logs = "/api/logs"

api_register_client = "/register-client"
api_upload_bulk = "/upload-bulk-zip"
api_get_model = "/get-model"
api_get_reference = "/get-reference-embeddings"
api_submit_attendance = "/submit-attendance"
api_ping = "/ping"

api_video = "/api/video"
api_video_upload = f"{api_video}/upload"
api_video_list = f"{api_video}/list"
api_video_stream = f"{api_video}/stream/{{video_name}}"
api_video_cache = f"{api_video}/cache/{{video_name}}"
api_video_delete = f"{api_video}/delete/{{video_name}}"

if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR, exist_ok=True)
image_pretrained_path = "app/model/global_model_v0.pth"
if not os.path.exists(PRETRAINED_PATH) and os.path.exists(image_pretrained_path):
    cl_manager.logger.info(f"Menyalin model dasar v0 dari {image_pretrained_path} ke {PRETRAINED_PATH}...")
    shutil.copy2(image_pretrained_path, PRETRAINED_PATH)

if not os.path.exists(MODEL_PATH):
    if os.path.exists(PRETRAINED_PATH):
        cl_manager.logger.info("Membuat model global awal dari model dasar v0...")
        shutil.copy2(PRETRAINED_PATH, MODEL_PATH)
    else:
        cl_manager.logger.info("Membuat model global awal v0...")
        tmp_model_init = MODEL_PATH + ".tmp"
        torch.save(MobileFaceNet().state_dict(), tmp_model_init)
        os.replace(tmp_model_init, MODEL_PATH)

if not os.path.exists(REF_PATH):
    cl_manager.logger.info("Membuat basis data referensi awal...")
    tmp_ref_init = REF_PATH + ".tmp"
    torch.save({}, tmp_ref_init)
    os.replace(tmp_ref_init, REF_PATH)

# Konfigurasi Aplikasi Dashboard Server
app = FastAPI(title="Centralized Attendance Server")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Dashboard Utama
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, dbs: Session = Depends(db.get_db)):
    status = cl_manager.get_status(db=dbs)
    return templates.TemplateResponse("index.html", {"request": request, "title": "Dashboard", "status": status})

# Halaman Rekapitulasi Presensi
@app.get("/records", response_class=HTMLResponse)
async def view_records(request: Request):
    return templates.TemplateResponse("records.html", {"request": request, "title": "Attendance Board"})

# Halaman Hasil Evaluasi (Copy-Paste friendly)
@app.get("/results", response_class=HTMLResponse)
async def view_results(request: Request):
    status = cl_manager.get_status()
    return templates.TemplateResponse("results.html", {"request": request, "title": "Evaluation Results", "status": status})

# Halaman Pengaturan (Settings)
@app.get("/settings", response_class=HTMLResponse)
async def view_settings(request: Request):
    status = cl_manager.get_status()
    return templates.TemplateResponse("settings.html", {"request": request, "title": "Settings", "status": status})

# Halaman Upload & Manajemen Video Simulasi
@app.get("/video-simulasi", response_class=HTMLResponse)
async def view_video_simulasi(request: Request):
    """Merender halaman manajemen video simulasi"""
    return templates.TemplateResponse("video_simulasi.html", {"request": request, "title": "Video Simulasi"})

# API Update Pengaturan
@app.post(api_settings)
async def update_settings(data: dict):
    success = cl_manager.save_settings(data)
    if success:
        return {"status": "success", "message": "Settings updated"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save settings")

@app.post(api_clients_ready)
async def report_client_ready(data: dict):
    # Digunakan untuk koordinasi sinkronisasi data
    return {"status": "ok"}

# Endpoint Baru: Ambil Log Jarak Jauh dari Client (Proxy)
@app.get(api_clients_logs)
async def get_client_logs(client_id: str, dbs: Session = Depends(db.get_db)):
    """Mengambil log aktivitas dari client tertentu secara remote."""
    client = dbs.query(models.Client).filter_by(edge_id=client_id).first()
    
    if not client or not client.ip_address:
        raise HTTPException(status_code=404, detail="Client tidak ditemukan atau IP tidak terdaftar.")
    
    try:
        # Panggil endpoint /api/logs di sisi client
        res = requests.get(f"http://{client.ip_address}:8080{api_logs}", timeout=3)
        if res.status_code == 200:
            return res.json()
        return {"logs": f"Gagal mengambil log dari client: HTTP {res.status_code}"}
    except Exception as e:
        return {"logs": f"Client {client_id} tidak merespon: {str(e)}"}

@app.post("/api/clients/delete/{client_id}")
def delete_client(client_id: str, dbs: Session = Depends(db.get_db)):
    client = dbs.query(models.Client).filter_by(edge_id=client_id).first()
    if client:
        dbs.delete(client)
        dbs.commit()
    if client_id in cl_manager.registered_clients:
        del cl_manager.registered_clients[client_id]
    return {"status": "success", "message": f"Client {client_id} berhasil dihapus"}


# API Status Sistem
@app.get(api_status)
async def get_status(dbs: Session = Depends(db.get_db)):
    return cl_manager.get_status(db=dbs)

# API Hasil Pelatihan (Metrik)
@app.get(api_results)
async def get_results(version: str = None):
    if version:
        if version == "v0" or version == "0":
            return {"epoch_history": [], "accuracy": 0.0, "loss": 0.0}
        elif version == "active":
            return cl_manager.metrics
            
        version_metrics = cl_manager.load_version_metrics(version)
        if version_metrics:
            return version_metrics
    return cl_manager.metrics

@app.get(api_attendance)
async def get_attendance_recap(dataset: str = None, dbs: Session = Depends(db.get_db)):
    tz_wib = timezone(timedelta(hours=7))
    today = datetime.now(tz_wib).date()
    start_of_day = datetime.combine(today, dt_time.min)
    
    query = dbs.query(models.UserGlobal)
    if dataset:
        query = query.filter(models.UserGlobal.dataset == dataset)
    users = query.all()
    recap = []
    
    for user in users:
        last_entry = dbs.query(models.AttendanceRecap).filter(
            models.AttendanceRecap.user_id == user.user_id,
            models.AttendanceRecap.timestamp >= start_of_day
        ).order_by(models.AttendanceRecap.timestamp.desc()).first()
        
        time_str = "--:--"
        if last_entry:
            dt = last_entry.timestamp
            if dt.tzinfo is None:
                wib_dt = dt.replace(tzinfo=timezone.utc).astimezone(tz_wib)
            else:
                wib_dt = dt.astimezone(tz_wib)
            time_str = wib_dt.strftime("%H:%M:%S")
            
        recap.append({
            "name": user.name,
            "nrp": user.nrp,
            "status": "HADIR" if last_entry else "MENUNGGU",
            "time": time_str,
            "confidence": f"{int(last_entry.confidence * 100)}%" if last_entry else "--",
            "registered_edge_id": user.registered_edge_id or "client-1"
        })
    
    return recap

# Pemeriksaan Kesehatan (Health Check) bagi Terminal
@app.get(api_ping)
async def health_check():
    return {
        "status": "online", 
        "upload_requested": cl_manager.upload_requested, 
        "model_version": cl_manager.model_version_str,
        "inference_threshold": cl_manager.inference_threshold
    }

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
    
    log_entry = {
        "timestamp": timestamp,
        "client_id": client_id,
        "user_id": user_id,
        "confidence": f"{confidence:.4f}",
        "latency": latency,
        "status": status
    }
    
    if "inference_logs" not in cl_manager.metrics:
        cl_manager.metrics["inference_logs"] = []
        
    cl_manager.metrics["inference_logs"].insert(0, log_entry)
    cl_manager.metrics["inference_logs"] = cl_manager.metrics["inference_logs"][:10000]
    cl_manager.save_inference_logs()
    return {"status": "logged"}

# Endpoint Baru: Menerima data energi dari Client (untuk audit nilai ekonomi)
@app.post(api_logs_energy)
async def receive_energy_log(data: dict):
    """Menerima konsumsi energi riil dari terminal untuk penghitungan biaya operasional."""
    client_id = data.get("client_id", "unknown")
    energy_kwh = data.get("energy_kwh", 0.0)
    
    current_energy = cl_manager.metrics.get("compute_energy_kwh", 0)
    cl_manager.update_metrics({"compute_energy_kwh": current_energy + energy_kwh})
    
    cl_manager.logger.info(f"Energi diterima dari {client_id}: {energy_kwh:.6f} kWh")
    return {"status": "success", "energy_total": cl_manager.metrics["compute_energy_kwh"]}

@app.post(api_register_client, response_model=schemas.ClientResponse)
async def register_client(client_data: schemas.ClientBase, request: Request, dbs: Session = Depends(db.get_db)):
    client_ip = request.client.host
    
    try:
        raw_data = await request.json()
    except Exception:
        raw_data = {}
        
    cid = client_data.edge_id
    if cid:
        raw_data["ip_address"] = client_ip
        cl_manager.registered_clients[cid] = raw_data
        
    existing = dbs.query(models.Client).filter(models.Client.edge_id == client_data.edge_id).first()
    if existing:
        existing.last_seen = datetime.now(timezone(timedelta(hours=7)))
        existing.ip_address = client_ip
        existing.status = client_data.status
        dbs.commit()
        dbs.refresh(existing)
        return existing
    client_data.ip_address = client_ip
    return crud.register_client(dbs, client_data)

# Menerima Unggahan Dataset Bulk (ZIP) dari Terminal
@app.post(api_upload_bulk)
async def upload_bulk_zip(file: UploadFile = File(...)):
    edge_id = file.filename.split("_")[0]
    UPLOAD_TEMP = f"data/upload_{edge_id}_{int(time.time())}.zip"
    os.makedirs("data", exist_ok=True)
    
    # Hitung Ukuran File Ril
    file_size_bytes = 0
    with open(UPLOAD_TEMP, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024) # 1MB chunks
            if not chunk: break
            buffer.write(chunk)
            file_size_bytes += len(chunk)
    
    # Update Metrik Transmisi Ril
    upload_mb = round(file_size_bytes / (1024 * 1024), 2)
    cl_manager.update_metrics({"upload_volume_mb": cl_manager.metrics.get("upload_volume_mb", 0) + upload_mb})
        
    try:
        with zipfile.ZipFile(UPLOAD_TEMP, 'r') as zip_ref:
            # Dapatkan daftar folder (NRP) sebelum diekstrak untuk pemetaan
            names = zip_ref.namelist()
            # Asumsi folder adalah level pertama: "NRP_Name/"
            top_level = {n.split('/')[0] for n in names if '/' in n}
            nrps = [t.split('_')[0] for t in top_level if t]
            
            zip_ref.extractall("data/students")
            
            client_specific_dir = f"data/students_{edge_id}"
            os.makedirs(client_specific_dir, exist_ok=True)
            zip_ref.extractall(client_specific_dir)
            
            # Daftarkan pemetaan di manajer
            cl_manager.register_upload(edge_id, nrps)
            cl_manager.logger.success(f"Berhasil mengekstrak {len(nrps)} folder mahasiswa ({upload_mb} MB) ke data/students dan {client_specific_dir}.")
            
        os.remove(UPLOAD_TEMP)
        cl_manager.update_logs(f"Menerima {len(nrps)} data mahasiswa ({upload_mb} MB) dari {edge_id}")
        return {"status": "success", "filename": file.filename, "edge_id": edge_id, "size_mb": upload_mb}
    except Exception as e:
        cl_manager.logger.error(f"Gagal ekstrak dataset ZIP: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
# Alur kerja pelatihan terpusat (research workflow)

# Helper/Internal workflow functions
def internal_workflow_import(dbs: Session):
    start_t = time.time()
    cl_manager.start_phase("Import Data")
    cl_manager.received_data = [] 
    cl_manager.uploader_map = {} 
    if os.path.exists(UPLOAD_DIR): shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    cl_manager.upload_requested = True
    cl_manager.logger.info("Menunggu client mengirimkan dataset...")
    try:
        online_count = dbs.query(models.Client).filter(models.Client.status == "online").count()
        res = training_controller.fetch_data(wait_timeout=3600, expected_clients=online_count)
        
        if res['status'] == 'success':
            cl_manager.update_metrics({
                "upload_volume_mb": res['upload_volume_mb'],
                "preprocess_upload_mb": res['upload_volume_mb']
            })
            folders = [f for f in os.listdir(UPLOAD_DIR) if os.path.isdir(os.path.join(UPLOAD_DIR, f))]
            for folder in folders:
                parts = folder.split("_", 1)
                nrp = parts[0].strip()
                name = parts[1].strip() if len(parts) > 1 else "Unknown"
                
                existing = dbs.query(models.UserGlobal).filter(models.UserGlobal.nrp == nrp).first()
                if not existing:
                    edge_id = cl_manager.uploader_map.get(nrp)
                    if not edge_id:
                        client = dbs.query(models.Client).first()
                        edge_id = client.edge_id if client else None
                    
                    new_student = models.UserGlobal(name=name, nrp=nrp, registered_edge_id=edge_id)
                    dbs.add(new_student)
                else:
                    if not existing.registered_edge_id:
                        existing.registered_edge_id = cl_manager.uploader_map.get(nrp)
            dbs.commit()
            cl_manager.update_received_data(UPLOAD_DIR)
            
        return res
    finally:
        cl_manager.update_metrics({
            "preprocess_duration_s": cl_manager.metrics.get("preprocess_duration_s", 0.0) + (time.time() - start_t)
        })
        cl_manager.upload_requested = False
        cl_manager.end_phase()

def internal_workflow_preprocess(dataset: str, dbs: Session):
    start_t = time.time()
    cl_manager.start_phase("Preprocess & Balance")
    cl_manager.logger.info(f"Memulai tahap pra-pemrosesan dataset {dataset}...")
    try:
        res = training_controller.workflow_preprocess(dataset=dataset)
        if res.get('status') == 'success':
            training_controller.sync_nrp_from_processed(dbs=dbs, dataset=dataset)
            energy_kwh = cl_manager.metrics.get("compute_energy_kwh", 0.0)
            cl_manager.update_metrics({
                "preprocess_energy_kwh": cl_manager.metrics.get("preprocess_energy_kwh", 0.0) + energy_kwh
            })
        return res
    finally:
        cl_manager.update_metrics({
            "preprocess_duration_s": cl_manager.metrics.get("preprocess_duration_s", 0.0) + (time.time() - start_t)
        })
        cl_manager.end_phase()

def internal_workflow_train(epochs: int, dataset: str, dbs: Session, model_version_id: int = None):
    cl_manager.start_phase("Training")
    cl_manager.reset_training_metrics()
    cl_manager.logger.info(f"Memulai pelatihan model ({epochs} epoch) dataset {dataset}...")
    
    if dbs is not None:
        try:
            from datetime import datetime, timezone, timedelta
            from app.db import models
            if model_version_id is None:
                # Create new ModelVersion at start of training
                now_wib = datetime.now(timezone(timedelta(hours=7)))
                new_v = models.ModelVersion(notes=f"Dataset: {dataset} | Epochs: {epochs} | Dibuat pada {now_wib.strftime('%Y-%m-%d %H:%M:%S')}")
                dbs.add(new_v)
                dbs.commit()
                dbs.refresh(new_v)
                model_version_id = new_v.version_id
                cl_manager.logger.info(f"Membuat versi model baru v{model_version_id} di database.")
            else:
                # Reuse existing model_version_id: clear old rounds
                dbs.query(models.TrainingRound).filter_by(model_version_id=model_version_id).delete()
                dbs.commit()
                cl_manager.logger.info(f"Menggunakan versi model {model_version_id} yang sudah ada. Menghapus ronde pelatihan lama.")
                
            cl_manager.current_db_version_id = model_version_id
        except Exception as db_err:
            cl_manager.logger.error(f"Gagal mempersiapkan versi model di database: {db_err}")
            dbs.rollback()
            
    try:
        res = training_controller.train_model(epochs=epochs, dataset=dataset, dbs=dbs, model_version_id=model_version_id)
        if res['status'] == 'success':
            history = res.get('epoch_history', [])
            cl_manager.update_metrics({
                "accuracy": res['accuracy'],
                "training_duration_s": res['duration_s'],
                "compute_energy_kwh": res.get('compute_energy_kwh', 0),
                "training_energy_kwh": res.get('compute_energy_kwh', 0),
                "epoch_history": history
            })
            
            # Perbarui metrik akumulatif akhir pada ronde terakhir di basis data
            if dbs is not None and history:
                try:
                    last_epoch_num = len(history)
                    last_round = dbs.query(models.TrainingRound).filter_by(
                        model_version_id=model_version_id,
                        round_number=last_epoch_num
                    ).first()
                    if last_round:
                        last_round.compute_energy_kwh = float(res.get('compute_energy_kwh', 0))
                        last_round.upload_volume_mb = float(cl_manager.metrics.get('upload_volume_mb', 0))
                        dbs.commit()
                except Exception as db_err:
                    cl_manager.logger.error(f"Gagal memperbarui metrik akhir pada ronde terakhir: {db_err}")
                    dbs.rollback()
        return res
    finally:
        cl_manager.end_phase()

def internal_workflow_export(dataset: str, dbs: Session, epochs: int = None):
    cl_manager.start_phase("Export & Eval")
    cl_manager.logger.info(f"Memulai tahap ekspor dan evaluasi model dataset {dataset}...")
    try:
        res = training_controller.generate_reference_and_eval(dbs=dbs, dataset=dataset)
        if res['status'] == 'success':
            if dbs is not None:
                cl_manager.increment_version(dbs=dbs)
            download_mb = res.get('download_volume_mb', 0)
            cl_manager.update_metrics({
                "download_volume_mb": download_mb,
                "total_round_time_s": round(time.time() - cl_manager.start_time, 2)
            })
            
            actual_epochs = epochs if epochs is not None else cl_manager.default_epochs
            from app.utils.versioning import get_or_create_model_version
            version_num, version_str, _ = get_or_create_model_version("cl", actual_epochs, dataset=dataset)
            cl_manager.model_version = version_num
            cl_manager.model_version_str = version_str
            cl_manager.save_version_metrics(version_str)
            
            # Check if version files already exist and log warning
            versioned_model_path = os.path.join(MODEL_DIR, f"global_model_{version_str}.pth")
            versioned_ref_path = os.path.join(MODEL_DIR, f"reference_embeddings_{version_str}.pth")
            
            if os.path.exists(versioned_model_path) or os.path.exists(versioned_ref_path):
                msg = f"[WARNING] Berkas versi model {version_str} sudah ada di disk dan akan ditimpa."
                cl_manager.logger.warn(msg)
                cl_manager.update_logs(msg)
                
            shutil.copy2(MODEL_PATH, versioned_model_path)
            if os.path.exists(REF_PATH):
                shutil.copy2(REF_PATH, versioned_ref_path)
                
            cl_manager.update_logs(f"[SUCCESS] Model disimpan sebagai {os.path.basename(versioned_model_path)}")
            cl_manager.update_logs(f"[SUCCESS] Referensi disimpan sebagai {os.path.basename(versioned_ref_path)}")
            
            cl_manager.save_version_metrics(version_str)
            
            try:
                if os.path.exists(VIDEO_CACHES_DIR):
                    for f in os.listdir(VIDEO_CACHES_DIR):
                        if f.endswith(".json"):
                            os.remove(os.path.join(VIDEO_CACHES_DIR, f))
                    cl_manager.logger.info("Menghapus cache deteksi video lama karena model diperbarui.")
            except Exception as cache_del_err:
                cl_manager.logger.warning(f"Gagal menghapus cache video lama: {cache_del_err}")
 
            if dbs is not None:
                try:
                    last_round = dbs.query(models.TrainingRound).order_by(models.TrainingRound.round_id.desc()).first()
                    if last_round:
                        last_round.download_volume_mb = float(download_mb)
                        dbs.commit()
                except: pass
        return res
    finally:
        cl_manager.end_phase()


# Tahap 1: Impor Data dari Terminal
@app.post("/workflow/import")
def workflow_import(dbs: Session = Depends(db.get_db)):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    return internal_workflow_import(dbs)

# Tahap 2: Pra-pemrosesan & Penyeimbangan Dataset
@app.post("/workflow/preprocess")
def workflow_preprocess(dataset: str = "students", dbs: Session = Depends(db.get_db)):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    return internal_workflow_preprocess(dataset, dbs)

# Tahap 3: Pelatihan Model Global
@app.post("/workflow/train")
def workflow_train(epochs: int = TRAINING_PARAMS["epochs"], dataset: str = "students", dbs: Session = Depends(db.get_db)):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    return internal_workflow_train(epochs, dataset, dbs)

# Tahap 4: Ekspor Model & Evaluasi Akhir
@app.post("/workflow/export")
def workflow_export(dataset: str = "students", epochs: int = None, dbs: Session = Depends(db.get_db)):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    return internal_workflow_export(dataset, dbs, epochs=epochs)

# Menjalankan Seluruh Siklus Hidup Pelatihan dari Awal sampai Akhir
@app.post("/workflow/full-lifecycle")
def workflow_full_lifecycle(dataset: str = "students", epochs: int = None, dbs: Session = Depends(db.get_db)):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    cl_manager.start_task()
    try:
        cl_manager.current_dataset = dataset
        cl_manager.update_logs(f"Memulai siklus pelatihan penuh (Full Lifecycle) dataset {dataset}...")
        if dataset == "students":
            res_import = internal_workflow_import(dbs=dbs)
            if res_import.get('status') != 'success': return res_import
        
        res_pre = internal_workflow_preprocess(dataset=dataset, dbs=dbs)
        if res_pre.get('status') != 'success': return res_pre
        
        training_epochs = epochs if epochs is not None else cl_manager.default_epochs
        res_train = internal_workflow_train(epochs=training_epochs, dataset=dataset, dbs=dbs)
        if res_train.get('status') != 'success': return res_train
        
        res_export = internal_workflow_export(dataset=dataset, dbs=dbs, epochs=training_epochs)
        cl_manager.update_logs(f"Siklus pelatihan penuh dataset {dataset} berhasil diselesaikan.")
        return res_export
        
    except Exception as e:
        cl_manager.update_logs(f"[ERROR] Kesalahan fatal dalam siklus: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        cl_manager.end_task()
        cl_manager.current_phase = "Standby"

@app.post("/workflow/preprocess-only")
def workflow_preprocess_only(dataset: str = "students", dbs: Session = Depends(db.get_db)):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    cl_manager.start_task()
    try:
        cl_manager.current_dataset = dataset
        cl_manager.update_logs(f"Memulai tahap preprocessing saja dataset {dataset}...")
        if dataset == "students":
            res_import = internal_workflow_import(dbs=dbs)
            if res_import.get('status') != 'success': 
                return res_import
        
        res_pre = internal_workflow_preprocess(dataset=dataset, dbs=dbs)
        cl_manager.update_logs(f"[SUCCESS] Preprocessing dataset {dataset} selesai. Data siap dilatih.")
        return res_pre
    except Exception as e:
        cl_manager.update_logs(f"[ERROR] Kesalahan fatal dalam preprocessing-only: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        cl_manager.end_task()
        cl_manager.current_phase = "Standby"

@app.post("/workflow/train-only")
def workflow_train_only(epochs: int = None, dataset: str = "students", dbs: Session = Depends(db.get_db)):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    cl_manager.start_task()
    if epochs is None:
        epochs = cl_manager.default_epochs
        
    try:
        cl_manager.current_dataset = dataset
        cl_manager.update_logs(f"Memulai pelatihan model terpisah (Train Only) dataset {dataset} dengan {epochs} epoch...")
        
        # Determine the model version based on the configuration and dataset
        from app.utils.versioning import get_or_create_model_version
        version_num, version_str, is_new = get_or_create_model_version("cl", epochs, dataset=dataset)
        
        if is_new:
            cl_manager.update_logs(f"Konfigurasi baru ({epochs} epoch, dataset {dataset}). Menggunakan versi baru: {version_str}")
        else:
            cl_manager.update_logs(f"Konfigurasi lama ({epochs} epoch, dataset {dataset}) dideteksi. Menggunakan kembali versi: {version_str}. Melatih ulang dari v0...")
            
        # Run training
        res_train = internal_workflow_train(epochs=epochs, dataset=dataset, dbs=dbs, model_version_id=None if is_new else version_num)
        if res_train.get('status') != 'success': 
            return res_train
            
        # Run export (we pass dbs if is_new, otherwise None to reuse database entry)
        res_export = internal_workflow_export(dataset=dataset, dbs=dbs if is_new else None, epochs=epochs)
        if res_export.get('status') != 'success':
            return res_export
            
        versioned_model_path = os.path.join(MODEL_DIR, f"global_model_{version_str}.pth")
        versioned_ref_path = os.path.join(MODEL_DIR, f"reference_embeddings_{version_str}.pth")
        
        return {
            "status": "success",
            "version": version_str,
            "epochs": epochs,
            "model_file": os.path.basename(versioned_model_path),
            "ref_file": os.path.basename(versioned_ref_path)
        }
    except Exception as e:
        cl_manager.update_logs(f"[ERROR] Kesalahan fatal dalam train-only: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        cl_manager.end_task()
        cl_manager.current_phase = "Standby"

# API status dataset dan status preprocessing
@app.get("/api/datasets/status")
def get_datasets_status():
    status_list = []
    
    # Kumpulkan semua dataset unik yang dilaporkan oleh client
    all_datasets = set()
    for cid, c_data in cl_manager.registered_clients.items():
        avail = c_data.get("available_datasets", {})
        for ds in avail.keys():
            all_datasets.add(ds)
        ds = c_data.get("dataset")
        if ds:
            all_datasets.add(ds)
            
    # Tambahkan dataset client yang terdeteksi di folder data (students_*)
    try:
        data_parent = os.path.dirname(UPLOAD_DIR) # "data"
        if os.path.exists(data_parent):
            for item in os.listdir(data_parent):
                if item.startswith("students_") and os.path.isdir(os.path.join(data_parent, item)):
                    ds_name = item.replace("students_", "")
                    all_datasets.add(ds_name)
    except Exception:
        pass
        
    all_datasets.add("students")
    datasets_to_check = sorted(list(all_datasets))
    
    for ds in datasets_to_check:
        raw_dir = UPLOAD_DIR
        processed_dir = PROCESSED_DATA
        
        if ds != "students":
            opt_datasets_path = f"/app/datasets/{ds}/students"
            direct_data_path = os.path.join(os.path.dirname(UPLOAD_DIR), ds)
            alt_path = os.path.join(os.path.dirname(UPLOAD_DIR), f"students_{ds}")
            sub_path = os.path.join(UPLOAD_DIR, ds)
            
            if os.path.exists(opt_datasets_path):
                raw_dir = opt_datasets_path
            elif os.path.exists(direct_data_path):
                raw_dir = direct_data_path
            elif os.path.exists(alt_path):
                raw_dir = alt_path
            else:
                raw_dir = sub_path
            
            processed_dir = os.path.join(os.path.dirname(PROCESSED_DATA), f"datasets_processed_{ds}")
        
        raw_exists = os.path.exists(raw_dir)
        processed_exists = os.path.exists(processed_dir)
        
        raw_classes = 0
        raw_images = 0
        if raw_exists:
            try:
                subdirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
                raw_classes = len(subdirs)
                for dirpath, _, filenames in os.walk(raw_dir):
                    raw_images += len([f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            except Exception: pass
                
        proc_classes = 0
        proc_images = 0
        if processed_exists:
            try:
                subdirs = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
                proc_classes = len(subdirs)
                for dirpath, _, filenames in os.walk(processed_dir):
                    proc_images += len([f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            except Exception: pass
                
        is_preprocessed = processed_exists and proc_images > 0
        
        status_list.append({
            "dataset": ds,
            "raw_exists": raw_exists,
            "raw_classes": raw_classes,
            "raw_images": raw_images,
            "processed_exists": processed_exists,
            "processed_classes": proc_classes,
            "processed_images": proc_images,
            "is_preprocessed": is_preprocessed,
            "status_label": "Sudah Diproses" if is_preprocessed else "Belum Diproses",
            "info": f"{raw_classes} klp, {raw_images} gbr (Raw) -> {proc_classes} klp, {proc_images} gbr (Aligned)" if is_preprocessed else f"{raw_classes} klp, {raw_images} gbr (Raw)"
        })
        
    return {"status": "success", "datasets": status_list}

# API pendukung dan akses aset

@app.get("/api/models/list")
def list_models():
    # Mengembalikan daftar model yang tersedia untuk uji coba atau presensi
    models_list = []
    models_list.append({"version": "v0", "epochs": 0, "file": "global_model_v0.pth", "label": "Model v0 (Base Model)"})
    from app.config import MODEL_DIR
    if os.path.exists(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
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
                    "epochs": epochs_val,
                    "rounds": rounds_val,
                    "file": f,
                    "label": label
                })
    
    # Custom sorting: versioned (descending), v0 (last)
    v0_models = [m for m in models_list if m["version"] == "v0"]
    other_models = [m for m in models_list if m["version"] != "v0"]
    other_models.sort(key=lambda x: x["version"], reverse=True)
    
    models_list = other_models + v0_models
    return {"status": "success", "models": models_list}

@app.post("/api/models/delete-specific")
def delete_specific_model(version: str, dbs: Session = Depends(db.get_db)):
    try:
        if version == "v0" or version == "0" or version == "active":
            raise HTTPException(400, "Tidak dapat menghapus model dasar atau model aktif")
            
        # 1. Hapus file fisik
        from app.config import MODEL_DIR
        model_file = os.path.join(MODEL_DIR, f"global_model_{version}.pth")
        ref_file = os.path.join(MODEL_DIR, f"reference_embeddings_{version}.pth")
        metrics_file = f"data/metrics_{version}.json"
        
        if os.path.exists(model_file):
            try: os.remove(model_file)
            except Exception as e: print(f"Error removing model file: {e}")
            
        if os.path.exists(ref_file):
            try: os.remove(ref_file)
            except Exception as e: print(f"Error removing ref file: {e}")
            
        if os.path.exists(metrics_file):
            try: os.remove(metrics_file)
            except Exception as e: print(f"Error removing metrics file: {e}")
            
        cl_manager.update_logs(f"[INFO] Model versi {version} berhasil dihapus dari sistem.")
        return {"status": "success", "message": f"Model versi {version} berhasil dihapus."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/models/reset")
def reset_models(dbs: Session = Depends(db.get_db)):
    try:
        # 1. Hapus semua versi model dari DB
        dbs.query(models.TrainingRound).delete()
        dbs.query(models.ModelVersion).delete()
        dbs.commit()
        
        # 2. Hapus file fisik model cl/fl dan file metrics
        from app.config import MODEL_DIR, MODEL_PATH, REF_PATH
        if os.path.exists(MODEL_DIR):
            for f in os.listdir(MODEL_DIR):
                if f.startswith("global_model_") and f.endswith(".pth") and f not in ("global_model_v0.pth", "global_model.pth"):
                    try: os.remove(os.path.join(MODEL_DIR, f))
                    except: pass
                if f.startswith("reference_embeddings_") and f.endswith(".pth") and f != "reference_embeddings.pth":
                    try: os.remove(os.path.join(MODEL_DIR, f))
                    except: pass
        
        # Hapus file metrics di folder data/
        if os.path.exists("data"):
            for f in os.listdir("data"):
                if f.startswith("metrics_") and f.endswith(".json"):
                    try: os.remove(os.path.join("data", f))
                    except: pass
                    
        # 3. Kembalikan model aktif ke v0
        v0_path = os.path.join(MODEL_DIR, "global_model_v0.pth")
        if os.path.exists(v0_path):
            shutil.copy2(v0_path, MODEL_PATH)
            
        # Hapus reference embeddings aktif karena v0 tidak memiliki wajah terdaftar
        if os.path.exists(REF_PATH):
            try: os.remove(REF_PATH)
            except: pass
            
        # 4. Reset state manager
        cl_manager.model_version = 0
        cl_manager.model_version_str = "v0"
        cl_manager.metrics = {
            "epoch_history": [],
            "accuracy": 0.0,
            "loss": 0.0,
            "training_duration_s": 0,
            "compute_energy_kwh": 0,
            "upload_volume_mb": 0,
            "download_volume_mb": 0,
            "transmission_cost_idr": 0,
            "compute_cost_idr": 0
        }
        
        # Update logs
        cl_manager.update_logs("[INFO] Server di-reset ke model dasar v0. Semua versi model terlatih telah dihapus.")
        
        return {"status": "success", "message": "Model berhasil di-reset ke v0."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/model")
@app.get(api_get_model)
async def get_model(version: str = None):
    # Terminal memanggil endpoint ini untuk mengambil bobot model terbaru atau versi spesifik
    return inference_controller.get_model(version)

@app.get("/api/reference")
@app.get(api_get_reference)
async def get_reference_embeddings(version: str = None):
    # Terminal memanggil endpoint ini untuk mengambil basis data wajah terbaru atau versi spesifik
    return inference_controller.get_reference(version)

@app.post(api_submit_attendance)
async def submit_attendance(recap: schemas.AttendanceRecapBase, dbs: Session = Depends(db.get_db)):
    # Menerima laporan identifikasi kehadiran dari terminal
    label = str(recap.user_id)
    parts = label.split("_", 1)
    nrp = label.split("_", 1)[0].strip()
    # Jangan tampilkan di konsol utama agar tidak kotor
    # cl_manager.logger.info(f"Sync Attendance | Client: {recap.edge_id} | NRP: {nrp} | Sim: {recap.confidence:.4f}")
    

    student = dbs.query(models.UserGlobal).filter(models.UserGlobal.nrp == nrp).first()
    if not student:
        name = parts[1].strip() if len(parts) > 1 else "Unknown"
        student = models.UserGlobal(name=name, nrp=nrp, registered_edge_id=recap.edge_id)
        dbs.add(student)
        dbs.commit()
        dbs.refresh(student)
        
    attendance = models.AttendanceRecap(
        user_id=student.user_id, edge_id=recap.edge_id,
        confidence=recap.confidence, lecture_id=recap.lecture_id,
        timestamp=datetime.now(timezone(timedelta(hours=7)))
    )
    dbs.add(attendance)
    dbs.commit()
    return {"status": "success", "student": student.name}

# Endpoints streaming video dan cache server
STORED_VIDEOS_DIR = "stored_videos"
VIDEO_CACHES_DIR = "video_caches"
os.makedirs(STORED_VIDEOS_DIR, exist_ok=True)
os.makedirs(VIDEO_CACHES_DIR, exist_ok=True)

def strip_audio_from_video(file_path: str, logger):
    temp_output = file_path + ".noaudio.mp4"
    try:
        # Hilangkan audio secara instan menggunakan salinan stream ffmpeg (-vcodec copy -an)
        cmd = ["ffmpeg", "-y", "-i", file_path, "-an", "-vcodec", "copy", temp_output]
        logger.info(f"Mencoba menghapus audio dari video: {file_path}")
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=15)
        
        # Replace original file with audio-stripped file
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
        strip_audio_from_video(file_path, cl_manager.logger)
                
        cl_manager.logger.info(f"Video {filename} berhasil diunggah ke server.")
        return {"status": "success", "message": f"Video {filename} uploaded to server"}
    except Exception as e:
        cl_manager.logger.error(f"Gagal mengunggah video ke server: {e}")
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
        
        cl_manager.logger.success(f"Video {video_name} dan cache pendeteksian dihapus dari disk server.")
        return {
            "status": "success", 
            "message": f"Berhasil menghapus video {video_name} dan cache-nya.",
            "deleted": deleted_files
        }
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Gagal menghapus file: {e}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, access_log=False)