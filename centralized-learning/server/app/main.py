# --- STANDAR LIBRARY ---
import os
import shutil
import time
import json
import zipfile
from datetime import datetime, time as dt_time, timedelta, timezone

# --- THIRD PARTY ---
import torch
import requests
import uvicorn
from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

# --- APP MODULES ---
from app.db import models, db, schemas, crud
models.Base.metadata.create_all(bind=db.engine)
from app.utils.mobilefacenet import MobileFaceNet
from app.config import (
    MODEL_PATH, REF_PATH, UPLOAD_DIR, TRAINING_PARAMS
)
from app.server_manager_instance import cl_manager
from app.controllers.student import student_controller
from app.controllers.training import training_controller
from app.controllers.inference import inference_controller

if not os.path.exists(os.path.dirname(MODEL_PATH)): os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
if not os.path.exists(MODEL_PATH):
    cl_manager.logger.info("Membuat model global awal v0...")
    torch.save(MobileFaceNet().state_dict(), MODEL_PATH)
if not os.path.exists(REF_PATH):
    cl_manager.logger.info("Membuat basis data referensi awal...")
    torch.save({}, REF_PATH)

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

# API Update Pengaturan
@app.post("/api/settings")
async def update_settings(data: dict):
    success = cl_manager.save_settings(data)
    if success:
        return {"status": "success", "message": "Settings updated"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save settings")

@app.post("/api/clients/ready")
async def report_client_ready(data: dict):
    # Digunakan untuk koordinasi sinkronisasi data
    return {"status": "ok"}

# Endpoint Baru: Ambil Log Jarak Jauh dari Client (Proxy)
@app.get("/api/clients/logs/{client_id}")
async def get_client_logs(client_id: str, dbs: Session = Depends(db.get_db)):
    """Mengambil log aktivitas dari client tertentu secara remote."""
    client = dbs.query(models.Client).filter_by(edge_id=client_id).first()
    
    if not client or not client.ip_address:
        raise HTTPException(status_code=404, detail="Client tidak ditemukan atau IP tidak terdaftar.")
    
    try:
        # Panggil endpoint /api/logs di sisi client
        res = requests.get(f"http://{client.ip_address}:8080/api/logs", timeout=3)
        if res.status_code == 200:
            return res.json()
        return {"logs": f"Gagal mengambil log dari client: HTTP {res.status_code}"}
    except Exception as e:
        return {"logs": f"Client {client_id} tidak merespon: {str(e)}"}

# API Status Sistem
@app.get("/api/status")
async def get_status(dbs: Session = Depends(db.get_db)):
    return cl_manager.get_status(db=dbs)

# API Hasil Pelatihan (Metrik)
@app.get("/api/results")
async def get_results():
    return cl_manager.metrics

@app.get("/api/attendance")
async def get_attendance_recap(dbs: Session = Depends(db.get_db)):
    tz_wib = timezone(timedelta(hours=7))
    today = datetime.now(tz_wib).date()
    start_of_day = datetime.combine(today, dt_time.min)
    
    users = dbs.query(models.UserGlobal).all()
    recap = []
    
    for user in users:
        last_entry = dbs.query(models.AttendanceRecap).filter(
            models.AttendanceRecap.user_id == user.user_id,
            models.AttendanceRecap.timestamp >= start_of_day
        ).order_by(models.AttendanceRecap.timestamp.desc()).first()
        
        recap.append({
            "name": user.name,
            "nrp": user.nrp,
            "status": "HADIR" if last_entry else "MENUNGGU",
            "time": last_entry.timestamp.strftime("%H:%M:%S") if last_entry else "--:--",
            "confidence": f"{int(last_entry.confidence * 100)}%" if last_entry else "--",
            "registered_edge_id": user.registered_edge_id or "client-1"
        })
    
    return recap

# Pemeriksaan Kesehatan (Health Check) bagi Terminal
@app.get("/ping")
async def health_check():
    return {
        "status": "online", 
        "upload_requested": cl_manager.upload_requested, 
        "model_version": cl_manager.model_version,
        "inference_threshold": cl_manager.inference_threshold
    }

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
    
    # Simpan ke memori manager
    log_entry = {
        "timestamp": timestamp,
        "client_id": client_id,
        "user_id": user_id,
        "confidence": f"{confidence:.4f}",
        "latency": f"{latency}ms",
        "status": status
    }
    
    if "inference_logs" not in cl_manager.metrics:
        cl_manager.metrics["inference_logs"] = []
        
    cl_manager.metrics["inference_logs"].insert(0, log_entry)
    # Simpan hingga 10.000 baris
    cl_manager.metrics["inference_logs"] = cl_manager.metrics["inference_logs"][:10000]
    
    # Simpan ke disk agar persisten
    cl_manager.save_inference_logs()
    
    return {"status": "logged"}

# Pendaftaran Terminal (Client) Baru
@app.post("/register-client", response_model=schemas.ClientResponse)
async def register_client(client_data: schemas.ClientBase, request: Request, dbs: Session = Depends(db.get_db)):
    client_ip = request.client.host
    cl_manager.logger.info(f"Mendaftarkan terminal {client_data.edge_id} dari {client_ip}")
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
@app.post("/upload-bulk-zip")
async def upload_bulk_zip(file: UploadFile = File(...)):
    edge_id = file.filename.split("_")[0]
    UPLOAD_TEMP = f"data/upload_{edge_id}_{int(time.time())}.zip"
    os.makedirs("data", exist_ok=True)
    
    with open(UPLOAD_TEMP, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        with zipfile.ZipFile(UPLOAD_TEMP, 'r') as zip_ref:
            # Dapatkan daftar folder (NRP) sebelum diekstrak untuk pemetaan
            names = zip_ref.namelist()
            # Asumsi folder adalah level pertama: "NRP_Name/"
            top_level = {n.split('/')[0] for n in names if '/' in n}
            nrps = [t.split('_')[0] for t in top_level if t]
            
            zip_ref.extractall("data/students")
            
            # Daftarkan pemetaan di manajer
            cl_manager.register_upload(edge_id, nrps)
            cl_manager.logger.success(f"Berhasil mengekstrak {len(nrps)} folder mahasiswa.")
            
        os.remove(UPLOAD_TEMP)
        cl_manager.update_logs(f"Menerima {len(nrps)} data mahasiswa dari {edge_id}")
        return {"status": "success", "filename": file.filename, "edge_id": edge_id}
    except Exception as e:
        cl_manager.logger.error(f"Gagal ekstrak dataset ZIP: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

# --- ALUR KERJA PELATIHAN TERPUSAT (RESEARCH WORKFLOW) ---

# Tahap 1: Impor Data dari Terminal
@app.post("/workflow/import")
def workflow_import(dbs: Session = Depends(db.get_db)):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    
    cl_manager.start_phase("Import Data")
    cl_manager.received_data = [] 
    cl_manager.uploader_map = {} 
    if os.path.exists(UPLOAD_DIR): shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    cl_manager.upload_requested = True
    cl_manager.logger.info("Menunggu client mengirimkan dataset...")
    try:
        online_count = dbs.query(models.Client).filter(models.Client.status == "online").count()
        res = training_controller.fetch_data(wait_timeout=600, expected_clients=online_count)
        
        if res['status'] == 'success':
            cl_manager.update_metrics({"upload_volume_mb": res['upload_volume_mb']})
            folders = [f for f in os.listdir(UPLOAD_DIR) if os.path.isdir(os.path.join(UPLOAD_DIR, f))]
            for folder in folders:
                parts = folder.split("_", 1)
                nrp = parts[0].strip()
                name = parts[1].strip() if len(parts) > 1 else "Unknown"
                
                existing = dbs.query(models.UserGlobal).filter(models.UserGlobal.nrp == nrp).first()
                if not existing:
                    # Ambil edge_id dari pemetaan uploader, atau fallback ke yang pertama ada
                    edge_id = cl_manager.uploader_map.get(nrp)
                    if not edge_id:
                        client = dbs.query(models.Client).first()
                        edge_id = client.edge_id if client else None
                    
                    new_student = models.UserGlobal(name=name, nrp=nrp, registered_edge_id=edge_id)
                    dbs.add(new_student)
                else:
                    # Opsi: Perbarui registered_edge_id jika belum ada (opsional)
                    if not existing.registered_edge_id:
                        existing.registered_edge_id = cl_manager.uploader_map.get(nrp)
            dbs.commit()
            cl_manager.update_received_data(UPLOAD_DIR)
            
        return res
    finally:
        cl_manager.upload_requested = False
        cl_manager.end_phase()

# Tahap 2: Pra-pemrosesan & Penyeimbangan Dataset
@app.post("/workflow/preprocess")
def workflow_preprocess(dbs: Session = Depends(db.get_db)):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    cl_manager.start_phase("Preprocess & Balance")
    cl_manager.logger.info("Memulai tahap pra-pemrosesan...")
    try:
        res = training_controller.preprocess_and_balance()
        if res.get('status') == 'success':
            training_controller.sync_nrp_from_processed(dbs=dbs)
        return res
    finally:
        cl_manager.end_phase()

# Tahap 3: Pelatihan Model Global
@app.post("/workflow/train")
def workflow_train(epochs: int = TRAINING_PARAMS["epochs"], dbs: Session = Depends(db.get_db)):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    cl_manager.start_phase("Training")
    cl_manager.logger.info(f"Memulai pelatihan model ({epochs} epoch)...")
    try:
        res = training_controller.train_model(epochs=epochs)
        if res['status'] == 'success':
            history = res.get('epoch_history', [])
            cl_manager.update_metrics({
                "accuracy": res['accuracy'],
                "training_duration_s": res['duration_s'],
                "compute_energy_kwh": res.get('compute_energy_kwh', 0),
                "epoch_history": history
            })
            
            # Simpan riwayat ronde ke database
            # Hanya simpan metrik sesi (durasi, energi, bandwidth) pada ronde TERAKHIR agar tidak dobel saat agregasi
            total_epochs = len(history)
            for i, h in enumerate(history):
                is_last = (i == total_epochs - 1)
                cl_manager.save_training_round(
                    dbs, 
                    h['epoch'], 
                    h['loss'], 
                    h['accuracy'],
                    duration=res['duration_s'] if is_last else 0,
                    energy=res.get('compute_energy_kwh', 0) if is_last else 0,
                    upload=cl_manager.metrics.get('upload_volume_mb', 0) if is_last else 0,
                    download=0 # Akan diisi di workflow_export
                )
        return res
    finally:
        cl_manager.end_phase()

# Tahap 4: Ekspor Model & Evaluasi Akhir
@app.post("/workflow/export")
def workflow_export(dbs: Session = Depends(db.get_db)):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    cl_manager.start_phase("Export & Eval")
    cl_manager.logger.info("Memulai tahap ekspor dan evaluasi model...")
    try:
        res = training_controller.generate_reference_and_eval(dbs=dbs)
        if res['status'] == 'success':
            cl_manager.increment_version(dbs=dbs)
            download_mb = res.get('download_volume_mb', 0)
            cl_manager.update_metrics({
                "download_volume_mb": download_mb,
                "total_round_time_s": round(time.time() - cl_manager.start_time, 2)
            })
            
            # Update data download_mb ke ronde terakhir di database
            try:
                last_round = dbs.query(models.TrainingRound).order_by(models.TrainingRound.round_id.desc()).first()
                if last_round:
                    last_round.download_volume_mb = float(download_mb)
                    dbs.commit()
            except: pass
        return res
    finally:
        cl_manager.end_phase()

# Menjalankan Seluruh Siklus Hidup Pelatihan dari Awal sampai Akhir
@app.post("/workflow/full-lifecycle")
def workflow_full_lifecycle(dbs: Session = Depends(db.get_db)):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    
    try:
        cl_manager.update_logs("Memulai siklus pelatihan penuh (Full Lifecycle)...")
        res_import = workflow_import(dbs=dbs)
        if res_import.get('status') != 'success': return res_import
        
        res_pre = workflow_preprocess(dbs=dbs)
        if res_pre.get('status') != 'success': return res_pre
        
        res_train = workflow_train(dbs=dbs)
        if res_train.get('status') != 'success': return res_train
        
        res_export = workflow_export(dbs=dbs)
        cl_manager.update_logs("Siklus pelatihan penuh berhasil diselesaikan.")
        return res_export

        
    except Exception as e:
        cl_manager.update_logs(f"[ERROR] Kesalahan fatal dalam siklus: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        cl_manager.is_running = False
        cl_manager.current_phase = "Standby"

# --- API PENDUKUNG & AKSES ASET ---

@app.get("/get-model")
async def get_model():
    # Terminal memanggil endpoint ini untuk mengambil bobot model terbaru
    if not os.path.exists(MODEL_PATH): 
        raise HTTPException(404, "Berkas model tidak ditemukan")
    return inference_controller.get_model()

@app.get("/get-reference-embeddings")
async def get_reference_embeddings():
    # Terminal memanggil endpoint ini untuk mengambil basis data wajah terbaru
    if not os.path.exists(REF_PATH): 
        raise HTTPException(404, "Berkas referensi tidak ditemukan")
    return inference_controller.get_reference()

@app.post("/submit-attendance")
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)