import os
import shutil
import time
import torch
from datetime import datetime, time as dt_time, timedelta, timezone
from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import uvicorn
import zipfile

from app.db import models, db, schemas, crud
from app.controllers.student import student_controller
from app.controllers.training import training_controller
from app.controllers.inference import inference_controller
from app.server_manager_instance import cl_manager
from app.utils.mobilefacenet import MobileFaceNet
from app.config import (
    MODEL_PATH, REF_PATH, UPLOAD_DIR, TRAINING_PARAMS
)
models.Base.metadata.create_all(bind=db.engine)

if not os.path.exists(os.path.dirname(MODEL_PATH)): os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("[INIT] Membuat model global awal v0...", flush=True)
    torch.save(MobileFaceNet().state_dict(), MODEL_PATH)
if not os.path.exists(REF_PATH):
    print("[INIT] Membuat basis data referensi awal...", flush=True)
    torch.save({}, REF_PATH)

# Konfigurasi Aplikasi Dashboard Server
app = FastAPI(title="Centralized Attendance Server")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Dashboard Utama
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    status = cl_manager.get_status()
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

# API Status Sistem
@app.get("/api/status")
async def get_status():
    return cl_manager.get_status()

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
    return {"status": "online", "upload_requested": cl_manager.upload_requested, "model_version": cl_manager.model_version}

# Pendaftaran Terminal (Client) Baru
@app.post("/register-client", response_model=schemas.ClientResponse)
async def register_client(client_data: schemas.ClientBase, request: Request, dbs: Session = Depends(db.get_db)):
    client_ip = request.client.host
    print(f"[REG] Mendaftarkan terminal {client_data.edge_id} dari {client_ip}", flush=True)
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
    print(f"[UPLOAD] Menerima dataset bulk: {file.filename}", flush=True)
    UPLOAD_TEMP = "data/upload.zip"
    os.makedirs("data", exist_ok=True)
    
    with open(UPLOAD_TEMP, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        edge_id = file.filename.split("_")[0]
        
        with zipfile.ZipFile(UPLOAD_TEMP, 'r') as zip_ref:
            # Dapatkan daftar folder (NRP) sebelum diekstrak untuk pemetaan
            names = zip_ref.namelist()
            # Asumsi folder adalah level pertama: "NRP_Name/"
            top_level = {n.split('/')[0] for n in names if '/' in n}
            nrps = [t.split('_')[0] for t in top_level if t]
            
            zip_ref.extractall("data/students")
            
            # Daftarkan pemetaan di manajer
            cl_manager.register_upload(edge_id, nrps)
            
        os.remove(UPLOAD_TEMP)
        cl_manager.update_logs(f"Menerima {len(nrps)} data mahasiswa dari {edge_id}")
        return {"status": "success", "filename": file.filename, "edge_id": edge_id}
    except Exception as e:
        print(f"[UPLOAD ERROR] Gagal ekstrak: {e}", flush=True)
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
    print("[INFO] Menunggu client mengirimkan dataset...", flush=True)
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
def workflow_preprocess():
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    cl_manager.start_phase("Preprocess & Balance")
    print("[INFO] Memulai tahap pra-pemrosesan...", flush=True)
    try:
        return training_controller.preprocess_and_balance()
    finally:
        cl_manager.end_phase()

# Tahap 3: Pelatihan Model Global
@app.post("/workflow/train")
def workflow_train(epochs: int = TRAINING_PARAMS["total_epochs"]):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    cl_manager.start_phase("Training")
    print(f"[INFO] Memulai pelatihan model ({epochs} epoch)...", flush=True)
    try:
        res = training_controller.train_model(epochs=epochs)
        if res['status'] == 'success':
            cl_manager.update_metrics({
                "accuracy": res['accuracy'],
                "training_duration_s": res['duration_s'],
                "compute_energy_kwh": res.get('compute_energy_kwh', 0),
                "epoch_history": res.get('epoch_history', [])
            })
        return res
    finally:
        cl_manager.end_phase()

# Tahap 4: Ekspor Model & Evaluasi Akhir
@app.post("/workflow/export")
def workflow_export():
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    cl_manager.start_phase("Export & Eval")
    print("[INFO] Memulai tahap ekspor dan evaluasi model...", flush=True)
    try:
        res = training_controller.generate_reference_and_eval()
        if res['status'] == 'success':
            cl_manager.increment_version()
            cl_manager.update_metrics({
                "download_volume_mb": res.get('download_volume_mb', 0),
                "total_round_time_s": round(time.time() - cl_manager.start_time, 2)
            })
        return res
    finally:
        cl_manager.end_phase()

# Menjalankan Seluruh Siklus Hidup Pelatihan dari Awal sampai Akhir
@app.post("/workflow/full-lifecycle")
def workflow_full_lifecycle(dbs: Session = Depends(db.get_db)):
    if cl_manager.is_running: raise HTTPException(400, "Server sedang sibuk")
    
    try:
        cl_manager.update_logs("Memulai siklus pelatihan penuh (Full Lifecycle)...")
        res_import = workflow_import(dbs)
        if res_import.get('status') != 'success': return res_import
        
        res_pre = workflow_preprocess()
        if res_pre.get('status') != 'success': return res_pre
        
        res_train = workflow_train()
        if res_train.get('status') != 'success': return res_train
        
        res_export = workflow_export()
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
    print(f"[DEBUG] Submit Attendance from {recap.edge_id}: label='{label}' -> nrp='{nrp}'", flush=True)
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