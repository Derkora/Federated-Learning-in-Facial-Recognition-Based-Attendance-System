from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

import os
import requests
import uvicorn

from .db.db import engine, get_db, Base
from .client_manager_instance import fl_manager
from .controllers.attendance_controller import AttendanceController

# Inisialisasi Database SQLite Lokal
Base.metadata.create_all(bind=engine)

# Konfigurasi Aplikasi FastAPI
app = FastAPI(title="Federated Face Recognition Terminal")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
templates.env.globals.update(os=os)

# Inisialisasi Controller
attendance_ptr = AttendanceController(fl_manager)

# Halaman Utama Presensi
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("attendance.html", {
        "request": request,
        "client_id": fl_manager.client_id,
        "model_version": fl_manager.model_version,
        "title": "Edge Terminal"
    })

# API Pengenalan Wajah (Inference)
# Endpoint ini dipanggil oleh frontend untuk memproses frame gambar dari kamera.
# Logika deteksi dan identitas dikelola oleh AttendanceController.
@app.post("/api/inference")
async def api_inference(data: dict, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    try:
        return await attendance_ptr.process_inference(data['image'], db, background_tasks)
    except Exception as e:
        print(f"[ERROR] Gagal memproses inference: {e}")
        return JSONResponse({"matched": "Error", "error": str(e)}, status_code=400)

# API Registrasi Mahasiswa
# Digunakan untuk mendaftarkan mahasiswa baru secara lokal di terminal ini.
@app.post("/api/register")
async def register_user(user_id: str, name: str, image_base64: str, db: Session = Depends(get_db)):
    try:
        return attendance_ptr.process_registration(user_id, name, image_base64, db)
    except Exception as e:
        print(f"[ERROR] Gagal registrasi: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# --- ORKESTRASI FEDERATED LEARNING ---

# Tahap Discovery: Sinkronisasi daftar mahasiswa dengan server
@app.post("/api/request-data")
@app.post("/api/request-discovery")
async def request_discovery(background_tasks: BackgroundTasks):
    background_tasks.add_task(fl_manager.run_discovery_phase)
    return {"status": "success", "message": "Discovery started"}

# Tahap Preprocessing: Menyiapkan dataset (MTCNN Crop & Laplacian Filter)
@app.post("/api/request-preprocess")
async def request_preprocess(background_tasks: BackgroundTasks):
    def do_prep():
        fl_manager.run_preprocess_phase()
        # Melaporkan status READY ke server setelah preprocessing selesai secara otomatis.
        try:
            requests.post(f"{fl_manager.server_api_url}/api/clients/ready", json={"client_id": fl_manager.client_id}, timeout=5)
            print(f"[OK] Melaporkan status READY ({fl_manager.client_id}) ke server.")
        except Exception as e:
            print(f"[ERROR] Gagal melaporkan status READY: {e}")
            
    background_tasks.add_task(do_prep)
    return {"status": "success", "message": "Preprocessing started"}

# Tahap Registry: Pembuatan galeri identitas global (Centroids)
@app.post("/api/request-registry")
async def request_registry(background_tasks: BackgroundTasks):
    background_tasks.add_task(fl_manager.run_registry_phase)
    return {"status": "success", "message": "Registry generation started"}

# Event saat aplikasi dinyalakan
@app.on_event("startup")
def startup_event():
    fl_manager.start_background_tasks()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
