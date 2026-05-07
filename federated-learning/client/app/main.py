import os
import time
import io
import base64

import cv2
import numpy as np
import requests
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.db.db import engine, Base, get_db, init_db
from app.db import models

init_db()

from app.manager import fl_manager
from app.controllers.management import router as management_router
from app.controllers.attendance import AttendanceController


# Inisialisasi Database SQLite Lokal
Base.metadata.create_all(bind=engine)

# Konfigurasi Aplikasi FastAPI
app = FastAPI(title="Federated Face Recognition Terminal")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
templates.env.globals.update(os=os)

# Include Routers
app.include_router(management_router)

# --- ENDPOINTS DASAR ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    # Cek apakah ada index.html (Modern) atau attendance.html (Legacy)
    return templates.TemplateResponse("attendance.html", {
        "request": request,
        "client_id": fl_manager.client_id,
        "model_version": fl_manager.model_version,
        "title": "Edge Client"
    })

@app.get("/video_feed")
@app.get("/api/attendance/frame")
async def video_feed():
    def gen_frames():
        while True:
            if fl_manager.latest_frame is not None:
                # Resize ke 640x480 untuk optimasi streaming MJPEG
                frame_small = cv2.resize(fl_manager.latest_frame, (640, 480))
                ret, buffer = cv2.imencode('.jpg', frame_small, [cv2.IMWRITE_JPEG_QUALITY, 40])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.15)
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/api/request-discovery")
async def request_discovery_legacy(background_tasks: BackgroundTasks):
    background_tasks.add_task(fl_manager.run_discovery_phase)
    return {"status": "success"}

@app.post("/api/request-preprocess")
async def request_preprocess_legacy(background_tasks: BackgroundTasks):
    background_tasks.add_task(fl_manager.run_preprocess_phase)
    return {"status": "success"}

@app.post("/api/request-registry")
async def request_registry_legacy(background_tasks: BackgroundTasks):
    background_tasks.add_task(fl_manager.run_registry_phase)
    return {"status": "success"}

@app.post("/api/inference")
async def api_inference(data: dict, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    attendance_ptr = AttendanceController(fl_manager)
    try:
        res = await attendance_ptr.process_inference(data['image'], db, background_tasks)
        
        # Penyelarasan UI: Masukkan frame kamera browser ke aliran video utama (Mirroring)
        try:
            img_bytes = base64.b64decode(data['image'])
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                fl_manager.latest_frame = frame
        except: pass
        
        fl_manager.latest_result = res
        return res
    except Exception as e:
        fl_manager.logger.error(f"Inference processing failed: {e}")
        return JSONResponse({"matched": "Error", "error": str(e)}, status_code=400)

@app.post("/api/register")
async def register_user(user_id: str, name: str, image_base64: str, db: Session = Depends(get_db)):
    
    attendance_ptr = AttendanceController(fl_manager)
    return attendance_ptr.process_registration(user_id, name, image_base64, db)

@app.get("/api/results/latest")
async def get_latest_result():
    return fl_manager.latest_result

@app.post("/api/camera/toggle")
async def toggle_camera():
    is_on = fl_manager.toggle_camera()
    return {"status": "success", "is_on": is_on}

@app.get("/api/logs")
async def get_logs():
    """Mengambil log dari memori logger global."""
    try:
        logs = fl_manager.logger.get_logs()
        return {"logs": "\n".join(logs)}
    except Exception as e:
        return {"logs": f"Error membaca log: {str(e)}"}

@app.on_event("startup")
def startup_event():
    fl_manager.start_time = time.time()
    fl_manager.start_background_tasks()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
