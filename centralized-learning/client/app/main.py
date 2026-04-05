import os
import base64
import io
import time
import uvicorn
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.client_manager_instance import cl_client

# Konfigurasi Aplikasi Terminal Terpusat (Centralized)
app = FastAPI(title="Centralized Edge Terminal")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Halaman Utama Presensi
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("attendance.html", {
        "request": request, 
        "client_id": cl_client.client_id,
        "model_version": cl_client.current_model_version
    })

# API Pengenalan Wajah (Inference)
@app.post("/api/inference")
async def api_inference(data: dict):
    # Memastikan model dan aset sudah terunduh dari server
    if not cl_client.has_assets:
        return JSONResponse({"matched": "Unknown", "confidence": 0, "message": "Model belum siap"})
    
    start = time.time()
    try:
        img_data = base64.b64decode(data['image'])
        img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        # Set as latest frame for MJPEG monitoring
        cl_client.latest_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Proses pengenalan wajah melalui controller attendance
        matched, confidence = cl_client.attendance.recognize_and_submit(
            img_pil, cl_client.model, cl_client.reference_embeddings
        )
        
        latency = int((time.time() - start) * 1000)
        res = {
            "matched": matched, 
            "confidence": confidence, 
            "latency_ms": latency,
            "model_version": cl_client.current_model_version,
            "is_virtual": False # Remote frame is considered "Real" in UI
        }
        cl_client.latest_result = res
        return res
    except Exception as e:
        print(f"[ERROR] Gagal memproses inference: {e}")
        return JSONResponse({"matched": "Error", "error": str(e)}, status_code=400)
# Endpoint Monitoring Video (MJPEG Stream)
@app.get("/video_feed")
async def video_feed():
    def gen_frames():
        while True:
            if cl_client.latest_frame is not None:
                # Kompres ke JPEG untuk efisiensi bandwidth
                ret, buffer = cv2.imencode('.jpg', cl_client.latest_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.2)

    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

# API Hasil Terakhir (untuk Polling UI Remote)
@app.get("/api/results/latest")
async def get_latest_result():
    return cl_client.latest_result

# API Permintaan Data dari Server
@app.post("/api/request-data")
async def request_data():
    print(f"[INFO] Server meminta unggah data dataset.")
    success, msg = cl_client.management.package_and_upload()
    return {"status": "success" if success else "error", "message": msg}

# Event saat Startup
@app.on_event("startup")
def startup_event():
    print(f"[STARTUP] Inisialisasi Terminal: {cl_client.client_id}", flush=True)
    cl_client.start_background_tasks()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)