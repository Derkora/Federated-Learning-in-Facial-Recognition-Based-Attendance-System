from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uvicorn
import base64
import io
import time
from PIL import Image

from app.client_manager_instance import cl_client

app = FastAPI(title="Centralized Edge Terminal")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("attendance.html", {
        "request": request, 
        "client_id": cl_client.client_id,
        "model_version": cl_client.current_model_version
    })

@app.post("/api/inference")
async def api_inference(data: dict):
    if not cl_client.has_assets:
        return JSONResponse({"matched": "Unknown", "confidence": 0, "message": "Model not ready"})
    
    start = time.time()
    try:
        img_data = base64.b64decode(data['image'])
        img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
        
        matched, confidence = cl_client.attendance.recognize_and_submit(
            img_pil, cl_client.model, cl_client.reference_embeddings
        )
        
        latency = int((time.time() - start) * 1000)
        return {"matched": matched, "confidence": confidence, "latency_ms": latency}
    except Exception as e:
        return JSONResponse({"matched": "Error", "error": str(e)}, status_code=400)

@app.post("/api/request-data")
async def request_data():
    print(f"[{cl_client.client_id}] Server requested data. Packaging...")
    success, msg = cl_client.management.package_and_upload()
    return {"status": "success" if success else "error", "message": msg}

@app.on_event("startup")
def startup_event():
    print(f"[STARTUP] Initializing Centralized Client: {cl_client.client_id}", flush=True)
    cl_client.start_background_tasks()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)