from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from datetime import datetime
import os
import uvicorn
import shutil

from app.db import models, db, schemas, crud
from app.controllers.student import student_controller
from app.controllers.training import training_controller
from app.controllers.inference import inference_controller
from app.server_manager_instance import cl_manager

# Inisialisasi Database
models.Base.metadata.create_all(bind=db.engine)

# Inisialisasi Model Awal (v0)
MODEL_DIR = "app/model"
MODEL_PATH = f"{MODEL_DIR}/global_model.pth"
REF_PATH = f"{MODEL_DIR}/reference_embeddings.pth"
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    import torch
    from app.utils.mobilefacenet import MobileFaceNet
    print("[INIT] Creating initial global model v0...", flush=True)
    torch.save(MobileFaceNet().state_dict(), MODEL_PATH)
if not os.path.exists(REF_PATH):
    import torch
    print("[INIT] Creating initial reference database...", flush=True)
    torch.save({}, REF_PATH)

app = FastAPI(title="Centralized Attendance Server")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    status = cl_manager.get_status()
    return templates.TemplateResponse("index.html", {"request": request, "title": "Dashboard", "status": status})

@app.get("/ping")
async def health_check():
    # Diagnostic Log: Prove clients are polling
    # print(f"[POLL] Edge terminal checking in...", flush=True)
    return {"status": "online", "upload_requested": cl_manager.upload_requested}

@app.post("/register-client", response_model=schemas.ClientResponse)
async def register_client(client_data: schemas.ClientBase, request: Request, dbs: Session = Depends(db.get_db)):
    client_ip = request.client.host
    print(f"[REG] Registering client {client_data.edge_id} from {client_ip}", flush=True)
    existing = dbs.query(models.Client).filter(models.Client.edge_id == client_data.edge_id).first()
    if existing:
        existing.last_seen = datetime.utcnow()
        existing.ip_address = client_ip
        existing.status = client_data.status
        dbs.commit()
        dbs.refresh(existing)
        return existing
    client_data.ip_address = client_ip
    return crud.register_client(dbs, client_data)

# --- PHASED RESEARCH WORKFLOW ---

@app.post("/workflow/import")
async def workflow_import():
    if cl_manager.is_busy: raise HTTPException(400, "Server is busy")
    
    cl_manager.start_phase("Import Data")
    UPLOAD_DIR = "data/students"
    if os.path.exists(UPLOAD_DIR): shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    cl_manager.upload_requested = True
    print("[ORCHESTRATOR] Training mode activated. Waiting for clients to check-in and upload...", flush=True)
    try:
        res = training_controller.fetch_data(wait_timeout=600)
        if res['status'] == 'success':
            cl_manager.update_metrics({"payload_size_mb": res['payload_mb']})
        return res
    finally:
        cl_manager.upload_requested = False
        cl_manager.end_phase()

@app.post("/workflow/preprocess")
async def workflow_preprocess():
    if cl_manager.is_busy: raise HTTPException(400, "Server is busy")
    cl_manager.start_phase("Preprocess & Balance")
    print("[ORCHESTRATOR] Starting Preprocessing Phase...", flush=True)
    try:
        return training_controller.preprocess_and_balance()
    finally:
        cl_manager.end_phase()

@app.post("/workflow/train")
async def workflow_train(epochs: int = 10):
    if cl_manager.is_busy: raise HTTPException(400, "Server is busy")
    cl_manager.start_phase("Training")
    print(f"[ORCHESTRATOR] Starting Training Phase ({epochs} epochs)...", flush=True)
    try:
        res = training_controller.train_model(epochs=epochs)
        if res['status'] == 'success':
            cl_manager.update_metrics({
                "accuracy": res['accuracy'],
                "training_duration_s": res['duration_s']
            })
        return res
    finally:
        cl_manager.end_phase()

@app.post("/workflow/export")
async def workflow_export():
    if cl_manager.is_busy: raise HTTPException(400, "Server is busy")
    cl_manager.start_phase("Export & Eval")
    print("[ORCHESTRATOR] Starting Export & Evaluation Phase...", flush=True)
    try:
        res = training_controller.generate_reference_and_eval()
        if res['status'] == 'success':
            import time
            cl_manager.update_metrics({
                "tar": res['tar'],
                "far": res['far'],
                "eer": res['eer'],
                "total_round_time_s": round(time.time() - cl_manager.start_time, 2)
            })
        return res
    finally:
        cl_manager.end_phase()

# --- Legacy & Support ---
@app.post("/upload-bulk-zip")
async def upload_bulk_zip(file: UploadFile = File(...)):
    import zipfile
    UPLOAD_DIR = "data/students"
    print(f"[UPLOAD] Bulk zip received: {file.filename}", flush=True)
    temp_path = f"data/temp_{file.filename}"
    with open(temp_path, "wb") as f: shutil.copyfileobj(file.file, f)
    with zipfile.ZipFile(temp_path, 'r') as zip_ref: zip_ref.extractall(UPLOAD_DIR)
    os.remove(temp_path)
    print(f"[UPLOAD] Successfully extracted {file.filename} to {UPLOAD_DIR}", flush=True)
    return {"status": "success"}

@app.get("/get-model")
async def get_model():
    print("[GET-MODEL] Client requesting global model...", flush=True)
    if not os.path.exists(MODEL_PATH): 
        print("[GET-MODEL] ERROR: Model file missing!", flush=True)
        raise HTTPException(404, "Model not found")
    return inference_controller.get_model()

@app.get("/get-reference-embeddings")
async def get_reference_embeddings():
    print("[GET-REF] Client requesting reference database...", flush=True)
    if not os.path.exists(REF_PATH): 
        print("[GET-REF] ERROR: Reference file missing!", flush=True)
        raise HTTPException(404, "Reference not found")
    return inference_controller.get_reference()

@app.post("/submit-attendance")
async def submit_attendance(recap: schemas.AttendanceRecapBase, dbs: Session = Depends(db.get_db)):
    label = str(recap.user_id)
    parts = label.split("_", 1)
    nrp = parts[0].strip()
    student = dbs.query(models.UserGlobal).filter(models.UserGlobal.nrp == nrp).first()
    if not student:
        name = parts[1].strip() if len(parts) > 1 else "Unknown"
        student = models.UserGlobal(name=name, nrp=nrp)
        dbs.add(student)
        dbs.commit()
        dbs.refresh(student)
    attendance = models.AttendanceRecap(
        user_id=student.user_id, edge_id=recap.edge_id,
        confidence=recap.confidence, lecture_id=recap.lecture_id,
        timestamp=datetime.utcnow()
    )
    dbs.add(attendance)
    dbs.commit()
    return {"status": "success", "student": student.name}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)