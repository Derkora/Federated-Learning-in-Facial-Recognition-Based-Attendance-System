from fastapi import FastAPI, Depends, Request, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from datetime import datetime
import os
import uvicorn

from app.db import models, db, schemas, crud
from app.controllers.student import student_controller
from app.controllers.training import training_controller
from app.controllers.inference import inference_controller
from app.server_manager_instance import cl_manager

# Inisialisasi Database
models.Base.metadata.create_all(bind=db.engine)

app = FastAPI(title="Centralized Attendance Server")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    status = cl_manager.get_status()
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "title": "Dashboard",
        "status": status
    })

@app.get("/records", response_class=HTMLResponse)
async def records(request: Request, dbs: Session = Depends(db.get_db)):
    attendances = dbs.query(models.AttendanceRecap).order_by(models.AttendanceRecap.timestamp.desc()).all()
    return templates.TemplateResponse("records.html", {"request": request, "title": "Records", "attendances": attendances})

@app.get("/ping")
async def health_check():
    return {"status": "online", "message": "Server Pusat Standby!"}

@app.post("/register-client", response_model=schemas.ClientResponse)
async def register_client(client_data: schemas.ClientBase, dbs: Session = Depends(db.get_db)):
    existing = dbs.query(models.Client).filter(models.Client.edge_id == client_data.id).first()
    if existing:
        existing.last_seen = datetime.utcnow()
        existing.ip_address = client_data.ip_address
        existing.status = client_data.cl_status
        dbs.commit()
        dbs.refresh(existing)
        return existing
    return crud.register_client(dbs, client_data)

@app.post("/register-student")
async def register_student(name: str, dbs: Session = Depends(db.get_db)):
    return student_controller.register_student(name, name, dbs)

@app.post("/upload-photo/{user_id}")
async def upload_photo(user_id: int, file: UploadFile = File(...), dbs: Session = Depends(db.get_db)):
    return await student_controller.upload_photo(user_id, file, dbs)

@app.post("/upload-bulk-zip")
async def upload_bulk_zip(file: UploadFile = File(...)):
    import shutil
    import zipfile
    UPLOAD_DIR = "data/students"
    temp_path = f"data/temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    with zipfile.ZipFile(temp_path, 'r') as zip_ref:
        zip_ref.extractall(UPLOAD_DIR)
    os.remove(temp_path)
    return {"status": "success"}

@app.post("/train")
async def train(dbs: Session = Depends(db.get_db)):
    # Trigger all clients first
    clients = dbs.query(models.Client).all()
    import requests as req_ext
    for cl in clients:
        try:
            base_url = cl.ip_address if ":" in cl.ip_address else f"{cl.ip_address}:8080"
            if not base_url.startswith("http"): base_url = f"http://{base_url}"
            req_ext.post(f"{base_url}/api/request-data", timeout=2)
        except: pass
    
    # Use manager to orchestrate
    return cl_manager.start_training_flow(training_controller)

@app.get("/get-model")
async def get_model():
    return inference_controller.get_model()

@app.get("/get-reference-embeddings")
async def get_reference_embeddings():
    return inference_controller.get_reference()

@app.post("/submit-attendance")
async def submit_attendance(recap: schemas.AttendanceRecapBase, dbs: Session = Depends(db.get_db)):
    # recap.user_id is coming as a string (nrp_nama) from the client
    # but the DB expects an Integer ID.
    label = str(recap.user_id)
    parts = label.split("_", 1)
    nrp = parts[0].strip()
    
    # Find or Create Student
    student = dbs.query(models.UserGlobal).filter(models.UserGlobal.nrp == nrp).first()
    if not student:
        # Auto-register if not exists
        name = parts[1].strip() if len(parts) > 1 else "Unknown"
        student = models.UserGlobal(name=name, nrp=nrp)
        dbs.add(student)
        dbs.commit()
        dbs.refresh(student)
    
    # Record Attendance
    attendance = models.AttendanceRecap(
        user_id=student.user_id,
        edge_id=recap.edge_id,
        confidence=recap.confidence,
        lecture_id=recap.lecture_id,
        timestamp=datetime.utcnow()
    )
    dbs.add(attendance)
    dbs.commit()
    return {"status": "success", "student": student.name}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)