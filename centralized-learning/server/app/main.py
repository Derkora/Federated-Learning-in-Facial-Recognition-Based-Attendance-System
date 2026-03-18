# app/main.py
from fastapi import FastAPI, Request, File, UploadFile, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import torch
import pickle
import numpy as np
import os
import shutil

from .database import engine, Base, get_db
from .models import User, FaceEmbedding
from .face_engine import engine as face_engine
from .trainer import train_central_model, SimpleClassifier

# Init Database
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Mount Static & Templates
# Pastikan folder 'app/static' ada jika ingin pakai CSS
# app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# --- HALAMAN UI (GET) ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/recognize", response_class=HTMLResponse)
async def recognize_page(request: Request):
    return templates.TemplateResponse("recognize.html", {"request": request})

# --- API LOGIC (POST) ---

@app.post("/api/register")
async def register_user(
    name: str = Form(...),
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    # 1. Cek atau Buat User Baru
    user = db.query(User).filter(User.name == name).first()
    if not user:
        user = User(name=name)
        db.add(user)
        db.commit()
        db.refresh(user)
    
    success_count = 0
    errors = []

    # 2. Proses Setiap Foto yang Diupload
    for file in files:
        content = await file.read()
        
        # Ekstrak Embedding pakai Pretrained MobileFaceNet
        emb, msg = face_engine.get_embedding(content)
        
        if emb is not None:
            # Simpan embedding (numpy array) ke Database
            db_emb = FaceEmbedding(user_id=user.id, embedding=emb)
            db.add(db_emb)
            success_count += 1
        else:
            errors.append(f"{file.filename}: {msg}")
            
    db.commit()
    
    # Optional: Auto-train setiap kali ada data baru (untuk kemudahan)
    train_res = train_central_model(db)
    
    return JSONResponse({
        "status": "success",
        "message": f"Berhasil mendaftarkan {success_count} foto untuk user '{name}'",
        "train_info": train_res,
        "errors": errors
    })

@app.post("/api/train")
async def manual_train(db: Session = Depends(get_db)):
    result = train_central_model(db)
    return JSONResponse(result)

@app.post("/api/recognize")
async def recognize_face(file: UploadFile = File(...)):
    # Cek apakah model classifier sudah dilatih
    if not os.path.exists("models/classifier_head.pth"):
        return JSONResponse({"status": "error", "message": "Model belum dilatih! Silakan registrasi data dulu."})
    
    # 1. Proses Gambar -> Embedding
    content = await file.read()
    emb, msg = face_engine.get_embedding(content)
    
    if emb is None:
        return JSONResponse({"status": "error", "message": f"Wajah tidak terdeteksi: {msg}"})
    
    # 2. Load Mapping (Index -> Nama User)
    if not os.path.exists("models/class_mapping.pkl"):
        return JSONResponse({"status": "error", "message": "Mapping class tidak ditemukan."})

    with open("models/class_mapping.pkl", "rb") as f:
        idx_to_name = pickle.load(f)
    
    # 3. Load Model Classifier Head
    num_classes = len(idx_to_name)
    model = SimpleClassifier(num_classes=num_classes)
    try:
        model.load_state_dict(torch.load("models/classifier_head.pth"))
    except:
        return JSONResponse({"status": "error", "message": "Struktur model tidak cocok, coba train ulang."})
        
    model.eval()
    
    # 4. Prediksi
    emb_tensor = torch.tensor(np.array([emb]), dtype=torch.float32)
    with torch.no_grad():
        outputs = model(emb_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
        
    idx = pred_idx.item()
    conf_score = confidence.item()
    
    predicted_name = idx_to_name.get(idx, "Unknown")
    
    # Thresholding (Misal 70%)
    threshold = 0.70
    if conf_score > threshold:
        return JSONResponse({
            "status": "success", 
            "name": predicted_name, 
            "confidence": f"{conf_score:.2%}"
        })
    else:
        return JSONResponse({
            "status": "unknown", 
            "name": "Tidak Dikenali", 
            "confidence": f"{conf_score:.2%}",
            "closest_match": predicted_name
        })