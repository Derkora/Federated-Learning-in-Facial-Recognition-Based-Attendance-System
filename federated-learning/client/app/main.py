import os
import threading
from fastapi import FastAPI, UploadFile, File, Form, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import List
import torch
import numpy as np

from app.db.db import get_db, engine, Base
from app.db.models import UserLocal, Embedding, AttendanceLocal
from app.utils.face_pipeline import face_pipeline
from app.utils.security import EmbeddingEncryptor
from app.utils.classifier import load_backbone, LocalClassifierHead, build_local_model
from app.utils.trainer import LocalTrainer
from app.config import config

from app.client import start_flower_client, get_global_label, sync_users_from_server

CLIENT_ID = os.getenv("HOSTNAME", "client-unknown")
MAX_USERS_CAPACITY = 100

# Init DB Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="FL Edge Client - Local Mode")

@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=start_flower_client, daemon=True)
    thread.start()
    sync_thread = threading.Thread(target=sync_users_from_server, daemon=True)
    sync_thread.start()
    
# Mount Static & Templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

encryptor = EmbeddingEncryptor()
def is_model_ready():
    return os.path.exists("local_backbone.pth")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    user_count = db.query(UserLocal).count()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_count": user_count,
        "model_exists": is_model_ready()
    })

@app.get("/register-page", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/attendance-page", response_class=HTMLResponse)
async def attendance_page(request: Request):
    return templates.TemplateResponse("attendance_live.html", {"request": request})

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "server_api_url": config.get_server_url(),
        "fl_server_address": config.get_fl_server_address()
    })

@app.post("/settings")
async def settings_update(
    request: Request,
    server_api_url: str = Form(...),
    fl_server_address: str = Form(...)
):
    success = config.save_settings(server_api_url, fl_server_address)
    
    msg = "Pengaturan berhasil disimpan. Silahkan Restart Client jika diperlukan." if success else "Gagal menyimpan pengaturan."
    status = "success" if success else "error"

    return templates.TemplateResponse("result.html", {
        "request": request,
        "status": status,
        "message": msg,
        "next_url": "/settings",
        "next_label": "Kembali ke Settings"
    })

@app.post("/api/attendance/live")
async def attendance_live(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """API untuk Live Facial Recognition (JSON Response) - Tanpa Login/Absen ke DB (Hanya Identifikasi)"""
    if not is_model_ready():
        return {"status": "error", "message": "Model not ready"}

    backbone, model_head = build_local_model(num_classes=MAX_USERS_CAPACITY, backbone_path="local_backbone.pth", use_quantized=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(backbone, torch.jit.ScriptModule):
         device = torch.device('cpu') 
         
    backbone.to(device).eval()
    model_head.to(device).eval()
    
    content = await file.read()
    emb_numpy, box, msg = face_pipeline.process_live_frame(content, model=backbone)
    
    if emb_numpy is None:
        return {"status": "no_face", "message": msg}

    # Inferensi
    emb_tensor = torch.tensor(emb_numpy, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model_head(emb_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
        
    class_idx = predicted_class.item()
    softmax_score = confidence.item()
    
    # Identify User
    matched_name = "Unknown"
    
    users = db.query(UserLocal).all()
    matched_user = None
    for user in users:
        lbl_data = get_global_label(user.nrp, client_id=CLIENT_ID)
        if lbl_data and lbl_data.get("label") == class_idx:
            matched_user = user
            matched_name = user.name
            break
            
    if not matched_user:
         return {
             "status": "unknown", 
             "box": box.tolist(), 
             "name": "Unknown (Label Missing)", 
             "confidence": softmax_score
         }

    # Verification (Cosine)
    stored_embeddings = db.query(Embedding).filter(Embedding.user_id == matched_user.user_id).all()
    max_similarity = -1.0
    
    if stored_embeddings:
        for db_emb in stored_embeddings:
            try:
                vec_stored = encryptor.decrypt_embedding(db_emb.encrypted_embedding, db_emb.iv)
                sim = np.dot(emb_numpy, vec_stored)
                if sim > max_similarity: max_similarity = sim
            except: continue
            
    final_score = float(max_similarity) if max_similarity != -1.0 else softmax_score
    THRESHOLD = 0.50
    
    if final_score > THRESHOLD:
        return {
            "status": "match",
            "box": box.tolist(),
            "name": matched_name,
            "confidence": final_score
        }
    else:
         return {
            "status": "unknown",
            "box": box.tolist(),
            "name": "Unknown",
            "confidence": final_score
         }

@app.post("/register")
async def register_user(
    request: Request,
    name: str = Form(...),
    nrp: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    valid_embeddings = []
    errors = []
    # Log untuk debug visual
    registration_logs = []

    for file in files:
        content = await file.read()
        
        # Sekarang mengembalikan [(emb, img), ...], pesan, dan log debug
        data_list, msg, file_debug_logs = face_pipeline.process_with_augmentation(content, filename=file.filename)
        registration_logs.extend(file_debug_logs)
        
        if data_list:
            valid_embeddings.extend(data_list)
        else:
            print(f"[REGISTER ERROR] File {file.filename}: {msg}") 
            errors.append(f"{file.filename}: {msg}")

    if len(valid_embeddings) == 0:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "error",
            "message": "Gagal Registrasi: Wajah tidak terdeteksi di semua foto (atau setelah augmentasi).",
            "details": {"Error Log": errors}, 
            "registration_logs": registration_logs, # Kirim log debug
            "next_url": "/register-page",
            "next_label": "Coba Lagi"
        })
    
    lbl_data = get_global_label(nrp, name, client_id=CLIENT_ID) 
    
    if lbl_data is None:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "error",
            "message": "Gagal menghubungi Server Pusat untuk sinkronisasi Label atau Server Penuh.",
            "registration_logs": registration_logs, # Kirim log debug
            "next_url": "/register-page",
            "next_label": "Coba Lagi Nanti"
        })
        
    # PERUBAHAN: Lakukan pengecekan pembatasan perangkat
    if lbl_data.get("registered_edge_id") and lbl_data.get("registered_edge_id") != CLIENT_ID:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "error",
            "message": f"Registrasi Gagal: NRP {nrp} sudah terdaftar di perangkat lain ({lbl_data.get('registered_edge_id')}).",
            "registration_logs": registration_logs, # Kirim log debug
            "next_url": "/register-page",
            "next_label": "Coba Lagi"
        })

    existing_user = db.query(UserLocal).filter(UserLocal.nrp == nrp).first()
    
    if existing_user:
        new_user = existing_user
        print(f"[REGISTER] User {name} sudah ada, menambahkan data wajah baru.")
    else:
        new_user = UserLocal(name=name, nrp=nrp)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

    for vec, img_np in valid_embeddings:
        # Enkripsi Embedding (AES-256)
        enc_data, iv, salt = encryptor.encrypt_embedding(vec)
        
        # Enkripsi Image (AES-256) - Reuse encrypt_embedding karena support numpy
        enc_img, img_iv, _ = encryptor.encrypt_embedding(img_np)
        
        # GABUNGKAN IV + Image Encrypted agar bisa didekripsi tanpa kolom DB baru
        # IV adalah 16 bytes pertama
        final_image_blob = img_iv + enc_img
        
        db_emb = Embedding(
            user_id=new_user.user_id,
            encrypted_embedding=enc_data,
            encrypted_image=final_image_blob, # Simpan IV + Ciphertext
            iv=iv, # Ini IV untuk embedding, biarkan saja
            salt=salt
        )
        db.add(db_emb)
            
    db.commit()
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "status": "registered",
        "message": f"Registrasi Berhasil untuk {name}",
        "details": {
            "Global Label ID": lbl_data.get("label"),
            "Total Sampel Wajah": len(valid_embeddings), 
            "NRP": nrp,
            "Registered on": CLIENT_ID
        },
        "registration_logs": registration_logs, # Kirim log visual registrasi
        "next_url": "/register-page",
        "next_label": "Daftar Lagi"
    })

@app.post("/train")
async def trigger_training(request: Request, db: Session = Depends(get_db)):
    # LocalTrainer sudah diubah untuk melatih Backbone + Head
    trainer = LocalTrainer(db)
    result = trainer.train_local()
    
    status = "success" if result.get("status") == "success" else "error"
    msg = "Model Backbone Lokal Berhasil Dilatih!" if status == "success" else f"Gagal: {result}"
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "status": status,
        "message": msg,
        "details": result,
        "next_url": "/",
        "next_label": "Kembali ke Dashboard"
    })

@app.post("/attendance")
async def attendance(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not is_model_ready():
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "error",
            "message": "Model belum dilatih! Lakukan training di Dashboard dulu.",
            "next_url": "/",
            "next_label": "Ke Dashboard"
        })

    # Load Backbone dan buat Head Lokal (Head hanya untuk inferensi klasifikasi)
    backbone, model_head = build_local_model(num_classes=MAX_USERS_CAPACITY, backbone_path="local_backbone.pth")
    backbone.eval()
    model_head.eval()
    
    # Proses Image
    content = await file.read()
    emb_numpy, drawn_img_b64, msg = face_pipeline.process_image(content, model=backbone)
    
    attendance_log = {"Drawn Image B64": drawn_img_b64, "Detection Status": msg}
    
    if emb_numpy is None:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "error",
            "message": f"Wajah tidak terdeteksi: {msg}",
            "details": attendance_log, # Kirim log deteksi BB
            "next_url": "/attendance-page",
            "next_label": "Coba Lagi"
        })

    # Inferensi (Head Classifier)
    # emb_numpy sudah berupa embedding (128-dim), cukup ubah ke tensor
    emb_tensor = torch.tensor(emb_numpy, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        # Masukkan langsung embedding ke Head (karena Backbone sudah dijalankan di face_pipeline)
        outputs = model_head(emb_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
        
    class_idx = predicted_class.item()
    softmax_score = confidence.item()
    
    # Ambil semua user lokal
    users = db.query(UserLocal).all()
    matched_user = None
    
    # Cari user lokal yang label global-nya cocok dengan hasil klasifikasi
    for user in users:
        lbl_data = get_global_label(user.nrp, client_id=CLIENT_ID) # Ambil label global lagi
        if lbl_data and lbl_data.get("label") == class_idx:
            matched_user = user
            attendance_log["Predicted Label"] = class_idx
            attendance_log["Predicted User NRP"] = user.nrp
            attendance_log["Predicted User Name"] = user.name
            break
    
    if not matched_user:
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "unknown",
            "message": "Wajah terdeteksi, tapi data user (label) belum tersinkronisasi.",
            "details": attendance_log, # Kirim log deteksi BB
            "next_url": "/attendance-page",
            "next_label": "Coba Lagi"
        })
    
    # Metrik final untuk absensi (TAR/FAR/EER)
    stored_embeddings = db.query(Embedding).filter(Embedding.user_id == matched_user.user_id).all()
    
    max_similarity = -1.0
    
    if not stored_embeddings:
        print("[ATTENDANCE] User ditemukan tapi tidak punya sampel wajah untuk verifikasi.")
        final_score = softmax_score
    else:
        for db_emb in stored_embeddings:
            try:
                # Decrypt vektor dari DB
                vec_stored = encryptor.decrypt_embedding(db_emb.encrypted_embedding, db_emb.iv)
                
                # Hitung Cosine Similarity 
                sim = np.dot(emb_numpy, vec_stored)
                
                if sim > max_similarity:
                    max_similarity = sim
            except Exception as e:
                print(f"[VERIFY ERROR] Gagal decrypt sample: {e}")
                continue
        
        final_score = float(max_similarity)
    
    THRESHOLD = 0.50 
    
    print(f"[ATTENDANCE] Kandidat: {matched_user.name} | Softmax: {softmax_score:.2f} | Cosine Sim: {final_score:.2f}")

    attendance_log["Softmax Score"] = f"{softmax_score:.2f}"
    attendance_log["Cosine Similarity"] = f"{final_score:.2f}"
    attendance_log["Threshold"] = THRESHOLD
    
    if final_score > THRESHOLD:
        # REKAM LOG
        log = AttendanceLocal(
            user_id=matched_user.user_id,
            confidence=final_score,
            sent_to_server=False
        )
        db.add(log)
        db.commit()
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "success",
            "message": f"Halo, {matched_user.name}!",
            "details": attendance_log, # Kirim log deteksi BB
            "next_url": "/attendance-page",
            "next_label": "Absen Lagi"
        })
    else:
        msg = "Wajah tidak terverifikasi."
        if final_score > 0.3:
            msg = "Wajah agak mirip, tapi kurang meyakinkan. Coba lepas kacamata/masker."
            
        return templates.TemplateResponse("result.html", {
            "request": request,
            "status": "unknown",
            "message": f"Maaf, {msg}",
            "details": attendance_log, # Kirim log deteksi BB
            "next_url": "/attendance-page",
            "next_label": "Coba Lagi"
        })