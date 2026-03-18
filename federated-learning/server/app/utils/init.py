# server/app/utils/init_head.py
import torch
import torch.nn as nn
import pickle
import io
import sys
import os

from .mobilefacenet import MobileFaceNet

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from app.db.db import SessionLocal
from app.db.models import ModelVersion

def create_initial_head():
    print("[INIT] Membuat inisialisasi MobileFaceNet Backbone (Global Model)")
    
    # Inisialisasi MobileFaceNet (Backbone)
    # Model MobileFaceNet sudah diimpor secara lokal
    model = MobileFaceNet(embedding_size=128) 
    
    # Ambil state_dict (bobot & bias) dari MobileFaceNet
    weights = model.state_dict()
    
    # Serialisasi ke bytes
    buffer = io.BytesIO()
    torch.save(weights, buffer)
    blob = buffer.getvalue()
    
    # Simpan ke Database Server
    db = SessionLocal()
    try:
        # Cek apakah sudah ada versi awal
        existing = db.query(ModelVersion).first()
        if existing:
            print("[INIT] Model versi awal sudah ada di DB. Skip.")
            return

        new_version = ModelVersion(
            head_blob=blob, 
            notes="Initial Random MobileFaceNet Backbone (Version 0)"
        )
        db.add(new_version)
        db.commit()
        print(f"[INIT] Berhasil menyimpan Backbone Version 0 ({len(blob)} bytes) ke Database.")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan ke DB: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_initial_head()