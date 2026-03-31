import os
from fastapi import HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from datetime import datetime
from ..db import models, schemas

MODEL_DIR = "app/model"
MODEL_PATH = f"{MODEL_DIR}/global_model.pth"
REF_PATH = f"{MODEL_DIR}/reference_embeddings.pth"

class InferenceController:
    # Kontroler untuk distribusi aset model dan manajemen laporan presensi.
    
    @staticmethod
    def get_model():
        # Mengirimkan berkas bobot model global ke terminal yang meminta.
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=404, detail="Berkas model tidak ditemukan")
        return FileResponse(MODEL_PATH, media_type='application/octet-stream', filename="global_model.pth")

    @staticmethod
    def get_reference():
        # Mengirimkan berkas basis data referensi wajah ke terminal yang meminta.
        if not os.path.exists(REF_PATH):
            raise HTTPException(status_code=404, detail="Berkas referensi tidak ditemukan")
        return FileResponse(REF_PATH, media_type='application/octet-stream', filename="reference_embeddings.pth")

    @staticmethod
    def submit_attendance(recap: schemas.AttendanceRecapBase, dbs: Session):
        # Mencatat kehadiran mahasiswa ke dalam database hasil inferensi edge.
        attendance = models.AttendanceRecap(
            user_id=recap.user_id,
            edge_id=recap.edge_id,
            confidence=recap.confidence,
            lecture_id=recap.lecture_id,
            timestamp=datetime.utcnow()
        )
        dbs.add(attendance)
        dbs.commit()
        return {"status": "success", "message": "Absensi berhasil dicatat secara terpusat."}

inference_controller = InferenceController()
