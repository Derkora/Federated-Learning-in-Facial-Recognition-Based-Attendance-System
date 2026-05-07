import os
import shutil
from fastapi import HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from datetime import datetime
from ..db import models, schemas, crud

UPLOAD_DIR = "data/students"
os.makedirs(UPLOAD_DIR, exist_ok=True)

from ..utils.logging import get_logger

class StudentController:
    # Kontroler untuk manajemen data mahasiswa dan unggah foto.
    
    def __init__(self):
        self.logger = get_logger()
    
    def register_student(self, name: str, nrp: str, dbs: Session):
        # Mendaftarkan mahasiswa baru ke database global jika belum ada.
        try:
            existing = dbs.query(models.UserGlobal).filter(models.UserGlobal.nrp == nrp).first()
            if existing:
                return {"status": "already_exists", "user_id": existing.user_id}
            
            student = models.UserGlobal(name=name, nrp=nrp)
            dbs.add(student)
            dbs.commit()
            dbs.refresh(student)
            self.logger.success(f"Mahasiswa Terdaftar: {name} ({nrp})")
            return {"status": "success", "user_id": student.user_id}
        except Exception as e:
            dbs.rollback()
            self.logger.error(f"Gagal mendaftarkan mahasiswa {nrp}: {e}")
            return {"status": "error", "message": str(e)}

    async def upload_photo(self, user_id: int, file: UploadFile, dbs: Session):
        # Menyimpan berkas foto mahasiswa ke direktori penyimpanan server.
        try:
            student = dbs.query(models.UserGlobal).filter(models.UserGlobal.user_id == user_id).first()
            if not student:
                self.logger.warn(f"Upload Gagal: User ID {user_id} tidak ditemukan.")
                raise HTTPException(status_code=404, detail="Mahasiswa tidak ditemukan")
            
            student_dir = os.path.join(UPLOAD_DIR, student.nrp)
            os.makedirs(student_dir, exist_ok=True)
            
            file_path = os.path.join(student_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            student.photo_path = student_dir
            dbs.commit()
            self.logger.success(f"Foto Disimpan: {student.nrp}/{file.filename}")
            return {"status": "success", "message": f"Foto {file.filename} berhasil disimpan."}
        except Exception as e:
            dbs.rollback()
            self.logger.error(f"Gagal upload foto user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

student_controller = StudentController()

