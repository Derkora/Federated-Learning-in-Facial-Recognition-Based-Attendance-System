import os
import shutil
from fastapi import HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from datetime import datetime
from app.db import models, schemas, crud

UPLOAD_DIR = "data/students"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class StudentController:
    @staticmethod
    def register_student(name: str, nrp: str, dbs: Session):
        existing = dbs.query(models.UserGlobal).filter(models.UserGlobal.nrp == nrp).first()
        if existing:
            return {"status": "already_exists", "user_id": existing.user_id}
        
        student = models.UserGlobal(name=name, nrp=nrp)
        dbs.add(student)
        dbs.commit()
        dbs.refresh(student)
        return {"status": "success", "user_id": student.user_id}

    @staticmethod
    async def upload_photo(user_id: int, file: UploadFile, dbs: Session):
        student = dbs.query(models.UserGlobal).filter(models.UserGlobal.user_id == user_id).first()
        if not student:
            raise HTTPException(status_code=404, detail="Student not found")
        
        student_dir = os.path.join(UPLOAD_DIR, student.nrp)
        os.makedirs(student_dir, exist_ok=True)
        
        file_path = os.path.join(student_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        student.photo_path = student_dir
        dbs.commit()
        return {"status": "success", "message": f"Foto {file.filename} tersimpan di {student_dir}"}

student_controller = StudentController()
