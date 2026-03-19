import os
from fastapi import HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from datetime import datetime
from app.db import models, schemas

MODEL_DIR = "app/model"
MODEL_PATH = f"{MODEL_DIR}/global_model.pth"
REF_PATH = f"{MODEL_DIR}/reference_embeddings.pth"

class InferenceController:
    @staticmethod
    def get_model():
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=404, detail="Model file not found")
        return FileResponse(MODEL_PATH, media_type='application/octet-stream', filename="global_model.pth")

    @staticmethod
    def get_reference():
        if not os.path.exists(REF_PATH):
            raise HTTPException(status_code=404, detail="Reference file not found")
        return FileResponse(REF_PATH, media_type='application/octet-stream', filename="reference_embeddings.pth")

    @staticmethod
    def submit_attendance(recap: schemas.AttendanceRecapBase, dbs: Session):
        attendance = models.AttendanceRecap(
            user_id=recap.user_id,
            edge_id=recap.edge_id,
            confidence=recap.confidence,
            lecture_id=recap.lecture_id,
            timestamp=datetime.utcnow()
        )
        dbs.add(attendance)
        dbs.commit()
        return {"status": "success", "message": "Absensi tercatat!"}

inference_controller = InferenceController()
