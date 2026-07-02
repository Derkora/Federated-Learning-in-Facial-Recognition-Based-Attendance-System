import os
from fastapi import HTTPException, Response
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from app.db import models, schemas
from app.config import MODEL_PATH, REF_PATH
from app.server_manager_instance import cl_manager

class InferenceController:
    # Kontroler untuk distribusi aset model dan manajemen laporan presensi.
    
    @staticmethod
    def get_model(version: str = None):
        # Mengirimkan berkas bobot model global ke terminal yang meminta.
        path = MODEL_PATH
        if version == "active":
            pass # path tetap MODEL_PATH
        elif version and version != "v0" and version != "0":
            # format versi: v1_10
            from app.config import MODEL_DIR
            versioned_path = os.path.join(MODEL_DIR, f"global_model_{version}.pth")
            if os.path.exists(versioned_path):
                path = versioned_path
            else:
                raise HTTPException(status_code=404, detail=f"Berkas model versi {version} tidak ditemukan")
        elif version == "v0" or version == "0":
            from app.config import PRETRAINED_PATH
            path = PRETRAINED_PATH
        else:
            # Jika versi masih 0, berarti belum pernah ditraining (Model Tidak Tersedia)
            if cl_manager.model_version == 0:
                raise HTTPException(status_code=403, detail="MODEL NOT AVAILABLE: Versi masih v0 (Belum ditraining)")
            
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Berkas model tidak ditemukan")
        return FileResponse(path, media_type='application/octet-stream', filename=os.path.basename(path))

    @staticmethod
    def get_reference(version: str = None):
        # Mengirimkan berkas basis data referensi wajah ke terminal yang meminta.
        path = REF_PATH
        if version == "active":
            pass # path tetap REF_PATH
        elif version and version != "v0" and version != "0":
            from app.config import MODEL_DIR
            versioned_path = os.path.join(MODEL_DIR, f"reference_embeddings_{version}.pth")
            if os.path.exists(versioned_path):
                path = versioned_path
            else:
                raise HTTPException(status_code=404, detail=f"Berkas referensi versi {version} tidak ditemukan")
        elif version == "v0" or version == "0":
            # Untuk v0, kita buat file referensi kosong
            import io
            import torch
            buf = io.BytesIO()
            torch.save({}, buf)
            return Response(content=buf.getvalue(), media_type='application/octet-stream', headers={"Content-Disposition": "attachment; filename=reference_embeddings.pth"})
            
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Berkas referensi tidak ditemukan")
        return FileResponse(path, media_type='application/octet-stream', filename=os.path.basename(path))

    @staticmethod
    def submit_attendance(recap: schemas.AttendanceRecapBase, dbs: Session):
        # Mencatat kehadiran mahasiswa ke dalam database hasil inferensi edge.
        attendance = models.AttendanceRecap(
            user_id=recap.user_id,
            edge_id=recap.edge_id,
            confidence=recap.confidence,
            lecture_id=recap.lecture_id,
            timestamp=datetime.now(timezone(timedelta(hours=7)))
        )
        dbs.add(attendance)
        dbs.commit()
        return {"status": "success", "message": "Absensi berhasil dicatat secara terpusat."}

inference_controller = InferenceController()
