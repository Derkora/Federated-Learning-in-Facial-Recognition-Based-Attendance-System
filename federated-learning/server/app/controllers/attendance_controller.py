from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

# Simulasi database sederhana
attendance_logs = []
registered_users = {}

class RegisterUser(BaseModel):
    user_id: str
    name: str
    assigned_edge: str   


@router.post("/register")
async def register_face(user: RegisterUser):
    registered_users[user.user_id] = user.dict()
    return {"message": "User registered", "user": user}


@router.post("/attendance")
async def attendance(user_id: str = Form(...), timestamp: str = Form(...)):
    if user_id not in registered_users:
        return {"error": "User not registered"}

    log = {
        "user_id": user_id,
        "timestamp": timestamp,
        "name": registered_users[user_id]["name"]
    }
    attendance_logs.append(log)
    return {"message": "Attendance recorded", "log": log}


@router.get("/logs")
def get_logs():
    return {"attendance": attendance_logs}
