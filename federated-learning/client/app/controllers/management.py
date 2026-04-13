from fastapi import APIRouter, Depends, BackgroundTasks
from app.manager import fl_manager
import time
import os

router = APIRouter(prefix="/api/management", tags=["management"])

@router.get("/status")
async def get_status():
    return {
        "client_id": fl_manager.client_id,
        "status": fl_manager.fl_status,
        "phase": getattr(fl_manager, 'current_phase', 'idle'),
        "model_version": fl_manager.model_version,
        "camera_active": fl_manager.is_camera_running,
        "is_training": fl_manager.is_training,
        "uptime": int(time.time() - getattr(fl_manager, 'start_time', time.time()))
    }

@router.get("/config")
async def get_config():
    return {
        "camera_index": fl_manager.camera_index,
        "fl_server": fl_manager.fl_server_address,
        "api_url": fl_manager.server_api_url,
        "device": str(fl_manager.device)
    }

@router.post("/discovery")
async def request_discovery(background_tasks: BackgroundTasks):
    background_tasks.add_task(fl_manager.run_discovery_phase)
    return {"status": "success"}

@router.post("/preprocess")
async def request_preprocess(background_tasks: BackgroundTasks):
    background_tasks.add_task(fl_manager.run_preprocess_phase)
    return {"status": "success"}

@router.post("/registry")
async def request_registry(background_tasks: BackgroundTasks):
    background_tasks.add_task(fl_manager.run_registry_phase)
    return {"status": "success"}
