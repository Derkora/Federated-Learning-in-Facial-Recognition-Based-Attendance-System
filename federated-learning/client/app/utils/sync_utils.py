import os
import time
import json
import threading
from datetime import datetime, timedelta, timezone
import requests

from app.utils.logging import get_logger

# Konfigurasi endpoint API
api_attendance_sync = "/api/attendance/sync"
api_logs_inference = "/api/logs/inference"

QUEUE_FILE = "/app/data/offline_attendance_queue.json"
queue_lock = threading.Lock()

INFERENCE_QUEUE_FILE = "/app/data/offline_inference_logs.json"
inference_queue_lock = threading.Lock()

def save_to_offline_queue(payload_item):
    """Menyimpan data absensi yang gagal dikirim ke file antrean lokal (JSON)."""
    logger = get_logger()
    os.makedirs(os.path.dirname(QUEUE_FILE), exist_ok=True)
    with queue_lock:
        queue = []
        if os.path.exists(QUEUE_FILE):
            try:
                with open(QUEUE_FILE, "r") as f:
                    queue = json.load(f)
            except Exception:
                queue = []
        
        queue.append(payload_item)
        
        try:
            with open(QUEUE_FILE, "w") as f:
                json.dump(queue, f, indent=4)
            logger.info(f"Absensi untuk {payload_item['user_id']} disimpan ke antrean offline lokal (Total: {len(queue)} item).")
        except Exception as e:
            logger.error(f"Gagal menulis ke file antrean offline: {e}")

def process_offline_queue():
    """Mengirim ulang semua antrean absensi offline yang tertunda ke server pusat."""
    if not os.path.exists(QUEUE_FILE):
        return
        
    logger = get_logger()
    server_url = os.getenv("SERVER_API_URL", "http://server-fl:8080")
    
    with queue_lock:
        try:
            with open(QUEUE_FILE, "r") as f:
                queue = json.load(f)
        except Exception:
            return
            
        if not queue:
            return
            
        logger.info(f"Mencoba mengirim {len(queue)} rekaman absensi offline tertunda...")
        
        try:
            response = requests.post(f"{server_url}{api_attendance_sync}", json=queue, timeout=10)
            if response.status_code == 200:
                logger.success(f"Berhasil sinkronisasi {len(queue)} rekaman offline ke server!")
                # Kosongkan antrean
                with open(QUEUE_FILE, "w") as f:
                    json.dump([], f)
            else:
                logger.error(f"Gagal sinkronisasi antrean offline ke server (Status: {response.status_code})")
        except Exception as e:
            # Server masih mati, biarkan antrean tetap tersimpan
            logger.debug(f"Sinkronisasi antrean offline tertunda (server masih offline): {e}")

def process_offline_inference_logs():
    """Mengirim ulang semua log inferensi offline yang tertunda ke server pusat."""
    if not os.path.exists(INFERENCE_QUEUE_FILE):
        return
        
    logger = get_logger()
    server_url = os.getenv("SERVER_API_URL", "http://server-fl:8080")
    
    with inference_queue_lock:
        try:
            with open(INFERENCE_QUEUE_FILE, "r") as f:
                queue_data = json.load(f)
        except Exception:
            return
            
        if not queue_data:
            return
            
        logger.info(f"Mencoba mengirim {len(queue_data)} log inferensi offline tertunda...")
        
        succeeded = []
        for item in queue_data:
            try:
                response = requests.post(f"{server_url}{api_logs_inference}", json=item, timeout=3)
                if response.status_code == 200:
                    succeeded.append(item)
                else:
                    break
            except Exception:
                break
                
        if succeeded:
            remaining = [x for x in queue_data if x not in succeeded]
            try:
                with open(INFERENCE_QUEUE_FILE, "w") as f:
                    json.dump(remaining, f, indent=4)
                logger.success(f"Berhasil sinkronisasi {len(succeeded)} log inferensi offline tertunda!")
            except Exception as e:
                logger.error(f"Gagal memperbarui berkas log inferensi offline: {e}")

# Sinkronisasi data ke server pusat
# Fungsi ini dijalankan di background untuk mengirim data absensi yang baru dicatat
# ke server Federated Learning agar bisa direkap secara global.
def sync_record_to_server(user_id, name, confidence, client_id, latency=0):
    logger = get_logger()
    server_url = os.getenv("SERVER_API_URL", "http://server-fl:8080")
    payload_item = {
        "user_id": user_id,
        "name": name,
        "client_id": client_id,
        "timestamp": datetime.now(timezone(timedelta(hours=7))).strftime('%Y-%m-%dT%H:%M:%S'),
        "confidence": confidence,
        "latency_ms": latency
    }
    
    try:
        process_offline_queue()
    except Exception:
        pass
        
    try:
        # Mengirim data ke endpoint sync server
        response = requests.post(f"{server_url}{api_attendance_sync}", json=[payload_item], timeout=5)
        if response.status_code == 200:
            logger.success(f"Berhasil sinkronisasi absensi {user_id} ke server.")
        else:
            logger.error(f"Gagal sinkronisasi ke server (Status: {response.status_code}). Menyimpan ke antrean offline.")
            save_to_offline_queue(payload_item)
            
    except Exception as e:
        logger.error(f"Gagal menghubungi server untuk sinkronisasi. Menyimpan ke antrean offline: {e}")
        save_to_offline_queue(payload_item)
