import os
import time
from datetime import datetime, timedelta, timezone
import requests

from app.utils.logging import get_logger

# Sinkronisasi data ke server pusat
# Fungsi ini dijalankan di background untuk mengirim data absensi yang baru dicatat
# ke server Federated Learning agar bisa direkap secara global.
def sync_record_to_server(user_id, name, confidence, client_id):
    logger = get_logger()
    server_url = os.getenv("SERVER_API_URL", "http://server-fl:8080")
    payload = [{
        "user_id": user_id,
        "name": name,
        "client_id": client_id,
        "timestamp": datetime.now(timezone(timedelta(hours=7))).strftime('%Y-%m-%dT%H:%M:%S'),
        "confidence": confidence
    }]
    
    try:
        # Mengirim data ke endpoint sync server
        response = requests.post(f"{server_url}/api/attendance/sync", json=payload, timeout=5)
        if response.status_code == 200:
            logger.success(f"Berhasil sinkronisasi absensi {user_id} ke server.")
        else:
            logger.error(f"Gagal sinkronisasi ke server (Status: {response.status_code})")
            
    except Exception as e:
        logger.error(f"Gagal menghubungi server untuk sinkronisasi: {e}")
