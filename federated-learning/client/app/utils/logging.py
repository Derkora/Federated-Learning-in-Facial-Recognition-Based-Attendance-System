import os
import time
from datetime import datetime, timezone, timedelta

class Logger:
    """
    Utility Logging Terpusat untuk Riset Federated & Centralized Learning.
    Mendukung penyimpanan memori (untuk Dashboard UI) dan penyimpanan file persisten.
    """
    def __init__(self, log_path, max_memory_logs=1000, tag="SYSTEM"):
        self.log_path = log_path
        self.max_memory_logs = max_memory_logs
        self.memory_logs = []
        self.tag = tag
        
        # Pastikan direktori log tersedia
        log_dir = os.path.dirname(self.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        # Muat history terakhir dari file jika ada (agar UI tidak kosong saat restart)
        self._load_history()

    def _load_history(self):
        # Muat dari .old dulu jika ada, lalu file utama
        old_path = self.log_path + ".old"
        all_lines = []
        
        if os.path.exists(old_path):
            try:
                with open(old_path, "r") as f:
                    all_lines.extend(f.readlines())
            except: pass
            
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, "r") as f:
                    all_lines.extend(f.readlines())
            except: pass
            
        # Ambil baris terakhir yang sesuai limit memori
        for line in all_lines[-self.max_memory_logs:]:
            self.memory_logs.append(line.strip())

    def info(self, msg):
        self._log("INFO", msg)

    def success(self, msg):
        self._log("SUCCESS", msg)

    def warn(self, msg):
        self._log("WARN", msg)

    def error(self, msg):
        self._log("ERROR", msg)

    def train(self, msg):
        self._log("TRAIN", msg)

    def _log(self, level, msg):
        # Gunakan WIB (UTC+7) agar sesuai dengan waktu Indonesia
        tz_wib = timezone(timedelta(hours=7))
        now = datetime.now(tz_wib)
        ts_display = now.strftime('%H:%M:%S')
        ts_file = now.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format untuk tampilan Dashboard (Ringkas)
        log_entry = f"[{ts_display}] [{level}] {msg}"
        # Format untuk File Persisten (Lengkap)
        file_entry = f"[{ts_file}] [{self.tag}] [{level}] {msg}\n"
        
        # 1. Update Memori (FIFO)
        self.memory_logs.append(log_entry)
        if len(self.memory_logs) > self.max_memory_logs:
            self.memory_logs.pop(0)
            
        # 2. Simpan ke File dengan Log Rotation (Max 5MB)
        try:
            if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 5 * 1024 * 1024:
                old_path = self.log_path + ".old"
                if os.path.exists(old_path):
                    os.remove(old_path)
                os.rename(self.log_path, old_path)
            
            with open(self.log_path, "a") as f:
                f.write(file_entry)
        except:
            pass
            
        # 3. Output ke Console (Docker Logs)
        print(f"{self.tag} LOG: {log_entry}", flush=True)


    def get_logs(self):
        """Mengambil daftar log untuk Dashboard UI."""
        return self.memory_logs

# Instance global untuk akses mudah dari modul lain
_global_logger = None

def init_logger(log_path, max_memory_logs=1000, tag="SYSTEM"):
    global _global_logger
    _global_logger = Logger(log_path, max_memory_logs, tag)
    return _global_logger

def get_logger():
    global _global_logger
    if _global_logger is None:
        # Fallback jika belum diinisialisasi
        return Logger("/app/data/fallback.log", tag="FALLBACK")
    return _global_logger
