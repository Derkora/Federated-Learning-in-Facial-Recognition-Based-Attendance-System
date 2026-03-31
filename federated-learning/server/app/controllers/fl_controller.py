import os
import time
import shutil
import requests
from threading import Thread
from datetime import datetime
from sqlalchemy.orm import Session
from ..db.db import SessionLocal
from ..db.models import FLSession

class FLController:
    # Orkestrator Pembelajaran Terfederasi (Federated Learning)
    # Kelas ini menangani seluruh alur kerja server, mulai dari pendaftaran terminal,
    # sinkronisasi ID mahasiswa, hingga penggabungan bobot model global.
    
    def __init__(self, fl_manager):
        self.fl_manager = fl_manager

    def start_lifecycle(self, rounds: int, min_clients: int, epochs: int = None):
        # Memulai siklus lengkap FL dalam satu tombol.
        # Proses ini berjalan di thread terpisah agar tidak memblokir dashboard.
        
        if self.fl_manager.is_busy or self.fl_manager.is_running:
            return {"status": "already_running"}
            
        session_id = f"session_{int(time.time())}"
        self.fl_manager.session_id = session_id
        self.fl_manager.current_logs = []
        self.fl_manager.received_data = []
        
        if epochs: self.fl_manager.default_epochs = epochs
        
        # Simpan sesi ke database
        db = SessionLocal()
        new_session = FLSession(session_id=session_id)
        db.add(new_session)
        db.commit()
        db.close()

        # Jalankan orkestrasi di background
        Thread(target=self._orchestrate_routine, args=(session_id, rounds, min_clients), daemon=True).start()
        return {"status": "started", "session_id": session_id}

    def _orchestrate_routine(self, session_id: str, rounds: int, min_clients: int):
        # Alur kerja teknis per fase:
        # 1. Konektivitas: Memastikan terminal cukup untuk memulai.
        # 2. Discovery: Sinkronisasi daftar mahasiswa global.
        # 3. Preprocessing: Cropping wajah di sisi terminal.
        # 4. Training: Pelatihan model menggunakan algoritma Flower.
        # 5. Registry: Penggabungan fitur wajah (Centroids) untuk pengenalan.
        
        try:
            db = SessionLocal()
            self.fl_manager.start_phase("Data Preparation")
            self.fl_manager.ensure_model_seeded(db)
            db.close()
            
            # Reset folder pengumpulan fitur
            if os.path.exists("data/submissions"):
                shutil.rmtree("data/submissions")
            os.makedirs("data/submissions", exist_ok=True)
            
            self._log("Memulai siklus penuh Pembelajaran Terfederasi...")
            
            # Fase 0: Menunggu terminal terhubung
            self._log(f"Menunggu {min_clients} terminal terhubung...")
            if not self._wait_for_condition(lambda: len(self.fl_manager.registered_clients) >= min_clients, timeout=300):
                self._log("[ERROR] Gagal: Terminal tidak mencukupi setelah 5 menit.")
                return

            # Fase 1a: Discovery (Sinkronisasi ID)
            self._log("Fase 1a: Sinkronisasi ID Mahasiswa antar terminal...")
            self.fl_manager.discovery_clients.clear()
            self._trigger_clients("/api/request-discovery")
            
            if not self._wait_for_condition(lambda: len(self.fl_manager.discovery_clients) >= min_clients, timeout=300):
                self._log("[ERROR] Gagal: Tahap Discovery melampaui batas waktu.")
                return

            # Fase 1b: Preprocessing (Deteksi & Crop)
            self._log("Fase 1b: Pemrosesan gambar di sisi terminal...")
            self.fl_manager.ready_clients.clear()
            self._trigger_clients("/api/request-preprocess")
            
            # Menunggu terminal siap (READY)
            if not self._wait_for_ready_clients(min_clients, timeout=600):
                self._log("[ERROR] Gagal: Tahap Preprocessing melampaui batas waktu.")
                return
            
            # Fase 2: Pelatihan Federated (Flower)
            self._log(f"Memulai pelatihan Flower dengan {len(self.fl_manager.ready_clients)} terminal...")
            self.fl_manager.is_running = True  
            self.fl_manager.start_training(session_id, rounds=rounds, min_clients=min_clients)
            
            # Fase 3: Pembuatan Registry Global
            self.fl_manager.start_phase("Registry Generation")
            self._log("Fase 3: Menggabungkan fitur wajah (Centroids) secara global...")
            
            if self._wait_for_registry_submissions(len(self.fl_manager.ready_clients), timeout=300):
                self._log("[OK] Semua data fitur wajah telah diterima.")
            else:
                self._log("[WARN] Batas waktu habis, memproses data fitur yang tersedia.")
            
            self._aggregate_registry_logic()
            
            self.fl_manager.start_phase("Completed")
            self._log("[OK] Seluruh siklus Pembelajaran Terfederasi selesai.")

        except Exception as e:
            self._log(f"[ERROR] Kesalahan pada orkestrasi: {e}")
        finally:
            self.fl_manager.end_phase()
            self._mark_session_completed(session_id)

    def _log(self, msg):
        self.fl_manager.update_logs(msg)

    def _trigger_clients(self, endpoint):
        for cid, data in self.fl_manager.registered_clients.items():
            ip = data.get("ip_address")
            if ip:
                try:
                    requests.post(f"http://{ip}:8080{endpoint}", timeout=2)
                except: pass

    def _wait_for_condition(self, condition_func, timeout):
        start = time.time()
        while not condition_func():
            if time.time() - start > timeout: return False
            time.sleep(5)
        return True

    def _wait_for_ready_clients(self, min_clients, timeout):
        start = time.time()
        while len(self.fl_manager.ready_clients) < min_clients:
            if time.time() - start > timeout: return False
            
            # Cek status via heartbeat sebagai cadangan jika sinyal API terlewat
            for cid, data in list(self.fl_manager.registered_clients.items()):
                if data.get("fl_status") == "Siap Training":
                    self.fl_manager.ready_clients.add(cid)
            time.sleep(5)
        return True

    def _wait_for_registry_submissions(self, expected_count, timeout):
        start = time.time()
        submission_dir = "data/submissions"
        while True:
            files = [f for f in os.listdir(submission_dir) if f.endswith("_assets.pth")] if os.path.exists(submission_dir) else []
            if len(files) >= expected_count: return True
            if time.time() - start > timeout: return False
            time.sleep(5)

    def _mark_session_completed(self, session_id):
        db = SessionLocal()
        session = db.query(FLSession).filter_by(session_id=session_id).first()
        if session:
            session.status = "completed"
            db.commit()
        db.close()

    def _aggregate_registry_logic(self):
        # Memanggil fungsi agregasi fitur dari utilitas terpisah
        from ..utils.aggregation_utils import aggregate_and_save_registry_assets
        aggregate_and_save_registry_assets(self._log)
