import os
import time
import shutil
import requests
from threading import Thread
from datetime import datetime
from sqlalchemy.orm import Session
from app.db.db import SessionLocal
from app.db.models import FLSession
from app.config import ECONOMICS, CODECARBON_AVAILABLE, DATA_ROOT, REGISTRY_PATH, BN_PATH, SUBMISSIONS_DIR
from app.utils.mobilefacenet import MobileFaceNet
from app.utils.aggregation_utils import aggregate_and_save_registry_assets

OfflineEmissionsTracker = None
if CODECARBON_AVAILABLE:
    try:
        from codecarbon import OfflineEmissionsTracker
    except ImportError:
        CODECARBON_AVAILABLE = False

class FLController:
    # Orkestrator Pembelajaran Terfederasi (Federated Learning)
    # Kelas ini menangani seluruh alur kerja server, mulai dari pendaftaran terminal,
    # sinkronisasi ID mahasiswa, hingga penggabungan bobot model global.
    
    def __init__(self, fl_manager):
        self.fl_manager = fl_manager

    def start_lifecycle(self, rounds: int = 10, min_clients: int = 2, epochs: int = 1):
        # Memulai siklus lengkap FL dalam satu tombol.
        self.fl_manager.start_time = time.time()
        # Proses ini berjalan di thread terpisah agar tidak memblokir dashboard.
        
        if self.fl_manager.is_running:
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
        # Konektivitas: Memastikan terminal cukup untuk memulai.
        # Discovery: Sinkronisasi daftar mahasiswa global.
        # Preprocessing: Cropping wajah di sisi terminal.
        # Training: Pelatihan model menggunakan algoritma Flower.
        # Registry: Penggabungan fitur wajah (Centroids) untuk pengenalan.
        
        try:
            db = SessionLocal()
            self.fl_manager.start_phase("discovery")
            self.fl_manager.ensure_model_seeded(db)
            db.close()

            # Inisialisasi Emission Tracker (CodeCarbon)
            tracker = None
            if CODECARBON_AVAILABLE and OfflineEmissionsTracker is not None:
                try:
                    emissions_dir = os.path.join(DATA_ROOT, "emissions")
                    os.makedirs(emissions_dir, exist_ok=True)
                    tracker = OfflineEmissionsTracker(
                         country_iso_code="IDN", 
                         log_level="error", 
                         save_to_file=True, 
                         output_dir=emissions_dir
                    )
                    tracker.start()
                except: pass

            # Reset folder pengumpulan fitur
            if os.path.exists(SUBMISSIONS_DIR):
                shutil.rmtree(SUBMISSIONS_DIR)
            os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
            
            self._log("Memulai siklus penuh Pembelajaran Terfederasi...")
            
            # Fase 0: Menunggu terminal terhubung
            self._log(f"Menunggu {min_clients} terminal terhubung...")
            if not self._wait_for_condition(lambda: len(self.fl_manager.registered_clients) >= min_clients, timeout=3600):
                self._log("[ERROR] Gagal: Terminal tidak mencukupi setelah 1 jam.")
                return

            # Fase 1a: Discovery (Sinkronisasi ID)
            self._log("Fase 1a: Sinkronisasi ID Mahasiswa antar terminal...")
            self.fl_manager.discovery_clients.clear()
            self._trigger_clients("/api/request-discovery")
            
            if not self._wait_for_condition(lambda: len(self.fl_manager.discovery_clients) >= min_clients, timeout=3600):
                self._log("[ERROR] Gagal: Tahap Discovery melampaui batas waktu (1 Jam).")
                return

            # Fase 1b: Preprocessing (Deteksi & Crop)
            self.fl_manager.start_phase("syncing")
            self._log("Fase 1b: Pemrosesan gambar di sisi terminal...")
            self.fl_manager.ready_clients.clear()
            self._trigger_clients("/api/request-preprocess")
            
            # Tunggu client lapor 'Ready' (Preprocessing selesai)
            self._log("SERVER LOG: Menunggu laporan 'Ready' dari seluruh terminal (Timeout: 3 Jam)...")
            if not self._wait_for_ready_clients(min_clients, timeout=10800):
                self._log("[ERROR] Gagal: Tahap Preprocessing melampaui batas waktu (3 Jam).")
                return
            
            # Fase 2: Pelatihan Federated (Flower)
            self.fl_manager.start_phase("training")
            self._log(f"Memulai pelatihan Flower dengan {len(self.fl_manager.ready_clients)} terminal...")
            self.fl_manager.is_running = True  
            self.fl_manager.start_training(session_id, rounds=rounds, min_clients=min_clients)
            
            # Fase 3: Pembuatan Registry Global
            self.fl_manager.start_phase("Registry Generation")
            self._log("Fase 3: Menggabungkan fitur wajah (Centroids) secara global...")
            
            # Tunggu pengumpulan aset (Timeout ditingkatkan ke 1 Jam / 3600 detik)
            if self._wait_for_registry_submissions(len(self.fl_manager.ready_clients), timeout=3600):
                self._log("[SUCCESS] Semua data fitur wajah telah diterima.")
            else:
                self._log("[WARNING] Batas waktu habis, memproses data fitur yang tersedia.")
            
            self._aggregate_registry_logic()
            
            # [INCREMENT VERSION] Naikkan versi SETELAH registry selesai agar sinkron dengan client
            self.fl_manager.increment_version()
            
            # Hitung Real Volume Transmisi
            num_clients = len(self.fl_manager.ready_clients)
            
            # Hitung Real Backbone dari jumlah parameter model
            try:
                temp_model = MobileFaceNet()
                # Total parameter * 4 byte (float32) / 1024^2
                param_size_mb = sum(p.numel() for p in temp_model.parameters()) * 4 / (1024 * 1024)
            except:
                param_size_mb = ECONOMICS.get("estimated_backbone_size_mb", 4.5)
            
            # Volume Backbone = (Down + Up) * Clients * Rounds
            backbone_mb = param_size_mb * 2 * num_clients * rounds
            
            # Hitung Real Registry (Aset yang dikirim ke client)
            registry_mb = 0
            
            if os.path.exists(REGISTRY_PATH):
                registry_mb += os.path.getsize(REGISTRY_PATH) / (1024 * 1024)
            if os.path.exists(BN_PATH):
                registry_mb += os.path.getsize(BN_PATH) / (1024 * 1024)
            
            # Real volume registri total adalah (size aset) * jumlah client yang mengunduh
            registry_mb = registry_mb * num_clients
            
            # Ambil Real Energy (CodeCarbon) jika tersedia
            energy_kwh = 0
            if tracker:
                try:
                    emissions_data = tracker.stop()
                    if emissions_data:
                        energy_kwh = float(emissions_data)
                except: pass

            self.fl_manager.update_metrics({
                "backbone_sync_mb": round(backbone_mb, 2),
                "registry_sync_mb": round(registry_mb, 2),
                "compute_energy_kwh": round(energy_kwh, 6) if energy_kwh > 0 else 0,
                "total_round_time_s": round(time.time() - self.fl_manager.start_time, 2)
            })
            
            self.fl_manager.start_phase("Completed")
            self._log("[SUCCESS] Seluruh siklus Pembelajaran Terfederasi selesai.")
            
            # Hapus file cache deteksi video lama karena model telah diperbarui
            try:
                video_caches_dir = "video_caches"
                if os.path.exists(video_caches_dir):
                    for f in os.listdir(video_caches_dir):
                        if f.endswith(".json"):
                            os.remove(os.path.join(video_caches_dir, f))
                    self._log("[INFO] Menghapus cache deteksi video lama karena model diperbarui.")
            except Exception as cache_del_err:
                self._log(f"[WARNING] Gagal menghapus cache video lama: {cache_del_err}")

            # Beri jeda agar heartbeat client sempat menangkap fase 'Completed'
            time.sleep(5)

        except Exception as e:
            self._log(f"[ERROR] Kesalahan pada orkestrasi: {e}")
        finally:
            self.fl_manager.end_phase()
            self._mark_session_completed(session_id)

    def _log(self, msg):
        if "[ERROR]" in msg:
            self.fl_manager.logger.error(msg.replace("[ERROR] ", ""))
        elif "[SUCCESS]" in msg:
            self.fl_manager.logger.success(msg.replace("[SUCCESS] ", ""))
        elif "[INFO]" in msg:
            self.fl_manager.logger.info(msg.replace("[INFO] ", ""))
        elif "[OK]" in msg:
            self.fl_manager.logger.success(msg.replace("[OK] ", ""))
        else:
            self.fl_manager.logger.info(msg)

    def _trigger_clients(self, endpoint):
        self.fl_manager.logger.info(f"Memicu endpoint {endpoint} ke seluruh klien...")
        for cid, data in self.fl_manager.registered_clients.items():
            ip = data.get("ip_address")
            port = data.get("port", 8080)
            if ip:
                try:
                    # Gunakan port yang dilaporkan client saat registrasi
                    requests.post(f"http://{ip}:{port}{endpoint}", timeout=2)
                except Exception as e:
                    self.fl_manager.logger.warn(f"Gagal memicu {cid} pada {ip}: {e}")

    def _wait_for_condition(self, condition_func, timeout):
        start = time.time()
        while not condition_func():
            if time.time() - start > timeout: return False
            time.sleep(5)
        return True

    def _wait_for_ready_clients(self, min_clients, timeout):
        start = time.time()
        last_log = 0
        last_ready_count = -1
        while len(self.fl_manager.ready_clients) < min_clients:
            if time.time() - start > timeout: return False
            
            current_ready_count = len(self.fl_manager.ready_clients)
            # Log hanya jika jumlah berubah ATAU setiap 10 menit (600 detik)
            if current_ready_count != last_ready_count or time.time() - last_log > 600:
                self._log(f"SERVER LOG: Status Terminal: {current_ready_count}/{min_clients} siap.")
                last_log = time.time()
                last_ready_count = current_ready_count

            # Cek status via heartbeat sebagai cadangan jika sinyal API terlewat
            for cid, data in list(self.fl_manager.registered_clients.items()):
                if data.get("fl_status") == "Siap Training" or "READY" in str(data.get("fl_status")):
                    self.fl_manager.ready_clients.add(cid)
            
            time.sleep(5)
        return True

    def _wait_for_registry_submissions(self, expected_count, timeout):
        start = time.time()
        last_log = 0
        last_count = -1
        while True:
            files = [f for f in os.listdir(SUBMISSIONS_DIR) if f.endswith("_assets.pth")] if os.path.exists(SUBMISSIONS_DIR) else []
            current_count = len(files)
            
            # Log progres setiap ada perubahan atau setiap 60 detik
            if current_count != last_count or time.time() - last_log > 60:
                self._log(f"SERVER LOG: Aset fitur diterima: {current_count}/{expected_count}")
                last_log = time.time()
                last_count = current_count
                
            if current_count >= expected_count: return True
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
        aggregate_and_save_registry_assets(self.fl_manager.logger)
