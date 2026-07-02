import requests
import time 
import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import threading
import queue
import gc
from app.utils.preprocessing import image_processor, DEVICE

from app.utils.logging import get_logger

# Konfigurasi endpoint API
api_logs_inference = "/api/logs/inference"
api_submit_attendance = "/submit-attendance"

ATTENDANCE_QUEUE_FILE = "/app/data/offline_attendance_queue.json"
INFERENCE_QUEUE_FILE = "/app/data/offline_inference_logs.json"
queue_lock = threading.Lock()

# Thread-safe global background logging worker to eliminate per-frame thread leaks
_inference_log_queue = queue.Queue(maxsize=1000)
_worker_started = False
_worker_lock = threading.Lock()

def _bg_inference_log_worker():
    while True:
        try:
            task = _inference_log_queue.get()
            if task is None:
                break
            server_url, payload, callback_offline = task
            try:
                res = requests.post(f"{server_url}{api_logs_inference}", json=payload, timeout=2)
                if res.status_code != 200 and callback_offline:
                    callback_offline(payload)
            except Exception:
                if callback_offline:
                    callback_offline(payload)
        except Exception:
            pass
        finally:
            _inference_log_queue.task_done()

import atexit

def queue_inference_log(server_url, payload, callback_offline=None):
    global _worker_started
    if not _worker_started:
        with _worker_lock:
            if not _worker_started:
                t = threading.Thread(target=_bg_inference_log_worker, daemon=True, name="CLInferenceLoggerThread")
                t.start()
                _worker_started = True
    try:
        _inference_log_queue.put_nowait((server_url, payload, callback_offline))
    except queue.Full:
        try:
            # Save oldest log offline before replacing it in the queue
            old_task = _inference_log_queue.get_nowait()
            if old_task is not None:
                _, old_payload, cb = old_task
                if cb:
                    cb(old_payload)
            _inference_log_queue.task_done()
        except Exception:
            pass
        try:
            _inference_log_queue.put_nowait((server_url, payload, callback_offline))
        except Exception:
            if callback_offline:
                callback_offline(payload)

def cleanup_on_exit():
    logs_to_save = []
    while not _inference_log_queue.empty():
        try:
            task = _inference_log_queue.get_nowait()
            if task is not None:
                _, payload, _ = task
                logs_to_save.append(payload)
            _inference_log_queue.task_done()
        except Exception:
            break
            
    if logs_to_save:
        os.makedirs(os.path.dirname(INFERENCE_QUEUE_FILE), exist_ok=True)
        with queue_lock:
            queue_data = []
            if os.path.exists(INFERENCE_QUEUE_FILE):
                try:
                    with open(INFERENCE_QUEUE_FILE, "r") as f:
                        queue_data = json.load(f)
                except Exception:
                    pass
            queue_data.extend(logs_to_save)
            if len(queue_data) > 1000:
                queue_data = queue_data[-1000:]
            try:
                with open(INFERENCE_QUEUE_FILE, "w") as f:
                    json.dump(queue_data, f, indent=4)
            except Exception:
                pass

atexit.register(cleanup_on_exit)

class AttendanceController:
    # Kontroler untuk proses Pengenalan Wajah dan Pelaporan Presensi.
    
    def __init__(self, manager):
        self.manager = manager
        self.server_url = manager.server_url
        self.client_id = manager.client_id
        self._last_version_loaded = -1 # Melacak versi model yang sedang aktif di session
        self.logger = get_logger()

    def save_attendance_offline(self, payload):
        os.makedirs(os.path.dirname(ATTENDANCE_QUEUE_FILE), exist_ok=True)
        with queue_lock:
            queue = []
            if os.path.exists(ATTENDANCE_QUEUE_FILE):
                try:
                    with open(ATTENDANCE_QUEUE_FILE, "r") as f:
                        queue = json.load(f)
                except Exception: pass
            queue.append(payload)
            try:
                with open(ATTENDANCE_QUEUE_FILE, "w") as f:
                    json.dump(queue, f, indent=4)
                self.logger.info(f"[Offline Queue] Presensi {payload['user_id']} disimpan secara lokal (Total: {len(queue)}).")
            except Exception as e:
                self.logger.error(f"Gagal menulis antrean presensi offline: {e}")

    def save_inference_offline(self, payload):
        os.makedirs(os.path.dirname(INFERENCE_QUEUE_FILE), exist_ok=True)
        with queue_lock:
            queue = []
            if os.path.exists(INFERENCE_QUEUE_FILE):
                try:
                    with open(INFERENCE_QUEUE_FILE, "r") as f:
                        queue = json.load(f)
                except Exception: pass
            queue.append(payload)
            # Batasi ukuran antrean log inferensi agar tidak memakan memori berlebih
            if len(queue) > 1000:
                queue = queue[-1000:]
            try:
                with open(INFERENCE_QUEUE_FILE, "w") as f:
                    json.dump(queue, f, indent=4)
            except Exception: pass

    def process_offline_queues(self):
        """Mengirim ulang seluruh antrean presensi dan log inferensi yang tertunda."""
        # Proses Presensi Offline
        if os.path.exists(ATTENDANCE_QUEUE_FILE):
            with queue_lock:
                try:
                    with open(ATTENDANCE_QUEUE_FILE, "r") as f:
                        attendance_queue = json.load(f)
                except Exception:
                    attendance_queue = []
            
            if attendance_queue:
                self.logger.info(f"Mencoba mengirim {len(attendance_queue)} presensi offline tertunda...")
                succeeded = []
                for item in attendance_queue:
                    try:
                        res = requests.post(f"{self.server_url}{api_submit_attendance}", json=item, timeout=5)
                        if res.status_code == 200:
                            succeeded.append(item)
                    except Exception:
                        break # Koneksi masih terputus, hentikan loop
                
                if succeeded:
                    with queue_lock:
                        # Hapus item yang berhasil dikirim
                        remaining = [x for x in attendance_queue if x not in succeeded]
                        try:
                            with open(ATTENDANCE_QUEUE_FILE, "w") as f:
                                json.dump(remaining, f, indent=4)
                            self.logger.success(f"Berhasil sinkronisasi {len(succeeded)} presensi offline tertunda!")
                        except Exception: pass

        # Proses Log Inferensi Offline
        if os.path.exists(INFERENCE_QUEUE_FILE):
            with queue_lock:
                try:
                    with open(INFERENCE_QUEUE_FILE, "r") as f:
                        inference_queue = json.load(f)
                except Exception:
                    inference_queue = []
            
            if inference_queue:
                succeeded_logs = []
                for item in inference_queue:
                    try:
                        res = requests.post(f"{self.server_url}{api_logs_inference}", json=item, timeout=3)
                        if res.status_code == 200:
                            succeeded_logs.append(item)
                    except Exception:
                        break
                
                if succeeded_logs:
                    with queue_lock:
                        remaining_logs = [x for x in inference_queue if x not in succeeded_logs]
                        try:
                            with open(INFERENCE_QUEUE_FILE, "w") as f:
                                json.dump(remaining_logs, f, indent=4)
                        except Exception: pass

    def process_inference(self, img_pil, model, reference_embeddings):
        start_time = time.perf_counter()
        # Melakukan inferensi wajah dengan Temporal Voting (Buffer Rata-rata)
        threshold = self.manager.threshold
        
        # Deteksi & MTCNN Square Crop (dengan margin 20px)
        mtcnn_start = time.perf_counter()
        face_tensor, _, _ = image_processor.detect_face(img_pil)
        mtcnn_time = int((time.perf_counter() - mtcnn_start) * 1000)
        
        if face_tensor is not None:

            # prepare_for_model menangani squash 112x112 ke 96x112 dan normalisasi [-1, 1]
            face_tensor_ready = image_processor.prepare_for_model(face_tensor)
            
            # Mendapatkan versi model saat ini dari manager
            current_v = getattr(self.manager, 'current_model_version', 0)
            current_v_clean = str(current_v).lstrip('v')
            
            # Inferensi murni PyTorch untuk optimasi edge
            backbone_start = time.perf_counter()
            try:
                if current_v != self._last_version_loaded:
                    self.logger.info(f"Menggunakan model PyTorch v{current_v_clean}. Input: {face_tensor_ready.shape}")
                    self._last_version_loaded = current_v
                
                with torch.no_grad():
                    model.eval() # PENTING: Cegah error BN "Expected more than 1 value"
                    # Proses flip trick alignment dengan registry
                    # Embedding Citra Asli
                    emb_orig = model(face_tensor_ready)
                    
                    # Embedding Citra Mirror (Horizontal Flip)
                    face_flipped = torch.flip(face_tensor_ready, dims=[3])
                    emb_mirror = model(face_flipped)
                    
                    # Rata-rata dan normalisasi akhir ke unit vector (Flip-Only ITA)
                    query_emb_tensor = F.normalize((emb_orig + emb_mirror) / 2, p=2, dim=1)
            except Exception as e:
                self.logger.error(f"Kegagalan inferensi: {e}")
                return "Unknown", 0, False
            finally:
                backbone_time = int((time.perf_counter() - backbone_start) * 1000)
            
            # Pastikan dimensi query_emb_tensor adalah (1, 128)
            query_emb_tensor = query_emb_tensor.view(1, -1)
            
            # Logika temporal voting dan confident instant match
            # Cek Kemiripan Instant Frame (Vektorisasi)
            ref_start = time.perf_counter()
            user_ids = list(reference_embeddings.keys())
            ref_list = []
            for nrp in user_ids:
                ref = reference_embeddings[nrp]
                if not isinstance(ref, torch.Tensor):
                    ref = torch.tensor(ref).to(DEVICE)
                ref_list.append(ref.view(1, -1))
            
            # Stack semua referensi menjadi matriks (N, 128)
            if not ref_list:
                self.logger.warn("Tidak ada referensi identitas (Registry Kosong).")
                return "Unknown", 0, False
            
            ref_matrix = torch.cat(ref_list, dim=0)
            ref_matrix = F.normalize(ref_matrix, p=2, dim=1)
            
            # Logika temporal voting sederhana
            now = time.time()
            if now - self.manager.last_face_time > 3.0:
                self.manager.prediction_buffer.clear()
            
            self.manager.prediction_buffer.append(query_emb_tensor)
            self.manager.last_face_time = now
            
            mean_emb_tensor = torch.stack(list(self.manager.prediction_buffer)).mean(0)
            mean_emb_tensor = F.normalize(mean_emb_tensor, p=2, dim=1)
            
            # Hitung skor temporal sekaligus
            scores_temporal = torch.mm(mean_emb_tensor, ref_matrix.t())
            max_sim_temp, max_idx_temp = torch.max(scores_temporal, dim=1)
            max_sim = max_sim_temp.item()
            best_match = user_ids[max_idx_temp.item()]
            matching_time = int((time.perf_counter() - ref_start) * 1000)
            
            latency = int((time.perf_counter() - start_time) * 1000)
            
            # Queue log report asynchronously without thread spawning leaks
            self._log_inference_to_server(best_match, max_sim, latency, threshold)
 
            # Log performa detail per langkah (Research-ready detailed log)
            self.logger.info(
                f"[PERF] MTCNN: {mtcnn_time}ms | Backbone: {backbone_time}ms | Search: {matching_time}ms | Total Latency: {latency}ms"
            )

            # Bebaskan memori secara eksplisit untuk mencegah memory leakage
            del face_tensor, face_tensor_ready, face_flipped, emb_orig, emb_mirror, query_emb_tensor, mean_emb_tensor, ref_matrix
            gc.collect()

            if max_sim > threshold:
                self.logger.success(f"{best_match} terdeteksi (Sim: {max_sim:.4f}) [Model: v{current_v_clean}]")
                nrp_only = best_match.split("_")[0] if "_" in best_match else best_match
                self.submit_attendance(nrp_only, max_sim, latency)
                return nrp_only, max_sim, True # True = Terverifikasi
            else:
                if max_sim > 0.1:
                    self.logger.info(f"Model v{current_v_clean} | Terbaik: {best_match} | Sim: {max_sim:.4f} | Thres: {threshold:.2f}")
                return "Unknown", max_sim, False # False = Tidak Terverifikasi
        
        return "Unknown", 0, False
 
    def submit_attendance(self, user_id, confidence, latency=0):
        # Mengirimkan laporan presensi mahasiswa ke server pusat.
        payload = {
            "user_id": user_id, 
            "edge_id": self.client_id, 
            "confidence": float(confidence), 
            "lecture_id": "L123",
            "latency_ms": int(latency)
        }
        
        # Coba bersihkan antrean offline terlebih dahulu
        try:
            self.process_offline_queues()
        except Exception: pass

        try:
            res = requests.post(f"{self.server_url}{api_submit_attendance}", json=payload, timeout=5)
            if res.status_code == 200:
                self.logger.success(f"Presensi berhasil dikirim untuk: {user_id}")
            else:
                self.logger.error(f"Gagal mengirim presensi ke server (Status: {res.status_code}). Menyimpan offline.")
                self.save_attendance_offline(payload)
        except Exception as e:
            self.logger.error(f"Gagal mengirim presensi ke server. Menyimpan offline: {e}")
            self.save_attendance_offline(payload)
 
    def _log_inference_to_server(self, user_id, confidence, latency_ms, threshold):
        """Mengirim data inferensi mentah ke server untuk pemantauan terpusat."""
        nrp_only = user_id.split("_")[0] if "_" in user_id else user_id
        status = "KNOWN" if float(confidence) >= threshold else "UNKNOWN"
        payload = {
            "client_id": self.client_id,
            "user_id": nrp_only,
            "confidence": float(confidence),
            "latency_ms": int(latency_ms),
            "status": status
        }
        try:
            queue_inference_log(self.server_url, payload, self.save_inference_offline)
        except Exception as e:
            self.logger.error(f"[CIM] Gagal mengantrekan log inferensi ke queue: {e}")
