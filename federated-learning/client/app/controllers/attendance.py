import os
import time
import base64
import io
import gc
import numpy as np
import torch
from PIL import Image
from sqlalchemy.orm import Session

from app.db.db import SessionLocal
from app.db.models import EmbeddingLocal, UserLocal, AttendanceLocal
from app.utils.preprocessing import image_processor
from app.utils.classifier import identify_user_globally
from app.utils.sync_utils import sync_record_to_server
from app.utils.security import encryptor

import queue
import threading
import requests

import json
import atexit

# Konfigurasi endpoint API
api_logs_inference = "/api/logs/inference"

INFERENCE_QUEUE_FILE = "/app/data/offline_inference_logs.json"
inference_queue_lock = threading.Lock()

def save_inference_offline(payload):
    os.makedirs(os.path.dirname(INFERENCE_QUEUE_FILE), exist_ok=True)
    with inference_queue_lock:
        queue_data = []
        if os.path.exists(INFERENCE_QUEUE_FILE):
            try:
                with open(INFERENCE_QUEUE_FILE, "r") as f:
                    queue_data = json.load(f)
            except Exception:
                pass
        queue_data.append(payload)
        if len(queue_data) > 1000:
            queue_data = queue_data[-1000:]
        try:
            with open(INFERENCE_QUEUE_FILE, "w") as f:
                json.dump(queue_data, f, indent=4)
        except Exception:
            pass

# Queue background log murni non-blocking thread
_log_queue = queue.Queue(maxsize=1000)
_worker_thread = None

def _bg_inference_log_worker():
    while True:
        try:
            item = _log_queue.get()
            if item is None:
                break
            server_url, payload, logger = item
            try:
                res = requests.post(f"{server_url}{api_logs_inference}", json=payload, timeout=2)
                if res.status_code != 200:
                    logger.warn(f"Gagal mengirim log inferensi ke server aggregator (status {res.status_code}). Menyimpan offline.")
                    save_inference_offline(payload)
            except Exception as e:
                logger.warn(f"Gagal mengirim log inferensi ke server aggregator: {e}. Menyimpan offline.")
                save_inference_offline(payload)
            finally:
                _log_queue.task_done()
        except Exception:
            time.sleep(1)

def queue_inference_log(server_url, payload, logger):
    global _worker_thread
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_bg_inference_log_worker, daemon=True)
        _worker_thread.start()
    try:
        _log_queue.put_nowait((server_url, payload, logger))
    except queue.Full:
        try:
            # Drop oldest and save offline
            old_item = _log_queue.get_nowait()
            if old_item is not None:
                _, old_payload, _ = old_item
                save_inference_offline(old_payload)
            _log_queue.task_done()
        except Exception:
            pass
        try:
            _log_queue.put_nowait((server_url, payload, logger))
        except Exception:
            save_inference_offline(payload)

def cleanup_on_exit():
    logs_to_save = []
    while not _log_queue.empty():
        try:
            item = _log_queue.get_nowait()
            if item is not None:
                _, payload, _ = item
                logs_to_save.append(payload)
            _log_queue.task_done()
        except Exception:
            break
            
    if logs_to_save:
        os.makedirs(os.path.dirname(INFERENCE_QUEUE_FILE), exist_ok=True)
        with inference_queue_lock:
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
    # Kontroler untuk manajemen pengenalan wajah offline dan absensi perangkat edge
    
    def __init__(self, fl_manager):
        self.fl_manager = fl_manager
        self._last_version_loaded = -1 # Melacak versi model yang aktif di session

    # Inferensi Presensi Wajah di Sisi Klien (Edge Attendance Recognition)
    async def process_inference(self, image_b64: str, db: Session, background_tasks):
        # Mengekstrak representasi wajah offline murni menggunakan MTCNN, Flip Trick, dan voting temporal
        start_time = time.perf_counter()
        
        # Dekode gambar dari base64
        img_bytes = base64.b64decode(image_b64)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Deteksi + Crop wajah menggunakan MTCNN
        mtcnn_start = time.perf_counter()
        face_tensor, box, prob = image_processor.detect_face(img_pil)
        mtcnn_time = int((time.perf_counter() - mtcnn_start) * 1000)
        
        if face_tensor is None:
            img_pil.close()
            return {"matched": "Unknown", "confidence": 0, "message": "Wajah tidak terdeteksi"}
        
        threshold = self.fl_manager.inference_threshold

        # Preprocessing: Squash ke 112x96 dan normalisasi [-1, 1]
        input_tensor = image_processor.prepare_for_model(face_tensor).to(self.fl_manager.device)
        
        current_v = getattr(self.fl_manager, 'model_version', 0)

        # Inferensi model murni PyTorch untuk optimalisasi perangkat edge
        backbone_start = time.perf_counter()
        try:
            if current_v != self._last_version_loaded:
                self.fl_manager.logger.info(f"Menggunakan model PyTorch v{current_v}. Input: {input_tensor.shape}")
                self._last_version_loaded = current_v

            with torch.no_grad():
                # Gunakan inference_backbone agar stabil terhadap drift ronde
                target_backbone = getattr(self.fl_manager, 'inference_backbone', self.fl_manager.backbone)
                if target_backbone is None: return {"matched": "Error", "confidence": 0, "message": "Model belum dimuat"}
                
                target_backbone.eval()
                
                # Ekstraksi embedding wajah dari citra asli
                emb_orig = target_backbone(input_tensor)
                
                # Ekstraksi embedding wajah dari citra cermin horizontal
                face_flipped = torch.flip(input_tensor, dims=[3])
                emb_mirror = target_backbone(face_flipped)
                
                # Rata-rata dan normalisasi akhir ke unit vector
                query_embedding_tensor = torch.nn.functional.normalize((emb_orig + emb_mirror) / 2, p=2, dim=1)
        except Exception as e:
            self.fl_manager.logger.error(f"Kegagalan inferensi: {e}")
            return {"matched": "Error", "confidence": 0, "message": str(e)}
        finally:
            backbone_time = int((time.perf_counter() - backbone_start) * 1000)
            
        # Manajemen Cache Identitas
        ref_start = time.perf_counter()
        local_refs = self._get_cached_identities(db)

        # Logika temporal voting dengan buffer frame untuk kestabilan akurasi
        now = time.time()
        if now - self.fl_manager.last_face_time > 3.0:
            self.fl_manager.prediction_buffer.clear()
            
        self.fl_manager.prediction_buffer.append(query_embedding_tensor)
        self.fl_manager.last_face_time = now
        
        # Rata-rata temporal embedding wajah dalam antrean buffer
        mean_embedding_tensor = torch.stack(list(self.fl_manager.prediction_buffer)).mean(0)
        mean_embedding_tensor = torch.nn.functional.normalize(mean_embedding_tensor, p=2, dim=1)
        mean_embedding = mean_embedding_tensor.cpu().numpy()[0]
        
        # Pencocokan identitas tunggal dengan Cosine Similarity
        best_match, confidence = identify_user_globally(mean_embedding, local_refs, threshold=threshold)
        matching_time = int((time.perf_counter() - ref_start) * 1000)
        
        # Tentukan ID hasil klasifikasi berdasarkan ambang batas
        user_id = best_match if confidence >= threshold else "Unknown"
        
        # Pencatatan absensi jika wajah berhasil terverifikasi
        if user_id != "Unknown":
            user_name = self._ensure_user_and_get_name(user_id, db)
            
            try:
                new_attendance = AttendanceLocal(
                    user_id=user_id,
                    confidence=confidence,
                    device_id=os.getenv("HOSTNAME", "terminal-1")
                )
                db.add(new_attendance)
                db.commit()
                
                # Sinkronisasi Riwayat Absensi ke Server Pusat
                # Mengirimkan rekapitulasi kehadiran lokal ke API server pusat secara asinkron
                latency = int((time.perf_counter() - start_time) * 1000)
                background_tasks.add_task(
                    sync_record_to_server, 
                    user_id, user_name, float(confidence), os.getenv("HOSTNAME", "client-1"),
                    latency
                )
                self.fl_manager.logger.success(f"Wajah Terverifikasi: {user_id} (Sim: {confidence:.4f})")
            except Exception as e:
                db.rollback()
                self.fl_manager.logger.error(f"Gagal mencatat presensi {user_id} ke DB lokal: {e}")

        else:
            if confidence > 0.1: 
                self.fl_manager.logger.info(f"Model {current_v} | Terbaik: {best_match} | Sim: {confidence:.4f} | Thres: {threshold:.2f}")
            
        latency = int((time.perf_counter() - start_time) * 1000)
        
        # Log performa detail per langkah
        self.fl_manager.logger.info(
            f"[PERF] MTCNN: {mtcnn_time}ms | Backbone: {backbone_time}ms | Search: {matching_time}ms | Total Latency: {latency}ms"
        )

        # Mengantrekan log ke server latar belakang
        self._log_inference_to_server(best_match, float(confidence), latency, threshold)

        # Bebaskan memori secara eksplisit untuk mencegah memory leakage
        img_pil.close()
        del face_tensor, input_tensor, face_flipped, emb_orig, emb_mirror, query_embedding_tensor, mean_embedding_tensor
        gc.collect()

        return {
            "matched": user_id if user_id != "Unknown" else "Unknown", 
            "is_confirmed": user_id != "Unknown",
            "confidence": float(confidence), 
            "box": box.tolist() if box is not None else None,
            "latency_ms": latency,
            "model_version": self.fl_manager.model_version
        }

    def recognize_directly(self, img_pil):
        # Inferensi langsung dari frame kamera
        target_backbone = getattr(self.fl_manager, 'inference_backbone', self.fl_manager.backbone)
        if target_backbone is None: return "Unknown", 0.0
        
        try:
            mtcnn_start = time.perf_counter()
            face_tensor, _, _ = image_processor.detect_face(img_pil)
            mtcnn_time = int((time.perf_counter() - mtcnn_start) * 1000)
            if face_tensor is None: return "Unknown", 0.0
            
            threshold = self.fl_manager.inference_threshold

            input_tensor = image_processor.prepare_for_model(face_tensor).to(self.fl_manager.device)
            backbone_start = time.perf_counter()
            with torch.no_grad():
                target_backbone.eval()
                
                # Ekstraksi embedding citra wajah asli
                emb_orig = target_backbone(input_tensor)
                
                # Ekstraksi embedding citra wajah cermin horizontal
                face_flipped = torch.flip(input_tensor, dims=[3])
                emb_mirror = target_backbone(face_flipped)
                
                # Rata-rata dan normalisasi akhir ke unit vector
                query_embedding_tensor = torch.nn.functional.normalize((emb_orig + emb_mirror) / 2, p=2, dim=1)
            backbone_time = int((time.perf_counter() - backbone_start) * 1000)
                
            # Logika temporal voting dengan buffer frame
            ref_start = time.perf_counter()
            local_refs = getattr(self.fl_manager, 'cached_refs', {})
            if not local_refs:
                db = SessionLocal()
                try:
                    local_refs = self._get_cached_identities(db)
                finally:
                    db.close()

            now = time.time()
            if now - self.fl_manager.last_face_time > 3.0:
                self.fl_manager.prediction_buffer.clear()
            self.fl_manager.prediction_buffer.append(query_embedding_tensor)
            self.fl_manager.last_face_time = now
            
            mean_embedding_tensor = torch.stack(list(self.fl_manager.prediction_buffer)).mean(0)
            mean_embedding_tensor = torch.nn.functional.normalize(mean_embedding_tensor, p=2, dim=1)
            mean_embedding = mean_embedding_tensor.cpu().numpy()[0]
            best_match, confidence = identify_user_globally(
                mean_embedding, 
                local_refs, 
                threshold=threshold
            )
            matching_time = int((time.perf_counter() - ref_start) * 1000)
            
            # Tentukan status akhir untuk antarmuka pengguna
            matched = best_match if confidence >= threshold else "Unknown"
            
            latency = int((time.perf_counter() - mtcnn_start) * 1000)
            
            # Kirim log statistik asinkron ke server
            self._log_inference_to_server(best_match, float(confidence), latency, threshold)

            # Log performa detail per langkah
            self.fl_manager.logger.info(
                f"[PERF-DIRECT] MTCNN: {mtcnn_time}ms | Backbone: {backbone_time}ms | Search: {matching_time}ms | Total Latency: {latency}ms"
            )

            # Bebaskan memori secara eksplisit untuk mencegah memory leakage
            del face_tensor, input_tensor, face_flipped, emb_orig, emb_mirror, query_embedding_tensor, mean_embedding_tensor
            gc.collect()

            return matched, float(confidence)
        except Exception as e:
            self.fl_manager.logger.error(f"Kegagalan sistem pengenalan wajah: {e}")
            return "Unknown", 0.0

    def _log_inference_to_server(self, user_id, confidence, latency_ms, threshold):
        # Mengirim data inferensi mentah ke server untuk pemantauan terpusat
        try:
            server_url = getattr(self.fl_manager, 'server_api_url', "http://server-fl:8080")
            
            # Pastikan hanya NRP yang dikirim
            nrp_only = user_id.split("_")[0] if "_" in user_id else user_id
            
            # Tentukan status berdasarkan ambang batas keputusan
            status = "KNOWN" if confidence >= threshold else "UNKNOWN"

            payload = {
                "client_id": os.getenv("HOSTNAME", "client-1"),
                "user_id": nrp_only,
                "confidence": float(confidence),
                "latency_ms": int(latency_ms),
                "status": status
            }
            queue_inference_log(server_url, payload, self.fl_manager.logger)
        except Exception as e:
            self.fl_manager.logger.error(f"[CIM] Gagal mengantrekan log inferensi ke queue: {e}")

    def _get_cached_identities(self, db):
        # Memperbarui cache identitas dari database lokal dan registri global server
        if not hasattr(self.fl_manager, 'cached_refs') or time.time() - getattr(self.fl_manager, 'last_cache_update', 0) > 30:
            local_refs = {}
            self.fl_manager.logger.info("Memperbarui cache identitas dari database dan registri global...")

            # Identitas mahasiswa global dari database sinkronisasi sebelumnya
            try:
                global_db_embs = db.query(EmbeddingLocal).filter_by(is_global=True).all()
                db_global_count = 0
                for emb in global_db_embs:
                    try:
                        dec_emb = np.frombuffer(emb.embedding_data, dtype=np.float32).copy()
                        v = torch.from_numpy(dec_emb).float()
                        v = torch.nn.functional.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)
                        local_refs[emb.user_id] = v.to(self.fl_manager.device)
                        db_global_count += 1
                    except Exception as e:
                        self.fl_manager.logger.error(f"Dekripsi global {emb.user_id}: {e}")
                if db_global_count > 0:
                    self.fl_manager.logger.info(f"Metrik: {db_global_count} embedding lintas klien dari database global.")
            except Exception as e:
                self.fl_manager.logger.error(f"Kueri database global: {e}")

            # Registri embedding wajah global terbaru dari server
            local_user_ids = set()
            try:
                local_ids_res = db.query(EmbeddingLocal.user_id).filter_by(is_global=False).all()
                local_user_ids = {row[0] for row in local_ids_res}
            except Exception: pass

            version = getattr(self.fl_manager, 'model_version', 0)
            if version and version != "v0" and version != "0" and version != 0:
                registry_path = os.path.join(self.fl_manager.data_path, "models", f"registry_embeddings_{version}.pth")
            else:
                registry_path = os.path.join(self.fl_manager.data_path, "models", "global_embedding_registry.pth")
                
            if not os.path.exists(registry_path):
                registry_path = os.path.join(self.fl_manager.data_path, "models", "global_embedding_registry.pth")
                
            if os.path.exists(registry_path):
                try:
                    registry = torch.load(registry_path, map_location="cpu")
                    reg_count = 0
                    for nrp, vec in registry.items():
                        v = torch.from_numpy(np.array(vec, dtype=np.float32).copy()) if not isinstance(vec, torch.Tensor) else vec.float()
                        v = torch.nn.functional.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)
                        local_refs[nrp] = v.to(self.fl_manager.device)
                        reg_count += 1
                    self.fl_manager.logger.info(f"Metrik: {reg_count} embedding lintas klien dari berkas registri global.")
                except Exception as e:
                    self.fl_manager.logger.error(f"Gagal membaca registri global: {e}")
            else:
                self.fl_manager.logger.info("Registri global belum tersedia di perangkat.")

            # Embedding wajah lokal dari database edge untuk mahasiswa terdaftar baru
            try:
                local_db_embs = db.query(EmbeddingLocal).filter_by(is_global=False).all()
                db_local_count = 0
                for emb in local_db_embs:
                    if emb.user_id in local_refs:
                        continue 
                    try:
                        dec_emb = encryptor.decrypt_embedding(emb.embedding_data, emb.iv).copy()
                        v = torch.from_numpy(dec_emb).float()
                        v = torch.nn.functional.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)
                        local_refs[emb.user_id] = v.to(self.fl_manager.device)
                        db_local_count += 1
                    except Exception as e:
                        self.fl_manager.logger.error(f"Dekripsi lokal {emb.user_id}: {e}")
                self.fl_manager.logger.info(f"Metrik: {db_local_count} embedding lokal dari database perangkat.")
            except Exception as e:
                self.fl_manager.logger.error(f"Kueri database lokal: {e}")

            self.fl_manager.logger.info(f"Total referensi wajah: {len(local_refs)} identitas.")
            self.fl_manager.cached_refs = local_refs
            self.fl_manager.last_cache_update = time.time()

        return self.fl_manager.cached_refs

    def _ensure_user_and_get_name(self, user_id, db):
        user = db.query(UserLocal).filter_by(user_id=user_id).first()
        if not user:
            user = UserLocal(user_id=user_id, name="Global Student")
            db.add(user)
            db.flush()
        return user.name
