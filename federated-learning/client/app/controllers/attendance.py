import os
import time
import base64
import io
import torch
import numpy as np
import traceback
from PIL import Image
from sqlalchemy.orm import Session

from app.db.models import UserLocal, EmbeddingLocal, AttendanceLocal
from app.utils.preprocessing import image_processor
from app.utils.classifier import identify_user_globally
from app.utils.security import encryptor
from app.utils.sync_utils import sync_record_to_server

from app.db.db import SessionLocal

class AttendanceController:
    # Kontroler untuk proses Pengenalan Wajah dan Pencatatan Presensi.
    # Menangani alur inferensi dari gambar kamera hingga pelaporan ke server.
    
    def __init__(self, fl_manager):
        self.fl_manager = fl_manager
        self._last_version_loaded = -1 # Melacak versi model yang aktif di session

    async def process_inference(self, image_b64: str, db: Session, background_tasks):
        start_time = time.perf_counter()
        
        # Dekode gambar dari base64
        img_bytes = base64.b64decode(image_b64)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Deteksi + Crop wajah menggunakan MTCNN
        face_tensor, box, prob = image_processor.detect_face(img_pil)
        if face_tensor is None:
            return {"matched": "Unknown", "confidence": 0, "message": "Wajah tidak terdeteksi"}
        
        # Preprocessing: Squash ke 112x96 dan normalisasi [-1, 1]
        input_tensor = image_processor.prepare_for_model(face_tensor).to(self.fl_manager.device)
        
        current_v = getattr(self.fl_manager, 'model_version', 0)

        # --- INFERENSI PURE PYTORCH (Optimasi Edge) ---
        try:
            if current_v != self._last_version_loaded:
                self.fl_manager.logger.info(f"Menggunakan model PyTorch v{current_v}. Input: {input_tensor.shape}")
                self._last_version_loaded = current_v

            with torch.no_grad():
                # Gunakan inference_backbone agar stabil terhadap drift ronde
                target_backbone = getattr(self.fl_manager, 'inference_backbone', self.fl_manager.backbone)
                if target_backbone is None: return {"matched": "Error", "confidence": 0, "message": "Model not loaded"}
                
                target_backbone.eval()
                
                # --- FLIP TRICK (Alignment dengan Registry) ---
                # 1. Embedding Citra Asli
                emb_orig = target_backbone(input_tensor)
                
                # 2. Embedding Citra Mirror (Horizontal Flip)
                face_flipped = torch.flip(input_tensor, dims=[3])
                emb_mirror = target_backbone(face_flipped)
                
                # 3. Rata-rata dan Normalisasi
                query_embedding_tensor = torch.nn.functional.normalize((emb_orig + emb_mirror) / 2, p=2, dim=1)
        except Exception as e:
            self.fl_manager.logger.error(f"Kegagalan inferensi: {e}")
            return {"matched": "Error", "confidence": 0, "message": str(e)}
            
        # Manajemen Cache Identitas
        local_refs = self._get_cached_identities(db)

        # --- LOGIKA TEMPORAL VOTING (Sederhana) ---
        # 1. Update Buffer Temporal
        now = time.time()
        if now - self.fl_manager.last_face_time > 1.0:
            self.fl_manager.prediction_buffer.clear()
            
        self.fl_manager.prediction_buffer.append(query_embedding_tensor)
        self.fl_manager.last_face_time = now
        
        # 2. Rata-rata temporal (Voting)
        mean_embedding_tensor = torch.stack(list(self.fl_manager.prediction_buffer)).mean(0)
        mean_embedding_tensor = torch.nn.functional.normalize(mean_embedding_tensor, p=2, dim=1)
        mean_embedding = mean_embedding_tensor.cpu().numpy()[0]
        
        # 3. Pencocokan identitas tunggal
        user_id, confidence = identify_user_globally(mean_embedding, local_refs, threshold=self.fl_manager.inference_threshold)
        
        # Pencatatan Absensi
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
                
                latency = int((time.perf_counter() - start_time) * 1000)
                background_tasks.add_task(
                    sync_record_to_server, 
                    user_id, user_name, float(confidence), os.getenv("HOSTNAME", "client-1"),
                    latency # Kirim latency asli ke fungsi sync
                )
                self.fl_manager.logger.success(f"Wajah Terverifikasi: {user_id} (Sim: {confidence:.4f})")
            except Exception as e:
                db.rollback()
                self.fl_manager.logger.error(f"Gagal mencatat presensi {user_id} ke DB lokal: {e}")

        else:
            if confidence > 0.1: 
                self.fl_manager.logger.info(f"Model v{current_v} | Kecocokan: Unknown | Sim: {confidence:.4f} | Thres: {self.fl_manager.inference_threshold}")
            
        latency = int((time.perf_counter() - start_time) * 1000)
        
        # --- NEW: Log Inferensi ke Server untuk Riset (FAR/TAR) ---
        import threading
        threading.Thread(target=self._log_inference_to_server, args=(user_id, float(confidence), latency)).start()

        return {
            "matched": user_id if user_id != "Unknown" else "Unknown", 
            "is_confirmed": user_id != "Unknown",
            "confidence": float(confidence), 
            "box": box.tolist() if box is not None else None,
            "latency_ms": latency,
            "model_version": self.fl_manager.model_version
        }

    def recognize_directly(self, img_pil):
        # Inferensi langsung dari frame kamera (Optimasi Edge)
        # Gunakan inference_backbone agar stabil terhadap drift ronde
        target_backbone = getattr(self.fl_manager, 'inference_backbone', self.fl_manager.backbone)
        if target_backbone is None: return "Unknown", 0.0
        
        try:
            face_tensor, _, _ = image_processor.detect_face(img_pil)
            if face_tensor is None: return "Unknown", 0.0
            
            input_tensor = image_processor.prepare_for_model(face_tensor).to(self.fl_manager.device)
            with torch.no_grad():
                target_backbone.eval()
                
                # --- FLIP TRICK (Alignment dengan Registry) ---
                # 1. Embedding Citra Asli
                emb_orig = target_backbone(input_tensor)
                
                # 2. Embedding Citra Mirror (Horizontal Flip)
                face_flipped = torch.flip(input_tensor, dims=[3])
                emb_mirror = target_backbone(face_flipped)
                
                # 3. Rata-rata dan Normalisasi
                query_embedding_tensor = torch.nn.functional.normalize((emb_orig + emb_mirror) / 2, p=2, dim=1)
                
            # --- LOGIKA TEMPORAL VOTING (Sederhana) ---
            local_refs = getattr(self.fl_manager, 'cached_refs', {})
            if not local_refs:
                
                db = SessionLocal()
                try:
                    local_refs = self._get_cached_identities(db)
                finally:
                    db.close()

            # Temporal Voting
            now = time.time()
            if now - self.fl_manager.last_face_time > 1.0:
                self.fl_manager.prediction_buffer.clear()
            self.fl_manager.prediction_buffer.append(query_embedding_tensor)
            self.fl_manager.last_face_time = now
            
            mean_embedding_tensor = torch.stack(list(self.fl_manager.prediction_buffer)).mean(0)
            mean_embedding_tensor = torch.nn.functional.normalize(mean_embedding_tensor, p=2, dim=1)
            mean_embedding = mean_embedding_tensor.cpu().numpy()[0]
            
            matched, confidence = identify_user_globally(
                mean_embedding, 
                local_refs, 
                threshold=self.fl_manager.inference_threshold
            )
            
            latency = int((time.time() - now) * 1000)
            
            # Log ke server secara background (Gunakan thread/requests simple)
            import threading
            threading.Thread(target=self._log_inference_to_server, args=(matched, float(confidence), latency)).start()

            return matched, float(confidence)
        except Exception as e:
            self.fl_manager.logger.error(f"Kegagalan sistem pengenalan wajah: {e}")
            return "Unknown", 0.0

    def _log_inference_to_server(self, user_id, confidence, latency_ms):
        """Mengirim data inferensi mentah ke server untuk pemantauan terpusat."""
        import requests
        try:
            server_url = getattr(self.fl_manager, 'server_api_url', "http://server-fl:8080")
            
            # Pastikan hanya NRP yang dikirim (potong nama jika ada)
            nrp_only = user_id.split("_")[0] if "_" in user_id else user_id
            
            # Tentukan status berdasarkan threshold sistem
            threshold = getattr(self.fl_manager, 'inference_threshold', 0.7)
            status = "KNOWN" if confidence >= threshold else "UNKNOWN"

            payload = {
                "client_id": os.getenv("HOSTNAME", "client-1"),
                "user_id": nrp_only,
                "confidence": float(confidence),
                "latency_ms": int(latency_ms),
                "status": status
            }
            requests.post(f"{server_url}/api/logs/inference", json=payload, timeout=2)
        except Exception as e:
            self.fl_manager.logger.error(f"[CIM] Gagal mengirim log inferensi ke server: {e}")

    def _get_cached_identities(self, db):
        # Memperbarui cache identitas (Setiap 30 detik atau jika data kosong)
        if not hasattr(self.fl_manager, 'cached_refs') or time.time() - getattr(self.fl_manager, 'last_cache_update', 0) > 30:
            local_refs = {}
            self.fl_manager.logger.info("Memperbarui cache identitas dari database dan registri global...")

            # ── TIER 1 (Terendah): Identitas cross-client dari DB (is_global=True) ──
            # Ini adalah fallback stale dari sesi sebelumnya. Bisa di-override oleh tier 2 dan 3.
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
                    self.fl_manager.logger.info(f"Tier 1: {db_global_count} embedding cross-client (DB global).")
            except Exception as e:
                self.fl_manager.logger.error(f"Query DB global: {e}")

            # ── TIER 2 (Sedang): global_embedding_registry.pth dari server ──
            # Dihitung dengan inference_backbone terkini → lebih akurat dari DB global stale.
            # Override SEMUA entri dari Tier 1, kecuali mahasiswa LOKAL (Tier 3 yang akan menang).
            # Lacak dulu siapa mahasiswa lokal:
            local_user_ids = set()
            try:
                local_ids_res = db.query(EmbeddingLocal.user_id).filter_by(is_global=False).all()
                local_user_ids = {row[0] for row in local_ids_res}
            except Exception: pass

            registry_path = os.path.join(self.fl_manager.data_path, "models", "global_embedding_registry.pth")
            if os.path.exists(registry_path):
                try:
                    registry = torch.load(registry_path, map_location="cpu")
                    reg_count = 0
                    for nrp, vec in registry.items():
                        # PENGUMUMAN PENTING: Jangan skip user lokal. 
                        # Gunakan embedding Global yang lebih robust (hasil agregasi pFedFace)
                        # biarpun mahasiswa tersebut ada di database lokal perangkat ini.
                        # Ini untuk menghindari 'Local Overfitting' (Skor rendah 0.3).
                        v = torch.from_numpy(np.array(vec, dtype=np.float32).copy()) if not isinstance(vec, torch.Tensor) else vec.float()
                        v = torch.nn.functional.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)
                        local_refs[nrp] = v.to(self.fl_manager.device)  # Override Tier 1 stale
                        reg_count += 1
                    self.fl_manager.logger.info(f"Tier 2: {reg_count} embedding cross-client dari registry.pth (fresh).")
                except Exception as e:
                    self.fl_manager.logger.error(f"Registry.pth: {e}")
            else:
                self.fl_manager.logger.info("Registry global belum tersedia (hanya Tier 1).")

            # ── TIER 3 (Tertinggi): Embedding LOKAL dari DB (is_global=False) ──
            # Dihasilkan oleh refresh_local_embeddings dengan backbone inferensi terkini.
            # Override Tier 1 dan 2 untuk semua mahasiswa lokal.
            try:
                local_db_embs = db.query(EmbeddingLocal).filter_by(is_global=False).all()
                db_local_count = 0
                for emb in local_db_embs:
                    if emb.user_id in local_refs:
                        # Jika sudah ada di Registry (Tier 2), gunakan versi global (Robust).
                        # Versi lokal (Tier 3) hanya digunakan jika mahasiswa benar-benar baru
                        # dan belum sempat ter-agregasi ke server (Fase awal).
                        continue 
                    try:
                        dec_emb = encryptor.decrypt_embedding(emb.embedding_data, emb.iv).copy()
                        v = torch.from_numpy(dec_emb).float()
                        v = torch.nn.functional.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)
                        local_refs[emb.user_id] = v.to(self.fl_manager.device)
                        db_local_count += 1
                    except Exception as e:
                        self.fl_manager.logger.error(f"Dekripsi lokal {emb.user_id}: {e}")
                self.fl_manager.logger.info(f"Tier 3: {db_local_count} embedding lokal baru (identitas baru).")
            except Exception as e:
                self.fl_manager.logger.error(f"Query DB lokal: {e}")

            self.fl_manager.logger.info(f"Total refs: {len(local_refs)} identitas.")
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
