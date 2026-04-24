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

class AttendanceController:
    # Kontroler untuk proses Pengenalan Wajah dan Pencatatan Presensi.
    # Menangani alur inferensi dari gambar kamera hingga pelaporan ke server.
    
    def __init__(self, fl_manager):
        self.fl_manager = fl_manager
        self._last_version_loaded = -1 # Melacak versi model yang aktif di session

    async def process_inference(self, image_b64: str, db: Session, background_tasks):
        start_time = time.time()
        
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
                print(f"[DEBUG] Menggunakan model PyTorch v{current_v}. Input: {input_tensor.shape}")
                self._last_version_loaded = current_v

            with torch.no_grad():
                # Gunakan inference_backbone agar stabil terhadap drift ronde
                target_backbone = getattr(self.fl_manager, 'inference_backbone', self.fl_manager.backbone)
                if target_backbone is None: return {"matched": "Error", "confidence": 0, "message": "Model not loaded"}
                
                target_backbone.eval()
                query_embedding_tensor = target_backbone(input_tensor)
                query_embedding_tensor = torch.nn.functional.normalize(query_embedding_tensor, p=2, dim=1)
        except Exception as e:
            print(f"[ERROR] Kegagalan inferensi: {e}")
            return {"matched": "Error", "confidence": 0, "message": str(e)}
            
        # Manajemen Cache Identitas
        local_refs = self._get_cached_identities(db)

        # Proses Voting Temporal
        now = time.time()
        if now - self.fl_manager.last_face_time > 1.0:
            self.fl_manager.prediction_buffer.clear()
            
        self.fl_manager.prediction_buffer.append(query_embedding_tensor)
        self.fl_manager.last_face_time = now
        
        mean_embedding_tensor = torch.stack(list(self.fl_manager.prediction_buffer)).mean(0)
        mean_embedding_tensor = torch.nn.functional.normalize(mean_embedding_tensor, p=2, dim=1)
        mean_embedding = mean_embedding_tensor.cpu().numpy()[0]
        
        # Pencocokan identitas dengan threshold produksi
        user_id, confidence = identify_user_globally(mean_embedding, local_refs, threshold=self.fl_manager.inference_threshold)
        
        # Pencatatan Absensi
        if user_id != "Unknown":
            user_name = self._ensure_user_and_get_name(user_id, db)
            
            new_attendance = AttendanceLocal(
                user_id=user_id,
                confidence=confidence,
                device_id=os.getenv("HOSTNAME", "terminal-1")
            )
            db.add(new_attendance)
            db.commit()
            
            background_tasks.add_task(
                sync_record_to_server, 
                user_id, user_name, float(confidence), os.getenv("HOSTNAME", "client-1")
            )
            print(f"[SUCCESS] Terdeteksi (Voted): {user_id} (Sim: {confidence:.4f})")
        else:
            if confidence > 0.1: 
                print(f"[DEBUG] Model v{current_v} | Kecocokan: Unknown | Sim: {confidence:.4f} | Thres: {self.fl_manager.inference_threshold}")
            
        latency = int((time.time() - start_time) * 1000)
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
                query_embedding_tensor = target_backbone(input_tensor)
                query_embedding_tensor = torch.nn.functional.normalize(query_embedding_tensor, p=2, dim=1)
                
            # Temporal Voting
            now = time.time()
            if now - self.fl_manager.last_face_time > 1.0:
                self.fl_manager.prediction_buffer.clear()
            self.fl_manager.prediction_buffer.append(query_embedding_tensor)
            self.fl_manager.last_face_time = now
            
            mean_embedding_tensor = torch.stack(list(self.fl_manager.prediction_buffer)).mean(0)
            mean_embedding_tensor = torch.nn.functional.normalize(mean_embedding_tensor, p=2, dim=1)
            mean_embedding = mean_embedding_tensor.cpu().numpy()[0]
            
            local_refs = getattr(self.fl_manager, 'cached_refs', {})
            if not local_refs:
                from app.db.db import SessionLocal
                db = SessionLocal()
                try:
                    local_refs = self._get_cached_identities(db)
                finally:
                    db.close()

            matched, confidence = identify_user_globally(
                mean_embedding, 
                local_refs, 
                threshold=self.fl_manager.inference_threshold
            )
            return matched, float(confidence)
        except Exception as e:
            print(f"[ERROR] Kegagalan pengenalan wajah: {e}")
            return "Unknown", 0.0

    def _get_cached_identities(self, db):
        # Memperbarui cache identitas (Setiap 30 detik atau jika data kosong)
        if not hasattr(self.fl_manager, 'cached_refs') or time.time() - getattr(self.fl_manager, 'last_cache_update', 0) > 30:
            local_refs = {}
            print(f"[ATTENDANCE] Memperbarui cache identitas dari database dan registri global...")
            
            # 1. Memuat identitas dari database lokal
            try:
                embeddings = db.query(EmbeddingLocal).all()
                db_count = 0
                for emb in embeddings:
                    try:
                        # Prioritas: Identitas global di DB (is_global=True) dimuat terakhir agar menimpa lokal jika ada konflik
                        dec_emb = np.frombuffer(emb.embedding_data, dtype=np.float32).copy() if emb.is_global else encryptor.decrypt_embedding(emb.embedding_data, emb.iv).copy()
                        v = torch.from_numpy(dec_emb).float()
                        v = torch.nn.functional.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)
                        local_refs[emb.user_id] = v.to(self.fl_manager.device)
                        db_count += 1
                    except Exception as e: 
                        print(f"[ATTENDANCE ERROR] Gagal dekripsi embedding user {emb.user_id}: {e}")
                        continue
                print(f"[ATTENDANCE] Berhasil memuat {db_count} identitas dari database.")
            except Exception as db_err:
                print(f"[ATTENDANCE ERROR] Gagal query database: {db_err}")

            # 2. PUSTAKA GLOBAL (SUMBER KEBENARAN UTAMA / Centroids)
            # Dimuat SETELAH database agar menimpa identitas lama dengan centroid terbaru dari sinkronisasi global
            registry_path = os.path.join(self.fl_manager.data_path, "models", "global_embedding_registry.pth")
            if os.path.exists(registry_path):
                try:
                    registry = torch.load(registry_path, map_location="cpu")
                    for nrp, vec in registry.items():
                        v = torch.from_numpy(np.array(vec, dtype=np.float32).copy()) if not isinstance(vec, torch.Tensor) else vec.float()
                        v = torch.nn.functional.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)
                        local_refs[nrp] = v.to(self.fl_manager.device)
                    print(f"[ATTENDANCE] [SUCCESS] Registri global berhasil dimuat dengan {len(registry)} identitas.")
                except Exception as e:
                    print(f"[ATTENDANCE ERROR] Gagal memuat berkas registry.pth: {e}")
            else:
                print(f"[ATTENDANCE] Registry global ({registry_path}) belum tersedia.")
            
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
