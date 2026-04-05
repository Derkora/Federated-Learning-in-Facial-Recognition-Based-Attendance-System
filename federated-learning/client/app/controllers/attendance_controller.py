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
    # Logika Utama Pengenalan Wajah dan Absensi
    # Bagian ini menangani alur dari penerimaan gambar, deteksi wajah,
    # ekstraksi ciri (embedding), hingga pencocokan dengan database global.
    
    def __init__(self, fl_manager):
        self.fl_manager = fl_manager

    async def process_inference(self, image_b64: str, db: Session, background_tasks):
        start_time = time.time()
        
        # Dekode gambar dari base64
        img_bytes = base64.b64decode(image_b64)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Deteksi wajah menggunakan MTCNN
        face_tensor, box, prob = image_processor.detect_face(img_pil)
        if face_tensor is None:
            return {"matched": "Unknown", "confidence": 0, "message": "Wajah tidak terdeteksi"}
        
        # Preprocessing dan ekstraksi embedding menggunakan model backbone (MobileFaceNet)
        input_tensor = image_processor.prepare_for_model(face_tensor).to(self.fl_manager.device)
        
        with torch.no_grad():
            self.fl_manager.backbone.eval()
            query_embedding_tensor = self.fl_manager.backbone(input_tensor)
            # Normalisasi L2 adalah standar untuk perbandingan vektor wajah (Cosine Similarity)
            query_embedding_tensor = torch.nn.functional.normalize(query_embedding_tensor, p=2, dim=1)
            query_embedding = query_embedding_tensor.cpu().numpy()[0]
            
        # Manajemen Cache Identitas
        # Library pengenal (Registry) diperbarui secara berkala dari server untuk memastikan
        # terminal memiliki data terbaru hasil pelatihan federated learning.
        local_refs = self._get_cached_identities(db)

        # Proses Voting Temporal
        # Menggunakan rata-rata dari beberapa frame terakhir untuk meningkatkan stabilitas
        # dan akurasi identifikasi saat wajah bergerak atau pencahayaan berubah.
        now = time.time()
        if now - self.fl_manager.last_face_time > 1.0:
            self.fl_manager.prediction_buffer.clear()
            
        self.fl_manager.prediction_buffer.append(query_embedding_tensor)
        self.fl_manager.last_face_time = now
        
        mean_embedding_tensor = torch.stack(list(self.fl_manager.prediction_buffer)).mean(0)
        mean_embedding_tensor = torch.nn.functional.normalize(mean_embedding_tensor, p=2, dim=1)
        mean_embedding = mean_embedding_tensor.cpu().numpy()[0]
        
        # Pencocokan identitas dengan threshold produksi
        user_id, confidence = identify_user_globally(mean_embedding, local_refs, threshold=0.50)
        
        # Pencatatan Absensi
        # Jika mahasiswa dikenali, data akan disimpan di database lokal dan dikirim ke server.
        # Jika mahasiswa berasal dari terminal lain, entri placeholder akan dibuat otomatis.
        if user_id != "Unknown":
            user_name = self._ensure_user_and_get_name(user_id, db)
            
            new_attendance = AttendanceLocal(
                user_id=user_id,
                confidence=confidence,
                device_id=os.getenv("HOSTNAME", "terminal-1")
            )
            db.add(new_attendance)
            db.commit()
            
            # Sinkronisasi ke server dilakukan di background agar tidak menghambat aliran video
            background_tasks.add_task(
                sync_record_to_server, 
                user_id, user_name, float(confidence), os.getenv("HOSTNAME", "terminal-1")
            )
            print(f"[OK] Absensi Berhasil: {user_id} ({confidence:.2f})")
            
        latency = int((time.time() - start_time) * 1000)
        return {
            "matched": user_id, 
            "is_confirmed": user_id != "Unknown",
            "confidence": float(confidence), 
            "box": box.tolist() if box is not None else None,
            "latency_ms": latency,
            "model_version": self.fl_manager.model_version
        }

    def recognize_directly(self, img_pil):
        # Memproses pengenalan secara langsung tanpa melalui API/DB
        try:
            matched, confidence, _ = identify_user_globally(
                img_pil, self.fl_manager.backbone, self.fl_manager.detector, 
                self.fl_manager.data_path, self.fl_manager.device
            )
            return matched, float(confidence)
        except:
            return "Unknown", 0.0

    def process_registration(self, user_id: str, name: str, image_b64: str, db: Session):
        # Registrasi Mahasiswa Baru
        # Data wajah awal diambil dan dikonversi menjadi embedding untuk diarsip secara lokal
        # dan dikirim ke server sebagai dataset awal pelatihan mandiri (training data).
        
        try:
            # Simpan data user dasar
            new_user = UserLocal(user_id=user_id, name=name)
            db.add(new_user)
            db.commit()

            # Simpan file gambar asli untuk audit/pelatihan ulang
            img_bytes = base64.b64decode(image_b64)
            img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            user_dir = os.path.join(self.fl_manager.data_path, name)
            os.makedirs(user_dir, exist_ok=True)
            target_path = os.path.join(user_dir, f"{int(time.time())}.jpg")
            img_pil.save(target_path)
            
            # Ekstraksi embedding awal
            face_tensor, _, _ = image_processor.detect_face(img_pil)
            if face_tensor is not None:
                input_tensor = image_processor.prepare_for_model(face_tensor)
                with torch.no_grad():
                    self.fl_manager.backbone.eval()
                    embedding = self.fl_manager.backbone(input_tensor)
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                    embedding_np = embedding.cpu().numpy()[0]
                    
                    # Enkripsi dan simpan embedding secara lokal
                    encrypted_data, iv = encryptor.encrypt_embedding(embedding_np)
                    new_emb = EmbeddingLocal(user_id=user_id, embedding_data=encrypted_data, iv=iv, is_global=False)
                    db.add(new_emb)
                    db.commit()
                
                # Kirim data identitas ke server pusat (Label Matching)
                self._signal_server_new_identity(user_id, name, embedding_np)
                
            return {"status": "success", "user_id": user_id}
            
        except Exception as e:
            db.rollback()
            print(f"[ERROR] Registrasi Gagal: {e}")
            raise e

    def _get_cached_identities(self, db):
        # Refresh cache identitas setiap 30 detik
        if not hasattr(self.fl_manager, 'cached_refs') or time.time() - getattr(self.fl_manager, 'last_cache_update', 0) > 30:
            local_refs = {}
            
            # Prioritas 1: Menggunakan Global Registry hasil Federated Learning
            registry_path = os.path.join(self.fl_manager.data_path, "models", "global_embedding_registry.pth")
            if os.path.exists(registry_path):
                try:
                    registry = torch.load(registry_path, map_location="cpu")
                    for nrp, vec in registry.items():
                        if isinstance(vec, torch.Tensor):
                            local_refs[nrp] = vec.to(self.fl_manager.device)
                        else:
                            local_refs[nrp] = torch.from_numpy(np.array(vec).copy()).to(self.fl_manager.device)
                    print(f"[OK] Memuat {len(local_refs)} identitas global dari registry.")
                except Exception as e:
                    print(f"[ERROR] Gagal memuat registry global: {e}")
            
            # Prioritas 2: Fallback ke database lokal jika registry belum tersedia
            if not local_refs:
                embeddings = db.query(EmbeddingLocal).all()
                for emb in embeddings:
                    if emb.user_id in local_refs: continue 
                    try:
                        if emb.is_global:
                            dec_emb = np.frombuffer(emb.embedding_data, dtype=np.float32).copy()
                        else:
                            dec_emb = encryptor.decrypt_embedding(emb.embedding_data, emb.iv).copy()
                        local_refs[emb.user_id] = torch.from_numpy(dec_emb).to(self.fl_manager.device)
                    except: continue
                if local_refs:
                    print(f"[OK] Menggunakan {len(local_refs)} identitas lokal (fallback).")
            
            self.fl_manager.cached_refs = local_refs
            self.fl_manager.last_cache_update = time.time()
            
        return self.fl_manager.cached_refs

    def _ensure_user_and_get_name(self, user_id, db):
        # Pastikan user ada di DB lokal (Global Identity handling)
        user = db.query(UserLocal).filter_by(user_id=user_id).first()
        if not user:
            print(f"[OK] Membuat entri placeholder untuk identitas global: {user_id}")
            user = UserLocal(user_id=user_id, name="Global Student")
            db.add(user)
            db.flush()
        return user.name

    def _signal_server_new_identity(self, user_id, name, embedding_np):
        server_url = os.getenv("SERVER_API_URL", "http://server-fl:8080")
        try:
            embedding_b64 = base64.b64encode(embedding_np.tobytes()).decode('utf-8')
            requests.post(f"{server_url}/api/training/get_label", json={
                "nrp": user_id,
                "name": name,
                "client_id": os.getenv("HOSTNAME", "terminal-1"),
                "embedding": embedding_b64
            }, timeout=5)
        except Exception as e:
            print(f"[ERROR] Gagal sinkronisasi identitas baru ke server: {e}")
