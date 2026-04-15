import os
import time
import base64
import io
import torch
import numpy as np
import traceback
import onnxruntime as ort
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
        self._last_version_loaded = -1 # Melacak versi model yang aktif di session

    async def process_inference(self, image_b64: str, db: Session, background_tasks):
        start_time = time.time()
        
        # Dekode gambar dari base64
        img_bytes = base64.b64decode(image_b64)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Deteksi + Crop wajah menggunakan MTCNN
        # Pipeline ini KONSISTEN dengan gambar training di data/processed/:
        # MTCNN -> tensor -> prepare_for_model(squash ke 112x96)
        face_tensor, box, prob = image_processor.detect_face(img_pil)
        if face_tensor is None:
            return {"matched": "Unknown", "confidence": 0, "message": "Wajah tidak terdeteksi"}
        
        input_tensor = image_processor.prepare_for_model(face_tensor)
        input_np = input_tensor.cpu().numpy()
        
        # ONNX akan diaktifkan kembali hanya setelah export ulang setelah training selesai.
        # PRIORITAS: backbone.onnx (FP32) untuk menjaga akurasi (~0.8 similarity)
        onnx_dir = os.path.join(self.fl_manager.data_path, "models")
        onnx_path = os.path.join(onnx_dir, "backbone.onnx")
        torch_path = os.path.join(onnx_dir, "backbone.pth")
        
        if not os.path.exists(onnx_path):
            onnx_path = os.path.join(onnx_dir, "backbone_quantized.onnx")
        
        # Gunakan ONNX HANYA jika backbone.pth juga ada DAN versi ONNX sinkron
        onnx_is_fresh = (
            os.path.exists(onnx_path) and
            os.path.exists(torch_path) and
            os.path.getmtime(onnx_path) >= os.path.getmtime(torch_path)
        )
        
        current_v = getattr(self.fl_manager, 'model_version', 0)

        if onnx_is_fresh:
            # --- PROTEKSI SINKRONISASI: Reload session jika versi berubah ---
            if current_v != self._last_version_loaded:
                if hasattr(self, 'ort_session'):
                    print(f"[SYNC-FL] Reloading ONNX session for Version {current_v}...")
                    del self.ort_session
                self._last_version_loaded = current_v

            try:
                if not hasattr(self, 'ort_session'):
                    self.ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                
                outputs = self.ort_session.run(None, {'input': input_np})
                query_embedding_tensor = torch.from_numpy(outputs[0][0]).view(1, -1)
                query_embedding_tensor = torch.nn.functional.normalize(query_embedding_tensor, p=2, dim=1)
            except Exception as e:
                print(f"[ONNX ERROR] Fallback to Torch: {e}")
                if hasattr(self, 'ort_session'): del self.ort_session
                input_tensor = input_tensor.to(self.fl_manager.device)
                with torch.no_grad():
                    self.fl_manager.backbone.eval()
                    query_embedding_tensor = self.fl_manager.backbone(input_tensor)
                    query_embedding_tensor = torch.nn.functional.normalize(query_embedding_tensor, p=2, dim=1)
                    query_embedding = query_embedding_tensor.cpu().numpy()[0]
        else:
            # Backbone PyTorch terkini — selalu konsisten dengan Registry
            if hasattr(self, 'ort_session'): del self.ort_session
            input_tensor = input_tensor.to(self.fl_manager.device)
            with torch.no_grad():
                self.fl_manager.backbone.eval()
                query_embedding_tensor = self.fl_manager.backbone(input_tensor)
                query_embedding_tensor = torch.nn.functional.normalize(query_embedding_tensor, p=2, dim=1)
                query_embedding = query_embedding_tensor.cpu().numpy()[0]
            
        # Manajemen Cache Identitas
        local_refs = self._get_cached_identities(db)

        # Proses Voting Temporal — DI AKTIFKAN KEMBALI untuk stabilitas
        # Gunakan rata-rata dari beberapa frame terakhir untuk meningkatkan stabilitas
        now = time.time()
        if now - self.fl_manager.last_face_time > 1.0:
            self.fl_manager.prediction_buffer.clear()
            
        self.fl_manager.prediction_buffer.append(query_embedding_tensor)
        self.fl_manager.last_face_time = now
        
        mean_embedding_tensor = torch.stack(list(self.fl_manager.prediction_buffer)).mean(0)
        mean_embedding_tensor = torch.nn.functional.normalize(mean_embedding_tensor, p=2, dim=1)
        mean_embedding = mean_embedding_tensor.cpu().numpy()[0]
        
        # Log diagnostik (sekali per batch saja)
        query_norm = float(mean_embedding_tensor.norm())
        print(f"[INFER-FL] Model v{current_v} | Norm: {query_norm:.2f} | Buf: {len(self.fl_manager.prediction_buffer)} | Refs: {len(local_refs)}")
        
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
            
            # Sinkronisasi ke server dilakukan di background agar tidak menghambat aliran video
            background_tasks.add_task(
                sync_record_to_server, 
                user_id, user_name, float(confidence), os.getenv("HOSTNAME", "client-1")
            )
            print(f"[OK] Terdeteksi (Voted): {user_id} (Sim: {confidence:.4f})")
        else:
            if confidence > 0.1: # Log lebih rendah untuk riset
                print(f"[DEBUG-FL] Model v{current_v} | Match: Unknown | Sim: {confidence:.4f} | Thres: {self.fl_manager.inference_threshold}")
            
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
        # Memproses pengenalan secara langsung tanpa melalui API/DB
        try:
            # 1. Ekstraksi wajah
            face_tensor, _, _ = image_processor.detect_face(img_pil)
            if face_tensor is None: return "Unknown", 0.0
            
            # 2. Preprocess & Embedding
            input_tensor = image_processor.prepare_for_model(face_tensor).to(self.fl_manager.device)
            with torch.no_grad():
                self.fl_manager.backbone.eval()
                query_embedding_tensor = self.fl_manager.backbone(input_tensor)
                query_embedding_tensor = torch.nn.functional.normalize(query_embedding_tensor, p=2, dim=1)
                query_embedding = query_embedding_tensor.cpu().numpy()[0]
            
            # 3. Dapatkan Cache Identitas
            # Gunakan db dummy atau ambil dari manager jika sudah ada
            local_refs = getattr(self.fl_manager, 'cached_refs', {})
            if not local_refs:
                # Jika belum ada cache, coba inisialisasi minimal
                from app.db.db import SessionLocal
                db = SessionLocal()
                try:
                    local_refs = self._get_cached_identities(db)
                finally:
                    db.close()

            # 4. Pencocokan
            matched, confidence = identify_user_globally(
                query_embedding, 
                local_refs, 
                threshold=self.fl_manager.inference_threshold
            )
            return matched, float(confidence)
        except Exception as e:
            print(f"[RECOG ERROR] {e}")
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
            
            # Pastikan folder students di raw_data tersedia
            user_dir = os.path.join(self.fl_manager.raw_data_path, "students", name)
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
            
            # 1. PUSTAKA GLOBAL: Muat dari File Registry (Aset Identitas Luar Terminal)
            registry_path = os.path.join(self.fl_manager.data_path, "models", "global_embedding_registry.pth")
            if os.path.exists(registry_path):
                try:
                    registry = torch.load(registry_path, map_location="cpu")
                    for nrp, vec in registry.items():
                        if isinstance(vec, torch.Tensor):
                            v = vec.float()
                        else:
                            v = torch.from_numpy(np.array(vec, dtype=np.float32).copy())
                        # PENTING: Normalisasi L2 agar cosine similarity valid
                        v = torch.nn.functional.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)
                        local_refs[nrp] = v.to(self.fl_manager.device)
                    print(f"[OK] Memuat {len(local_refs)} identitas global dari file.")
                except Exception as e:
                    print(f"[ERROR] Gagal memuat file registry: {e}")
            
            # 2. IDENTITAS SEGAR: Gabungkan dengan Database Lokal
            # Embedding di database lokal di-refresh setiap kali BN Global masuk,
            # sehingga lebih akurat daripada file registry statis.
            try:
                embeddings = db.query(EmbeddingLocal).all()
                db_count = 0
                for emb in embeddings:
                    try:
                        if emb.is_global:
                            dec_emb = np.frombuffer(emb.embedding_data, dtype=np.float32).copy()
                        else:
                            dec_emb = encryptor.decrypt_embedding(emb.embedding_data, emb.iv).copy()
                        
                        v = torch.from_numpy(dec_emb).float()
                        v = torch.nn.functional.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0)
                        local_refs[emb.user_id] = v.to(self.fl_manager.device)
                        db_count += 1
                    except: continue
                if db_count > 0:
                    print(f"[OK] Sinkronisasi {db_count} identitas segar dari database lokal.")
            except: pass
            
            self.fl_manager.cached_refs = local_refs
            self.fl_manager.last_cache_update = time.time()
            
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
