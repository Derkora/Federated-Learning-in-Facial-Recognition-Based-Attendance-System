import requests
import time 
import torch
import torch.nn.functional as F
import numpy as np
import os
from app.utils.preprocessing import image_processor, DEVICE

class AttendanceController:
    # Kontroler untuk proses Pengenalan Wajah dan Pelaporan Presensi (Optimasi Edge: No ONNX)
    
    def __init__(self, manager):
        self.manager = manager
        self.server_url = manager.server_url
        self.client_id = manager.client_id
        self._last_version_loaded = -1 # Melacak versi model yang sedang aktif di session

    def process_inference(self, img_pil, model, reference_embeddings):
        # Melakukan inferensi wajah dengan Temporal Voting (Buffer Rata-rata)
        threshold = self.manager.threshold
        
        # 1. Deteksi & MTCNN Square Crop (dengan margin 20px)
        face_tensor, _, _ = image_processor.detect_face(img_pil)
        
        if face_tensor is not None:
            # prepare_for_model menangani squash 112x112 ke 96x112 dan normalisasi [-1, 1]
            face_tensor_ready = image_processor.prepare_for_model(face_tensor)
            
            # Mendapatkan versi model saat ini dari manager
            current_v = getattr(self.manager, 'current_model_version', 0)
            
            # --- INFERENSI PURE PYTORCH (Optimasi Edge) ---
            try:
                if current_v != self._last_version_loaded:
                    print(f"[DEBUG] Menggunakan model PyTorch v{current_v}. Input: {face_tensor_ready.shape}")
                    self._last_version_loaded = current_v
                
                with torch.no_grad():
                    # --- FLIP TRICK (Alignment dengan Server Registry) ---
                    # 1. Embedding Citra Asli
                    emb_orig = model(face_tensor_ready)
                    
                    # 2. Embedding Citra Mirror (Horizontal Flip)
                    face_flipped = torch.flip(face_tensor_ready, dims=[3])
                    emb_mirror = model(face_flipped)
                    
                    # 3. Rata-rata dan Normalisasi
                    query_emb_tensor = F.normalize((emb_orig + emb_mirror) / 2, p=2, dim=1)
            except Exception as e:
                print(f"[ERROR] Kegagalan inferensi: {e}")
                return "Unknown", 0
            
            # Pastikan dimensi query_emb_tensor adalah (1, 128)
            query_emb_tensor = query_emb_tensor.view(1, -1)
            
            # --- LOGIKA TEMPORAL VOTING & CIM (Confident Instant Match) ---
            # 1. Cek Kemiripan Instant Frame (Untuk Menghilangkan "Pemanasan")
            best_match_instant, max_sim_instant = "Unknown", -1
            for nrp, ref_emb in reference_embeddings.items():
                if not isinstance(ref_emb, torch.Tensor):
                    ref_emb = torch.tensor(ref_emb).to(DEVICE)
                ref_emb = ref_emb.view(1, -1)
                sim = F.cosine_similarity(query_emb_tensor, ref_emb).item()
                if sim > max_sim_instant:
                    max_sim_instant, best_match_instant = sim, nrp

            # CIM Bypass: Jika skor sangat tinggi (> 0.85), anggap valid langsung dan reset buffer
            if max_sim_instant > 0.85:
                print(f"[CIM] Instant Match Confident! {best_match_instant} (Sim: {max_sim_instant:.4f})")
                self.manager.prediction_buffer.clear()
                self.manager.prediction_buffer.append(query_emb_tensor)
                best_match, max_sim = best_match_instant, max_sim_instant
            else:
                # Jika tidak sangat tinggi, gunakan rata-rata temporal (Normal logic)
                now = time.time()
                if now - self.manager.last_face_time > 1.0:
                    self.manager.prediction_buffer.clear()
                
                self.manager.prediction_buffer.append(query_emb_tensor)
                self.manager.last_face_time = now
                
                mean_emb_tensor = torch.stack(list(self.manager.prediction_buffer)).mean(0)
                mean_emb_tensor = F.normalize(mean_emb_tensor, p=2, dim=1)
                
                best_match, max_sim = "Unknown", -1
                for nrp, ref_emb in reference_embeddings.items():
                    if not isinstance(ref_emb, torch.Tensor):
                        ref_emb = torch.tensor(ref_emb).to(DEVICE)
                    ref_emb = ref_emb.view(1, -1)
                    sim = F.cosine_similarity(mean_emb_tensor, ref_emb).item()
                    if sim > max_sim:
                        max_sim, best_match = sim, nrp
            
            if max_sim > threshold:
                print(f"[SUCCESS] {best_match} (Sim: {max_sim:.4f}) [Model: v{current_v}]")
                nrp_only = best_match.split("_")[0] if "_" in best_match else best_match
                self.submit_attendance(nrp_only, max_sim)
                return nrp_only, max_sim
            else:
                if max_sim > 0.1:
                    print(f"[DEBUG] Model v{current_v} | Terbaik: {best_match} | Sim: {max_sim:.4f} | Thres: {threshold}")
                
        return "Unknown", 0

    def submit_attendance(self, user_id, confidence):
        # Mengirimkan laporan presensi mahasiswa ke server pusat.
        try:
            payload = {
                "user_id": user_id, 
                "edge_id": self.client_id, 
                "confidence": confidence, 
                "lecture_id": "L123"
            }
            res = requests.post(f"{self.server_url}/submit-attendance", json=payload, timeout=5)
            if res.status_code == 200:
                print(f"[SUCCESS] Presensi berhasil dikirim untuk: {user_id}")
        except Exception as e:
            print(f"[ERROR] Gagal mengirim presensi ke server: {e}")
