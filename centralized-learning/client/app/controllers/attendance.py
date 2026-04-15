import requests
import time 
import torch
import torch.nn.functional as F
import onnxruntime as ort
import numpy as np
import os
from app.utils.preprocessing import image_processor, DEVICE

class AttendanceController:
    # Kontroler untuk proses Pengenalan Wajah dan Pelaporan Presensi.
    
    def __init__(self, manager):
        self.manager = manager
        self.server_url = manager.server_url
        self.client_id = manager.client_id
        self._last_version_loaded = -1 # Melacak versi model yang sedang aktif di session

    def recognize_and_submit(self, img_pil, model, reference_embeddings):
        # Melakukan inferensi wajah dengan Temporal Voting (Buffer Rata-rata)
        threshold = self.manager.threshold
        
        
        face, box, prob = image_processor.detect_face(img_pil)
        if face is not None:
            face_tensor = image_processor.prepare_for_model(face)
            input_np = face_tensor.cpu().numpy()
            
            # Mendapatkan versi model saat ini dari manager
            current_v = getattr(self.manager, 'current_model_version', 0)
            
            # Gunakan ONNX jika tersedia. Prioritaskan FP32 untuk akurasi (~0.8 sim).
            onnx_dir = os.path.join(self.manager.data_path, "models")
            onnx_path = os.path.join(onnx_dir, "backbone.onnx")
            
            # Fallback ke quantized jika versi FP32 belum ada (legacy/compatibility)
            if not os.path.exists(onnx_path):
                onnx_path = os.path.join(onnx_dir, "backbone_quantized.onnx")
            
            # --- PROTEKSI SINKRONISASI: Reload session jika versi berubah atau file berubah ---
            if current_v != self._last_version_loaded:
                if hasattr(self, 'ort_session'):
                    print(f"[SYNC] Reloading ONNX session for Model v{current_v}...")
                    del self.ort_session
                self._last_version_loaded = current_v

            if os.path.exists(onnx_path):
                try:
                    if not hasattr(self, 'ort_session'):
                        print(f"[DEBUG] Initializing ONNX session from {onnx_path}")
                        self.ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                    
                    # Log Input Info (hanya saat awal muat/ganti versi)
                    if current_v != self._last_version_loaded:
                        print(f"[DEBUG] Input Shape: {input_np.shape} | Range: [{input_np.min():.2f}, {input_np.max():.2f}]")

                    outputs = self.ort_session.run(None, {'input': input_np})
                    
                    # Log Output Info
                    if current_v != self._last_version_loaded:
                        print(f"[DEBUG] Raw Output Shape: {outputs[0].shape}")

                    query_emb_tensor = torch.from_numpy(outputs[0][0]).view(1, -1).to(DEVICE)
                    # PURE TORCH NORMALIZE
                    query_emb_tensor = F.normalize(query_emb_tensor, p=2, dim=1)
                except Exception as e:
                    print(f"[ONNX ERROR] CL Fallback: {e}")
                    with torch.no_grad():
                        query_emb_tensor = F.normalize(model(face_tensor))
            else:
                if current_v != self._last_version_loaded:
                    print(f"[DEBUG] Using PyTorch model. Input shape: {face_tensor.shape}")
                with torch.no_grad():
                    query_emb_tensor = F.normalize(model(face_tensor))
            
            # Pastikan dimensi query_emb_tensor adalah (1, 128)
            query_emb_tensor = query_emb_tensor.view(1, -1)
            
            # --- LOGIKA TEMPORAL VOTING ---
            now = time.time()
            if now - self.manager.last_face_time > 1.0:
                self.manager.prediction_buffer.clear()
            
            self.manager.prediction_buffer.append(query_emb_tensor)
            self.manager.last_face_time = now
            
            # Hitung rata-rata embedding di dalam buffer dan re-normalisasi
            mean_emb_tensor = torch.stack(list(self.manager.prediction_buffer)).mean(0)
            mean_emb_tensor = F.normalize(mean_emb_tensor, p=2, dim=1)
            
            best_match, max_sim = "Unknown", -1
            for nrp, ref_emb in reference_embeddings.items():
                if not isinstance(ref_emb, torch.Tensor):
                    ref_emb = torch.tensor(ref_emb).to(DEVICE)
                
                # Pastikan ref_emb adalah (1, 128) untuk similarity
                ref_emb = ref_emb.view(1, -1)
                
                sim = F.cosine_similarity(mean_emb_tensor, ref_emb).item()
                if sim > max_sim:
                    max_sim, best_match = sim, nrp
            
            # Mendapatkan versi model saat ini untuk konteks log
            m_ver = current_v

            if max_sim > threshold:
                print(f"[OK] {best_match} (Sim: {max_sim:.4f}) [Model: v{m_ver}]")
                nrp_only = best_match.split("_")[0] if "_" in best_match else best_match
                self.submit_attendance(nrp_only, max_sim)
                return nrp_only, max_sim
            else:
                # Log lebih detail untuk riset
                if max_sim > 0.1:
                    query_norm = float(mean_emb_tensor.norm())
                    print(f"[DEBUG] Model v{m_ver} | Match: {best_match} | Sim: {max_sim:.4f} | Norm: {query_norm:.2f} | Thres: {threshold}")
                
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
                print(f"[OK] Presensi berhasil dikirim untuk: {user_id}")
        except Exception as e:
            print(f"[ERROR] Gagal mengirim presensi ke server: {e}")
