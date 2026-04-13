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

    def recognize_and_submit(self, img_pil, model, reference_embeddings):
        # Melakukan inferensi wajah dengan Temporal Voting (Buffer Rata-rata)
        threshold = self.manager.threshold
        
        
        face, box, prob = image_processor.detect_face(img_pil)
        if face is not None:
            face_tensor = image_processor.prepare_for_model(face)
            input_np = face_tensor.cpu().numpy()
            
            # Gunakan ONNX jika tersedia
            onnx_path = os.path.join("app/data", "models", "backbone_quantized.onnx")
            if os.path.exists(onnx_path):
                try:
                    if not hasattr(self, 'ort_session'):
                        self.ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                    
                    outputs = self.ort_session.run(None, {'input': input_np})
                    query_emb_np = outputs[0][0]
                    # Normalisasi L2
                    norm = np.linalg.norm(query_emb_np)
                    query_emb_np = query_emb_np / (norm + 1e-6)
                    query_emb_tensor = torch.from_numpy(query_emb_np).unsqueeze(0).to(DEVICE)
                except Exception as e:
                    print(f"[ONNX ERROR] CL Fallback: {e}")
                    with torch.no_grad():
                        query_emb_tensor = F.normalize(model(face_tensor))
            else:
                with torch.no_grad():
                    query_emb_tensor = F.normalize(model(face_tensor))
            
            # --- LOGIKA TEMPORAL VOTING ---
            now = time.time()
            # Jika jeda antar wajah > 1s, reset buffer (asumsi orang berbeda atau sesi berbeda)
            if now - self.manager.last_face_time > 1.0:
                self.manager.prediction_buffer.clear()
            
            self.manager.prediction_buffer.append(query_emb_tensor)
            self.manager.last_face_time = now
            
            # Hitung rata-rata embedding di dalam buffer
            mean_emb_tensor = torch.stack(list(self.manager.prediction_buffer)).mean(0)
            mean_emb_tensor = F.normalize(mean_emb_tensor, p=2, dim=1)
            
            best_match, max_sim = "Unknown", -1
            for nrp, ref_emb in reference_embeddings.items():
                # Pastikan ref_emb adalah tensor dengan device yang sama
                if not isinstance(ref_emb, torch.Tensor):
                    ref_emb = torch.tensor(ref_emb).to(DEVICE)
                
                sim = F.cosine_similarity(mean_emb_tensor, ref_emb).item()
                if sim > max_sim:
                    max_sim, best_match = sim, nrp
            
            if max_sim > threshold:
                print(f"[OK] Terdeteksi (Voted): {best_match} (Sim: {max_sim:.2f})")
                nrp_only = best_match.split("_")[0] if "_" in best_match else best_match
                self.submit_attendance(nrp_only, max_sim)
                return nrp_only, max_sim
                
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
