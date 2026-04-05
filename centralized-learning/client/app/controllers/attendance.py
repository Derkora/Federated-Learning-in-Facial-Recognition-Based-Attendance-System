import requests
import torch
import torch.nn.functional as F
from app.utils.image_processing import image_processor, DEVICE

class AttendanceController:
    # Kontroler untuk proses Pengenalan Wajah dan Pelaporan Presensi.
    
    def __init__(self, server_url, client_id):
        self.server_url = server_url
        self.client_id = client_id

    def recognize_and_submit(self, img_pil, model, reference_embeddings, threshold=0.50):
        # Melakukan inferensi wajah dan mengirimkan data jika ditemukan kecocokan.
        face = image_processor.detect_face(img_pil)
        if face is not None:
            face_tensor = image_processor.prepare_for_model(face)
            with torch.no_grad():
                test_emb = F.normalize(model(face_tensor))
            
            best_match, max_sim = "Unknown", -1
            for nrp, ref_emb in reference_embeddings.items():
                sim = F.cosine_similarity(test_emb, ref_emb).item()
                if sim > max_sim:
                    max_sim, best_match = sim, nrp
            
            if max_sim > threshold:
                print(f"[INFO] Terdeteksi: {best_match} (Kemiripan: {max_sim:.2f})")
                self.submit_attendance(best_match, max_sim)
                return best_match, max_sim
        return "Unknown", 0

    def submit_attendance(self, user_id, confidence):
        # Mengirimkan laporan presensi mahasiswa ke database pusat.
        try:
            payload = {
                "user_id": user_id, 
                "edge_id": self.client_id, 
                "confidence": confidence, 
                "lecture_id": "L123"
            }
            res = requests.post(f"{self.server_url}/submit-attendance", json=payload, timeout=5)
            if res.status_code == 200:
                print(f"[OK] Presensi berhasil dikirim untuk {user_id}.")
        except Exception as e:
            print(f"[ERROR] Gagal mengirim presensi ke server: {e}")
