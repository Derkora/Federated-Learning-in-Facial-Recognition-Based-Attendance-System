import os
import shutil
import zipfile
import requests
import torch
import traceback
from ..utils.image_processing import DEVICE

MODEL_DIR = "app/model"
DATA_DIR = os.getenv("RAW_DATA_PATH", "raw_data") + "/students"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

class ManagementController:
    # Kontroler untuk manajemen aset (model/referensi) dan pengunggahan dataset.
    
    def __init__(self, server_url, client_id):
        self.server_url = server_url
        self.client_id = client_id

    def register_client(self, ip_address):
        # Mendaftarkan terminal ke database server pusat.
        try:
            payload = {"edge_id": self.client_id, "ip_address": ip_address, "status": "online"}
            response = requests.post(f"{self.server_url}/register-client", json=payload, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"[ERROR] Gagal registrasi client: {e}")
            return False

    def check_training_request(self):
        # Memeriksa apakah server meminta pengunggahan data dataset.
        try:
            res = requests.get(f"{self.server_url}/ping", timeout=3)
            if res.status_code == 200:
                data = res.json()
                return data.get("upload_requested", False)
        except: pass
        return False

    def sync_assets(self, model):
        # Sinkronisasi Bobot Model Global dan Basis Data Referensi Wajah Mahasiswa.
        try:
            # 1. Sinkronisasi Model
            url_m = f"{self.server_url}/get-model"
            res_m = requests.get(url_m, stream=True, timeout=15)
            
            if res_m.status_code == 200:
                path_m = f"{MODEL_DIR}/global_model.pth"
                with open(path_m, "wb") as f:
                    for chunk in res_m.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                
                state_dict = torch.load(path_m, map_location=DEVICE)
                model.load_state_dict(state_dict)
                model.eval()
            else:
                print(f"[WARN] Gagal mendapatkan model. HTTP: {res_m.status_code}")
                return False, None

            # 2. Sinkronisasi Referensi (Embeddings)
            url_r = f"{self.server_url}/get-reference-embeddings"
            res_r = requests.get(url_r, stream=True, timeout=15)
            if res_r.status_code == 200:
                path_r = f"{MODEL_DIR}/reference_embeddings.pth"
                with open(path_r, "wb") as f:
                    for chunk in res_r.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                
                refs = torch.load(path_r, map_location=DEVICE)
                return True, refs
            
            print(f"[WARN] Gagal mendapatkan referensi. HTTP: {res_r.status_code}")
            return False, None
        except Exception as e:
            print(f"[ERROR] Terjadi kesalahan saat sinkronisasi aset: {e}")
            return False, None

    def package_and_upload(self):
        # Mengunggah dataset lokal (ZIP) ke server pusat untuk proses pelatihan.
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
            return False, "Data tidak ditemukan."
        
        zip_path = "data/upload.zip"
        try:
            # Hitung statistik dataset sebelum dikemas
            folder_count = 0
            file_count = 0
            for root, dirs, files in os.walk(DATA_DIR):
                folder_count += len(dirs)
                file_count += len(files)
            
            print(f"[INFO] Mengemas {file_count} gambar dari {folder_count} mahasiswa.")

            if file_count == 0:
                return False, "Tidak ada gambar untuk diunggah."
                
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(DATA_DIR):
                    for file in files:
                        zipf.write(os.path.join(root, file), 
                                   os.path.relpath(os.path.join(root, file), DATA_DIR))
            
            with open(zip_path, 'rb') as f:
                res = requests.post(
                    f"{self.server_url}/upload-bulk-zip", 
                    files={"file": (f"{self.client_id}_data.zip", f)}, 
                    timeout=120
                )
            os.remove(zip_path)
            return res.status_code == 200, res.text
        except Exception as e:
            print(f"[ERROR] Gagal mengemas/mengunggah data: {e}")
            if os.path.exists(zip_path): os.remove(zip_path)
            return False, str(e)
