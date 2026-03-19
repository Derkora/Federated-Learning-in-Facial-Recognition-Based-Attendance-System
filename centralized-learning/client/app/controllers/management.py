import os
import shutil
import zipfile
import requests
from app.utils.image_processing import DEVICE

MODEL_DIR = "app/model"
DATA_DIR = "data/students"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

class ManagementController:
    def __init__(self, server_url, client_id):
        self.server_url = server_url
        self.client_id = client_id

    def register_client(self, ip_address):
        try:
            payload = {"id": self.client_id, "ip_address": ip_address, "cl_status": "online"}
            response = requests.post(f"{self.server_url}/register-client", json=payload, timeout=5)
            return response.status_code == 200
        except: return False

    def sync_assets(self, model):
        """Sync global model and reference embeddings."""
        try:
            # Sync model
            res_m = requests.get(f"{self.server_url}/get-model", stream=True, timeout=10)
            if res_m.status_code == 200:
                path_m = f"{MODEL_DIR}/global_model.pth"
                with open(path_m, "wb") as f:
                    for chunk in res_m.iter_content(chunk_size=8192): f.write(chunk)
                model.load_state_dict(torch.load(path_m, map_location=DEVICE))
                model.eval()
            else: return False, None

            # Sync references
            res_r = requests.get(f"{self.server_url}/get-reference-embeddings", stream=True, timeout=10)
            if res_r.status_code == 200:
                path_r = f"{MODEL_DIR}/reference_embeddings.pth"
                with open(path_r, "wb") as f:
                    for chunk in res_r.iter_content(chunk_size=8192): f.write(chunk)
                import torch # Ensure torch is available for load
                return True, torch.load(path_r, map_location=DEVICE)
            return False, None
        except: return False, None

    def package_and_upload(self):
        """ZIP local student data and upload to server."""
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
            return False, "No data to upload"
        
        zip_path = "data/upload.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(DATA_DIR):
                for file in files:
                    zipf.write(os.path.join(root, file), 
                               os.path.relpath(os.path.join(root, file), DATA_DIR))
        
        try:
            with open(zip_path, 'rb') as f:
                res = requests.post(
                    f"{self.server_url}/upload-bulk-zip", 
                    files={"file": (f"{self.client_id}_data.zip", f)}, 
                    timeout=30
                )
            os.remove(zip_path)
            return res.status_code == 200, res.text
        except Exception as e:
            return False, str(e)
