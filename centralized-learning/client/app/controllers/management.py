import os
import shutil
import zipfile
import requests
import torch
import traceback
from app.utils.processing import DEVICE

MODEL_DIR = "app/model"
DATA_DIR = os.getenv("RAW_DATA_PATH", "raw_data") + "/students"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

class ManagementController:
    def __init__(self, server_url, client_id):
        self.server_url = server_url
        self.client_id = client_id

    def register_client(self, ip_address):
        try:
            payload = {"edge_id": self.client_id, "ip_address": ip_address, "status": "online"}
            response = requests.post(f"{self.server_url}/register-client", json=payload, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"[MGMT] Registration error: {e}", flush=True)
            return False

    def check_training_request(self):
        """Poll server for training/upload request."""
        try:
            res = requests.get(f"{self.server_url}/ping", timeout=3)
            if res.status_code == 200:
                data = res.json()
                return data.get("upload_requested", False)
        except: pass
        return False

    def sync_assets(self, model):
        """Sync global model and reference embeddings."""
        try:
            # Sync model
            print(f"[MGMT] TRACE: Starting sync_assets for model...", flush=True)
            url_m = f"{self.server_url}/get-model"
            print(f"[MGMT] TRACE: Requesting {url_m}", flush=True)
            res_m = requests.get(url_m, stream=True, timeout=15)
            
            if res_m.status_code == 200:
                path_m = f"{MODEL_DIR}/global_model.pth"
                print(f"[MGMT] TRACE: Saving model to {path_m}...", flush=True)
                with open(path_m, "wb") as f:
                    for chunk in res_m.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                
                print(f"[MGMT] TRACE: Model saved. Loading with torch (map: {DEVICE})...", flush=True)
                state_dict = torch.load(path_m, map_location=DEVICE)
                print(f"[MGMT] TRACE: State dict loaded. Applying to model...", flush=True)
                model.load_state_dict(state_dict)
                model.eval()
                print(f"[MGMT] TRACE: Model ready.", flush=True)
            else:
                print(f"[MGMT] TRACE: Failed to get model. Status: {res_m.status_code}", flush=True)
                return False, None

            # Sync references
            url_r = f"{self.server_url}/get-reference-embeddings"
            print(f"[MGMT] TRACE: Requesting references {url_r}...", flush=True)
            res_r = requests.get(url_r, stream=True, timeout=15)
            if res_r.status_code == 200:
                path_r = f"{MODEL_DIR}/reference_embeddings.pth"
                print(f"[MGMT] TRACE: Saving references to {path_r}...", flush=True)
                with open(path_r, "wb") as f:
                    for chunk in res_r.iter_content(chunk_size=8192):
                        if chunk: f.write(chunk)
                
                print(f"[MGMT] TRACE: References saved. Loading...", flush=True)
                refs = torch.load(path_r, map_location=DEVICE)
                print(f"[MGMT] TRACE: References synced successfully. Returning True.", flush=True)
                return True, refs
            
            print(f"[MGMT] TRACE: Failed to get references. Status: {res_r.status_code}", flush=True)
            return False, None
        except Exception as e:
            print(f"[MGMT] TRACE: CRITICAL ERROR during sync_assets:", flush=True)
            traceback.print_exc()
            return False, None

    def package_and_upload(self):
        """ZIP local student data and upload to server."""
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
            return False, "No data to upload"
        
        zip_path = "data/upload.zip"
        print(f"[MGMT] Packaging {DATA_DIR} into {zip_path}...", flush=True)
        try:
            folder_count = 0
            file_count = 0
            for root, dirs, files in os.walk(DATA_DIR):
                folder_count += len(dirs)
                file_count += len(files)
            
            print(f"[MGMT] DIAGNOSTIC: Found {folder_count} student folders and {file_count} total images.", flush=True)

            if file_count == 0:
                return False, "No images found to upload"
                
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
            print(f"[MGMT] Packaging error: {e}", flush=True)
            if os.path.exists(zip_path): os.remove(zip_path)
            return False, str(e)
