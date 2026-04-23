import os
import torch
import tempfile
import shutil
import numpy as np
from collections import defaultdict

# Agregasi Fitur dan Vektor Wajah (Centroids)
# Fungsi ini menggabungkan kiriman dari berbagai terminal untuk membuat sebuah
# "galeri global" atau registry yang berisi identitas seluruh mahasiswa.
def aggregate_and_save_registry_assets(log_func):
    try:
        log_func("[INFO] Memulai Agregasi Registry: Penggabungan data terminal...")
        submission_dir = "data/submissions"
        if not os.path.exists(submission_dir): return
        
        all_submissions = {}
        for f_name in os.listdir(submission_dir):
            if f_name.endswith("_assets.pth"):
                cid = f_name.split("_")[0]
                all_submissions[cid] = torch.load(os.path.join(submission_dir, f_name), map_location="cpu")

        if not all_submissions: return

        # 1. Agregasi Parameter Batch Normalization (BN)
        # Parameter BN dari setiap terminal diratakan untuk menjaga kestabilan model di berbagai kondisi pencahayaan terminal.
        clients_bn = [sub['bn'] for sub in all_submissions.values()]
        global_bn = {}
        bn_keys = list(clients_bn[0].keys())
        
        for key in bn_keys:
            if clients_bn[0][key].dtype == np.int64 or 'num_batches_tracked' in key:
                global_bn[key] = torch.from_numpy(clients_bn[0][key])
            else:
                tensors = [torch.from_numpy(bn[key]) for bn in clients_bn]
                global_bn[key] = torch.stack(tensors).mean(0)
        
        os.makedirs("data", exist_ok=True)
        torch.save(global_bn, "data/global_bn_combined.pth")
        log_func("[OK] File global_bn_combined.pth berhasil dibuat.")
        
        # 2. Agregasi Centroids (Fitur Wajah Unik)
        nrp_centroids_list = defaultdict(list)
        for client_id, sub in all_submissions.items():
            for nrp, vec in sub['centroids'].items():
                nrp_centroids_list[nrp].append(torch.from_numpy(vec.copy()))
        
        all_centroids = {}
        for nrp, vecs in nrp_centroids_list.items():
            if len(vecs) > 1:
                stack = torch.stack(vecs)
                avg_vec = torch.mean(stack, dim=0)
            else:
                avg_vec = vecs[0]
            all_centroids[nrp] = torch.nn.functional.normalize(avg_vec.unsqueeze(0), p=2, dim=1).squeeze(0)
                
        # ATOMIC WRITE: Preserve consistency
        os.makedirs("data", exist_ok=True)
        
        # Save BN
        with tempfile.NamedTemporaryFile(delete=False, dir="data") as tmp:
            torch.save(global_bn, tmp.name)
            shutil.move(tmp.name, "data/global_bn_combined.pth")
            
        # Save Registry
        with tempfile.NamedTemporaryFile(delete=False, dir="data") as tmp:
            torch.save(all_centroids, tmp.name)
            shutil.move(tmp.name, "data/global_embedding_registry.pth")
            
        log_func(f"[OK] Agregasi Asset Selesai ({len(all_centroids)} identitas).")
        
    except Exception as e:
        log_func(f"[ERROR] Gagal melakukan agregasi: {e}")
