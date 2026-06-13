import os

def run_discovery_phase(self):
    try:
        # Tentukan folder mahasiswa
        path = os.path.join(self.raw_data_path, "students")
        if not os.path.exists(path):
            path = self.raw_data_path
  
        # Membaca daftar sub-direktori mahasiswa
        folders = [
            f for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f))
        ]
        for folder in sorted(folders):
            # Memisah nama folder menjadi NRP dan Nama
            nrp = folder.split('_')[0] if "_" in folder else folder
            name = folder.split('_')[1] if "_" in folder else nrp
        
            # Kirim identitas ke server pusat menggunakan variabel
            url = f"{self.server_api_url}{api_training_get_label}"
            payload = {
                "nrp": nrp,
                "name": name,
                "client_id": self.client_id
            }
            self._safe_request("POST", url, json=payload)
        
            # Daftarkan ke SQLite lokal perangkat
            db = SessionLocal()
            try:
                user = db.query(UserLocal).filter_by(
                    user_id=nrp
                ).first()
                if not user:
                    db.add(UserLocal(user_id=nrp, name=name))
                    db.commit()
            finally: 
                db.close()
        
        # Beritahu server bahwa discovery selesai
        done_url = (
            f"{self.server_api_url}"
            f"{api_clients_discovery_done}"
        ) 
        done_payload = {"client_id": self.client_id}
        self._safe_request("POST", done_url, json=done_payload)
    except Exception:
        pass
