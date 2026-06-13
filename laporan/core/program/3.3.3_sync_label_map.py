import os
import json

def sync_label_map(self):
    try:
        self._ensure_models_loaded()
        
        # Request daftar label map dari server menggunakan variabel
        url = f"{self.server_api_url}{api_training_label_map}"
        res = self._safe_request("GET", url)
        
        if res and res.status_code == 200:
            self.client.label_map = res.json()
            self.num_classes = len(self.client.label_map)
            
            # Perluas classifier head jika kelas bertambah
            if self.head is not None:
                curr_classes = self.head.weight.shape[0]
                if curr_classes != self.num_classes:
                    new_map = {
                        nrp: idx for idx, nrp in 
                        enumerate(self.client.label_map)
                    }
                    self.head = (
                        self.client.trainer
                        .update_head(self.num_classes, new_map)
                    )
 
            # Simpan peta label ke file JSON lokal
            path = os.path.join(
                self.data_path, "models", "label_map.json"
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.client.label_map, f)
            return True
    except Exception:
        pass
    return False
