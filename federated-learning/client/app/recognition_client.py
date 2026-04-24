import flwr as fl
import torch
import numpy as np
from app.utils.trainer import LocalTrainer, TrainingNaNError
from app.utils.mobilefacenet import MobileFaceNet, ArcMarginProduct
import os
import json
import requests
import base64
import io
import gc

from app.db.db import SessionLocal
from app.db.models import EmbeddingLocal, UserLocal

class FaceRecognitionClient(fl.client.NumPyClient):
    def __init__(self, model, head, data_path="/app/data", device="cpu"):
        self.model = model
        self.head = head
        self.data_path = data_path
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.label_map = None
        
        # Prioritas: Muat dari arsip lokal (Penguasaan Label Permanen)
        map_path = os.path.join(self.data_path, "models", "label_map.json")
        if os.path.exists(map_path):
            import json
            with open(map_path, "r") as f:
                self.label_map = json.load(f)
                print(f"[CLIENT] Initialized with archived Global Label Map: {len(self.label_map)} identities.")
        
        processed_path = os.path.join(self.data_path, "processed")
        self.trainer = LocalTrainer(self.model, self.head, device=self.device, data_path=processed_path)
        
        head_path = os.path.join(self.data_path, "models", "local_head.pth")
        if os.path.exists(head_path) and self.head is not None:
            try:
                print(f"Loading local classifier head from {head_path}...")
                self.head.load_state_dict(torch.load(head_path, map_location=self.device))
            except Exception as e:
                print(f"[WARN] local_head.pth corrupt or incompatible, skipping: {e}")
                import os as _os
                _os.remove(head_path)  # Hapus file rusak agar tidak crash lagi

    def get_parameters(self, config):
        return self.trainer.get_backbone_parameters(personalized=True)

    def fit(self, parameters, config):
        if hasattr(self, 'fl_manager'):
            self.fl_manager._ensure_models_loaded()
            
        rnd = config.get("round", 0)
        print(f"FL Fit [Round {rnd}]: Receiving {len(parameters)} parameters...")
        
        # Sinkronisasi Redundan: Pastikan label map tersedia
        if not self.label_map and "label_map" in config:
            self.label_map = json.loads(config["label_map"])
            print(f"[CLIENT] Received label_map via fit config.")
        
        try:
            if len(parameters) > 0:
                self.trainer.set_backbone_parameters(parameters, personalized=True)
        except Exception as e:
            print(f"[ERROR] set_backbone_parameters failed: {e}")
            return parameters, 0, {"loss": 0.0, "accuracy": 0.0, "status": "ParameterError", "hostname": os.getenv("HOSTNAME", "unknown"), "epoch_history": "[]"}
        
        epochs = config.get("local_epochs", 3)
        lr = config.get("lr", 0.0001)
        mu = config.get("mu", 0.01)
        lam = config.get("lambda", 0.1)

        # Global embeddings: Load for Knowledge Sharing / catastrophic forgetting prevention
        global_embs = []
        db = SessionLocal()
        try:
            items = db.query(EmbeddingLocal).filter_by(is_global=True).all()
            for item in items:
                # Use .copy() on numpy array from buffer to ensure it is writable
                emb_np = np.frombuffer(item.embedding_data, dtype=np.float32).copy()
                global_embs.append({"nrp": item.user_id, "embedding": torch.from_numpy(emb_np)})
            
            local_user_count = db.query(UserLocal).count()
            print(f"[CLIENT] Dataset: {local_user_count} local users, {len(global_embs)} global memories.")
        except Exception as e:
            print(f"[CLIENT] Error loading dataset info: {e}")
        finally:
            db.close()

        # Update trainer nrp_to_idx agar sinkron dengan head saat ini (Kritis untuk Weight Copying)
        if self.label_map:
            self.trainer.nrp_to_idx = {nrp: idx for idx, nrp in enumerate(self.label_map)}

        loss, accuracy, num_samples, epoch_history = 0.0, 0.0, 0, []
        status = "Skipped"

        try:
            if hasattr(self, 'fl_manager'):
                total_rounds = config.get("total_rounds", 10)
                self.fl_manager.fl_round = rnd
                self.fl_manager.fl_status = f"Training: Ronde {rnd}/{total_rounds}"
            
            loss, accuracy, num_samples, epoch_history = self.trainer.train(
                epochs=epochs, lr=lr, round_num=rnd,
                global_embeddings=global_embs,
                label_map=self.label_map,
                mu=mu, lam=lam
            )

            status = "Success"
        except TrainingNaNError as e:
            print(f"[CLIENT] NaN error: {e}")
            status = "NaNError"
        except Exception as e:
            print(f"[CLIENT] Training exception (non-fatal): {e}")
            import traceback; traceback.print_exc()
            status = "TrainingError"

        # Simpan checkpoint lokal (Tanpa merubah nomor versi global di disk)
        try:
            model_dir = os.path.join(self.data_path, "models")
            os.makedirs(model_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(model_dir, "backbone.pth"))
            torch.save(self.trainer.head.state_dict(), os.path.join(model_dir, "local_head.pth"))
            # model_version.txt jangan diupdate di sini, biarkan manager yang menangani di akhir siklus
        except Exception as e:
            print(f"[CLIENT] Checkpoint save failed: {e}")

        # model_version.txt jangan diupdate di sini, biarkan manager yang menangani di akhir siklus
        except Exception as e:
            print(f"[CLIENT] Checkpoint save failed: {e}")

        # Evaluasi validasi (non-fatal)
        val_loss, val_accuracy = 0.0, 0.0
        try:
            val_loss, val_accuracy, _ = self.trainer.evaluate(global_embeddings=global_embs, label_map=self.label_map)
        except Exception as e:
            print(f"[CLIENT] Evaluate failed (non-fatal): {e}")

        # Memory Cleanup (Jetson specialized)
        del global_embs
        gc.collect()
            
        return self.trainer.get_backbone_parameters(personalized=True), num_samples, {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
            "status": status,
            "hostname": os.getenv("HOSTNAME", "unknown-client"),
            "epoch_history": json.dumps(epoch_history)
        }

    def evaluate(self, parameters, config):
        if hasattr(self, 'fl_manager'):
            self.fl_manager._ensure_models_loaded()
            
        try:
            self.trainer.set_backbone_parameters(parameters, personalized=True)
        except Exception as e:
            print(f"[CLIENT] evaluate set_params failed: {e}")
            return 0.0, 0, {"accuracy": 0.0}

        try:
            loss, accuracy, num_samples = self.trainer.evaluate(global_embeddings=[], label_map=self.label_map)
        except Exception as e:
            print(f"[CLIENT] evaluate failed: {e}")
            loss, accuracy, num_samples = 0.0, 0.0, 0

        gc.collect()
        return float(loss), num_samples, {"accuracy": float(accuracy), "loss": float(loss)}

def start_fl_client(server_address, model, head, data_path="/app/data", device="cpu"):
    client = FaceRecognitionClient(model, head, data_path, device)
    fl.client.start_numpy_client(server_address=server_address, client=client)
