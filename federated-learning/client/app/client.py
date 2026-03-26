import flwr as fl
import torch
import numpy as np
from .utils.trainer import LocalTrainer, TrainingNaNError
from .utils.mobilefacenet import MobileFaceNet, ArcMarginProduct
import os

from .db.db import SessionLocal
from .db.models import EmbeddingLocal, UserLocal

class FaceRecognitionClient(fl.client.NumPyClient):
    def __init__(self, model, head, artifacts_path="/app/artifacts", device="cpu"):
        self.model = model
        self.head = head
        self.artifacts_path = artifacts_path
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        processed_path = os.path.join(self.artifacts_path, "processed")
        self.trainer = LocalTrainer(self.model, self.head, device=self.device, data_path=processed_path)
        
        head_path = os.path.join(self.artifacts_path, "models", "local_head.pth")
        if os.path.exists(head_path):
            print(f"Loading local classifier head from {head_path}...")
            self.head.load_state_dict(torch.load(head_path, map_location=self.device))

    def get_parameters(self, config):
        return self.trainer.get_backbone_parameters(personalized=True)

    def fit(self, parameters, config):
        rnd = config.get("round", 0)
        print(f"FL Fit [Round {rnd}]: Receiving {len(parameters)} parameters...")
        
        try:
            if len(parameters) > 0:
                self.trainer.set_backbone_parameters(parameters, personalized=True)
            else:
                print("WARNING: Received empty parameters!")
        except Exception as e:
            print(f"ERROR in set_backbone_parameters: {e}")
            raise e
        
        epochs = config.get("local_epochs", 5)
        lr = config.get("lr", 0.0001)
        
        db = SessionLocal()
        global_embs = []
        try:
            items = db.query(EmbeddingLocal).filter_by(is_global=True).all()
            if not items:
                print("[WARNING] No global embeddings found! Forgetting prevention might be disabled.")
            
            for item in items:
                # Use .copy() on numpy array from buffer to ensure it is writable
                emb_np = np.frombuffer(item.embedding_data, dtype=np.float32).copy()
                global_embs.append({"nrp": item.user_id, "embedding": torch.from_numpy(emb_np)})
            
            local_user_count = db.query(UserLocal).count()
            print(f"[CLIENT] Dataset: {local_user_count} local users, {len(global_embs)} global memories.")
        finally:
            db.close()
            
        mu = config.get("mu", 0.01)
        lam = config.get("lambda", 0.1)
        
        try:
            loss, accuracy, num_samples = self.trainer.train(
                epochs=epochs, lr=lr, round_num=rnd, 
                global_embeddings=global_embs,
                mu=mu, lam=lam
            )
            status = "Success"
        except TrainingNaNError as e:
            print(f"[CLIENT] {e}. Reporting error to server and rolling back.")
            loss, accuracy, num_samples = 0.0, 0.0, 0
            status = "Training Error"
        
        model_dir = os.path.join(self.artifacts_path, "models")
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_dir, "backbone.pth"))
        torch.save(self.head.state_dict(), os.path.join(model_dir, "local_head.pth"))
        
        with open(os.path.join(model_dir, "model_version.txt"), "w") as f:
            f.write(str(rnd))
            
        return self.trainer.get_backbone_parameters(personalized=True), num_samples, {
            "loss": loss, 
            "accuracy": accuracy,
            "status": status
        }

    def evaluate(self, parameters, config):
        self.trainer.set_backbone_parameters(parameters, personalized=True)
        
        db = SessionLocal()
        global_embs = []
        try:
            items = db.query(EmbeddingLocal).filter_by(is_global=True).all()
            for item in items:
                # Use .copy() on numpy array from buffer to ensure it is writable
                emb_np = np.frombuffer(item.embedding_data, dtype=np.float32).copy()
                global_embs.append({"nrp": item.user_id, "embedding": torch.from_numpy(emb_np)})
        finally:
            db.close()
            
        loss, accuracy, num_samples = self.trainer.evaluate(global_embeddings=global_embs)
        return float(loss), num_samples, {"accuracy": float(accuracy)}

def start_fl_client(server_address, model, head, artifacts_path="/app/artifacts", device="cpu"):
    client = FaceRecognitionClient(model, head, artifacts_path, device)
    fl.client.start_numpy_client(server_address=server_address, client=client)
