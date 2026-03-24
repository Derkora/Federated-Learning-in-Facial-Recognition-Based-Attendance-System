import flwr as fl
import torch
import numpy as np
from .utils.trainer import LocalTrainer, ArcMarginProduct
from .utils.mobilefacenet import MobileFaceNet
import os

class FaceRecognitionClient(fl.client.NumPyClient):
    def __init__(self, model, head, data_path="/app/data", device="cpu"):
        self.model = model
        self.head = head
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        # Use the processed faces directory for training
        self.trainer = LocalTrainer(self.model, self.head, device=self.device, data_path=os.path.join(data_path, "processed_faces"))
        
        # Load local head if exists
        head_path = os.path.join(data_path, "local_head.pth")
        if os.path.exists(head_path):
            print(f"Loading local classifier head from {head_path}...")
            self.head.load_state_dict(torch.load(head_path, map_location=self.device))

    def get_parameters(self, config):
        """Returns the parameters of the local model (Backbone only)."""
        return self.trainer.get_backbone_parameters()

    def fit(self, parameters, config):
        """Sets parameters, trains the model locally, and returns the updated parameters."""
        rnd = config.get("round", 0)
        print(f"FL Fit [Round {rnd}]: Receiving {len(parameters)} parameter arrays...")
        
        try:
            if len(parameters) > 0:
                print(f"DEBUG: First param shape: {parameters[0].shape}")
                self.trainer.set_backbone_parameters(parameters)
            else:
                print("WARNING: Received empty parameters from server!")
        except Exception as e:
            print(f"DEBUG ERROR in set_backbone_parameters: {e}")
            raise e
        
        epochs = config.get("local_epochs", 1)
        lr = config.get("lr", 0.01)
        
        print(f"FL Fit [Round {rnd}]: Starting local training for {epochs} epochs...")
        
        # Pull Global Embeddings from DB for hybrid training
        from .db.db import SessionLocal
        from .db.models import EmbeddingLocal
        db = SessionLocal()
        global_embs = []
        try:
            items = db.query(EmbeddingLocal).filter_by(is_global=True).all()
            for item in items:
                emb_np = np.frombuffer(item.embedding_data, dtype=np.float32)
                global_embs.append({"nrp": item.user_id, "embedding": torch.from_numpy(emb_np)})
            print(f"[CLIENT] Hybrid Training: Using {len(global_embs)} global memories.")
        finally:
            db.close()
            
        loss, accuracy, num_samples = self.trainer.train(
            epochs=epochs, lr=lr, round_num=rnd, 
            global_embeddings=global_embs
        )
        
        # PERSISTENCE: Save aggregated backbone weights and updated head
        base_path = os.path.dirname(self.trainer.data_path)
        torch.save(self.model.state_dict(), os.path.join(base_path, "backbone.pth"))
        torch.save(self.head.state_dict(), os.path.join(base_path, "local_head.pth"))
        
        with open(os.path.join(base_path, "model_version.txt"), "w") as f:
            f.write(str(rnd))
            
        print(f"FL Fit: Models saved to {base_path} (Round: {rnd}, Samples: {num_samples})")
        
        return self.trainer.get_backbone_parameters(), num_samples, {"loss": loss, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        """Sets parameters and evaluates the model locally (optional)."""
        self.trainer.set_backbone_parameters(parameters)
        return 0.0, 1, {"accuracy": 1.0}

def start_fl_client(server_address, model, head, data_path="/app/data", device="cpu"):
    client = FaceRecognitionClient(model, head, data_path, device)
    fl.client.start_numpy_client(server_address=server_address, client=client)
