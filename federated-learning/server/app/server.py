import threading
import flwr as fl
import torch
import io
import time
from app.db.db import SessionLocal
from app.db.models import ModelVersion
from typing import List, Tuple
from flwr.common import Metrics, ndarrays_to_parameters

from app.utils.mobilefacenet import MobileFaceNet
    
class FLServerManager:
    def __init__(self):
        self.running = False
        self.current_round = 0
        self.total_rounds = 0
        self.server_thread = None
        
        # Metrics
        self.start_time = 0
        self.end_time = 0
        self.model_size_bytes = 0 
        self.clients_per_round = 2 
        self.metrics_history = [] # List of {"round": int, "loss": float, "accuracy": float}
        self.reset_counter = 0 

    def _get_initial_parameters(self):
        db = SessionLocal()
        try:
            # Ambil versi terakhir
            latest_model = db.query(ModelVersion).order_by(ModelVersion.version_id.desc()).first()
            
            if latest_model:
                self.model_size_bytes = len(latest_model.head_blob)
                print(f"[FL SERVER] Model Size Loaded: {self.model_size_bytes} bytes")

                buffer = io.BytesIO(latest_model.head_blob)
                
                # Load MobileFaceNet Architecture 
                backbone = MobileFaceNet(embedding_size=128) # Menggunakan MobileFaceNet yang sudah diimport
                state_dict = torch.load(buffer)
                
                params = [val.cpu().numpy() for _, val in state_dict.items()]
                return ndarrays_to_parameters(params)
            else:
                print("[FL SERVER] Belum ada model di DB. Jalankan init.py dulu!")
                return None
        except Exception as e:
            print(f"[FL SERVER] Error loading model: {e}")
            return None
        finally:
            db.close()

    def start_training(self, rounds: int = 10):
        if self.running:
            return

        self.running = True
        self.current_round = 0
        self.total_rounds = rounds
        self.start_time = time.time()
        self.end_time = 0 
        self.metrics_history = [] # Reset history saat mulai baru
        
        if self.model_size_bytes == 0:
             self._get_initial_parameters()

        print(f"[FL SERVER] Mulai Federated Learning: {rounds} rounds")

        self.server_thread = threading.Thread(
            target=self._run_flower_server,
            args=(rounds,),
            daemon=True
        )
        self.server_thread.start()

    def _run_flower_server(self, rounds: int):
        try:
            initial_params = self._get_initial_parameters()

            def fit_config(rnd):
                self.current_round = rnd
                return {"round": rnd}

            # Definisi Weighted Average dengan akses ke self.metrics_history
            def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
                if not metrics:
                    return {"accuracy": 0.0, "loss": 0.0}
                    
                accuracies = [num_examples * m.get("accuracy", 0) for num_examples, m in metrics]
                losses = [num_examples * m.get("loss", 0) for num_examples, m in metrics]
                examples = [num_examples for num_examples, _ in metrics]
                
                total_examples = sum(examples)
                print(f"[DEBUG AGGREGATE] Total Examples: {total_examples}. Metrics count: {len(metrics)}")
                if total_examples == 0:
                     print("[DEBUG AGGREGATE] Zero examples, returning 0.0 metrics.")
                     return {"accuracy": 0.0, "loss": 0.0}

                aggregated = {
                    "accuracy": sum(accuracies) / total_examples,
                    "loss": sum(losses) / total_examples,
                }
                
                existing_round = next((item for item in self.metrics_history if item["round"] == self.current_round), None)
                if existing_round:
                     existing_round.update(aggregated)
                else:
                     self.metrics_history.append({
                         "round": self.current_round,
                         "loss": aggregated["loss"],
                         "accuracy": aggregated["accuracy"]
                     })
                     
                return aggregated

            strategy = fl.server.strategy.FedAvg(
                fraction_fit=1.0,
                fraction_evaluate=1.0,
                min_fit_clients=self.clients_per_round,
                min_available_clients=self.clients_per_round,
                on_fit_config_fn=fit_config,
                initial_parameters=initial_params,
                fit_metrics_aggregation_fn=weighted_average,
                evaluate_metrics_aggregation_fn=weighted_average
            )

            fl.server.start_server(
                server_address="0.0.0.0:8085",
                strategy=strategy,
                config=fl.server.ServerConfig(num_rounds=rounds)
            )
            print("[FL SERVER] Training Selesai.")

        except Exception as e:
            print(f"[FL SERVER] Error: {e}")
        finally:

            self.running = False
            self.end_time = time.time() 
            print(f"[FL SERVER] Stopped at {self.end_time}")

    def stop(self):
        self.running = False

    def status(self):
        if self.running:
            elapsed = time.time() - self.start_time
        elif self.end_time > 0:
            elapsed = self.end_time - self.start_time
        else:
            elapsed = 0
            
        mins, secs = divmod(int(elapsed), 60)
        time_str = f"{mins:02d}:{secs:02d}"
        
        size_kb = self.model_size_bytes / 1024
        
        if self.current_round > 0:
            total_kb = size_kb * self.clients_per_round * 2 * self.current_round * 1.1
        else:
            total_kb = 0
        
        bandwidth_str = f"{total_kb:,.2f} KB"

        is_finished = (self.current_round >= self.total_rounds) and (self.total_rounds > 0)
        
        return {
            "running": self.running and not is_finished,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "elapsed_time": time_str, 
            "bandwidth_kb": bandwidth_str,
            "metrics_history": sorted(self.metrics_history, key=lambda x: x['round']) 
        }