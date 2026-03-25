import flwr as fl
import os
import io
import torch
import json
from datetime import datetime
from .db.db import SessionLocal
from .db.models import FLRound, GlobalModel

def weighted_average(metrics: list) -> dict:
    accuracies = [m[1]["accuracy"] * m[0] for m in metrics]
    examples = [m[0] for m in metrics]
    
    aggregated = {
        "accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0,
        "loss": sum([m[1]["loss"] * m[0] for m in metrics]) / sum(examples) if sum(examples) > 0 else 0.0
    }
    return aggregated

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, session_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_id = session_id

    def aggregate_fit(
        self,
        server_round: int,
        results: list,
        failures: list,
    ):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Round {server_round} aggregated. Saving to database...")
            params_np = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            final_loss = aggregated_metrics.get("loss", 0.0)
            
            # Save to DB
            db = SessionLocal()
            try:
                new_round = FLRound(
                    session_id=self.session_id,
                    round_number=server_round,
                    loss=final_loss,
                    metrics=json.dumps(aggregated_metrics)
                )
                db.add(new_round)
                
                buf = io.BytesIO()
                torch.save(params_np, buf)
                
                global_model = db.query(GlobalModel).first()
                if not global_model:
                    global_model = GlobalModel(version=server_round, weights=buf.getvalue())
                    db.add(global_model)
                else:
                    global_model.version = server_round
                    global_model.weights = buf.getvalue()
                    global_model.last_updated = datetime.utcnow()
                
                db.commit()
                
                try:
                    os.makedirs("data", exist_ok=True)
                    torch.save(params_np, "data/backbone.pth")
                except Exception as e:
                    print(f"Warning: Could not save backbone.pth to disk: {e}")
                    
            except Exception as e:
                print(f"Error saving round {server_round}: {e}")
                db.rollback()
            finally:
                db.close()

        return aggregated_parameters, aggregated_metrics

class FLServerManager:
    def __init__(self):
        self.session_id = None
        self.is_running = False

    def start_training(self, session_id: str, rounds: int = 3, min_clients: int = 2):
        self.session_id = session_id
        self.is_running = True
        
        initial_parameters = None
        db = SessionLocal()
        try:
            latest_model = db.query(GlobalModel).order_by(GlobalModel.last_updated.desc()).first()
            if latest_model and latest_model.weights:
                print("FL Server: Loading initial parameters from database...")
                weights_np = torch.load(io.BytesIO(latest_model.weights))
                initial_parameters = fl.common.ndarrays_to_parameters(weights_np)
        except Exception as e:
            print(f"Error loading initial parameters: {e}")
        finally:
            db.close()

        strategy = SaveModelStrategy(
            session_id=session_id,
            initial_parameters=initial_parameters,
            fraction_fit=1.0,
            min_fit_clients=min_clients,
            min_available_clients=min_clients,
            fit_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=lambda server_round: {"round": server_round},
        )

        print(f"FL Server: Starting on 0.0.0.0:8085 (Backbone-only Aggregation Mode)...")
        try:
            fl.server.start_server(
                server_address="0.0.0.0:8085",
                config=fl.server.ServerConfig(num_rounds=rounds),
                strategy=strategy,
            )
        finally:
            self.is_running = False
            print("FL Server: Stopped.")

# Global Instance
fl_manager = FLServerManager()
