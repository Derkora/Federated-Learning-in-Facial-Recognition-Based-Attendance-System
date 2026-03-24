import flwr as fl
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
import io
from .utils.mobilefacenet import MobileFaceNet
from .db.db import SessionLocal
from .db.models import FLRound, GlobalModel, FLSession
from datetime import datetime
import json

def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    accuracies = [m[1]["accuracy"] * m[0] for m in metrics]
    examples = [m[0] for m in metrics]
    
    # Check if accuracy is 0-1 or 0-100 and normalize to 0-100%
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
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Round {server_round} aggregated. Saving to database...")
            
            # Convert to ndarrays for serialization
            params_np = fl.common.parameters_to_ndarrays(aggregated_parameters)
            
            # Use aggregated_metrics if FitRes reported them and they were aggregated by weighted_average
            final_loss = aggregated_metrics.get("loss", 0.0)
            final_acc = aggregated_metrics.get("accuracy", 0.0)
            
            # Save to DB
            db = SessionLocal()
            try:
                # 1. Save Round Info
                new_round = FLRound(
                    session_id=self.session_id,
                    round_number=server_round,
                    loss=final_loss,
                    metrics=json.dumps(aggregated_metrics)
                )
                db.add(new_round)
                
                # 2. Update Global Model
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
            except Exception as e:
                print(f"Error saving round {server_round}: {e}")
                db.rollback()
            finally:
                db.close()

        return aggregated_parameters, aggregated_metrics

def start_flower_server(session_id: str, rounds: int = 3, min_clients: int = 2):
    # Load initial parameters from DB if available
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

    # Define strategy
    strategy = SaveModelStrategy(
        session_id=session_id,
        initial_parameters=initial_parameters,
        fraction_fit=1.0,
        min_fit_clients=min_clients,
        min_available_clients=min_clients,
        fit_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=lambda server_round: {"round": server_round},
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8085",
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )
