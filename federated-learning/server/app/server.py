import threading
import flwr as fl
import torch
import io
import time
from datetime import datetime
from collections import OrderedDict
from app.db.db import SessionLocal
from app.db.models import ModelVersion, TrainingRound, TrainingUpdate
from typing import List, Tuple
from flwr.common import Metrics, ndarrays_to_parameters

from app.utils.mobilefacenet import MobileFaceNet, ArcMarginProduct
    
class FLServerManager:
    def __init__(self):
        self.running = False
        self.current_round = 0
        self.total_rounds = 0
        self.server_thread = None
        self.current_phase = "idle" # phases: idle, syncing, preprocessing, training, completed
        self.phase_logs = []
        self.clients_per_round = 2
        self.metrics_history = []
        self.reset_counter = int(time.time())
        self.model_size_bytes = 0

    def set_phase(self, phase):
        if self.current_phase != phase:
            self.current_phase = phase
            self.add_log(f"System Phase Changed: {phase}")

    def run_lifecycle(self, rounds: int = 10):
        """Unified manual trigger for Sync -> Preprocess -> Train."""
        if self.running:
            return
        
        self.total_rounds = rounds
        self.add_log("🚀 Starting Unified FL Lifecycle...")
        
        def _lifecycle_task():
            try:
                # Helper to get ready clients
                def get_ready_info(target_substring):
                    active = self._get_active_clients_list(threshold=300)
                    ready = [c for c in active if target_substring.lower() in (c.get('fl_status') or "").lower()]
                    return len(ready), len(active)

                # 1. SYNCING
                self.set_phase("syncing")
                self.add_log("Phase 1: Instructing clients to synchronize student data...")
                
                # Wait for clients to report "Siap Preprocess"
                timeout = 600 
                start_sync = time.time()
                while time.time() - start_sync < timeout:
                    ready, total = get_ready_info("Siap Preprocess")
                    if total > 0 and ready >= total:
                        self.add_log(f"Phase 1 Complete: All {ready} clients synchronized.")
                        break
                    time.sleep(5)
                
                # 2. PREPROCESSING
                self.set_phase("preprocessing")
                self.add_log("Phase 2: Instructing clients to perform face extraction...")
                
                # Wait for clients to report "Siap Training"
                start_prep = time.time()
                while time.time() - start_prep < 3600: 
                    ready, total = get_ready_info("Siap Training")
                    if total > 0 and ready >= total:
                        self.add_log(f"Phase 2 Complete: All {ready} clients reported readiness.")
                        break
                    time.sleep(5)

                # 3. TRAINING
                active_now = self._get_active_clients_list(threshold=900)
                ready_clients = [c for c in active_now if "Siap Training" in (c.get('fl_status') or "")]
                
                if len(ready_clients) < 1:
                    raise Exception("No clients ready for training.")

                self.set_phase("training")
                self.clients_per_round = len(ready_clients)
                self.add_log(f"Phase 3: Starting Flower Server Training with {self.clients_per_round} clients.")
                
                self.start_training(rounds)
                
                while self.running:
                    time.sleep(2)
                
                if self.current_round >= self.total_rounds:
                    self.set_phase("completed")
                    self.add_log("✅ Federated Training Cycle Finished Successfully.")
                
            except Exception as e:
                self.add_log(f"❌ Lifecycle Error: {e}")
                self.running = False
                self.set_phase("idle")

        threading.Thread(target=_lifecycle_task, daemon=True).start()

    def _get_active_clients_list(self, threshold=30):
        from app.controllers.client_controller import registered_clients
        now = time.time()
        active = []
        for cid, cdata in registered_clients.items():
            if now - cdata.get('last_seen', 0) < threshold:
                active.append(cdata)
        return active

    def start_training(self, num_rounds):
        self.running = True
        self.current_round = 0
        self.metrics_history = []
        
        def _srv():
            try:
                def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
                    accuracies = [m[1]["accuracy"] * m[0] for m in metrics]
                    examples = [m[0] for m in metrics]
                    aggregated = {
                        "accuracy": sum(accuracies) / sum(examples) if sum(examples) > 0 else 0.0,
                        "loss": sum([m[1]["loss"] * m[0] for m in metrics]) / sum(examples) if sum(examples) > 0 else 0.0
                    }
                    
                    self.metrics_history.append({
                        "round": self.current_round,
                        "loss": aggregated["loss"],
                        "accuracy": aggregated["accuracy"]
                    })
                    return aggregated

                from flwr.common import parameters_to_ndarrays
                class SaveModelStrategy(fl.server.strategy.FedAvg):
                    def aggregate_fit(self, server_round, results, failures):
                        valid_results = [res for res in results if res[1].num_examples > 0]
                        if not valid_results: return None, {}
                        
                        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, valid_results, failures)
                        
                        if aggregated_parameters is not None:
                            ndarrays = parameters_to_ndarrays(aggregated_parameters)
                            model = MobileFaceNet(embedding_size=128)
                            metric_fc = ArcMarginProduct(128, 100)
                            all_keys = list(model.state_dict().keys()) + list(metric_fc.state_dict().keys())
                            
                            state_dict = OrderedDict()
                            for i, key in enumerate(all_keys):
                                if i < len(ndarrays): state_dict[key] = torch.tensor(ndarrays[i])
                            
                            db = SessionLocal()
                            try:
                                buffer = io.BytesIO()
                                torch.save(state_dict, buffer); blob = buffer.getvalue()
                                new_version = ModelVersion(head_blob=blob, notes=f"Aggregated Model Round {server_round}")
                                db.add(new_version)
                                tr_round = db.query(TrainingRound).filter(TrainingRound.round_number == server_round).first()
                                if not tr_round:
                                    tr_round = TrainingRound(round_number=server_round); db.add(tr_round)
                                tr_round.model_version = new_version
                                if aggregated_metrics:
                                    tr_round.global_accuracy = aggregated_metrics.get("accuracy", 0.0)
                                    tr_round.global_loss = aggregated_metrics.get("loss", 0.0)
                                db.commit()
                            except Exception as e: print(f"Save error: {e}")
                            finally: db.close()
                        return aggregated_parameters, aggregated_metrics

                def fit_config(rnd):
                    self.current_round = rnd; return {"round": rnd}

                strategy = SaveModelStrategy(
                    fraction_fit=1.0, min_fit_clients=self.clients_per_round,
                    on_fit_config_fn=fit_config,
                    fit_metrics_aggregation_fn=weighted_average,
                )
                
                fl.server.start_server(
                    server_address="0.0.0.0:8085",
                    config=fl.server.ServerConfig(num_rounds=num_rounds),
                    strategy=strategy,
                )
            except Exception as e: print(f"FL Error: {e}")
            finally: self.running = False

        self.server_thread = threading.Thread(target=_srv, daemon=True)
        self.server_thread.start()

    def _get_initial_parameters(self):
        db = SessionLocal()
        try:
            latest = db.query(ModelVersion).order_by(ModelVersion.version_id.desc()).first()
            if latest:
                buffer = io.BytesIO(latest.head_blob)
                full_state = torch.load(buffer, map_location="cpu")
                return ndarrays_to_parameters([val.cpu().numpy() for _, val in full_state.items()])
            else:
                model = MobileFaceNet(embedding_size=128)
                metric_fc = ArcMarginProduct(128, 100)
                params = [v.cpu().numpy() for v in model.state_dict().values()]
                params += [v.cpu().numpy() for v in metric_fc.state_dict().values()]
                return ndarrays_to_parameters(params)
        finally: db.close()

    def status(self):
        active_clients = self._get_active_clients_list(threshold=30)
        db = SessionLocal()
        latest = db.query(ModelVersion).order_by(ModelVersion.version_id.desc()).first()
        model_ver = (latest.version_id - 1) if latest else 0
        db.close()
        
        return {
            "running": self.running,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "current_phase": self.current_phase,
            "clients_connected": len(active_clients),
            "model_version": f"v{model_ver}",
            "metrics_history": self.metrics_history,
            "logs": self.phase_logs[-20:]
        }

    def stop(self): self.running = False
    def add_log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.phase_logs.append(f"[{ts}] {msg}")