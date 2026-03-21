import threading
import flwr as fl
import torch
import io
import time
from datetime import datetime
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
        
        # Lifecycle Phases: "idle", "syncing", "preprocessing", "training"
        self.current_phase = "idle"
        self.phase_logs = []
        self.model_version = 0 # Tracked from DB

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

    def run_lifecycle(self, rounds: int = 10):
        if self.running or self.current_phase != "idle":
            return
        
        self.total_rounds = rounds
        self.phase_logs = []
        self.add_log("🚀 Starting Full FL Training Cycle...")
        
        def _lifecycle_task():
            try:
                from app.controllers.client_controller import registered_clients

                def get_active_clients(threshold=60):
                    now = time.time()
                    active = []
                    for c in registered_clients.values():
                        try:
                            # Robust parsing for both float and string-encoded float
                            ls = float(c.get('last_seen', 0))
                            if (now - ls) < threshold:
                                active.append(c)
                        except: continue
                    return active

                # Phase 1: Syncing
                self.current_phase = "syncing"
                self.add_log("Phase 1: Client Synchronization (Global Label Mapping)...")
                
                # Wait for clients to report Syncing status
                start_wait = time.time()
                while time.time() - start_wait < 120: 
                    active = get_active_clients()
                    if active and all("Syncing" in c.get('fl_status', '') for c in active):
                        break
                    time.sleep(3)
                
                # Phase 2: Preprocessing
                self.current_phase = "preprocessing"
                self.add_log("Phase 2: Persistent Client Image Preprocessing (MTCNN 112x96)...")
                
                # Wait for clients to finish preprocessing and return to Standby
                start_wait = time.time()
                last_diag_time = 0
                while time.time() - start_wait < 3600: # Max 1 hour
                    # Relax threshold to 15 minutes during preprocessing
                    active = get_active_clients(threshold=900)
                    
                    # Periodic Diagnostic Log (every 10s)
                    if time.time() - last_diag_time > 10:
                        statuses = [f"{c.get('id','unk')[:6]}:{c.get('fl_status')}" for c in active]
                        print(f"[LIFECYCLE DIAG] Active (15m threshold): {len(active)} clients. Statuses: {statuses}")
                        last_diag_time = time.time()

                    if not active:
                        time.sleep(5)
                        continue

                    # Robust check for Standby or Ready status
                    is_ready = lambda s: s and ("Standby" in s or "Ready" in s or "Siap" in s)
                    if all(is_ready(c.get('fl_status')) for c in active):
                        self.add_log("✅ All clients ready for training.")
                        break
                    
                    if any("Error" in (c.get('fl_status') or "") for c in active):
                        self.add_log("⚠️ Some clients reported errors during preprocessing.")
                    time.sleep(5)

                # Phase 3: Training
                active_now = get_active_clients(threshold=900)
                self.clients_per_round = len(active_now)
                if self.clients_per_round == 0:
                    raise Exception("No active clients found for training phase (after 15m idle).")

                self.current_phase = "training"
                self.add_log(f"Phase 3: Starting FL Training with {self.clients_per_round} clients...")
                self.start_training(rounds)
                
                # Wait for training to finish
                while True:
                    time.sleep(2)
                    if not self.running:
                        # Success check: did we reach the target rounds?
                        if self.current_round >= self.total_rounds:
                            break
                        # Error check: did it stop with an error?
                        if self.end_time > 0 and self.current_round < self.total_rounds:
                            raise Exception(f"Training stopped prematurely at round {self.current_round}")
                        break
                
                self.add_log("✅ Full FL Cycle Completed successfully.")
                self.current_phase = "idle"
                
            except Exception as e:
                self.add_log(f"❌ Lifecycle Error: {e}")
                print(f"[LIFECYCLE ERROR] {e}")
                self.current_phase = "idle"
                self.running = False

        threading.Thread(target=_lifecycle_task, daemon=True).start()

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

            from flwr.common import parameters_to_ndarrays
            import torch

            class SaveModelStrategy(fl.server.strategy.FedAvg):
                def aggregate_fit(self, server_round, results, failures):
                    aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                    
                    if aggregated_parameters is not None:
                        # Convert Flower parameters to NDArrays
                        ndarrays = parameters_to_ndarrays(aggregated_parameters)
                        
                        # Load model architecture to get state_dict keys
                        temp_model = MobileFaceNet(embedding_size=128)
                        params_dict = zip(temp_model.state_dict().keys(), ndarrays)
                        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                        
                        # Save to Database
                        db = SessionLocal()
                        try:
                            buffer = io.BytesIO()
                            torch.save(state_dict, buffer)
                            blob = buffer.getvalue()
                            
                            new_version = ModelVersion(
                                head_blob=blob,
                                notes=f"Aggregated Model Round {server_round}"
                            )
                            db.add(new_version)
                            
                            # Update TrainingRound if exists
                            tr_round = db.query(TrainingRound).filter(TrainingRound.round_number == server_round).first()
                            if tr_round:
                                tr_round.model_version = new_version
                                if aggregated_metrics:
                                    tr_round.global_accuracy = aggregated_metrics.get("accuracy", 0.0)
                                    tr_round.global_loss = aggregated_metrics.get("loss", 0.0)
                            else:
                                tr_round = TrainingRound(
                                    round_number=server_round,
                                    model_version=new_version,
                                    global_accuracy=aggregated_metrics.get("accuracy", 0.0) if aggregated_metrics else 0.0,
                                    global_loss=aggregated_metrics.get("loss", 0.0) if aggregated_metrics else 0.0
                                )
                                db.add(tr_round)
                                
                            db.commit()
                            print(f"[FL SERVER] Round {server_round} aggregated model saved to DB.")
                        except Exception as e:
                            print(f"[FL SERVER] Error saving aggregated model: {e}")
                        finally:
                            db.close()
                            
                    return aggregated_parameters, aggregated_metrics

            strategy = SaveModelStrategy(
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

    def add_log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.phase_logs.append(f"[{timestamp}] {msg}")
        if len(self.phase_logs) > 50: self.phase_logs.pop(0)

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
        
        # Determine model version from DB 
        db = SessionLocal()
        latest = db.query(ModelVersion).order_by(ModelVersion.version_id.desc()).first()
        self.model_version = (latest.version_id - 1) if latest else 0
        db.close()

        return {
            "running": self.running and not is_finished,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "elapsed_time": time_str, 
            "bandwidth_kb": bandwidth_str,
            "metrics_history": sorted(self.metrics_history, key=lambda x: x['round']),
            "current_phase": self.current_phase,
            "phase_logs": self.phase_logs,
            "model_version": self.model_version
        }