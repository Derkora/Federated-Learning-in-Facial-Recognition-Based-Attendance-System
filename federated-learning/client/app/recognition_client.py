import flwr as fl
import torch
import numpy as np
import os
import os as _os
import json
import requests
import base64
import io
import time
import gc
import traceback
try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    OfflineEmissionsTracker = None
from app.utils.trainer import LocalTrainer, TrainingNaNError
from app.utils.mobilefacenet import MobileFaceNet, ArcMarginProduct
from app.db.db import SessionLocal
from app.db.models import EmbeddingLocal, UserLocal
from app.utils.logging import get_logger

class FaceRecognitionClient(fl.client.NumPyClient):
    def __init__(self, model, head, data_path="/app/data", device="cpu"):
        self.model = model
        self.head = head
        self.data_path = data_path
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.label_map = None
        self.logger = get_logger()
        
        # Prioritas: Muat dari arsip lokal (Penguasaan Label Permanen)
        map_path = os.path.join(self.data_path, "models", "label_map.json")
        if os.path.exists(map_path):
            with open(map_path, "r") as f:
                self.label_map = json.load(f)
                self.logger.info(f"Inisialisasi berhasil dengan Global Label Map arsip: {len(self.label_map)} identitas.")
        
        processed_path = os.path.join(self.data_path, "processed")
        self.trainer = LocalTrainer(self.model, self.head, device=self.device, data_path=processed_path)
        
        head_path = os.path.join(self.data_path, "models", "local_head.pth")
        if os.path.exists(head_path) and self.head is not None:
            try:
                self.logger.info(f"Memuat local classifier head dari {head_path}...")
                self.head.load_state_dict(torch.load(head_path, map_location=self.device))
            except Exception as e:
                self.logger.warn(f"Berkas local_head.pth rusak atau tidak kompatibel, dilewati: {e}")
                _os.remove(head_path)  # Hapus file rusak agar tidak crash lagi

    def get_parameters(self, config):
        return self.trainer.get_backbone_parameters(personalized=True)

    # Pelatihan Model Lokal (Fit)
    def fit(self, parameters, config):
        # Pastikan model inferensi telah termuat dengan aman ke RAM
        if hasattr(self, 'fl_manager'):
            self.fl_manager._ensure_models_loaded()
            
        rnd = config.get("round", 0)
        self.logger.info(f"FL Fit [Ronde {rnd}]: Menerima {len(parameters)} parameter global... Config: {config}")
        
        # Sinkronisasi pemetaan label kelas mahasiswa terpadu
        if not self.label_map and "label_map" in config:
            self.label_map = json.loads(config["label_map"])
            self.logger.info("Menerima label_map via konfigurasi fit server.")
        
        # Salin bobot model global terbaru ke backbone lokal perangkat
        try:
            if len(parameters) > 0:
                self.trainer.set_backbone_parameters(parameters, personalized=True)
        except Exception as e:
            self.logger.error(f"Gagal mengatur parameter backbone: {e}")
            return parameters, 0, {"loss": 0.0, "accuracy": 0.0, "status": "ParameterError", "hostname": os.getenv("HOSTNAME", "unknown"), "epoch_history": "[]"}
        
        # Ambil konfigurasi parameter pelatihan untuk ronde berjalan
        epochs = config.get("local_epochs", 3)
        lr = config.get("lr", 0.0001)
        mu = config.get("mu", 0.01)
        lam = config.get("lambda", 0.1)

        # Muat memori global (global embeddings) dari database SQLite lokal
        global_embs = []
        db = SessionLocal()
        try:
            items = db.query(EmbeddingLocal).filter_by(is_global=True).all()
            for item in items:
                emb_np = np.frombuffer(item.embedding_data, dtype=np.float32).copy()
                global_embs.append({"nrp": item.user_id, "embedding": torch.from_numpy(emb_np)})
            
            local_user_count = db.query(UserLocal).count()
            self.logger.info(f"Dataset: {local_user_count} pengguna lokal, {len(global_embs)} memori global terdaftar.")
        except Exception as e:
            self.logger.error(f"Gagal memuat informasi dataset: {e}")
        finally:
            db.close()

        # Sinkronkan indeks NRP pada classifier head agar klop
        if self.label_map:
            self.trainer.nrp_to_idx = {nrp: idx for idx, nrp in enumerate(self.label_map)}

        loss, accuracy, num_samples, epoch_history = 0.0, 0.0, 0, []
        train_duration = 0.0
        status = "Skipped"

        # Aktifkan pelacak konsumsi daya listrik perangkat edge (CodeCarbon)
        start_fit_time = time.time()
        tracker = None
        energy_kwh = 0.0
        if CODECARBON_AVAILABLE and OfflineEmissionsTracker is not None:
            try:
                tracker = OfflineEmissionsTracker(country_iso_code="IDN", measure_power_secs=15, log_level="error", save_to_file=False)
                tracker.start()
            except: pass

        session_id = config.get("session_id", "")
        active_dataset = config.get("dataset", "students")
        if active_dataset == "students":
            processed_path = os.path.join(self.data_path, "processed")
        else:
            processed_path = os.path.join(self.data_path, f"processed_{active_dataset}")
        self.trainer.data_path = processed_path

        # Jalankan proses pelatihan PyTorch (Local Training)
        try:
            if hasattr(self, 'fl_manager'):
                total_rounds = config.get("total_rounds", 10)
                self.fl_manager.fl_round = rnd
                self.fl_manager.fl_status = f"Pelatihan: Ronde {rnd}/{total_rounds}"
            
            loss, accuracy, num_samples, epoch_history, train_duration = self.trainer.train(
                epochs=epochs, lr=lr, round_num=rnd,
                global_embeddings=global_embs,
                label_map=self.label_map,
                mu=mu, lam=lam,
                session_id=session_id,
                status_callback=lambda e, total_e, l, a: self._update_training_status(rnd, e, total_e, l, a, config.get("total_rounds", 10))
            )

            status = "Success"
            # Hapus file checkpoint lama karena latihan ronde ini berhasil selesai
            try:
                if session_id:
                    checkpoint_name = f"training_checkpoint_{session_id}_round{rnd}.pth"
                else:
                    checkpoint_name = f"training_checkpoint_round{rnd}.pth"
                ckpt_path = os.path.join(self.data_path, "models", checkpoint_name)
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                    self.logger.success("Pembersihan checkpoint berhasil.")
            except: pass
        except TrainingNaNError as e:
            self.logger.error(f"Error nilai NaN terdeteksi: {e}")
            status = "NaNError"
        except Exception as e:
            self.logger.error(f"Kegagalan pelatihan lokal (non-fatal): {e}")
            self.logger.error(traceback.format_exc())
            status = "TrainingError"

        # Simpan file fisik model terbaru hasil training secara atomik (backbone & local head)
        try:
            model_dir = os.path.join(self.data_path, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            backbone_path = os.path.join(model_dir, "backbone.pth")
            tmp_backbone_path = backbone_path + ".tmp"
            torch.save(self.model.state_dict(), tmp_backbone_path)
            os.replace(tmp_backbone_path, backbone_path)
            
            head_path = os.path.join(model_dir, "local_head.pth")
            tmp_head_path = head_path + ".tmp"
            torch.save(self.trainer.head.state_dict(), tmp_head_path)
            os.replace(tmp_head_path, head_path)
        except Exception as e:
            self.logger.error(f"Gagal menyimpan checkpoint final secara atomik: {e}")

        # Lakukan uji coba validasi internal model terbaru
        val_loss, val_accuracy = 0.0, 0.0
        try:
            val_loss, val_accuracy, _ = self.trainer.evaluate(global_embeddings=global_embs, label_map=self.label_map)
        except Exception as e:
            self.logger.error(f"Evaluasi gagal (non-fatal): {e}")

        # Hentikan pelacak daya listrik dan bersihkan memori RAM dari tensor
        del global_embs
        gc.collect()
            
        fit_duration = train_duration
        if tracker:
            try:
                energy_kwh = tracker.stop()
                if energy_kwh is None: energy_kwh = 0.0
            except: pass

        return self.trainer.get_backbone_parameters(personalized=True), num_samples, {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
            "status": status,
            "duration_s": float(fit_duration),
            "energy_kwh": float(energy_kwh),
            "hostname": os.getenv("HOSTNAME", "unknown-client"),
            "epoch_history": json.dumps(epoch_history)
        }

    def _update_training_status(self, rnd, epoch, total_epochs, loss, acc, total_rounds):
        if hasattr(self, 'fl_manager'):
            msg = f"Pelatihan: Ronde {rnd}/{total_rounds} | Epoch {epoch}/{total_epochs} | Loss: {loss:.4f} | Akurasi: {acc*100:.1f}%"
            self.fl_manager.fl_status = msg
            self.logger.info(msg)

    # Evaluasi Validasi Lokal (Evaluate)
    def evaluate(self, parameters, config):
        if hasattr(self, 'fl_manager'):
            self.fl_manager._ensure_models_loaded()
            
        active_dataset = config.get("dataset")
        if active_dataset:
            if active_dataset == "students":
                processed_path = os.path.join(self.data_path, "processed")
            else:
                processed_path = os.path.join(self.data_path, f"processed_{active_dataset}")
            self.trainer.data_path = processed_path

        # Terapkan bobot model global terbaru ke backbone lokal
        try:
            self.trainer.set_backbone_parameters(parameters, personalized=True)
        except Exception as e:
            self.logger.error(f"Gagal menyetel parameter untuk evaluasi: {e}")
            return 0.0, 0, {"accuracy": 0.0}

        # Hitung statistik nilai loss dan akurasi pada dataset evaluasi
        try:
            loss, accuracy, num_samples = self.trainer.evaluate(global_embeddings=[], label_map=self.label_map)
        except Exception as e:
            self.logger.error(f"Evaluasi gagal: {e}")
            loss, accuracy, num_samples = 0.0, 0.0, 0

        gc.collect()
        return float(loss), num_samples, {"accuracy": float(accuracy), "loss": float(loss)}

def start_fl_client(server_address, model, head, data_path="/app/data", device="cpu"):
    client = FaceRecognitionClient(model, head, data_path, device)
    fl.client.start_numpy_client(server_address=server_address, client=client)
