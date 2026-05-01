import os

# Konfigurasi Pelatihan Lokal (Terminal)
TRAINING_CONFIG = {
    "epochs": int(os.getenv("LOCAL_EPOCHS", 5)),
    "learning_rate": float(os.getenv("LOCAL_LR", 0.0001)),
    "mu": float(os.getenv("LOCAL_MU", 0.05)),      # Penalti FedProx
    "lam": float(os.getenv("LOCAL_LAM", 0.1)),    # Parameter tambahan (jika ada)
    "batch_size": int(os.getenv("LOCAL_BATCH_SIZE", 16)),
    "weight_decay": float(os.getenv("LOCAL_WEIGHT_DECAY", 1e-4))
}

# Jalur Data & Model
PATHS = {
    "data_root": os.getenv("DATA_ROOT", "/app/data"),
    "model_dir": os.getenv("MODEL_DIR", "/app/data/models"),
    "raw_data": os.getenv("RAW_DATA", "/app/data/raw")
}
