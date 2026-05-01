import os

try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

# Konfigurasi Jalur Data
DATA_ROOT = os.getenv("DATA_ROOT", "data")
MODEL_DIR = os.getenv("MODEL_DIR", "app/model")
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "global_model_v0.pth")
REGISTRY_PATH = os.path.join(DATA_ROOT, "global_embedding_registry.pth")
BN_PATH = os.path.join(DATA_ROOT, "global_bn_combined.pth")
SUBMISSIONS_DIR = os.path.join(DATA_ROOT, "submissions")

# Parameter Pelatihan (Penyelarasan dengan CL)
# CL: 20 Epochs | FL: 10 Rounds x 2 Epochs = 20 Iterations
TRAINING_PARAMS = {
    "total_rounds": 10,
    "epochs_per_round": 2,
    "batch_size_per_client": 16,
    "mu": 0.05,
    "label_smoothing": 0.0,
    "lr_schedule": "cosine", # Diganti ke Cosine Annealing
    "initial_lr": 0.1,
    "min_lr": 1e-4,
    "swa_start_round": 9,     # SWA Snapshot mulai ronde 9 (Last 2 Rounds = 4 Epochs)
    "use_snapshot_averaging": True
}

# Parameter Ekonomi & Biaya
ECONOMICS = {
    "transmission_cost_per_mb": 3.25,
    "compute_cost_per_kwh": 1444.70,
    "estimated_server_power_kw": 0.1,
    "estimated_client_power_kw": 0.01, # 10W edge device (Raspberry Pi/Jetson)
    "estimated_backbone_size_mb": 4.5,
    "estimated_registry_size_mb": 1.0
}
