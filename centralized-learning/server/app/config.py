import os

try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

# Konfigurasi Jalur Data (Dataset & Model Paths)
DATA_ROOT = os.getenv("DATA_ROOT", "data")
UPLOAD_DIR = os.path.join(DATA_ROOT, "students")
PROCESSED_DATA = os.path.join(DATA_ROOT, "datasets_processed")
MODEL_DIR = os.getenv("MODEL_DIR", "app/model")
MODEL_PATH = os.path.join(MODEL_DIR, "global_model.pth")
REF_PATH = os.path.join(MODEL_DIR, "reference_embeddings.pth")
PRETRAINED_PATH = os.path.join(MODEL_DIR, "global_model_v0.pth")
EMISSIONS_DIR = os.path.join(DATA_ROOT, "emissions")

# Parameter Pelatihan (Penyelarasan dengan Optimized SGD Strategy)
TRAINING_PARAMS = {
    "total_epochs": 20,
    "batch_size": 32,
    "label_smoothing": 0.0, # Pure CrossEntropy for better margin expansion
    "lr_schedule": "cosine", # Diganti ke Cosine Annealing
    "initial_lr": 0.1,
    "min_lr": 1e-4,
    "swa_start_epoch": 15,    # SWA Aktif di 5 epoch terakhir
    "swa_lr": 0.01
}

# Parameter Ekonomi & Perhitungan Biaya
# Sesuai dengan tarif real-world (Rp 3,25/MB dan Rp 1.444,70/kWh)
ECONOMICS = {
    "transmission_cost_per_mb": 3.25,
    "compute_cost_per_kwh": 1444.70,
    "estimated_server_power_kw": 0.1,  # 100W laptop/server
    "estimated_backbone_size_mb": 4.5,
    "estimated_registry_size_mb": 1.0
}
