import time
from codecarbon import OfflineEmissionsTracker

# Inisialisasi awal waktu dan pencatat daya CodeCarbon
start_train_time = time.time()
tracker = OfflineEmissionsTracker(
    country_iso_code="IDN", 
    measure_power_secs=15, 
    log_level="error", 
    save_to_file=False
)
tracker.start()

# Dapatkan total konsumsi kWh dan durasi detik pelatihan
energy_kwh = tracker.stop()
train_duration = time.time() - start_train_time
