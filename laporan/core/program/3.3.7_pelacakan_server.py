import os
from datetime import datetime

# Hitung total durasi waktu per ronde global
round_duration = round(
    datetime.now().timestamp() - round_start_time, 2
)

# Hitung transfer bandwidth nyata dari berkas model
bb_size = ECONOMICS["estimated_backbone_size_mb"]
if os.path.exists("data/backbone_pure.pth"):
    bb_size = (
        os.path.getsize("data/backbone_pure.pth") 
        / (1024 * 1024)
    )
backbone_sync_mb = num_clients * bb_size * 2

# Akumulasi energi total server dan seluruh klien
if has_real_server_energy:
    server_energy = max_server_energy
else:
    server_energy = (
        (round_duration / 3600) 
        * server_power_kw
    )

client_energy_total = sum([
    c.get("energy_kwh", 0) for c in clients_data
])
total_energy = server_energy + client_energy_total
