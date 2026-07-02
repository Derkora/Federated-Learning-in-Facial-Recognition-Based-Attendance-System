import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter

# Create output directory if it doesn't exist
os.makedirs('../laporan/core/gambar', exist_ok=True)

# ──────────────────────────────────────────────
# Raw data
# ──────────────────────────────────────────────
rounds = list(range(1, 11))

# Federated Learning – Global (Server)
fl_server = {
    'loss':     [7.9577, 4.7032, 3.3332, 3.1460, 2.3292, 2.0002, 1.7754, 1.8225, 1.6027, 1.5366],
    'acc':      [91.64,  94.85,  96.78,  96.42,  97.43,  97.98,  97.24,  97.79,  98.25,  98.16],
    'val_loss': [2.4800, 1.6627, 0.9307, 0.7207, 0.5002, 0.3776, 0.3158, 0.2927, 0.3054, 0.3599],
    'val_acc':  [98.29,  97.59,  98.29,  98.28,  98.98,  98.97,  99.31,  98.97,  99.66,  99.66],
}

# Federated Learning – Client 1 (Jetson Nano)
fl_client1 = {
    'loss':     [8.2696, 5.2434, 3.6089, 2.9653, 2.3636, 2.0452, 1.9291, 1.7294, 1.4726, 1.5759],
    'acc':      [91.54,  93.57,  95.77,  96.14,  97.43,  97.98,  96.51,  98.16,  99.08,  98.16],
    'val_loss': [2.0359, 1.3771, 0.5229, 0.7914, 0.2929, 0.2478, 0.2320, 0.2113, 0.2119, 0.2532],
    'val_acc':  [99.30,  97.90,  99.30,  98.60, 100.00,  99.30,  99.30,  99.30, 100.00, 100.00],
}

# Federated Learning – Client 2 (Raspberry Pi)
fl_client2 = {
    'loss':     [7.6457, 4.1630, 3.0576, 3.3267, 2.2948, 1.9553, 1.6216, 1.9157, 1.7329, 1.4974],
    'acc':      [91.73,  96.14,  97.79,  96.69,  97.43,  97.98,  97.98,  97.43,  97.43,  98.16],
    'val_loss': [2.9241, 1.9483, 1.3384, 0.6501, 0.7075, 0.5075, 0.3997, 0.3741, 0.3990, 0.4666],
    'val_acc':  [97.28,  97.28,  97.28,  97.96,  97.96,  98.64,  99.32,  98.64,  99.32,  99.32],
}

# Centralized Learning
cl_data = {
    'loss':     [14.7202, 12.7373, 11.1575, 10.4583, 9.0467, 7.7791, 6.3492, 5.6795, 4.8377, 4.7318],
    'acc':      [81.32,   92.19,   92.72,   93.25,   95.09,  94.82,  96.75,  95.88,  97.28,  96.93],
    'val_loss': [7.5392,  3.7419,  1.9923,  1.1129,  0.8387, 0.6262, 0.6526, 0.5447, 0.5619, 0.5457],
    'val_acc':  [92.66,   97.20,   97.55,   98.95,   98.60,  99.30,  99.30,  98.95,  98.60,  98.95],
}

# ──────────────────────────────────────────────
# Helper: Format desimal koma untuk sumbu Y
# ──────────────────────────────────────────────
def comma_formatter(x, pos):
    return f"{x:g}".replace('.', ',')

# ──────────────────────────────────────────────
# Helper: Plot TERPISAH (Hanya Loss)
# ──────────────────────────────────────────────
def plot_loss(data, filename, title, x_label='Ronde'):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(rounds, data['loss'],     'b-o', label='Training Loss',     linewidth=2, markersize=5)
    ax.plot(rounds, data['val_loss'], 'r-s', label='Validation Loss',  linewidth=2, markersize=5)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)
    ax.set_xticks(rounds)
    ax.yaxis.set_major_formatter(FuncFormatter(comma_formatter))
    
    plt.tight_layout()
    out = os.path.join('../laporan/core/gambar', filename)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

# ──────────────────────────────────────────────
# Helper: Plot TERPISAH (Hanya Akurasi)
# ──────────────────────────────────────────────
def plot_acc(data, filename, title, x_label='Ronde'):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(rounds, data['acc'],     'g-o', label='Training Accuracy',     linewidth=2, markersize=5)
    ax.plot(rounds, data['val_acc'], 'm-s', label='Validation Accuracy',  linewidth=2, markersize=5)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel('Akurasi (%)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=9)
    ax.set_xticks(rounds)
    ax.set_ylim(75, 102)
    ax.yaxis.set_major_formatter(FuncFormatter(comma_formatter))
    
    plt.tight_layout()
    out = os.path.join('../laporan/core/gambar', filename)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")

# ──────────────────────────────────────────────
# Helper: Plot GABUNGAN (Loss & Akurasi berdampingan)
# ──────────────────────────────────────────────
def plot_combined(data, filename, x_label='Ronde'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Kiri: Kurva Loss
    ax1.plot(rounds, data['loss'],     'b-o', label='Training Loss',     linewidth=2, markersize=5)
    ax1.plot(rounds, data['val_loss'], 'r-s', label='Validation Loss',  linewidth=2, markersize=5)
    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Kurva Loss', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=9)
    ax1.set_xticks(rounds)
    ax1.yaxis.set_major_formatter(FuncFormatter(comma_formatter))
    
    # Kanan: Kurva Akurasi
    ax2.plot(rounds, data['acc'],     'g-o', label='Training Accuracy',     linewidth=2, markersize=5)
    ax2.plot(rounds, data['val_acc'], 'm-s', label='Validation Accuracy',  linewidth=2, markersize=5)
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel('Akurasi (%)', fontsize=11)
    ax2.set_title('Kurva Akurasi', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=9)
    ax2.set_xticks(rounds)
    ax2.set_ylim(75, 102)
    ax2.yaxis.set_major_formatter(FuncFormatter(comma_formatter))
    
    plt.tight_layout()
    out = os.path.join('../laporan/core/gambar', filename)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Combined: {out}")

# ──────────────────────────────────────────────
# Generate all charts (12 files total)
# ──────────────────────────────────────────────

# --- 1. Federated Learning Global (Server) ---
plot_loss(fl_server, 'chart_fl_server_loss.png', 'Kurva Loss Federated Learning')
plot_acc (fl_server, 'chart_fl_server_acc.png', 'Kurva Akurasi Federated Learning')
plot_combined(fl_server, 'chart_fl_server.png')

# --- 2. Federated Learning Client 1 (Jetson Nano) ---
plot_loss(fl_client1, 'chart_fl_client1_loss.png', 'Kurva Loss (Jetson Nano)')
plot_acc (fl_client1, 'chart_fl_client1_acc.png', 'Kurva Akurasi (Jetson Nano)')
plot_combined(fl_client1, 'chart_fl_client1.png')

# --- 3. Federated Learning Client 2 (Raspberry Pi) ---
plot_loss(fl_client2, 'chart_fl_client2_loss.png', 'Kurva Loss (Raspberry Pi)')
plot_acc (fl_client2, 'chart_fl_client2_acc.png', 'Kurva Akurasi (Raspberry Pi)')
plot_combined(fl_client2, 'chart_fl_client2.png')

# --- 4. Centralized Learning ---
plot_loss(cl_data, 'chart_cl_loss.png', 'Kurva Loss Centralized Learning', x_label='Epoch')
plot_acc (cl_data, 'chart_cl_acc.png', 'Kurva Akurasi Centralized Learning', x_label='Epoch')
plot_combined(cl_data, 'chart_cl.png', x_label='Epoch')