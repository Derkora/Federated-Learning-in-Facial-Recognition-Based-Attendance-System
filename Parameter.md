# Perbandingan Parameter: Federated vs Centralized Learning

Dokumen ini menyajikan perbandingan parameter teknis antara sistem **Federated Learning (FL)** dan **Centralized Learning (CL)** untuk memastikan keselarasan performa dan transparansi konfigurasi.

| Parameter | Federated Learning (FL) | Centralized Learning (CL) |
| :--- | :--- | :--- |
| **Arsitektur Model** | MobileFaceNet (128-d) | MobileFaceNet (128-d) |
| **Loss Function** | ArcFace (ArcMarginProduct) | ArcFace (ArcMarginProduct) |
| **Optimizer** | Adam | Adam |
| **Learning Rate (Initial)** | 1e-4 (Dynamic) | 1e-3 (Fixed) |
| **Batch Size** | 16 (per client) | 32 (total) |
| **Local Epochs / Epochs** | 3 (per round) | 10 (total) |
| **Training Rounds** | 5 Rounds (Default) | N/A (Single Session) |
| **Min Clients** | 2 | N/A (Direct Upload) |
| **Augmentasi (On-the-fly)** | Yes (Jitter, Rotate, Flip, Erasing) | Yes (Sync with FL Standard) |
| **Dataset Selection** | Laplacian Variance (Top 50) | Laplacian Variance (Top 50) |
| **Label Smoothing** | 0.1 | 0.0 (Standard CE) |
| **FedProx Mu** | 0.05 | N/A |
| **Inference Threshold** | 0.50 (Strict) | 0.50 (Strict) |
| **Input Resolution** | 112x96 (Margin 20) | 112x96 (Margin 20) |

> [!NOTE]
> Parameter CL telah diselaraskan dengan FL untuk menggunakan seleksi gambar berbasis variansi Laplacian (Blur Detection) dan augmentasi *on-the-fly* guna mencapai tingkat akurasi yang setara.
