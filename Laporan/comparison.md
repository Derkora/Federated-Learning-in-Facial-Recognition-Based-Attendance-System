# Perbandingan Sistem: Centralized Learning vs Federated Learning (Revisi Final)

Laporan ini membandingkan pendekatan **Centralized Learning (CL)** dengan sistem **Federated Learning (FL)** yang dioptimasi (pFedFace) untuk aplikasi absensi berbasis wajah.

## 1. Perbandingan Arsitektur & Alur Data

| Komponen | Centralized Learning (CL) | Federated Learning (FL - pFedFace) |
| :--- | :--- | :--- |
| **Aliran Data** | **Data-to-Model**: Citra wajah mentah dikirim ke server pusat. | **Model-to-Data**: Citra wajah mentah tidak pernah keluar dari terminal. |
| **Pusat Pengetahuan** | **Monolitik**: Satu model global untuk semua identitas. | **Hybrid**: Global Backbone (Fitur Umum) + Local Head (Identitas Spesifik). |
| **Keamanan Identitas** | Bergantung pada keamanan server pusat. | **Sangat Tinggi**: Identitas unik (Head) tidak pernah dibagikan. |
| **Optimasi Agregasi** | SWA (Last 5 Epochs). | Snapshot Averaging (Last 3 Rounds). |

## 2. Keselarasan Parameter (Parity)
Kedua sistem menggunakan parameter yang identik untuk memastikan perbandingan yang adil:

| Parameter | Centralized Learning (CL) | Federated Learning (FL) |
| :--- | :--- | :--- |
| **Total Iterasi** | 20 Epochs | 10 Rounds x 2 Local Epochs (Total 20) |
| **Partial Freezing**| **Mode: Early (Stage 1 & 2)** | **Mode: Early (Stage 1 & 2)** |
| **BN Adaptation** | **Client-side Calibration** | **Inherent (Local BN Storage)** |
| **Booster (SWA)**  | **SWA (Last 4 Epochs)** | **Snapshot Avg (Last 2 Rounds/4 Epochs)** |
| **Centroid Data** | **Full Dataset (50 Images)** | **Full Dataset (50 Images)** |
| **Threshold** | 0.75 (CIM: 0.85) | 0.75 (CIM: 0.85) |
| **Metode Inferensi** | Flip Trick + Temporal Voting | Flip Trick + Temporal Voting |

## 3. Analisis Perbedaan Operasional

### Keunggulan Centralized (CL)
- **Kemudahan Agregasi**: Karena seluruh data ada di satu tempat, perhitungan centroid dan normalisasi data sangat mudah dilakukan.
- **Konvergensi Cepat**: Model belajar dari distribusi data yang lengkap sejak awal (IID).

### Keunggulan Federated (FL)
- **Privasi Maksimal**: Menghilangkan risiko kebocoran data biometrik massal di server.
- **Adaptasi Lokal (Personalization)**: Melalui pFedFace, terminal bisa memiliki bobot BatchNorm yang spesifik untuk kondisi pencahayaan di lokasi tersebut tanpa mengganggu terminal lain.
- **Efisiensi Bandwidth**: Hanya mengirimkan bobot model (beberapa MB) alih-alih ribuan foto (ratusan MB).

## 4. Kesimpulan
Sistem **Federated Learning** dengan arsitektur **pFedFace** terbukti mampu menyamai performa sistem Centralized dalam hal stabilitas pengenalan wajah, sekaligus memberikan perlindungan privasi yang jauh lebih unggul bagi data biometrik mahasiswa.
