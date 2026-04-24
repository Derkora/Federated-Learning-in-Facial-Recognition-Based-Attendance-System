# Perbandingan Sistem: Centralized Learning vs Federated Learning

Laporan ini membandingkan pendekatan baseline **Centralized Learning (CL)** dengan sistem **Federated Learning (FL)** yang diusulkan untuk aplikasi absensi berbasis pengenalan wajah.

## 1. Perbandingan Arsitektur

| Komponen | Centralized Learning (Baseline) | Federated Learning (Diambil) |
| :--- | :--- | :--- |
| **Aliran Data** | **Data-to-Global**: Gambar wajah mentah dikirim ke server dalam bentuk berkas ZIP. | **Model-to-Data**: Data mentah tetap berada di perangkat edge. Hanya bobot model yang dibagikan. |
| **Struktur Model** | **Monolitik**: Satu model MobileFaceNet (128-dim) dilatih pada seluruh data gabungan. | **Hybrid (pFedFace)**: Global Backbone (MobileFaceNet) + Local Head (ArcMargin) untuk identitas unik. |
| **Strategi Pelatihan** | **Bulk Training**: Server melatih satu model global sekaligus setelah seluruh data diterima. | **Iterative Training**: Klien melatih secara lokal menggunakan FedProx dan agregasi melalui Flower. |
| **Privasi** | Rendah (Server melihat seluruh data wajah). | **Tinggi** (Server tidak pernah menerima gambar mentah). |

## 2. Keselarasan Parameter Pelatihan
Untuk memastikan validitas perbandingan, kedua sistem mengikuti konfigurasi pelatihan yang identik:

| Parameter | Centralized Learning (CL) | Federated Learning (FL) | Catatan |
| :--- | :--- | :--- | :--- |
| **Arsitektur Backbone** | MobileFaceNet (128-dim) | MobileFaceNet (128-dim) | Identik untuk validitas fitur. |
| **Loss Function** | ArcMarginProduct | ArcMarginProduct | Standar pengenalan wajah modern. |
| **Optimizer** | Adam | Adam | Konsistensi dalam pembaruan bobot. |
| **Learning Rate (LR)** | 1e-4 (E1-10), 5e-5 (E11-15), 1e-5 (E16-20) | 1e-4 (R1-5), 5e-5 (R6-8), 1e-5 (R9-10) | Menghindari bias kecepatan belajar. |
| **Total Iterasi Data** | 20 Epoch | (10 Ronde x 2 Epoch) = 20 Iterasi | Frekuensi melihat data yang setara. |
| **Batch Size** | 32 (Total) | 16 (per klien) = 32 (Total) | Beban gradien yang setara. |
| **Kualitas Input** | Top 50 (Laplacian Var) | Top 50 (Laplacian Var) | Standar kualitas input yang sama. |
| **Resolusi Input** | 112 x 96 (Portrait) | 112 x 96 (Portrait) | Tanpa distorsi aspek rasio. |
| **Engine Inferensi** | **Full PyTorch (CPU)** | **Full PyTorch (CPU)** | Migrasi dari ONNX demi stabilitas. |
| **Threshold** | 0.60 | 0.60 | Ambang batas kemiripan yang selaras. |
| **Sync Registry** | Centroid Embedding | Centroid Embedding | Metode pencocokan identitas yang sama. |

## 3. Inovasi Federated Learning (Diambil)

Sistem FL yang diusulkan memperkenalkan beberapa teknik tingkat lanjut:

*   **pFedFace (Personalized Federated Face)**: Memisahkan pengetahuan ekstraksi fitur (global) dari pengetahuan identitas (lokal).
*   **Global BN Merging**: Server menggabungkan statistik Batch Normalization (mean/variance) dari seluruh klien, yang secara signifikan menstabilkan akurasi inferensi lintas perangkat.
*   **Knowledge Sharing (Centroids)**: Klien bertukar centroid wajah yang telah anonim, memungkinkan model "mengenal" mahasiswa dari terminal lain tanpa melihat foto mereka.
*   **FedProx Optimization**: Menggunakan parameter proximal ($\mu$) untuk menangani data non-IID (Independent and Identically Distributed) yang umum pada pengenalan wajah.

## 4. Perbandingan Alur Kerja

### Alur Kerja Centralized
1.  **Klien**: Preprocessing -> Kemas ZIP -> Unggah ke Server.
2.  **Server**: Ekstrak ZIP -> Pelatihan Model -> Pembuatan Registri Embedding.
3.  **Klien**: Unduh model `.pth` dan registri `reference_embeddings.pth` untuk inferensi.

### Alur Kerja Federated (Dynamic Barrier Sync)
1.  **Fase Discovery**: Pendaftaran ID Mahasiswa ke Global Map di server.
2.  **Fase Preprocess**: Pemotongan wajah lokal dan pemilihan 50 gambar tertajam.
3.  **Fase Training**: Ronde pelatihan paralel menggunakan Flower dan FedProx.
4.  **Fase Registry**: Sinkronisasi Global BN diikuti pembuatan database identitas universal.

## Kesimpulan
Sistem **Federated Learning** menawarkan alternatif yang menjaga privasi dibandingkan baseline Centralized tanpa mengorbankan performa pengenalan. Dengan memanfaatkan **pFedFace**, **Global BN Merging**, dan **Knowledge Sharing**, sistem ini mampu mengatasi tantangan khas FL seperti heterogenitas identitas dan model drift.
