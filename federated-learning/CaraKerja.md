# Cara Kerja Sistem: Federated Face Attendance (Prime State)

Sistem ini dirancang untuk mencapai identifikasi wajah stabil melalui kolaborasi antar terminal tanpa memindahkan data mentah mahasiswa ke server (Privacy-Preserving).

---

## Tahap 1: Fase Persiapan & Sinkronisasi (Discovery)
Langkah awal untuk memastikan semua terminal memiliki "bahasa" yang sama sebelum belajar dimulai:
1.  **Discovery Identitas**: Setiap terminal memindai folder lokal (`raw_data/students`) untuk mendaftarkan ID (NRP) ke server.
2.  **Global Label Map Sync**: Server mengumpulkan semua ID unik dari seluruh terminal dan mengirimkan balik daftar urutan kelas (label map) yang seragam ke semua client. Ini mencegah index klasifikasi tertukar antar terminal.

---

## Tahap 2: Fase Pemrosesan Data (Preprocessing)
Langkah krusial untuk kualitas input yang seragam (Identik dengan CL):
1.  **Laplacian Selection (Top 50)**: Memilih maksimal 50 foto paling tajam per mahasiswa.
2.  **Hardware-Aware Downscaling (New)**: Resize gambar input ke max-width 640px sebelum deteksi untuk menjaga stabilitas RAM pada Raspi 3B.
3.  **MTCNN Face Detection**: Deteksi lokasi wajah pada gambar yang di-downscale.
4.  **Affine Landmark Alignment**: Menyejajarkan posisi mata dan mulut secara horizontal berdasarkan 5 titik landmark.
5.  **Portrait Crop (96x112)**: Memastikan area wajah fokus pada dimensi Portrait yang standar.
5.  **Initial Centroid Generation**: Menghasilkan embedding awal menggunakan model saat ini untuk dikirim ke server sebagai data referensi.

---

## Tahap 3: Pelatihan Terfederasi (Federated Learning)
1. **Low-Batch Local Training**: Client melatih model secara lokal dengan Batch Size **4** guna mencegah OOM pada perangkat Raspberry Pi 3B (1GB RAM) dan menjaga stabilitas gradien.
2. **Partial Freezing**: Client membekukan Stage 1 & 2 secara lokal untuk efisiensi komputasi dan menjaga fitur umum.
3. **Local Training (pFedFace)**: Client melatih Stage 3 dan Head menggunakan SGD (Nesterov) dengan Initial LR **0.01** (10 Ronde x 1 Local Epoch).
4. **Premium Augmentation**: Menggunakan `RandomPerspective`, `GaussianBlur`, `ColorJitter`, dan `RandomErasing` untuk menangani variasi lingkungan terminal.
5. **Hybrid Validation Metrics**: Menghitung akurasi berdasarkan data citra lokal (asli) dan **Global Embeddings** dari terminal lain untuk mencegah bias.
6. **Knowledge Sharing**: Server melakukan FedAvg pada parameter backbone bersama setiap ronde.
7. **Snapshot Averaging**: Menggunakan rata-rata snapshot pada 3 ronde terakhir (8, 9, 10) untuk stabilitas global.

---

## Tahap 4: Finalisasi Registry & Inferensi
1. **Registry Phase**: Client menghitung Centroid identitas menggunakan **50 gambar terbaik** dan mengirimkannya ke server.
2. **Global Integration**: Server menggabungkan centroid dari seluruh client menjadi `global_embedding_registry.pth`.
3. **Source of Truth Priority**: Client memprioritaskan penggunaan **Global Registry** dalam proses pengenalan wajah untuk menghilangkan bias overfitting lokal (pemanfaatan memori global).
4. **Local BN Preservation**: Statistik BatchNorm yang telah dikalibrasi lokal tetap dipertahankan untuk personalisasi normalisasi fitur.

---

## Tahap 5: Fase Inferensi (Live Attendance)
1.  **Eager Loading**: Memuat model versi terbaru ke RAM secara terisolasi.
2.  **Vectorized Identity Identification**: Pencarian identitas menggunakan operasi matriks (`torch.mm`) yang efisien.
3.  **Flip Trick Evaluation**: Merata-ratakan embedding wajah asli dan mirror.
4.  **Production Logging**: Mencatat hasil identifikasi terbaik dengan skor kepercayaan untuk auditabilitas presensi.
5.  **Temporal Voting Strategy**: Verifikasi identitas menggunakan rata-rata buffer temporal untuk stabilitas. Threshold standar ditetapkan pada **0.7**.
6.  **Explicit Memory Cleanup (GC)**: Pembersihan RAM secara rutin melalui `gc.collect()`.
