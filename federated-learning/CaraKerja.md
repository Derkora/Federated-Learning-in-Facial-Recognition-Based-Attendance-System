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
1. **Low-Batch Local Training**: Client melatih model secara lokal dengan Batch Size **8** guna mencegah OOM pada perangkat Raspberry Pi 3B (1GB RAM).
2. **Partial Freezing**: Client membekukan Stage 1 & 2 secara lokal untuk efisiensi komputasi dan menjaga fitur umum.
3. **Local Training (pFedFace)**: Client melatih Stage 3 dan Head menggunakan SGD (Nesterov) dengan Initial LR **0.05** (10 Ronde x 2 Local Epoch).
4. **Hybrid Validation Metrics**: Selama training, client menghitung akurasi berdasarkan data citra lokal (asli) dan **Global Embeddings** dari terminal lain. Hal ini memungkinkan terminal melaporkan akurasi tinggi (±90%) meskipun tidak memiliki data subjek tertentu secara lokal (Knowledge Sharing Proof).
5. **Knowledge Sharing**: Server melakukan FedAvg pada parameter backbone bersama (Shared Parameters) setiap ronde.
6. **Snapshot Averaging**: Menggunakan rata-rata snapshot pada 2 ronde terakhir (Setara 4 epoch) untuk stabilitas global.

---

## Tahap 4: Finalisasi Registry & Inferensi
1. **Registry Phase**: Client menghitung Centroid identitas menggunakan **50 gambar terbaik** dan mengirimkannya ke server.
2. **Global Integration**: Server menggabungkan centroid dari seluruh client menjadi `global_embedding_registry.pth`.
3. **Cross-Client Recognition**: Client mengunduh registry global sehingga bisa mengenali mahasiswa yang terdaftar di client lain dengan akurasi tinggi.
ru yang sudah melalui proses federasi.
2.  **Local BN Preservation**: Statistik Batch Normalization (mean/variance) yang telah dikalibrasi secara lokal tetap dipertahankan untuk memastikan normalisasi fitur yang optimal sesuai kondisi cahaya terminal (Inherent Personalization).
3.  **Centroid Re-calculation**: Terminal menghitung ulang "titik tengah" (centroid) embedding setiap mahasiswa menggunakan kombinasi Backbone Global + Local BN untuk disimpan sebagai registri identitas universal.

---

## Tahap 5: Fase Inferensi (Live Attendance)
1.  **Eager Loading**: Memuat model versi terbaru ke RAM secara terisolasi.
2.  **Vectorized Identity Identification (New)**: Pencarian identitas menggunakan operasi matriks (`torch.mm`) yang jauh lebih efisien daripada loop manual.
3.  **Flip Trick Evaluation**: Merata-ratakan embedding wajah asli dan mirror untuk hasil skor yang lebih stabil.
4.  **Temporal Voting & CIM**: Menggunakan buffer frame untuk konfirmasi identitas, mendukung *Instant Match* jika skor > 0.85.
5.  **Explicit Memory Cleanup (GC)**: Pembersihan RAM secara rutin melalui `gc.collect()` untuk menjaga ketersediaan memori di perangkat edge.
