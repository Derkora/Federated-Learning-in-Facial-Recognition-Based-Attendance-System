# Cara Kerja Sistem: Federated Face Attendance

Sistem ini dirancang untuk mencapai identifikasi wajah yang stabil dengan tingkat kepercayaan tinggi melalui kolaborasi antar perangkat tanpa memindahkan data mentah mahasiswa ke server.

## Tahap 1: Data Engineering & Mandatory Preprocessing
Sistem memastikan kualitas data lokal sebelum masuk ke fase pelatihan.
- **Multi-Trial Face Detection**: Sistem melakukan percobaan deteksi wajah hingga 5 kali per identitas. Jika deteksi gagal, sistem menggunakan gambar alternatif dengan tingkat ketajaman (Laplacian Variance) tertinggi berikutnya.
- **MTCNN Face Cropping**: Wajah dipotong secara presisi dengan margin tertentu untuk menghilangkan porsi latar belakang yang tidak relevan.
- **112x96 Dimensional Alignment**: Hasil potongan wajah diubah ukurannya ke dimensi **112x96** (Portrait). Standarisasi dimensi ini sangat krusial agar sinkron dengan input arsitektur MobileFaceNet.

## Tahap 2: Inisialisasi Arsitektur & Optimasi (pFedFace - Hybrid)
Struktur ini memisahkan pengetahuan global dan identitas lokal untuk akurasi maksimal.
- **Global Backbone (MobileFaceNet)**: Ekstraktor fitur universal yang menggunakan **SGD with Nesterov Momentum** dan per-layer weight decay untuk mempelajari struktur wajah secara general.
- **Global BN Merging**: Server menggabungkan statistik Batch Normalization (mean/variance) dari seluruh klien untuk menstabilkan ekstraksi fitur terhadap variasi pencahayaan antar terminal.
- **Local Head (ArcMargin Product)**: Komponen lokal yang menyimpan pengetahuan spesifik tentang identitas mahasiswa di terminal tersebut, memastikan pemisahan antar mahasiswa (class separation) sangat tajam.

## Tahap 3: Siklus Federated & Registry Generation
Sistem berjalan dalam fase terkoordinasi untuk membangun "World-Knowledge" yang aman:
1.  **Phase Discovery**: Pendaftaran ID Mahasiswa ke Global Map di server untuk sinkronisasi NRP dan Nama.
2.  **Phase Preprocessing**: Ekstraksi wajah secara lokal di semua terminal secara serentak.
3.  **Phase Training (Flower)**: Pelatihan paralel menggunakan **FedProx** ($\mu=0.05$) dan **Cosine Annealing LR**. Setiap terminal melakukan pelatihan lokal pada data pribadinya tanpa mengirimkan citra mentah ke server.
4.  **Snapshot Averaging (SWA Variant)**: Server menyimpan snapshot model dari ronde-ronde terakhir (misalnya ronde 8-10) dan melakukan rata-rata bobot (weight averaging). Hal ini menghasilkan model global yang jauh lebih stabil dan tahan terhadap fluktuasi data antar ronde.
5.  **Phase Registry Generation**: Pembuatan database identitas universal. Statistik **Global BN disinkronkan terlebih dahulu** sebelum perhitungan centroid dilakukan untuk memastikan ekstraksi fitur yang konsisten dan akurat.

## Tahap 4: Inferensi & Advanced Matching
Setelah pelatihan, sistem siap digunakan untuk absensi real-time dengan teknik canggih:
- **Flip Trick Evaluation**: Meningkatkan stabilitas skor dengan merata-ratakan embedding citra asli dan citra mirror (horizontal flip) sebelum pencocokan identitas.
- **Confident Instant Match (CIM)**: Menghilangkan delay "pemanasan". Jika skor kemiripan > 0.85, sistem langsung memverifikasi wajah tanpa menunggu buffer temporal.
- **Temporal Voting (0.75+ Threshold)**: Jika skor di antara 0.75 - 0.85, sistem menggunakan buffer 5 frame untuk memastikan stabilitas prediksi sebelum dicatat dalam database kehadiran.
