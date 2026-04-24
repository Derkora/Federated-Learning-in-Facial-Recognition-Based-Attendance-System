# Cara Kerja Sistem: Federated Face Attendance

Sistem ini dirancang untuk mencapai identifikasi wajah yang stabil dengan tingkat kepercayaan tinggi melalui kolaborasi antar perangkat tanpa memindahkan data mentah mahasiswa ke server.

## Tahap 1: Data Engineering & Mandatory Preprocessing
Sistem memastikan kualitas data lokal sebelum masuk ke fase pelatihan.
- **Multi-Trial Face Detection**: Sistem melakukan percobaan deteksi wajah hingga 5 kali per identitas. Jika deteksi gagal, sistem menggunakan gambar alternatif dengan tingkat ketajaman (Laplacian Variance) tertinggi berikutnya.
- **MTCNN Face Cropping**: Wajah dipotong secara presisi dengan margin tertentu untuk menghilangkan porsi latar belakang yang tidak relevan.
- **112x96 Dimensional Alignment**: Hasil potongan wajah diubah ukurannya ke dimensi **112x96** (Portrait). Standarisasi dimensi ini sangat krusial agar sinkron dengan input arsitektur MobileFaceNet.

## Tahap 2: Inisialisasi Arsitektur (pFedFace - Hybrid)
Struktur ini memisahkan pengetahuan global dan identitas lokal.
- **Global Backbone (MobileFaceNet)**: Ekstraktor fitur universal yang dikirim ke server untuk agregasi menggunakan algoritma FedProx.
- **Global BN Merging**: Server menggabungkan statistik Batch Normalization dari seluruh terminal. Statistik ini kemudian digunakan oleh semua terminal untuk menstabilkan ekstraksi fitur.
- **Local Head (ArcMargin Product)**: Komponen lokal yang menyimpan pengetahuan spesifik tentang identitas mahasiswa di terminal tersebut, dilengkapi fitur Weight Preservation untuk menjaga data lama.

## Tahap 3: Siklus Federated & Registry Generation
Sistem berjalan dalam fase terkoordinasi untuk membangun "World-Knowledge" yang aman:
1.  **Phase Discovery**: Pendaftaran ID Mahasiswa ke Global Map di server untuk sinkronisasi NRP dan Nama.
2.  **Phase Preprocessing**: Ekstraksi wajah secara lokal di semua terminal secara serentak.
3.  **Phase Training (Flower)**: Pelatihan paralel menggunakan FedProx ($\mu=0.05$). Terminal melakukan Knowledge Sharing dengan menyertakan centroid wajah mahasiswa dari terminal lain.
4.  **Phase Registry Generation**: Pembuatan database identitas universal. Statistik **Global BN disinkronkan terlebih dahulu** sebelum perhitungan centroid dilakukan untuk mencegah drift fitur.

## Tahap 4: Inferensi & Auto-Sync
Setelah pelatihan, sistem siap digunakan untuk absensi real-time:
- **Automatic Version Sync**: Background heartbeat secara otomatis mendeteksi jika ada versi model baru di server (berdasarkan database PostgreSQL).
- **Proactive Registry Refresh**: Terminal secara otomatis melakukan refresh embedding lokal jika mendeteksi lonjakan versi model global, memastikan biometric data selalu selaras dengan backbone terbaru.
- **Threshold 0.60+**: Ambang batas ditingkatkan menjadi **0.60** guna memastikan presisi tinggi dan meminimalkan False Positives.
- **Temporal Voting**: Hasil deteksi dirata-ratakan dari buffer 10 frame untuk stabilitas prediksi.
