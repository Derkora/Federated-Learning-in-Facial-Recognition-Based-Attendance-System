# Cara Kerja Sistem: Federated Face Attendance 

Sistem ini dirancang untuk mencapai identifikasi wajah yang stabil dengan tingkat kepercayaan (confidence) tinggi melalui kolaborasi antar perangkat tanpa memindahkan data mentah mahasiswa ke server.

## Tahap 1: Data Engineering & Mandatory Preprocessing
Sebelum training, sistem memastikan "bahan bakar" model bersih dan proporsional melalui standar pemrosesan lokal.
- **Laplacian Variance Selection**: Sistem memilih **50 foto tertajam** per mahasiswa untuk menghilangkan noise/blur.
- **MTCNN Face Cropping**: Wajah dipotong secara presisi dari frame asli. Ini menghilangkan 90% background noise.
- **112x96 Dimensional Alignment**: Hasil potongan wajah (crop) di-resize ke dimensi **112x96**. Ini adalah langkah **WAJIB** agar sinkron dengan input arsitektur MobileFaceNet.

## Tahap 2: Inisialisasi Arsitektur (pFedFace - Hybrid)
Struktur ini memisahkan komponen global yang bersifat kolaboratif dan komponen lokal yang bersifat unik.
- **Global Backbone (MobileFaceNet)**: Ekstraktor fitur universal (128-dim). Satu-satunya bagian yang dikirim ke server untuk agregasi FedAvg/FedProx.
- **Global BN Merging**: Server menggabungkan statistik Batch Normalization dari seluruh terminal menjadi satu set statistik global untuk meningkatkan stabilitas inferensi.
- **Local Head (ArcMargin Product)**: Menggunakan $s=32.0$ dan $m=0.50$. Sistem kini memiliki fitur **Weight Preservation** yang menjaga bobot identitas lama saat ada pendaftaran mahasiswa baru.

## Tahap 3: Siklus Federated (Dynamic Barrier Sync)
Sistem berjalan dalam 4 fase linear yang diawasi oleh Server secara ketat:
1.  **Fase Discovery**: Pendaftaran ID Mahasiswa ke Global Map server dan sinkronisasi identitas (NRP + Nama).
2.  **Fase Preprocessing**: Ekstraksi wajah (Crop -> Resize) secara serentak di semua terminal.
3.  **Fase Training (Flower)**: Pelatihan paralel menggunakan FedProx ($\mu=0.05$). Terminal kini melakukan **Knowledge Sharing** dengan menyertakan centroid wajah mahasiswa dari terminal lain dalam dataset pelatihan lokal.
4.  **Fase Registry Generation**: Pembuatan database identitas universal berbasis "World-Knowledge" global model yang sudah digabungkan dengan Global BN.

## Tahap 4: Inferensi & High-Fidelity Registry
Setelah training selesai, sistem siap melakukan absensi dengan tingkat kepercayaan sangat tinggi.
- **Unified Brain Sync**: Setiap terminal mendownload bobot backbone global yang sudah di-merge dengan Global BN, serta Registry identitas terbaru.
- **Double Normalization**: Centroid dihitung dan dicocokkan menggunakan normalisasi L2 ganda untuk akurasi maksimal.
- **Temporal Voting (Buffer = 10)**: Merata-ratakan hasil deteksi dari beberapa frame terakhir untuk menstabilkan skor similarity.
- **Threshold 0.70+**: Berkat optimasi pipeline, ambang batas kini ditingkatkan ke **0.70** untuk memastikan presisi tinggi dan meminimalkan kesalahan identifikasi (False Positives).
