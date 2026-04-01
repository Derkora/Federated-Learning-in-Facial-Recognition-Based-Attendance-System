# Cara Kerja Sistem: Federated Face Attendance 

Sistem ini dirancang untuk mencapai identifikasi wajah yang stabil dengan tingkat kepercayaan (confidence) tinggi melalui kolaborasi antar perangkat tanpa memindahkan data mentah mahasiswa.

## Tahap 1: Data Engineering & Mandatory Preprocessing
Sebelum training, kita memastikan "bahan bakar" model ini bersih dan proporsional melalui standar.
- **Laplacian Variance Selection**: Sistem memilih **50 foto tertajam** per mahasiswa untuk menghilangkan noise/blur.
- **MTCNN Face Cropping**: Wajah dipotong secara presisi dari frame 1080p asli menggunakan MTCNN. Ini menghilangkan 90% background noise.
- **112x96 Dimensional Alignment**: Hasil potongan wajah (crop) di-resize ke dimensi 112x96. Ini adalah langkah **HARUS** agar sinkron dengan input arsitektur MobileFaceNet (dimensi 128).

## Tahap 2: Inisialisasi Arsitektur (pFedFace - Hybrid)
Struktur ini memisahkan komponen global yang bersifat kolaboratif dan komponen lokal yang bersifat unik.
- **Global Backbone (MobileFaceNet)**: Ekstraktor fitur universal (128-dim). Satu-satunya bagian yang dikirim ke server.
- **Local Batch Normalization (BN)**: Tetap lokal untuk menangani perbedaan teknis sensor dan pencahayaan kamera di masing-masing perangkat.
- **Local Head (ArcMargin Product)**: Menggunakan $s=32.0$ dan $m=0.50$ untuk memetakan fitur ke identitas lokal mahasiswa.

## Tahap 3: Siklus Linear Federated (Strict Barrier Sync)
Sistem berjalan dalam 4 fase linear yang diawasi oleh Server secara ketat:
1.  **Fase Discovery**: Pendaftaran ID Mahasiswa ke Global Map server.
2.  **Fase Preprocessing**: Ekstraksi wajah (Crop -> Resize) secara serentak di semua terminal.
3.  **Fase Training (Flower)**: Pelatihan paralel menggunakan FedProx ($\mu=0.05$).
4.  **Fase Registry Generation**: Pembuatan database identitas universal berbasis "World-Knowledge" global model.

## Tahap 4: Inferensi & High-Fidelity Registry
Setelah training selesai, sistem siap melakukan absensi dengan tingkat kepercayaan sangat tinggi.
- **Unified Brain Sync**: Setiap terminal mendownload bobot backbone global, BN gabungan, dan Registry identitas terbaru dari server.
- **Double Normalization**: Centroid dihitung dan dicocokkan menggunakan normalisasi L2 ganda untuk akurasi maksimal.
- **Temporal Voting (Buffer = 10)**: Merata-ratakan hasil deteksi dari 10 frame terakhir untuk menstabilkan skor similarity.
- **Threshold 0.42+**: Berkat pendekatan **Crop-then-Resize**, ambang batas aman kini berada di angka **0.42 - 0.50** (jauh lebih ketat dan aman).

