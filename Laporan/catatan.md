# Catatan Implementasi & Revisi (Prime State)

Dokumen ini mencatat perubahan teknis krusial yang diimplementasikan selama pengembangan untuk mencapai stabilitas sistem. Poin-poin ini disarankan untuk diupdate pada Buku Laporan/Skripsi (Bab 3 dan Bab 4).

## 1. Upgrade Preprocessing: Affine Landmark Alignment
- **Semula (Proposal)**: Hanya menggunakan MTCNN Bounding Box Crop.
- **Implementasi (Update)**: Menggunakan **Affine Alignment** berdasarkan 5 titik landmark (mata, hidung, mulut).
- **Justifikasi**: Penyelarasan wajah secara geometris (memastikan posisi mata sejajar horizontal) secara signifikan mengurangi variasi fitur yang harus dipelajari model, sehingga akurasi meningkat drastis terutama pada wajah yang miring.

## 2. Adopsi Arsitektur pFedFace (Personalized FL)
- **Semula (Proposal)**: Full Model Synchronization (semua parameter disinkronkan).
- **Implementasi (Update)**: Hanya mensinkronkan **Backbone** (MobileFaceNet), sementara **BatchNorm (BN)** dan **Classifier Head** tetap lokal di setiap terminal.
- **Justifikasi**: Masalah *non-IID* (distribusi identitas yang berbeda antar terminal) dapat menyebabkan *Loss Spike* jika Head disinkronkan secara paksa. Dengan pFedFace, terminal mempertahankan "pengetahuan identitas lokal" yang unik bagi mahasiswanya sendiri.

## 3. Preservasi Statistik Global Batch Normalization (BN)
- **Teknis**: Statistik BN (mean/variance) tidak lagi direset setiap ronde di server. Server sekarang menjaga (*preserve*) BN global hasil sesi sebelumnya sebagai baseline.
- **Dampak**: Menghilangkan masalah "amnesia model" di mana akurasi sering kali drop secara misterius setelah ronde pelatihan selesai. Hal ini memastikan model selalu memulai dari titik optimal (Acc 0.9+ sejak ronde awal).

## 4. Standarisasi Dimensi Portrait 96x112
- **Detail**: Resolusi input dikunci pada **96 (width) x 112 (height)**.
- **Justifikasi**: Fokus pada area vertikal wajah (portrait) memberikan kepadatan fitur yang lebih baik untuk model MobileFaceNet dibandingkan resolusi landscape atau square standar.

## 5. Implementasi Flip Trick & CIM pada Inferensi
- **Flip Trick**: Mengevaluasi wajah asli dan mirror (rata-rata embedding) untuk stabilitas skor.
- **Confident Instant Match (CIM)**: Bypass temporal voting jika skor > 0.85 untuk respons instan.
- **Justifikasi**: Meningkatkan *User Experience* (kecepatan absensi) tanpa mengorbankan keamanan (keamanan tetap terjaga melalui threshold 0.75 untuk kasus umum).

## 6. Integrasi Database untuk Versioning Model
- **Sistem**: Pelacakan versi model (v1, v2, dst.) dikelola secara permanen di database PostgreSQL.
- **Manfaat**: Konsistensi versi tetap terjaga meskipun infrastruktur server (container) dimulai ulang, memudahkan terminal dalam melakukan sinkronisasi otomatis.

## 7. Partial Freezing (Stage-wise Training)
- **Penerapan**: Membekukan lapisan `conv1` hingga `conv_3` (indeks 0-11) pada kedua sistem (CL & FL).
- **Justifikasi**: Mencegah "Catastrophic Forgetting" pada fitur wajah dasar yang sudah sangat baik dari model pretrained. Dengan fokus hanya pada lapisan akhir, model lebih stabil dalam mengenali identitas spesifik mahasiswa tanpa merusak ekstraktor fitur umum.

## 8. Adaptasi Lingkungan (Client-side BN Calibration)
- **Update (Centralized)**: Menambahkan langkah kalibrasi BatchNorm pada client CL setelah download model global.
- **Justifikasi**: Menghilangkan bias "Global BN" yang sering kali menyebabkan CL gagal pada pencahayaan yang berbeda dari distribusi training. Dengan kalibrasi lokal, model CL memiliki kemampuan adaptasi lingkungan yang setara dengan model pFedFace (FL).

## 9. Penyelarasan Total Iterasi & SWA
- **Detail**: CL (20 Epoch) disetarakan dengan FL (10 Round x 2 Epoch). SWA pada CL (4 epoch terakhir) disetarakan dengan Snapshot Averaging FL (2 ronde terakhir, 4 epoch total).
- **Justifikasi**: Menjamin bahwa model pada kedua metode memiliki jumlah "pengalaman" yang sama terhadap data sebelum dibandingkan.

## 10. Paritas Algoritma Inferensi (Edge Side)
- **Detail**: Implementasi **Flip Trick** (Average Horizontal Flip) dan **Temporal Voting** (Buffer 10 frame) disamakan di tingkat kode Python.
- **Justifikasi**: Memastikan perbedaan akurasi benar-benar berasal dari metode pembelajaran (Centralized vs Federated), bukan karena perbedaan cara mesin inferensi bekerja di client.

## 11. Standarisasi Centroid Generation
- **Detail**: Jumlah gambar untuk ekstraksi fitur final (Centroid) ditetapkan sebanyak **50 gambar terbaik** (seleksi Laplacian) untuk kedua metode.
- **Justifikasi**: Menghilangkan bias pada Client 1 (pencipta data) yang sebelumnya hanya menggunakan 5 gambar (quick refresh), sehingga sekarang semua client memiliki kualitas referensi yang sama kuatnya (Premium Quality).

---

## 12. Evolusi dari Proposal Awal (Revisi Tugas Akhir)
Berikut adalah perubahan signifikan dibandingkan dengan dokumen **5027221021-Steven Figo-Revisi Proposal.pdf**:

| Fitur | Rencana Awal (Proposal) | Implementasi Saat Ini (Update) | Justifikasi Perubahan |
| :--- | :--- | :--- | :--- |
| **Hardware Edge** | Raspberry Pi 4 (RAM 4GB) | **Raspberry Pi 3B (RAM 1GB)** | Menguji ketahanan algoritma pada hardware yang lebih terbatas (*lower-end*). |
| **Preprocessing** | MTCNN Bbox Crop | **Affine Landmark Alignment** | Meningkatkan akurasi pada wajah miring secara signifikan. |
| **Resolusi Input** | Standar (112x112 / Square) | **Portrait (96x112)** | Fokus fitur pada area wajah, lebih optimal untuk MobileFaceNet. |
| **Framework FL** | Framework Flower (flwr) | **Custom Lightweight FL** | Mengurangi dependensi library berat agar lebih stabil di RAM 1GB Raspi 3B. |
| **Efisiensi Memori**| Tidak disebutkan secara detail | **Input Downscaling & Vectorization** | Wajib dilakukan agar MTCNN tidak menyebabkan OOM pada RAM 1GB. |
| **Inference Mode** | Standar Loop | **Vectorized (Matrix Multiplication)** | Mengurangi beban CPU dan latency saat pencocokan wajah massal. |
| **Adaptasi Lokal** | Sinkronisasi penuh | **pFedFace (Personalized FL)** | Mengatasi masalah *non-IID* data antar terminal yang berbeda. |
