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
