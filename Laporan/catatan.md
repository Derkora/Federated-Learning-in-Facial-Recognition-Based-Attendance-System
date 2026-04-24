# Catatan Implementasi & Perbedaan dari Proposal (Revisi)

Dokumen ini mencatat perubahan teknis dan optimalisasi yang dilakukan selama tahap implementasi yang mungkin berbeda dari proposal awal. Catatan ini dapat digunakan sebagai dasar pembaruan pada buku Laporan/Skripsi.

## 1. Optimalisasi Perangkat Edge (Stabilitas vs RAM)

Meskipun pada tahap awal sempat dieksplorasi penggunaan ONNX Runtime untuk menekan konsumsi RAM, sistem akhir diputuskan menggunakan **Full PyTorch (CPU Mode)** di terminal.
- **Alasan**: Menghindari ketidakcocokan metadata model saat pembaruan dinamis dan memastikan keselarasan 100% antara fase pelatihan dan inferensi.
- **Dampak**: Peningkatan penggunaan RAM tetap dalam batas wajar (1GB-2GB) dengan stabilitas arsitektur yang jauh lebih tinggi.

## 2. Peningkatan Robustness Registrasi (Detection Retry)

Terdapat penambahan mekanisme **Multi-Trial Face Detection** pada proses pendaftaran dan pembaruan embedding:
- **Perubahan**: Sistem tidak lagi hanya mengandalkan satu gambar tertajam. Jika deteksi wajah gagal pada gambar pertama, sistem akan mencoba hingga **5 kali** menggunakan urutan gambar tertajam berikutnya.
- **Hasil**: Menghilangkan masalah "missing user" (misalnya kasus 29/30 identitas) yang disebabkan oleh satu frame yang tidak terbaca MTCNN meskipun cukup tajam.

## 3. Sinkronisasi Global Batch Normalization (BN)

Terobosan utama dalam akurasi Federated Learning:
- **Konsep**: Sebelum menghitung centroid (registri), terminal **WAJIB** mensinkronisasikan statistik Global BN dari server dan menginjeksikannya ke dalam backbone model.
- **Justifikasi**: Tanpa sinkronisasi BN sebelum pembuatan registri, akan terjadi "feature drift" yang menyebabkan skor similarity rendah (~0.2). Dengan sinkronisasi ini, akurasi meningkat pesat (Score > 0.6).

## 4. Persistensi Versi Model Berbasis Database

- **Perubahan**: Pelacakan versi model (v1, v2, dst.) dipindahkan dari metadata file ke **Database PostgreSQL (tabel public.model_versions)** di sisi Centralized Server.
- **Manfaat**: Versi model tetap konsisten meskipun container dinyalakan ulang. Terminal dapat mendeteksi "lompatan versi" secara akurat dan memicu penyegaran (refresh) data biometrik lokal secara otomatis.

## 5. Standarisasi Preprocessing (112x96 Portrait)

- **Ketentuan**: Seluruh pipeline (CL, FL, Training, dan Inferensi) diselaraskan pada resolusi **112x96 portrait**.
- **Teknis**: Menggunakan margin deteksi 20px dan resize bilinear tanpa distorsi aspek rasio, memastikan wajah yang "dilihat" model selama absensi memiliki karakteristik yang sama dengan saat pelatihan.
