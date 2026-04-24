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

## 6. Optimasi Hyperparameter (Accuracy Boost)

- **Perubahan**: Mengganti optimizer Adam dengan **SGD with Nesterov Momentum** dan menerapkan **Per-Layer Weight Decay**.
- **Justifikasi**: Berdasarkan referensi implementasi *ArcFace*, SGD memberikan kurva konvergensi yang lebih tajam dan generalisasi fitur yang lebih baik pada model MobileFaceNet dibandingkan Adam yang cenderung datar.
- **Dampak**: Model lebih sensitif terhadap perbedaan fitur wajah antar individu yang mirip sekalipun.

## 7. Metode Inferensi Tingkat Lanjut (Flip Trick & CIM)

Sistem mengadopsi dua teknik baru untuk meningkatkan stabilitas dan kecepatan:
- **Flip Trick Evaluation**: Mengekstraksi fitur dari wajah asli dan wajah yang di-flip secara horizontal, lalu merata-ratakannya. Teknik ini secara signifikan menstabilkan skor similarity terhadap kemiringan wajah.
- **Confident Instant Match (CIM)**: Bypass temporal voting jika skor kemiripan frame tunggal > 0.85. Hal ini menghilangkan keluhan "pemanasan" atau delay saat pertama kali wajah terdeteksi.

## 8. Pengetatan Keamanan (Threshold 0.75)
- **Ketentuan**: Ambang batas (threshold) pengenalan ditingkatkan dari proposal awal (0.5 atau 0.6) menjadi **0.75**.
- **Tujuan**: Untuk meminimalisir *False Positive* (salah deteksi) pada lingkungan dengan banyak orang di latar belakang. Dengan model yang sudah dioptimasi, skor mahasiswa asli tetap mampu mencapai angka yang tinggi secara konsisten.

## 9. Penerapan Stochastic Weight Averaging (SWA)
- **Perubahan**: Menambahkan fase perataan bobot model (*weight averaging*) di akhir siklus pelatihan. Pada CL, hal ini dilakukan pada 5 epoch terakhir. Pada FL, dilakukan melalui *Snapshot Averaging* pada 3 ronde terakhir di sisi server.
- **Manfaat**: Teknik ini menghasilkan model yang lebih "generalis" dan tidak terpaku pada noise data di satu iterasi tertentu, sehingga hasil identifikasi jauh lebih stabil dan tidak fluktuatif.
