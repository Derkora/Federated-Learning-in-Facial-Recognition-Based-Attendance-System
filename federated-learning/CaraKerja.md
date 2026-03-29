# Cara Kerja Sistem: Federated Face Attendance (Verified Results)

Sistem ini dirancang untuk mencapai identifikasi wajah yang stabil dengan tingkat kepercayaan (confidence) tinggi melalui kolaborasi antar perangkat tanpa memindahkan data mentah mahasiswa.

## Tahap 1: Data Engineering (Kualitas & Presisi)
Sebelum training, kita memastikan "bahan bakar" model ini bersih dan tajam melalui seleksi berbasis kualitas.
- **Laplacian Variance Selection**: Menggunakan algoritma Laplace untuk menghitung tingkat ketajaman setiap foto. Sistem hanya memilih **50 foto tertajam** per mahasiswa. 
  - *Hasil*: Menghilangkan noise (blur) dan memastikan setiap kelas memiliki jumlah data yang adil (Equalization).
- **Local Preprocessing**: Deteksi wajah menggunakan MTCNN mandiri di tiap Client untuk alignment dan resize ke dimensi 112x96.
- **Data Splitting**: Membagi 50 foto terbaik tersebut menjadi porsi **40 Train (80%)** dan **10 Validation (20%)** untuk pemantauan akurasi lokal yang jujur.

## Tahap 2: Inisialisasi Arsitektur (pFedFace - Hybrid)
Struktur ini memisahkan komponen global yang bersifat kolaboratif dan komponen lokal yang bersifat unik.
- **Global Backbone (MobileFaceNet)**: Ekstraktor fitur universal (128-dim). Satu-satunya bagian yang dikirim ke server.
- **Local Batch Normalization (BN)**: Tetap lokal untuk menangani perbedaan teknis sensor dan pencahayaan kamera di masing-masing perangkat.
- **Local Head (ArcMargin Product)**: Menggunakan $s=32.0$ dan $m=0.50$ untuk memetakan fitur ke identitas lokal mahasiswa.

## Tahap 3: Siklus Training (Optimized Federated Flow)
Menggunakan parameter yang telah teruji dalam eksperimen untuk mencegah stagnasi akurasi.
- **Konfigurasi**: Menjalankan minimal **15-20 Round** dengan 3 Local Epochs.
- **FedProx Regularization**: Menggunakan penalti **$\mu=0.05$** (diperkuat) untuk menjaga agar client tidak "melenceng" terlalu jauh dari pengetahuan global server.
- **Learning Rate Schedule**:
    - Round 1-5: $1e-4$ (Fase eksplorasi cepat).
    - Round 6-10: $5e-5$ (Fase konsolidasi).
    - Round 11-15+: $1e-5$ (Fase kristalisasi untuk stabilitas akurasi >80%).
- **Aggregasi**: Server melakukan rata-rata bobot (FedAvg) pada Backbone di akhir setiap ronde.

## Tahap 4: Inferensi & Real-Time Voting (Universal Access)
Setelah training selesai, sistem siap melakukan absensi dengan tingkat kepercayaan tinggi.
- **Centroid Calculation**: Menghitung rata-rata vektor (embedding) dari 40 foto train untuk mendapatkan satu titik pusat (Centroid) yang mewakili identitas unik mahasiswa.
- **Temporal Voting (Buffer Size = 10)**: Sistem merata-ratakan hasil deteksi dari **10 frame terakhir** di kamera secara real-time.
  - *Hasil*: Berhasil menaikkan skor kemiripan (similarity) secara stabil ke rentang **0.50 - 0.75**.
- **Combined BN**: Menggunakan rata-rata statistik BN dari semua Client agar Backbone tetap adaptif.
- **Registry Global**: Menggabungkan seluruh Centroid dari semua Client menjadi satu database identitas universal.
- **Security Threshold**: Berkat kestabilan Temporal Voting, ambang batas (threshold) dapat ditingkatkan ke **0.45 - 0.50** untuk keamanan ekstra terhadap wajah orang asing (*Unknown*).

---
**Verified Achievement**: 
Eksperimen terakhir (2026-03-29) menunjukkan identifikasi stabil dengan skor **0.70 Confidence**, membuktikan efektivitas seleksi Laplacian dan Temporal Voting.
