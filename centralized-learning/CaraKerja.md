# Cara Kerja Sistem: Centralized Face Attendance (Prime State)

Sistem ini dirancang untuk mencapai performa identifikasi wajah maksimal melalui pengumpulan dataset ke server pusat untuk melatih satu model global yang presisi.

---

## Tahap 1: Data Engineering & Mandatory Preprocessing (Identik dengan FL)
Proses penyiapan data di sisi terminal sebelum pengiriman ke server pusat:
1.  **Laplacian Variance Selection**: Sistem secara otomatis memilih maksimal **50 citra wajah paling tajam** dari folder `raw_data/students`. Hal ini memastikan model hanya belajar dari data berkualitas tinggi.
2.  **Hardware-Aware Downscaling (New)**: Sebelum deteksi, gambar input di-resize ke lebar maksimal **640px**. Ini adalah langkah krusial untuk mencegah *Out of Memory* (OOM) pada perangkat seperti Raspberry Pi 3B.
3.  **Face Detection (MTCNN)**: Mendeteksi lokasi wajah pada gambar yang sudah di-downscale.
4.  **Affine Landmark Alignment**: Sistem mendeteksi 5 titik landmark (mata, hidung, mulut) dan melakukan transformasi Affine agar posisi mata dan mulut sejajar secara horizontal.
5.  **Portrait Resizing (112x96)**: Hasil alignment dipotong dan diubah ukurannya ke dimensi **112x96** (Portrait), standar MobileFaceNet.
6.  **Single Normalization**: Citra dikonversi ke tensor dan dinormalisasi ke rentang [-1, 1].

---

## Tahap 2: Arsitektur Model & Optimasi (Accuracy Boost)
- **MobileFaceNet Backbone**: Arsitektur CNN 128-dimensi yang sangat efisien untuk perangkat Edge (Raspberry Pi/Jetson).
- **ArcMargin Product (Head)**: Menggunakan fungsi ArcFace untuk memaksimalkan margin jarak antar identitas di ruang laten.
- **Optimizer: SGD with Nesterov Momentum**: Menggunakan momentum 0.9 untuk konvergensi yang lebih tajam dan stabil dibandingkan Adam.
- **Per-Layer Weight Decay**: Pinalti bobot yang berbeda antara Backbone (4e-5) dan Head (4e-4) untuk mencegah overfitting pada jumlah data yang terbatas.

---

### Tahap 3: Pelatihan Terpusat (Centralized Training)
1. **Resource-Aware Training**: Batch Size diatur ke **16** untuk menjaga stabilitas memori server/edge saat menangani dataset besar.
2. **Partial Freezing**: Server membekukan Stage 1 & 2 (`conv1` hingga `blocks[0-11]`) untuk menjaga fitur umum wajah.
3. **Transfer Learning**: Server melatih backbone Stage 3 dan ArcMargin head menggunakan SGD (Nesterov) & Cosine Annealing (20 Epoch) dengan Initial LR **0.05**.
4. **Optimasi Akhir**: Menggunakan **SWA (Stochastic Weight Averaging)** pada 4 epoch terakhir untuk stabilitas bobot.

### Tahap 4: Deployment & Adaptasi Lokal
1. **Sinkronisasi**: Client mengunduh model global dan referensi embedding identitas.
2. **BN Adaptation (Kalibrasi)**: Client menjalankan forward-pass (tanpa gradien) menggunakan data lokal untuk menyesuaikan statistik `running_mean` & `running_var` BatchNorm dengan kondisi pencahayaan spesifik lokasi.
3. **Inferensi Edge**: Menggunakan **Flip Trick** dan **Temporal Voting** untuk pengenalan wajah yang stabil.

---

## Tahap 4: Live Inference (Inference Engine)
Setelah terminal mengunduh model terbaru, sistem menjalankan mesin inferensi real-time:
- **Eager Loading & Isolation**: Memuat model ke RAM secara terpisah dari thread sistem agar absensi tetap berjalan meskipun ada proses background.
- **BN Adaptation (Client Calibration)**: Fitur terbaru untuk menyamai kemampuan adaptasi FL. Setelah mengunduh model global, terminal menjalankan kalibrasi statistik BatchNorm menggunakan data lokal. Ini menyesuaikan model pusat dengan kondisi pencahayaan spesifik di terminal tersebut.
- **Flip Trick Evaluation**: Mengambil embedding dari wajah asli dan wajah yang di-flip horizontal, lalu dirata-ratakan untuk stabilitas skor maksimal.
- **Vectorized Classifier (New)**: Proses pencocokan wajah menggunakan operasi matriks (`torch.mm`) alih-alih loop `for`. Ini mempercepat proses dan menghemat RAM secara signifikan.
- **Temporal Voting**: Mengumpulkan 5 frame berturut-turut untuk memastikan identitas sebelum mencatat presensi.
- **Confident Instant Match (CIM)**: Jika skor similarity > 0.85, sistem langsung memverifikasi wajah tanpa menunggu buffer frame.
- **Memory Management (GC)**: Sistem secara eksplisit memicu *Garbage Collection* di setiap akhir loop kamera untuk menjaga stabilitas RAM 1GB.
