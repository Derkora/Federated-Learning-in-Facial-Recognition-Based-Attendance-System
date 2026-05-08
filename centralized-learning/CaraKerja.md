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
1. **Resource-Aware Training**: Batch Size diatur ke **8** untuk menjaga stabilitas memori server/edge dan mencegah osilasi gradien pada dataset kecil.
2. **Partial Freezing**: Server membekukan Stage 1 & 2 (`conv1` hingga `blocks[0-11]`) untuk menjaga fitur umum wajah.
3. **Transfer Learning**: Server melatih backbone Stage 3 dan ArcMargin head menggunakan SGD (Nesterov) & Cosine Annealing (10 Ronde) dengan Initial LR **0.01**.
4. **Premium Augmentation**: Menggunakan `RandomPerspective`, `GaussianBlur`, `ColorJitter`, dan `RandomErasing` untuk ketangguhan ekstrem terhadap variasi kamera.
5. **Optimasi Akhir**: Menggunakan **SWA (Stochastic Weight Averaging)** pada 3 ronde terakhir (8, 9, 10) untuk stabilitas bobot.

---

## Tahap 4: Live Inference (Inference Engine)
Setelah terminal mengunduh model terbaru, sistem menjalankan mesin inferensi real-time:
- **Eager Loading & Isolation**: Memuat model ke RAM secara terpisah dari thread sistem agar absensi tetap berjalan meskipun ada proses background.
- **BN Adaptation (Client Calibration)**: Menyesuaikan model pusat dengan kondisi pencahayaan spesifik di terminal menggunakan data lokal.
- **Flip Trick Evaluation**: Mengambil rata-rata embedding dari wajah asli dan mirror untuk stabilitas skor.
- **Vectorized Classifier**: Pencocokan wajah menggunakan operasi matriks (`torch.mm`) yang cepat dan efisien RAM.
- **Production Logging**: Mencatat skor identitas terbaik dengan timestamp untuk auditabilitas data presensi.
- **Temporal Voting Strategy**: Verifikasi identitas menggunakan rata-rata buffer temporal untuk stabilitas. Threshold standar ditetapkan pada **0.7**.
- **Memory Management (GC)**: Pemicu *Garbage Collection* otomatis untuk menjaga stabilitas RAM 1GB.
