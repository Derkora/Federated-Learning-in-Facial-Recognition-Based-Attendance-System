# Cara Kerja Sistem: Centralized Face Attendance (Prime State)

Sistem ini dirancang untuk mencapai performa identifikasi wajah maksimal melalui pengumpulan dataset ke server pusat untuk melatih satu model global yang presisi.

---

## Tahap 1: Data Engineering & Mandatory Preprocessing (Identik dengan FL)
Proses penyiapan data di sisi terminal sebelum pengiriman ke server pusat:
1.  **Laplacian Variance Selection**: Sistem secara otomatis memilih maksimal **50 citra wajah paling tajam** dari folder `raw_data/students`. Hal ini memastikan model hanya belajar dari data berkualitas tinggi.
2.  **Face Detection (MTCNN)**: Mendeteksi lokasi wajah. Jika deteksi gagal pada citra tertajam, sistem akan mencoba gambar alternatif berikutnya (Multi-Trial).
3.  **Affine Landmark Alignment**: Ini adalah peningkatan krusial. Sistem mendeteksi 5 titik landmark (mata, hidung, mulut) dan melakukan transformasi Affine agar posisi mata dan mulut sejajar secara horizontal.
4.  **Portrait Resizing (112x96)**: Hasil alignment dipotong dan diubah ukurannya ke dimensi **112x96** (Portrait), yang merupakan standar emas untuk model MobileFaceNet.
5.  **Single Normalization**: Citra dikonversi ke tensor dan dinormalisasi tepat satu kali ke rentang [-1, 1] menggunakan standar MobileFaceNet: `(x - 127.5) / 128.0`.

---

## Tahap 2: Arsitektur Model & Optimasi (Accuracy Boost)
- **MobileFaceNet Backbone**: Arsitektur CNN 128-dimensi yang sangat efisien untuk perangkat Edge (Raspberry Pi/Jetson).
- **ArcMargin Product (Head)**: Menggunakan fungsi ArcFace untuk memaksimalkan margin jarak antar identitas di ruang laten.
- **Optimizer: SGD with Nesterov Momentum**: Menggunakan momentum 0.9 untuk konvergensi yang lebih tajam dan stabil dibandingkan Adam.
- **Per-Layer Weight Decay**: Pinalti bobot yang berbeda antara Backbone (4e-5) dan Head (4e-4) untuk mencegah overfitting pada jumlah data yang terbatas.

---

## Tahap 3: Siklus Pelatihan Terpusat (Centralized)
1.  **Bulk Data Upload**: Terminal mengompresi folder `processed` menjadi berkas ZIP dan mengunggahnya ke server.
2.  **Model Versioning**: Server secara otomatis menaikkan nomor versi model (misal: v1 ke v2) dalam database PostgreSQL untuk pelacakan permanen.
3.  **Cosine Annealing LR**: Pelatihan selama 20 epoch dengan skema penurunan Learning Rate mengikuti kurva kosinus (0.1 -> 0.0001) untuk transisi bobot yang sangat halus.
4.  **Stochastic Weight Averaging (SWA)**: Server mengambil snapshot model pada 5 epoch terakhir dan melakukan perataan bobot (*averaging*) untuk menghasilkan model final yang lebih tahan terhadap fluktuasi data.
5.  **Global Registry Generation**: Server menghitung *centroid* (rata-rata vektor) untuk setiap mahasiswa dan menyimpannya dalam `reference_embeddings.pth`.

---

## Tahap 4: Live Inference (Inference Engine)
Setelah terminal mengunduh model terbaru, sistem menjalankan mesin inferensi real-time:
- **Eager Loading & Isolation**: Memuat model ke RAM secara terpisah dari thread sistem agar absensi tetap berjalan meskipun ada proses background.
- **Flip Trick Evaluation**: Mengambil embedding dari wajah asli dan wajah yang di-flip horizontal, lalu dirata-ratakan untuk stabilitas skor maksimal.
- **Temporal Voting**: Mengumpulkan 5 frame berturut-turut untuk memastikan identitas sebelum mencatat presensi.
- **Confident Instant Match (CIM)**: Jika skor similarity > 0.85, sistem langsung memverifikasi wajah tanpa menunggu buffer frame.
