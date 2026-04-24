# Cara Kerja Sistem: Centralized Face Attendance

Sistem ini dirancang untuk mencapai identifikasi wajah dengan performa maksimal melalui pengumpulan seluruh dataset ke server pusat untuk melatih satu model global yang utuh.

---

## Tahap 1: Data Engineering & Mandatory Preprocessing
Proses penyiapan data yang dilakukan di sisi terminal sebelum pengiriman ke server pusat.
- **Multi-Trial Face Detection**: Sistem melakukan percobaan deteksi wajah hingga 5 kali per mahasiswa. Jika deteksi gagal pada gambar tertajam, sistem secara otomatis mencoba gambar berikutnya berdasarkan urutan nilai Laplacian Variance (sharpness) tertinggi.
- **MTCNN Face Cropping**: Wajah dipotong secara otomatis dari frame kamera asli dengan margin 20px untuk meminimalkan noise latar belakang.
- **112x96 Dimensional Alignment**: Wajah yang dipotong diubah ukurannya ke dimensi standar **112x96** (Portrait) menggunakan metode bilinear untuk memastikan konsistensi ekstraksi fitur tanpa distorsi aspek rasio.

---

## Tahap 2: Arsitektur Model & Optimasi (Accuracy Boost)
- **MobileFaceNet Backbone**: Menggunakan arsitektur 128-dimensi yang ringan namun tangguh untuk perangkat edge.
- **SGD with Nesterov Momentum**: Beralih dari Adam ke SGD dengan momentum Nesterov (0.9) untuk konvergensi yang lebih stabil dan generalisasi fitur yang lebih tajam.
- **Per-Layer Weight Decay**: Menerapkan bobot pinalti yang berbeda (Backbone: 4e-5, Head: 4e-4) untuk mencegah overfitting pada dataset yang kecil.
- **ArcMargin Product**: Margin denda (penalty) digunakan untuk mendorong pemisahan antar identitas mahasiswa secara maksimal di ruang laten.

---

## Tahap 3: Siklus Pelatihan Terpusat & Model Versioning
Alur kerja yang memindahkan dataset wajah dari terminal ke server pusat dan mengelola versi model:
1.  **Fase Import Data**: Terminal mengemas folder `raw_data/students` menjadi berkas ZIP (`data.zip`) dan mengirimkannya melalui endpoint `/upload-bulk-zip`.
2.  **Centralized Training**: Server mengekstrak data dan melatih model selama 20 epoch menggunakan jadwal **Cosine Annealing Learning Rate** untuk transisi bobot yang mulus.
3.  **Stochastic Weight Averaging (SWA)**: Untuk meningkatkan stabilitas fitur, sistem melakukan snapshot model pada 5 epoch terakhir dan merata-ratakannya menjadi satu model final yang lebih robust terhadap noise citra.
4.  **Model Version Persistence**: Setelah training, server menyimpan catatan versi model baru ke dalam database PostgreSQL. Hal ini memastikan nomor versi model (misalnya v1, v2) tetap terjaga meskipun container server dijalankan ulang.
5.  **Reference Generation**: Server menghasilkan registri embedding (`reference_embeddings.pth`) yang berisi centroid dari seluruh data mahasiswa yang telah dikumpulkan.

---

## Tahap 4: Inferensi & Advanced Matching
Setelah model disinkronisasi, terminal melakukan pengenalan real-time dengan teknik tingkat lanjut:
- **Flip Trick Evaluation**: Meningkatkan stabilitas skor dengan merata-ratakan embedding citra asli dan citra mirror (horizontal flip) sebelum pencocokan.
- **Confident Instant Match (CIM)**: Menghilangkan "pemanasan" kamera. Jika skor kemiripan > 0.85, sistem langsung memberikan hasil identifikasi tanpa menunggu buffer temporal.
- **Temporal Voting (0.75+ Threshold)**: Jika skor di bawah 0.85 namun di atas 0.75, sistem menggunakan buffer 5 frame untuk memastikan stabilitas prediksi sebelum dicatat sebagai kehadiran.
