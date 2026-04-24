# Cara Kerja Sistem: Centralized Face Attendance

Sistem ini dirancang untuk mencapai identifikasi wajah dengan performa maksimal melalui pengumpulan seluruh dataset ke server pusat untuk melatih satu model global yang utuh.

---

## Tahap 1: Data Engineering & Mandatory Preprocessing
Proses penyiapan data yang dilakukan di sisi terminal sebelum pengiriman ke server pusat.
- **Multi-Trial Face Detection**: Sistem melakukan percobaan deteksi wajah hingga 5 kali per mahasiswa. Jika deteksi gagal pada gambar tertajam, sistem secara otomatis mencoba gambar berikutnya berdasarkan urutan nilai Laplacian Variance (sharpness) tertinggi.
- **MTCNN Face Cropping**: Wajah dipotong secara otomatis dari frame kamera asli dengan margin 20px untuk meminimalkan noise latar belakang.
- **112x96 Dimensional Alignment**: Wajah yang dipotong diubah ukurannya ke dimensi standar **112x96** (Portrait) menggunakan metode bilinear untuk memastikan konsistensi ekstraksi fitur tanpa distorsi aspek rasio.

---

## Tahap 2: Arsitektur Model (MobileFaceNet - Global)
Berbeda dengan Federated Learning, sistem terpusat ini melatih satu model global yang digunakan oleh seluruh terminal.
- **Global Backbone**: Menggunakan arsitektur MobileFaceNet (128-dim) yang dilatih secara serentak di server pusat.
- **Shared Parameters**: Semua terminal menggunakan bobot yang identik yang disinkronisasi dari server pusat.
- **Loss Function (ArcMargin)**: Menggunakan ArcMargin Product pada server untuk membedakan identitas mahasiswa dengan margin yang ketat.

---

## Tahap 3: Siklus Pelatihan Terpusat & Model Versioning
Alur kerja yang memindahkan dataset wajah dari terminal ke server pusat dan mengelola versi model:
1.  **Fase Import Data**: Terminal mengemas folder `raw_data/students` menjadi berkas ZIP (`data.zip`) dan mengirimkannya melalui endpoint `/upload-bulk-zip`.
2.  **Centralized Training**: Server mengekstrak data dan melatih model MobileFaceNet selama 20 epoch dengan learning rate schedule yang teroptimasi (1e-4 ke 1e-5).
3.  **Model Version Persistence**: Setelah training, server menyimpan catatan versi model baru ke dalam database PostgreSQL. Hal ini memastikan nomor versi model (misalnya v1, v2) tetap terjaga meskipun container server dijalankan ulang.
4.  **Reference Generation**: Server menghasilkan registri embedding (`reference_embeddings.pth`) yang berisi centroid dari seluruh data mahasiswa yang telah dikumpulkan.

---

## Tahap 4: Inferensi & Cosine Similarity
Setelah model disinkronisasi, terminal melakukan pengenalan real-time:
- **Registry Synchronization**: Terminal mengunduh bobot model (`global_model.pth`) dan registri embedding terbaru dari server.
- **Cosine Similarity Matching**: Perbandingan dilakukan menggunakan skor Cosine Similarity dengan normalisasi L2 pada query dan reference embedding.
- **Threshold 0.60+**: Ambang batas kemiripan ditetapkan pada **0.60** untuk menyeimbangkan antara tingkat akurasi (True Positives) dan keamanan (False Positives).
- **Temporal Voting**: Menggunakan buffer 10 frame untuk menstabilkan prediksi wajah di depan kamera.
