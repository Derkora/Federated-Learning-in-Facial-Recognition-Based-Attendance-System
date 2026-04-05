# Cara Kerja Sistem: Centralized Face Attendance

Sistem ini dirancang untuk mencapai identifikasi wajah dengan performa maksimal melalui pengumpulan seluruh dataset ke server pusat untuk melatih satu model global yang utuh.

---

## Tahap 1: Data Engineering & Mandatory Preprocessing
Proses penyiapan data yang dilakukan di sisi terminal sebelum pengiriman ke server pusat.
- **Laplacian Variance Selection**: Sistem memilih **50 foto tertajam** per mahasiswa untuk menghilangkan noise/blur.
- **MTCNN Face Cropping**: Wajah dipotong secara otomatis dari frame kamera asli untuk meminimalkan noise latar belakang.
- **112x96 Dimensional Alignment**: Wajah yang dipotong diubah ukurannya ke dimensi standar 112x96 agar sesuai dengan input arsitektur MobileFaceNet.

---

## Tahap 2: Arsitektur Model (MobileFaceNet - Global)
Berbeda dengan Federated Learning, sistem terpusat ini tidak memisahkan komponen lokal.
- **Global Backbone**: Menggunakan arsitektur MobileFaceNet utuh (128-dim) yang dilatih secara serentak di server pusat.
- **Shared Parameters**: Semua terminal menggunakan bobot yang identik 100% yang disinkronisasi dari server pusat.
- **Loss Function (ArcMargin)**: Menggunakan ArcMargin Product pada server untuk membedakan ribuan identitas mahasiswa di satu lokasi penyimpanan pusat.

---

## Tahap 3: Siklus Pelatihan Terpusat (Bulk Data Transfer)
Alur kerja yang memindahkan data mentah (cropped images) dari terminal ke server pusat:
1.  **Fase Registrasi**: Terminal mendaftar ke server pusat sebagai node pengirim data.
2.  **Fase Data Collection & Upload**: Terminal mengumpulkan wajah mahasiswa, memaketkannya (ZIP), dan mengunggahnya ke server.
3.  **Fase Centralized Processing**: Server mengekstrak seluruh paket data dari semua terminal dan melatih model secara massal menggunakan GPU/CPU tinggi.
4.  **Fase Asset Distribution**: Setelah proses training selesai, server mengirimkan model hasil pelatihan kembali ke seluruh terminal yang terhubung.

---

## Tahap 4: Inferensi & Cosine Similarity
Setelah model disinkronisasi, terminal beralih ke mode pengenalan real-time:
- **Registry Synchronization**: Terminal mengunduh "World-Knowledge" (centroid wajah) terbaru dari server pusat.
- **Cosine Similarity Matching**: Wajah yang dideteksi di depan kamera dibandingkan dengan basis data registri menggunakan skor **Cosine Similarity**.
- **Average Result Voting**: Menggunakan buffer frame untuk menstabilkan skor deteksi sehingga lebih handal terhadap perubahan cahaya atau pose.
- **Threshold 0.45+**: Ambang batas minimal kemiripan untuk menentukan apakah mahasiswa dianggap "Hadir" atau "Unknown".
