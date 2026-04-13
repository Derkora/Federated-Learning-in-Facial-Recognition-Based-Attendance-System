# Cara Kerja Sistem: Centralized Face Attendance

Sistem ini dirancang untuk mencapai identifikasi wajah dengan performa maksimal melalui pengumpulan seluruh dataset ke server pusat untuk melatih satu model global yang utuh.

---

## Tahap 1: Data Engineering & Mandatory Preprocessing
Proses penyiapan data yang dilakukan di sisi terminal sebelum pengiriman ke server pusat.
- **Laplacian Variance Selection**: Sistem memilih **50 foto tertajam** per mahasiswa untuk menghilangkan noise/blur.
- **MTCNN Face Cropping**: Wajah dipotong secara otomatis dari frame kamera asli untuk meminimalkan noise latar belakang.
- **112x96 Dimensional Alignment**: Wajah yang dipotong diubah ukurannya ke dimensi standar **112x96** agar sesuai dengan input arsitektur MobileFaceNet.

---

## Tahap 2: Arsitektur Model (MobileFaceNet - Global)
Berbeda dengan Federated Learning, sistem terpusat ini tidak memisahkan komponen lokal.
- **Global Backbone**: Menggunakan arsitektur MobileFaceNet utuh (128-dim) yang dilatih secara serentak di server pusat.
- **Shared Parameters**: Semua terminal menggunakan bobot yang identik 100% yang disinkronisasi dari server pusat.
- **Loss Function (ArcMargin)**: Menggunakan ArcMargin Product pada server untuk membedakan ribuan identitas mahasiswa di satu lokasi penyimpanan pusat.

---

## Tahap 3: Siklus Pelatihan Terpusat (Bulk Data Transfer)
Alur kerja yang memindahkan dataset wajah dari terminal ke server pusat secara efisien:
1.  **Fase Registrasi & Signal**: Terminal memantau status server. Saat server memulai fase `Import Data`, terminal menerima sinyal untuk mulai mengirimkan dataset.
2.  **Bulk Packaging (ZIP)**: Terminal mengumpulkan seluruh foto mahasiswa yang telah di-crop, mengemas folder `raw_data/students` menjadi satu berkas ZIP (`data.zip`) untuk efisiensi transmisi.
3.  **Endpoint /upload-bulk-zip**: Terminal mengirimkan berkas ZIP tersebut ke server melalui endpoint khusus. Server secara otomatis mengekstrak berkas tersebut ke direktori `data/students/` di sisi server.
4.  **Centralized Processing**: Server memproses seluruh data gabungan dari berbagai terminal (pemeriksaan duplikat, balancing) dan melatih model MobileFaceNet secara terpusat menggunakan resource komputasi tinggi.
5.  **Direct Asset Deployment**: Setelah training selesai, terminal secara otomatis mengunduh bobot model (`.pth`) dan registri embedding terbaru untuk digunakan dalam inferensi real-time.

---

## Tahap 4: Inferensi & Cosine Similarity
Setelah model disinkronisasi, terminal beralih ke mode pengenalan real-time:
- **Registry Synchronization**: Terminal mengunduh "World-Knowledge" (centroid wajah) terbaru dari server pusat.
- **Cosine Similarity Matching**: Wajah yang dideteksi di depan kamera dibandingkan dengan basis data registri menggunakan skor **Cosine Similarity**.
- **Average Result Voting**: Menggunakan buffer frame untuk menstabilkan skor deteksi sehingga lebih handal terhadap perubahan cahaya atau pose.
- **Threshold 0.70+**: Ambang batas minimal kemiripan untuk menentukan apakah mahasiswa dianggap "Hadir" atau "Unknown". Angka ini memastikan presisi tinggi untuk lingkungan kampus.
