# Cara Kerja Sistem: Federated Face Attendance (Prime State)

Sistem ini dirancang untuk mencapai identifikasi wajah stabil melalui kolaborasi antar terminal tanpa memindahkan data mentah mahasiswa ke server (Privacy-Preserving).

---

## Tahap 1: Fase Persiapan & Sinkronisasi (Discovery)
Langkah awal untuk memastikan semua terminal memiliki "bahasa" yang sama sebelum belajar dimulai:
1.  **Discovery Identitas**: Setiap terminal memindai folder lokal (`raw_data/students`) untuk mendaftarkan ID (NRP) ke server.
2.  **Global Label Map Sync**: Server mengumpulkan semua ID unik dari seluruh terminal dan mengirimkan balik daftar urutan kelas (label map) yang seragam ke semua client. Ini mencegah index klasifikasi tertukar antar terminal.

---

## Tahap 2: Fase Pemrosesan Data (Preprocessing)
Langkah krusial untuk kualitas input yang seragam (Identik dengan CL):
1.  **Laplacian Selection (Top 50)**: Memilih maksimal 50 foto paling tajam per mahasiswa.
2.  **MTCNN Face Detection**: Deteksi lokasi wajah dengan mekanisme Multi-Trial.
3.  **Affine Landmark Alignment**: Menyejajarkan posisi mata dan mulut secara horizontal berdasarkan 5 titik landmark.
4.  **Portrait Crop (96x112)**: Memastikan area wajah fokus pada dimensi Portrait yang standar.
5.  **Initial Centroid Generation**: Menghasilkan embedding awal menggunakan model saat ini untuk dikirim ke server sebagai data referensi.

---

## Tahap 3: Fase Pelatihan Terfederasi (pFedFace - Hybrid)
Proses kolaborasi pengetahuan tanpa berbagi privasi:
1.  **Model Initialization**: Terminal memuat MobileFaceNet (backbone) dan mengekspansi ArcMarginProduct (head) sesuai label map global.
2.  **Backbone Only Sync (pFedFace)**: Selama ronde Flower, terminal **hanya mengirimkan parameter Backbone** ke server. Parameter BatchNorm (BN) dan Head (Classifier) tetap disimpan secara lokal agar model bisa beradaptasi dengan kondisi spesifik (kamera/cahaya) di terminal tersebut.
3.  **FedProx Optimization**: Menggunakan parameter proximal untuk menangani variasi data antar terminal (non-IID).
4.  **Snapshot Averaging**: Server merata-ratakan bobot backbone dari 3 ronde terakhir untuk stabilitas fitur yang lebih tinggi.

---

## Tahap 4: Fase Finalisasi & Pembangkitan Registry
Konsolidasi pengetahuan global setelah pelatihan Flower selesai:
1.  **Download Global Backbone**: Mengunduh bobot backbone hasil agregasi terbaru.
2.  **Download Global BN**: Mengunduh statistik Batch Normalization global ("tempel di belakang") untuk menstabilkan ekstraksi fitur saat inferensi.
3.  **Centroid Re-calculation**: Terminal menghitung ulang "titik tengah" (centroid) embedding setiap mahasiswa menggunakan kombinasi Backbone Global + BN Global untuk disimpan sebagai registri identitas universal.

---

## Tahap 5: Fase Inferensi (Live Attendance)
1.  **Eager Loading**: Memuat model versi terbaru ke RAM secara terisolasi.
2.  **Flip Trick Evaluation**: Merata-ratakan embedding wajah asli dan mirror untuk hasil skor yang lebih stabil.
3.  **Temporal Voting & CIM**: Menggunakan buffer frame untuk konfirmasi identitas, namun mendukung *Instant Match* jika skor > 0.85.
