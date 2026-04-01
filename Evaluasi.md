# Panduan Evaluasi & Skenario Uji Sistem

Dokumen ini merinci metrik utama untuk mengukur performa sistem absensi berbasis *Federated Learning*.

---

## 1. Evaluasi Training (Fase Belajar)
Mengukur seberapa efektif model belajar dari data yang tersebar di berbagai terminal (Client).

### A. Metrik Training
*   **Accuracy & Loss**: Mengukur ketepatan prediksi dan tingkat error model selama ronde federasi berlangsung.
*   **Convergence Speed**: Menghitung jumlah ronde yang dibutuhkan hingga akurasi mencapai titik stabil (misal >85%).
*   **Weight Divergence**: Mengukur perbedaan bobot antar client. Penting untuk membuktikan efektivitas algoritma pFedFace dalam menangani data yang tidak seimbang. (membandingkan FL dengan FL tidak seimbang)
* **Training Time**: Menghitung waktu yang dibutuhkan untuk melatih model.

> **Kenapa dievaluasi?** Untuk memastikan model tidak "menyimpang" dan bisa belajar secara merata dari semua terminal tanpa harus mengirim gambar mentah.

**Rumus Akurasi:**
$$ \text{Accuracy} = \frac{\text{Prediksi Benar}}{\text{Total Data}} \times 100\% $$

### B. Nilai Ekonomi (Resources)
*   **Transmission Volume (Bandwidth)**: Menghitung total data (MB) yang dikirim. FL hanya mengirim bobot model, jauh lebih hemat dibanding Centralized yang mengirim ribuan gambar.

* Biaya Operasional
    *   **Transmission Cost**: Menghitung total biaya (Rupiah) yang dikeluarkan untuk mengirim data ke server pusat. (tarif internet per MB)
    *   **Compute Cost**: Estimasi biaya operasional berdasarkan durasi training (CPU uptime) yang dikonversi ke tarif listrik (kWh).

* Biaya Pengadaan
*   **Infrastructure Cost**: Membandingkan biaya pengadaan server pusat vs pemanfaatan CPU terminal yang sudah ada (Biaya $\approx$ Rp0 di sisi client).

**Rumus Biaya Transmisi:**
$$ \text{Cost}_{\text{data}} = \text{Volume (MB)} \times \text{Rp 2,-} $$

**Rumus Biaya Listrik:**
$$ \text{Cost}_{\text{listrik}} = \frac{\text{Watt} \times \text{Jam}}{1000} \times \text{Rp 1.444,70} $$

---

## 2. Evaluasi Inferensi (Penggunaan Langsung)
Mengukur keandalan dan kecepatan sistem saat digunakan mahasiswa untuk absensi di depan kamera.

### A. Metrik Inferensi (Biometrik)
*   **False Acceptance Rate (FAR)**: Frekuensi orang asing salah dikenali sebagai user terdaftar.
*   **False Rejection Rate (FRR)**: Frekuensi user terdaftar malah ditolak oleh sistem.
*   **True Acceptance Rate (TAR)**: Frekuensi user asli berhasil dikenali dengan benar.
*   **Equal Error Rate (EER)**: Titik di mana nilai FAR sama dengan nilai FRR.

> **Kenapa perlu mengetahui EER?** EER digunakan untuk menentukan **Threshold (Ambang Batas)** paling optimal. Semakin rendah nilai EER, semakin akurat sistem dalam menyeimbangkan antara keamanan (tidak menerima penyusup) dan kenyamanan (tidak menolak user asli).

**Rumus FAR:**
$$ \text{FAR} = \frac{\text{Jumlah Orang Asing Diterima}}{\text{Total Percobaan Orang Asing}} \times 100\% $$

**Rumus FRR:**
$$ \text{FRR} = \frac{\text{Jumlah User Terdaftar Ditolak}}{\text{Total Percobaan User Terdaftar}} \times 100\% $$

**Rumus TAR:**
$$ \text{TAR} = \frac{\text{Jumlah User Terdaftar Diterima}}{\text{Total Percobaan User Terdaftar}} \times 100\% $$

**Kondisi EER:**
$$ \text{EER} = \text{Titik } \theta \text{ di mana } \text{FAR}(\theta) = \text{FRR}(\theta) $$

### B. Kecepatan (Latency)
*   **Inference Latency (ms)**: Waktu yang dibutuhkan dari wajah terdeteksi hingga nama muncul (Target: < 500ms).

> **Kenapa dievaluasi?** Untuk menjamin keamanan (tidak ada penyusup) dan pengalaman pengguna yang mulus (absen instan).

---

## 3. Skenario Uji

### A. 4 Bentuk Sistem (Kondisi Data)
1.  **Centralized (Baseline)**: Semua data ditarik ke satu server (Performa maksimal sebagai tolok ukur).
2.  **FL Ideal (Seimbang)**: Pembagian data mahasiswa seimbang antar client (15:15).
3.  **FL Non-Ideal (Tidak Seimbang)**: Pembagian data sedikit tidak seimbang (10:20).
4.  **FL Non-Ideal (Sangat Tidak Seimbang)**: Pembagian data sangat timpang (5:25) untuk menguji ketahanan model.

### B. 4 Skenario Uji Lapangan
1.  **Uji Konsistensi (Reliability)**: 1 orang melakukan absen berkali-kali untuk melihat stabilitas skor similarity.
2.  **Uji Lintas Perangkat (Cross-Device)**: Mahasiswa Client 1 mencoba absen di perangkat Client 2 untuk menguji universalitas model.
3.  **Uji Kecepatan (Latency)**: Mengukur waktu respon proses pada CPU.
4.  **Uji Penolakan**: Orang yang tidak terdaftar mencoba absen untuk memastikan sistem memberikan label "Unknown".
