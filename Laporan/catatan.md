# Catatan Perubahan & Metodologi Riset

## 1. Upgrade Preprocessing: Affine Landmark Alignment
- **Semula (Proposal)**: Hanya menggunakan MTCNN Bounding Box Crop.
- **Implementasi (Update)**: Menggunakan **Affine Alignment** berdasarkan 5 titik landmark (mata, hidung, mulut).
- **Justifikasi**: Penyelarasan wajah secara geometris (memastikan posisi mata sejajar horizontal) secara signifikan mengurangi variasi fitur yang harus dipelajari model, sehingga akurasi meningkat drastis terutama pada wajah yang miring.

## 2. Adopsi Arsitektur pFedFace (Personalized FL)
- **Semula (Proposal)**: Full Model Synchronization (semua parameter disinkronkan).
- **Implementasi (Update)**: Hanya mensinkronkan **Backbone** (MobileFaceNet), sementara **BatchNorm (BN)** dan **Classifier Head** tetap lokal di setiap terminal.
- **Justifikasi**: Masalah *non-IID* (distribusi identitas yang berbeda antar terminal) dapat menyebabkan *Loss Spike* jika Head disinkronkan secara paksa. Dengan pFedFace, terminal mempertahankan "pengetahuan identitas lokal" yang unik bagi mahasiswanya sendiri.

## 3. Preservasi Statistik Global Batch Normalization (BN)
- **Teknis**: Statistik BN (mean/variance) tidak lagi direset setiap ronde di server. Server sekarang menjaga (*preserve*) BN global hasil sesi sebelumnya sebagai baseline.
- **Dampak**: Menghilangkan masalah "amnesia model" di mana akurasi sering kali drop secara misterius setelah ronde pelatihan selesai. Hal ini memastikan model selalu memulai dari titik optimal (Acc 0.9+ sejak ronde awal).

## 4. Standarisasi Dimensi Portrait 96x112
- **Detail**: Resolusi input dikunci pada **96 (width) x 112 (height)**.
- **Justifikasi**: Fokus pada area vertikal wajah (portrait) memberikan kepadatan fitur yang lebih baik untuk model MobileFaceNet dibandingkan resolusi landscape atau square standar.

## 5. Implementasi Flip Trick & CIM pada Inferensi
- **Flip Trick**: Mengevaluasi wajah asli dan mirror (rata-rata embedding) untuk stabilitas skor.
- **Confident Instant Match (CIM)**: Bypass temporal voting jika skor > 0.85 untuk respons instan.
- **Justifikasi**: Meningkatkan *User Experience* (kecepatan absensi) tanpa mengorbankan keamanan (keamanan tetap terjaga melalui threshold 0.75 untuk kasus umum).

## 6. Integrasi Database untuk Versioning Model
- **Sistem**: Pelacakan versi model (v1, v2, dst.) dikelola secara permanen di database PostgreSQL.
- **Manfaat**: Konsistensi versi tetap terjaga meskipun infrastruktur server (container) dimulai ulang, memudahkan terminal dalam melakukan sinkronisasi otomatis.

## 7. Partial Freezing (Stage-wise Training)
- **Penerapan**: Membekukan lapisan `conv1` hingga `conv_3` (indeks 0-11) pada kedua sistem (CL & FL).
- **Justifikasi**: Mencegah "Catastrophic Forgetting" pada fitur wajah dasar yang sudah sangat baik dari model pretrained. Dengan fokus hanya pada lapisan akhir, model lebih stabil dalam mengenali identitas spesifik mahasiswa tanpa merusak ekstraktor fitur umum.

## 8. Adaptasi Lingkungan (Client-side BN Calibration)
- **Update (Centralized)**: Menambahkan langkah kalibrasi BatchNorm pada client CL setelah download model global.
- **Justifikasi**: Menghilangkan bias "Global BN" yang sering kali menyebabkan CL gagal pada pencahayaan yang berbeda dari distribusi training. Dengan kalibrasi lokal, model CL memiliki kemampuan adaptasi lingkungan yang setara dengan model pFedFace (FL).

## 9. Penyelarasan Total Iterasi & SWA
- **Detail**: CL (20 Epoch) disetarakan dengan FL (10 Round x 2 Epoch). SWA pada CL (4 epoch terakhir) disetarakan dengan Snapshot Averaging FL (2 ronde terakhir, 4 epoch total).
- **Justifikasi**: Menjamin bahwa model pada kedua metode memiliki jumlah "pengalaman" yang sama terhadap data sebelum dibandingkan.

## 10. Paritas Algoritma Inferensi (Edge Side)
- **Detail**: Implementasi **Flip Trick** (Average Horizontal Flip) dan **Temporal Voting** (Buffer 10 frame) disamakan di tingkat kode Python.
- **Justifikasi**: Memastikan perbedaan akurasi benar-benar berasal dari metode pembelajaran (Centralized vs Federated), bukan karena perbedaan cara mesin inferensi bekerja di client.

## 11. Standarisasi Centroid Generation
- **Detail**: Jumlah gambar untuk ekstraksi fitur final (Centroid) ditetapkan sebanyak **50 gambar terbaik** (seleksi Laplacian) untuk kedua metode.
- **Justifikasi**: Menghilangkan bias pada Client 1 (pencipta data) yang sebelumnya hanya menggunakan 5 gambar (quick refresh), sehingga sekarang semua client memiliki kualitas referensi yang sama kuatnya (Premium Quality).

## 12. Perbandingan dengan Proposal Awal
| Komponen | Proposal (Awal) | Implementasi (Aktual) | Alasan Perubahan |
|----------|-----------------|-----------------------|------------------|
| Hardware | Raspberry Pi 4 | Raspberry Pi 3B / Jetson | Menyesuaikan ketersediaan alat & menguji batas bawah hardware. |
| Perangkat | 3 Client | 2 Client | Fokus pada validasi orkestrasi & stabilitas jaringan. |
| Model | Facenet | MobileFaceNet (Xiaomi) | Efisiensi parameter & kompatibilitas Mobile SDK. |
| Dashboard | Streamlit | FastAPI + Vanilla JS | Latensi UI lebih rendah & kendali penuh atas request lifecycle. |

## 13. Metodologi Pengumpulan Data

### A. Perhitungan Bandwidth (Data Transmission)
Dihitung berdasarkan *Payload Size* pada lapisan aplikasi (Application Layer):
- **Centralized Learning (CL)**: 
  - Mengukur total ukuran gambar mentah (Images) yang diunggah dari terminal ke server.
  - Rumus: `Total = Ukuran_Gambar_x_Jumlah_Data`.
- **Federated Learning (FL)**:
  - **Sync Phase**: Mengukur transmisi bobot model (upload & download) selama N ronde.
  - **Registry Phase**: Mengukur ukuran file registri centroid yang diunduh client di akhir fase.
  - **Rumus FL**: `Total = (Model_Weights_Size * 2 * Num_Rounds * Num_Clients) + (Registry_Size * Num_Clients)`.

### B. Perhitungan Konsumsi Energi (kWh) via CodeCarbon
- **Metode**: Menggunakan pustaka **CodeCarbon** dengan tracker `OfflineEmissionsTracker`.
- **Cara Kerja**: Mendeteksi TDP (Thermal Design Power) dan utilisasi CPU/GPU secara real-time selama proses training/agregasi berlangsung.
- **Akurasi**: 
  - **Tinggi (Intel/AMD)**: Menggunakan sensor **RAPL** (Running Average Power Limit) untuk pembacaan konsumsi daya aktual.
  - **Estimasi (ARM/Edge)**: Menggunakan profil konsumsi daya referensi berdasarkan model chipset jika sensor hardware tidak tersedia.
- **Justifikasi**: Penggunaan kWh memberikan dimensi baru dalam riset, yaitu **Environmental Impact** dan **Operational Cost**, yang membuktikan bahwa FL tidak hanya lebih privat tetapi juga bisa lebih hemat biaya operasional jangka panjang dibandingkan CL yang memerlukan server GPU besar secara terus-menerus.

## 14. Persistensi Log & Auditabilitas Data (Riset)
Untuk menjamin integritas data selama pengujian berulang (10x trial), sistem dilengkapi mekanisme penyimpanan log permanen:

- **Log File System**: Semua aktivitas teknis (Training, Inference, Aggregation) disimpan dalam file `.log` di dalam direktori `/app/data/`. Log ini bersifat persisten dan tidak akan hilang meskipun container Docker dimatikan.
- **Centralized Client Monitoring**: Server dashboard kini memiliki kemampuan untuk menarik log dari terminal edge secara remote. Hal ini memudahkan peneliti (mahasiswa) dalam memantau kesehatan sistem di Raspberry Pi tanpa harus mengakses terminal SSH secara manual.
- **Inference Audit**: Setiap hasil pengenalan wajah yang berhasil (Inference Success) dicatat dengan timestamp dan skor kepercayaan, memberikan bukti autentik untuk hasil yang dilaporkan pada Bab 4.

## 15. Analisis Anomali Akurasi (FL Client 1 vs 2)
Dalam pengujian, ditemukan bahwa Client 2 (tanpa data pendaftar) memiliki akurasi lebih tinggi (±90%) dibanding Client 1 (±80%). Analisis teknis menunjukkan:

- **Validasi Citra Asli (Client 1)**: Client 1 menguji model menggunakan data citra asli (mentah) yang memiliki variansi sudut dan *noise*. Ini adalah pengujian yang "jujur" dan berat.
- **Validasi Hybrid Embedding (Client 2)**: Karena Client 2 tidak memiliki data lokal subjek tertentu, ia menggunakan **Global Embeddings** (fitur yang sudah bersih) untuk validasi kelas tersebut. Mengenali fitur matang jauh lebih mudah daripada citra mentah, sehingga skor akurasi terlihat lebih tinggi.
- **Kesimpulan**: Akurasi Client 1 lebih representatif terhadap performa di lapangan, sementara akurasi Client 2 mencerminkan keberhasilan mekanisme *Knowledge Sharing* dalam mempertahankan memori global.
