# Catatan Implementasi & Perbedaan dari Proposal

## 🚀 Optimalisasi Perangkat Edge (Low RAM)
Untuk mendukung perangkat dengan RAM terbatas (1GB - 4GB), implementasi teknis melakukan beberapa penyesuaian dari proposal awal:

1.  **Transition ke ONNX Runtime**:
    *   Meskipun proposal mungkin menyarankan penggunaan PyTorch secara langsung, sistem diubah menggunakan **ONNX Runtime** untuk proses inferensi absensi di latar belakang.
    *   **Alasan**: Mengurangi konsumsi RAM hingga 80% dan memperkecil ukuran library dari >1GB menjadi ~30MB.
    
2.  **Kuantisasi Model (Dynamic Quantization)**:
    *   Model `MobileFaceNet` dikonversi ke format **INT8 ONNX**.
    *   **Hasil**: Ukuran model turun dari ~100MB menjadi ~25MB dengan penurunan akurasi yang minimal (<1%), sangat krusial untuk perangkat edge dengan penyimpanan terbatas.

3.  **MTCNN Pre-Cropping & Resize**:
    *   Proses deteksi dilakukan satu kali di awal, lalu wajah di-crop dan disimpan. Training dan Inferensi dilakukan pada data yang sudah di-resize ke **112x96**.

---

## 🛡️ Stabilisasi & Fitur Mutakhir (Breakthroughs)
Beberapa fitur kritial ditambahkan untuk meningkatkan performa nyata sistem Federated Learning:

1.  **Global Batch Normalization (BN) Merging**:
    Server kini menggabungkan statistik BN dari seluruh terminal menjadi satu set parameter global yang di-merge ke backbone sebelum didistribusikan kembali.

2.  **Knowledge Sharing (Global Identity Sync)**:
    Terminal kini saling berbagi *feature centroids* (melalui server). Hal ini memungkinkan terminal A mengenali mahasiswa yang hanya pernah mendaftar di terminal B tanpa harus mengirim foto mentah mereka.

3.  **Dynamic Head Expansion with Weight Preservation**:
    *   Implementasi logika *Weight Copying* memastikan bobot mahasiswa lama tetap terjaga saat struktur klasifikasi melebar untuk menampung identitas baru.

4.  **Architectural Alignment (112x96 Portrait)**:
    *   Sistem secara ketat menggunakan dimensi **112x96** yang disesuaikan dengan kernel global depthwise convolution MobileFaceNet. Ini memastikan fitur wajah tidak terdistorsi antara fase training dan inferensi.
