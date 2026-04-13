# Panduan Menjalankan Skenario Eksperimen (Uji Coba)

Direktori ini digunakan untuk mensimulasikan operasional sistem dalam skenario eksperimen (Beda Perangkat).

---

## Persiapan Awal
1.  **Server IP**: Pastikan Anda mengetahui Alamat IP Fisik laptop yang akan menjadi server.
2.  **Dataset**: Letakkan dataset wajah mahasiswa di folder `client/data/students/[Nama_Mahasiswa]/`.

---

## Cara Menjalankan (Federated Learning)

### Langkah 1: Konfigurasi Client
1.  Masuk ke folder `client/`.
2.  Ubah nama `.env.example` menjadi `.env`.
3.  Isi berkas `.env` dengan IP Server Anda:
    ```env
    SERVER_IP=192.168.x.x
    CLIENT_ID=terminal-1
    RAW_DATA_PATH_CLIENT=./data/students
    ```

### Langkah 2: Jalankan Server
1.  Buka terminal di folder `uji-coba/server/`.
2.  Jalankan: `docker-compose up --build`
3.  Akses dashboard di: `http://localhost:8081`

### Langkah 3: Jalankan Client
1.  Buka terminal di folder `uji-coba/client/`.
2.  Jalankan: `docker-compose up --build`
3.  Akses UI Client di: `http://localhost:8081`

---

## Cara Menjalankan (Centralized Learning)

Jika Anda ingin menguji sistem terpusat (Centralized):
1.  **Server**: Jalankan `docker-compose -f docker-compose-cl.yml up --build` di folder `server/`.
2.  **Client**: Jalankan `docker-compose -f docker-compose-cl.yml up --build` di folder `client/`.
3.  Pastikan `.env` sudah terkonfigurasi dengan `SERVER_IP` yang benar.

---

## Catatan Penting
*   **Wipe Data**: Untuk mengulang eksperimen dari nol, hapus folder `data/` di masing-masing direktori `server/` atau `client/`.
*   **Network**: Kedua perangkat HARUS berada dalam satu jaringan WiFi/LAN yang sama.
*   **Port**: Jika dijalankan di satu laptop yang sama, pastikan tidak ada konflik port (disarankan menggunakan perangkat terpisah sesuai skenario Tugas Akhir).
