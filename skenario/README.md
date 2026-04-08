# Panduan Menjalankan Skenario Eksperimen

## Struktur Skenario
1.  **seimbang**: Simulasi data terdistribusi secara merata (digunakan untuk CL dan FL).
2.  **timpang**: Simulasi data tidak merata (FL).
3.  **ekstrim**: Simulasi data sangat tidak merata (FL).

---

## Cara Menjalankan (Beda Perangkat)

### Langkah 1: Persiapan Dataset
1.  Siapkan folder dataset wajah di masing-masing perangkat Client.
2.  Letakkan ke dalam folder: `skenario/[nama_skenario]/client/datasets/students/`.

### Langkah 2: Konfigurasi IP (Perangkat Client saja)
1.  Masuk ke `skenario/[nama_skenario]/client/`.
2.  Ubah nama `.env.example` menjadi `.env`.
3.  Isi `SERVER_IP` dengan alamat IP fisik laptop yang menjadi Server.

### Langkah 3: Menjalankan Container
**SERVER:**
1.  Masuk ke `skenario/[nama_skenario]/server/`.
2.  Jalankan: `docker-compose up --build`

**CLIENT:**
1.  Masuk ke `skenario/[nama_skenario]/client/`.
2.  Jalankan: `docker-compose up --build`

---

## Catatan Penting
*   **Database Isolasi**: Setiap skenario memiliki database Postgres sendiri di folder `data/db`. Jika ingin mengulang skenario dari nol (wipe), hapus folder `data/` di skenario tersebut.
*   **Urutan**: Selalu jalankan Server terlebih dahulu sebelum Client.
*   **Network**: Pastikan kedua laptop berada di jaringan WiFi/LAN yang sama dan port `8081` & `8085` tidak diblokir oleh Firewall (Windows Defender).
