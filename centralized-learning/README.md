# Centralized Learning Facial Recognition System

Sistem Pengenalan Wajah Berbasis Centralized Learning (CL). Seluruh dataset wajah dikirim ke server pusat untuk proses pelatihan model global secara terpusat.

## Cara Menjalankan (Development / Localhost)

Untuk menjalankan seluruh sistem (1 Server + 2 Client) dalam satu komputer menggunakan Docker:

```bash
docker-compose up --build
```

Akses layanan di:
- **Server Dashboard**: [http://localhost:8081](http://localhost:8081)
- **Client 1 Dashboard**: [http://localhost:8082](http://localhost:8082)
- **Client 2 Dashboard**: [http://localhost:8083](http://localhost:8083)

---

## Quick Start (Docker Deployment)

Gunakan file Docker Compose yang tersedia di root folder ini untuk deploy server dan client secara terpisah.

### 1. Persiapan Environment
Salin file `.env.example` menjadi `.env` dan sesuaikan nilainya:
```bash
cp .env.example .env
```
Pastikan `SERVER_IP` mengarah ke alamat IP server yang bisa dijangkau oleh client.

### 2. Jalankan Server
Device: Server PC / Laptop Utama
```bash
docker compose -f docker-compose-server.yml up --build -d
```
Server akan berjalan di port **8081**. Dashboard dapat diakses di `http://localhost:8081`.

### 3. Jalankan Client (Terminal)
Device: Edge Device (Raspberry Pi / Jetson Nano / Laptop)
```bash
docker compose -f docker-compose-client.yml up --build -d
```
Client akan berjalan dan melakukan streaming wajah ke server pusat jika dikonfigurasi.

## Struktur Proyek
- `/server`: Backend FastAPI untuk manajemen user dan proses training terpusat.
- `/client`: Frontend/Inference engine yang berjalan di perangkat edge untuk absensi.
- `docker-compose-server.yml`: Konfigurasi deployment untuk server pusat + Database.
- `docker-compose-client.yml`: Konfigurasi deployment untuk terminal pengenalan wajah.
