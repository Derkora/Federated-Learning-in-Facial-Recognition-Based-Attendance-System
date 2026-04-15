# Edge Federated Learning System

Sistem Federated Learning (FL) untuk Edge Devices dengan fitur Facial Recognition. Sistem ini memungkinkan training model wajah secara terdistribusi tanpa mengirimkan data mentah (foto wajah) ke server pusat, menjaga privasi pengguna.

## Fitur Utama
- **Federated Learning**: Training model di setiap Client (Edge Device), hanya update parameter model yang dikirim ke Server.
- **Privacy Preserving**: Data wajah tersimpan lokal dan terenkripsi.
- **Dynamic Configuration**: Atur alamat IP Server melalui UI tanpa perlu coding ulang.
- **Live Attendance**: Absensi wajah realtime.

---

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

## Cara Deployment (Production / Edge Devices)

Untuk deployment di perangkat terpisah (misal: 1 Server PC, 2 Jetson Nano/Raspberry Pi), gunakan file Docker Compose yang tersedia di root folder ini.

### 1. Server Device
Salin folder `/server` dan file `docker-compose-server.yml` ke perangkat server.

```bash
# Jalankan Server
docker compose -f docker-compose-server.yml up --build -d
```
- Server akan berjalan di Port **8081** (Dashboard/API) dan **8085** (Flower Aggregator).
- Dashboard dapat diakses di `http://localhost:8081`.

### 2. Client Devices (Edge)
Salin folder `/client` dan file `docker-compose-client.yml` ke perangkat edge.

```bash
# Jalankan Client
docker compose -f docker-compose-client.yml up --build -d
```
- Client akan mencoba melakukan registrasi ke IP Server yang dikonfigurasi di file `.env`.

---

## Struktur Folder
- `/server`: Kode backend Server (FastAPI + Flower Server Strategy).
- `/client`: Kode Edge Client (FastAPI + Flower Client + Face Recognition Pipeline).
- `docker-compose-server.yml`: Konfigurasi deployment untuk server Federated pusat.
- `docker-compose-client.yml`: Konfigurasi deployment untuk terminal Federated.
