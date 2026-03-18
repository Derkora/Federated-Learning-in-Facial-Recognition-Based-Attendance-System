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

Untuk deployment di perangkat terpisah (misal: 1 Server PC, 2 Jetson Nano/Raspberry Pi), gunakan file konfigurasi di folder `deployment/`.

### 1. Server Device
Copy folder `server/` dan `deployment/docker-compose-server.yaml` ke perangkat server.

```bash
# Jalankan Server
docker-compose -f deployment/docker-compose-server.yaml up --build -d
```
- Server akan berjalan di Port **8080** (API) dan **8085** (Flower).

### 2. Client Devices (Edge)
Copy folder `client/` dan `deployment/docker-compose-clientX.yaml` ke perangkat edge.

```bash
# Jalankan Client 1
docker-compose -f deployment/docker-compose-client1.yaml up --build -d
```

---

## Konfigurasi Koneksi (PENTING)

Saat client berjalan di perangkat yang berbeda dengan server, Anda perlu mengatur alamat IP Server.

1. Buka Dashboard Client di browser (misal: `http://ip-client:8080` atau `http://localhost:8082` jika simulasi).
2. Klik tombol **"Pengaturan Koneksi"**.
3. Masukkan IP Server:
   - **Server API URL**: `http://[IP_SERVER]:8080`
   - **Flower Server Address**: `[IP_SERVER]:8085`
4. Klik **Simpan**. Client akan otomatis mencoba koneksi ulang.

**Catatan**: Pengaturan ini disimpan di file `client/app/config.json`.

---

## Struktur Folder
- `/server`: Kode backend Server (FastAPI + Flower Server Strategy).
- `/client`: Kode Edge Client (FastAPI + Flower Client + Face Recognition Pipeline).
- `/deployment`: Konfigurasi Docker Compose terpisah untuk deployment fisik.
