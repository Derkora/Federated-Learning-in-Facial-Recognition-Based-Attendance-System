# 📖 Metodologi & Alur Proses Pelatihan Model (Model Training)

Berikut adalah urutan proses terstruktur untuk bab **3.1.2 Model Training** pada penulisan buku atau karya ilmiah Anda. Setiap tahapan dipetakan secara kronologis langsung ke logika dan potongan kode esensialnya:

---

## 1. Pelatihan Lokal (Sisi Klien)

Proses pelatihan lokal berjalan di dalam perangkat Edge (Klien) untuk memperbarui bobot model menggunakan data citra lokal tanpa memindahkan data sensitif tersebut ke server.

### a. Konfigurasi Layer Freezing
* **Berkas:** `federated-learning/client/app/utils/freezing.py`
* **Penjelasan:** Sebelum pelatihan dimulai, lapisan-lapisan awal backbone MobileFaceNet dibekukan agar parameter yang dilatih lebih sedikit. Ini menghemat penggunaan energi RAM/GPU perangkat Edge secara signifikan.

```python
def set_model_freeze(model, freeze_mode="early"):
    # Aktifkan seluruh parameter model untuk dilatih kembali
    for param in model.parameters():
        param.requires_grad = True

    # Bekukan parameter awal backbone untuk menghemat komputasi edge
    if freeze_mode == "early":
        # Nonaktifkan gradien untuk lapisan input awal MobileFaceNet
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.dw_conv1.parameters():
            param.requires_grad = False
        
        # Bekukan dua belas blok bottleneck awal pada backbone
        for i in range(12): 
            for param in model.blocks[i].parameters():
                param.requires_grad = False
```

---

### b. Augmentasi Gambar On-The-Fly
* **Berkas:** `federated-learning/client/app/utils/trainer.py`
* **Penjelasan:** Saat batch gambar dimuat dari penyimpanan fisik oleh `DataLoader`, citra wajah dimodifikasi secara dinamis di memori RAM menggunakan transformasi PyTorch. Hal ini menghasilkan variasi posisi wajah baru di setiap epoch tanpa mengubah berkas gambar asli di disk.

```python
# Pipeline Augmentasi di memori RAM
self.transform = transforms.Compose([
    transforms.Resize((112, 96), interpolation=InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196]),
    transforms.RandomErasing(p=0.1)
])

# Pemuatan Dinamis Citra Wajah pada Dataset Iterator
def __getitem__(self, idx):
    # Ambil sampel dan label kelas mahasiswa berdasarkan indeks
    sample = self.samples[idx]
    label = sample['label']
    
    # Proses citra mentah lokal jika tipe datanya berupa gambar
    if sample['type'] == "image":
        image = Image.open(sample['path']).convert('RGB')
        # Jalankan augmentasi gambar acak secara on-the-fly
        if self.transform:
            image = self.transform(image)
        return image, label, False
    else:
        # Kembalikan data embedding memori global lintas-klien
        return sample['data'], label, True
```

---

### c. Optimasi Lokal Dengan FedProx Proximal Term
* **Berkas:** `federated-learning/client/app/utils/trainer.py`
* **Penjelasan:** Di sinilah loop pelatihan utama PyTorch dijalankan. Selain meminimalkan loss klasifikasi wajah ArcFace, sistem menambahkan **FedProx Proximal Term** (penalti regulasi jarak matriks L2 terhadap model acuan global) untuk menjaga agar model lokal klien tidak menyimpang (*drift*) terlalu jauh dari model global.

```python
# Loop pembagian batch data latih di dalam LocalTrainer.train()
for imgs, embs, labels, is_embedding in dataloader:
    imgs, embs, labels = imgs.to(self.device), embs.to(self.device), labels.to(self.device)
    optimizer.zero_grad()
    
    features_local = torch.zeros((labels.size(0), 128), device=self.device)
    img_mask, emb_mask = ~is_embedding, is_embedding
    
    # Forward pass citra wajah lokal ke backbone MobileFaceNet
    if img_mask.any():
        features_local[img_mask] = self.backbone(imgs[img_mask])
    # Pemuatan data memori global lintas-klien (anti-forgetting)
    if emb_mask.any():
        features_local[emb_mask] = embs[emb_mask]
    
    # Hitung loss klasifikasi ArcFace
    outputs = self.head(features_local, labels)
    ce_loss = criterion(outputs, labels)
    
    # Hitung penalti regulasi jarak L2 (FedProx Proximal Term)
    prox_loss = torch.tensor(0.0, device=self.device)
    for name, param in self.backbone.named_parameters():
        if name in global_ref:
            prox_loss += (mu / 2) * torch.norm(param - global_ref[name])**2
    
    # Total Loss = ArcFace Loss + FedProx Proximal Term
    loss = ce_loss + prox_loss
    
    # Backward pass & pembaruan bobot optimizer
    loss.backward()
    optimizer.step()
```

---

### d. Ekstraksi Parameter dan Head Centroid
* **Berkas:** `federated-learning/client/app/recognition_client.py` & `trainer.py`
* **Penjelasan:** Setelah loop pelatihan selesai, klien mengekstrak parameter backbone yang telah disesuaikan dan mengambil centroid model klasifikasi lokal.

```python
# Ekstraksi parameter backbone di dalam LocalTrainer
def get_backbone_parameters(self, personalized=True):
    params = []
    for name, param in self.backbone.named_parameters():
        # Ambil bobot teradaptasi yang memerlukan gradien
        if param.requires_grad:
            params.append(param.data.cpu().numpy())
    return params
```

---

### e. Transmisi Parameter
* **Berkas:** `federated-learning/client/app/recognition_client.py`
* **Penjelasan:** Klien mengemas seluruh metrik performa lokal (Akurasi, Loss, kWh energi) serta parameter model backbone yang baru diekstrak untuk dikirimkan kembali secara asinkron ke server orchestrator melalui protokol Flower.

```python
# Pengembalian parameter hasil latih di dalam FaceRecognitionClient.fit()
return self.trainer.get_backbone_parameters(personalized=True), num_samples, {
    "loss": float(loss),
    "accuracy": float(accuracy),
    "val_loss": float(val_loss),
    "val_accuracy": float(val_accuracy),
    "duration_s": float(fit_duration),
    "energy_kwh": float(energy_kwh),
    "hostname": os.getenv("HOSTNAME", "unknown-client")
}
```

---

## 2. Agregator (Sisi Server)

Server orchestrator bertanggung jawab untuk menyeimbangkan, mengumpulkan, mengagregasikan, dan meredistribusikan model global baru ke seluruh terminal klien.

### a. Penyelarasan Pengumpulan Parameter
* **Berkas:** `federated-learning/server/app/server_manager_instance.py`
* **Penjelasan:** Server menyelaraskan penerimaan paket dari semua klien aktif, memetakan jumlah sampel data (`num_samples`) masing-masing klien, durasi pelatihan, dan memeriksa kegagalan transmisi sebelum memicu proses agregasi.

```python
# Verifikasi hasil pengumpulan di dalam SaveModelStrategy.aggregate_fit()
clients_data = {}
for i, (client_proxy, fit_res) in enumerate(results):
    cid = fit_res.metrics.get("hostname") or f"client-{i}"
    clients_data[cid] = {
        "num_samples": fit_res.num_examples,
        "accuracy": fit_res.metrics.get("accuracy", 0.0),
        "loss": fit_res.metrics.get("loss", 0.0),
        "duration_s": fit_res.metrics.get("duration_s", 0.0)
    }
```

---

### b. Agregasi Multi-Client FedProx
* **Berkas:** `federated-learning/server/app/server_manager_instance.py`
* **Penjelasan:** Menerapkan pembobotan rata-rata parameter model backbone berdasarkan rasio data sampel klien (FedAvg) dan memisahkannya dari statistik lapisan Batch Normalization (BN) lokal untuk menjaga personalisasi tingkat klien.

```python
# Agregasi terbobot di dalam SaveModelStrategy.aggregate_fit()
params_np = fl.common.parameters_to_ndarrays(aggregated_parameters)
all_keys = list(sd.keys())

# Pisahkan kunci lapisan konvolusi backbone dengan lapisan Batch Normalization (BN)
conv_keys = [k for k in all_keys if not any(x in k.lower() for x in ['bn', 'running_', 'num_batches_tracked'])]

# Agregasikan parameter backbone saja (Mode pFedFace)
target_keys = conv_keys if len(params_np) == len(conv_keys) else all_keys

backbone_params = {}
for i, k in enumerate(target_keys):
    if i < len(params_np):
        val = torch.from_numpy(params_np[i].copy())
        backbone_params[k] = val

# Perbarui state-dict model global hanya pada parameter backbone hasil agregasi
sd.update(backbone_params)
```

---

### c. Redistribusi Model Global
* **Berkas:** `federated-learning/server/app/server_manager_instance.py`
* **Penjelasan:** Menyimpan bobot global terbaru hasil agregasi ke database server (`GlobalModel`) dan mengekspor berkas fisiknya ke registri disk agar klien dapat mengunduhnya kembali sebagai baseline awal pada ronde berikutnya.

```python
# Menyimpan model global baru ke database di dalam SaveModelStrategy.aggregate_fit()
buffer = io.BytesIO()
torch.save(sd, buffer)
new_model = GlobalModel(
    version=target_version,
    weights=buffer.getvalue(),
    loss=final_loss,
    accuracy=val_acc
)
db.add(new_model)
db.commit()
```
