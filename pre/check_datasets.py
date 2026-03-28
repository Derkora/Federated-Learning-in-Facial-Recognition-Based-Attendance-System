import os
from tabulate import tabulate # Install dulu: pip install tabulate

def check_image_counts(src_dir):
    if not os.path.exists(src_dir):
        print(f"Ndan, folder {src_dir} tidak ditemukan!")
        return

    folders = sorted([f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, f))])
    
    data = []
    total_images = 0
    
    print(f"Menganalisis folder: {src_dir}...\n")
    
    for folder in folders:
        path = os.path.join(src_dir, folder)
        # Hitung file dengan ekstensi gambar
        images = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(images)
        data.append([folder, count])
        total_images += count

    # Urutkan berdasarkan jumlah foto terkecil (agar tau siapa yang jadi bottleneck)
    data.sort(key=lambda x: x[1])

    # Tampilkan dalam bentuk tabel biar gagah
    headers = ["Nama/NRP", "Jumlah Foto"]
    print(tabulate(data, headers=headers, tablefmt="grid"))
    
    if data:
        print(f"\n📊 RINGKASAN DATA:")
        print(f"🔹 Total Mahasiswa : {len(data)}")
        print(f"🔹 Total Foto      : {total_images}")
        print(f"🔹 Foto Tersedikit : {data[0][1]} ({data[0][0]})")
        print(f"🔹 Foto Terbanyak  : {data[-1][1]} ({data[-1][0]})")
        print(f"🔹 Rata-rata       : {total_images // len(data)} foto/orang")

if __name__ == "__main__":
    SOURCE_DATA = "datasets" 
    check_image_counts(SOURCE_DATA)