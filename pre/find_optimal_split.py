import os
import random
import shutil
import argparse

def get_image_count(folder_path):
    """Menghitung jumlah gambar dalam folder."""
    valid_exts = ('.jpg', '.jpeg', '.png')
    if not os.path.isdir(folder_path):
        return 0
    return len([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])

def find_best_split(students):
    """
    Mencari pembagian 15-15 mahasiswa yang paling optimal menggunakan algoritma 
    Randomized Hill Climbing untuk meminimalkan selisih jumlah gambar antar client.
    """
    n = len(students)
    half = n // 2
    
    best_diff = float('inf')
    best_c1 = []
    best_c2 = []
    
    # Lakukan 20.000 kali iterasi pencarian lokal acak (sangat cepat, < 0.2 detik)
    for _ in range(20000):
        shuffled = list(students)
        random.shuffle(shuffled)
        c1 = shuffled[:half]
        c2 = shuffled[half:]
        
        sum1 = sum(item[1] for item in c1)
        sum2 = sum(item[1] for item in c2)
        diff = abs(sum1 - sum2)
        
        improved = True
        while improved:
            improved = False
            for i in range(half):
                for j in range(half):
                    # Coba tukar c1[i] dengan c2[j]
                    new_sum1 = sum1 - c1[i][1] + c2[j][1]
                    new_sum2 = sum2 - c2[j][1] + c1[i][1]
                    new_diff = abs(new_sum1 - new_sum2)
                    
                    if new_diff < diff:
                        diff = new_diff
                        sum1 = new_sum1
                        sum2 = new_sum2
                        # Tukar elemen
                        c1[i], c2[j] = c2[j], c1[i]
                        improved = True
                        break
                if improved:
                    break
        
        if diff < best_diff:
            best_diff = diff
            best_c1 = list(c1)
            best_c2 = list(c2)
            if best_diff == 0:
                break # Ditemukan keseimbangan mutlak (selisih 0)
                
    return best_c1, best_c2, best_diff

def main():
    parser = argparse.ArgumentParser(description="Mencari pembagian 15-15 mahasiswa optimal untuk Simulasi FL.")
    parser.add_argument("--apply", action="store_true", help="Terapkan hasil pembagian ke folder client1_data dan client2_data.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, "datasets")
    
    if not os.path.exists(src_dir):
        # Fallback ke root repository jika tidak ditemukan secara relatif
        src_dir = os.path.join(base_dir, "..", "datasets")
        if not os.path.exists(src_dir):
            print(f"[ERROR] Folder '{src_dir}' tidak ditemukan!")
            return

    # Ambil semua folder mahasiswa
    folders = [f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, f)) and not f.startswith(".")]
    
    # Filter folder agar hanya memproses folder berformat NRP_Nama
    folders = [f for f in folders if "_" in f]
    
    if len(folders) != 30:
        print(f"[INFO] Jumlah folder mahasiswa terdeteksi: {len(folders)} (Target ideal: 30).")
        if len(folders) % 2 != 0:
            print("[ERROR] Jumlah folder ganjil! Tidak dapat dibagi 50:50 secara rata.")
            return

    # Ambil data jumlah gambar tiap mahasiswa
    students_data = []
    for f in folders:
        path = os.path.join(src_dir, f)
        count = get_image_count(path)
        students_data.append((f, count))

    print(f"=== MENGANALISIS {len(folders)} MAHASISWA DI: {src_dir} ===")
    total_imgs = sum(c for _, c in students_data)
    print(f"Total Gambar Keseluruhan : {total_imgs} foto")
    print(f"Rata-rata per Mahasiswa   : {total_imgs / len(folders):.2f} foto/orang\n")

    # Cari kombinasi pembagian terbaik
    c1, c2, diff = find_best_split(students_data)
    
    sum_c1 = sum(c for _, c in c1)
    sum_c2 = sum(c for _, c in c2)
    
    # Urutkan berdasarkan nama agar rapi saat ditampilkan
    c1.sort()
    c2.sort()
    
    print("=" * 70)
    print(f"📢 HASIL PEMBAGIAN OPTIMAL (Masing-masing {len(folders)//2} Mahasiswa)")
    print("=" * 70)
    print(f"🟢 CLIENT 1 (Total: {sum_c1} gambar):")
    for i, (f, c) in enumerate(c1, 1):
        print(f"  {i:2d}. {f:<50} ({c} foto)")
        
    print("\n" + "-" * 70)
    print(f"🔵 CLIENT 2 (Total: {sum_c2} gambar):")
    for i, (f, c) in enumerate(c2, 1):
        print(f"  {i:2d}. {f:<50} ({c} foto)")
        
    print("=" * 70)
    print(f"📊 ANALISIS KESEIMBANGAN BEBAN DATA:")
    print(f"  - Total Client 1 : {sum_c1} foto")
    print(f"  - Total Client 2 : {sum_c2} foto")
    print(f"  - Selisih Beban  : {diff} foto (Perbedaan hanya {diff/total_imgs*100:.3f}% dari total data!)")
    print("=" * 70)

    # Terapkan pembagian jika diinstruksikan
    if args.apply:
        print("\n⏳ Menerapkan pembagian ke folder client...")
        repo_root = os.path.abspath(os.path.join(base_dir, ".."))
        c1_dest = os.path.join(repo_root, "datasets", "client1_data", "students")
        c2_dest = os.path.join(repo_root, "datasets", "client2_data", "students")

        # Bersihkan & buat ulang folder tujuan
        for dest_dir, client_name in [(c1_dest, "Client 1"), (c2_dest, "Client 2")]:
            if os.path.exists(dest_dir):
                print(f"  - Membersihkan folder lama {client_name}...")
                shutil.rmtree(dest_dir)
            os.makedirs(dest_dir, exist_ok=True)

        # Salin data ke Client 1
        print("  - Menyalin data mahasiswa ke Client 1...")
        for f, _ in c1:
            shutil.copytree(os.path.join(src_dir, f), os.path.join(c1_dest, f))
            
        # Salin data ke Client 2
        print("  - Menyalin data mahasiswa ke Client 2...")
        for f, _ in c2:
            shutil.copytree(os.path.join(src_dir, f), os.path.join(c2_dest, f))

        print("\n✅ SUKSES! Data telah berhasil dibagi dan disalin secara seimbang ke:")
        print(f"  📍 Client 1: {c1_dest}")
        print(f"  📍 Client 2: {c2_dest}")
    else:
        print("\n💡 Tips: Jalankan dengan argumen `--apply` untuk langsung membagi dan memindahkan folder mahasiswa secara otomatis!")
        print("   Contoh: python pre/find_optimal_split.py --apply")

if __name__ == "__main__":
    main()
