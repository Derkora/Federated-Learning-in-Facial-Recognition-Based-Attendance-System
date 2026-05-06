import cv2
import numpy as np
import os
from PIL import Image

# Hardcoded camera parameters dari data EXIF asli (Detail Gambar)
IMAGE_METADATA = {
    "cukup cahaya.jpeg": {
        "aperture": 1.8,
        "exposure_time": 1/50,  # 0.02s
        "exposure_str": "1/50s",
        "iso": 56,
        "device": "ASUS_I003DD",
        "time": "02:27 PM",
        "description": "Kondisi Pencahayaan Cukup (Ideal)"
    },
    "minim cahaya.jpeg": {
        "aperture": 1.8,
        "exposure_time": 1/33,  # 0.0303s
        "exposure_str": "1/33s",
        "iso": 252,
        "device": "ASUS_I003DD",
        "time": "04:16 PM",
        "description": "Kondisi Pencahayaan Minim (Redup)"
    },
    "gelap.jpeg": {
        "aperture": 1.8,
        "exposure_time": 1/20,  # 0.05s
        "exposure_str": "1/20s",
        "iso": 1019,
        "device": "ASUS_I003DD",
        "time": "05:10 PM",
        "description": "Kondisi Pencahayaan Gelap"
    }
}

def calculate_exif_lux(avg_brightness, aperture, exposure_time, iso):
    """
    Menghitung estimasi Lux fisik berdasarkan formula APEX kamera:
    Lux (E) = C * (N^2) / (t * ISO) * (avg_brightness / 255)
    Di mana:
      - N = Aperture (F-Number)
      - t = Exposure Time (Shutter Speed dalam detik)
      - ISO = Kepekaan Sensor
      - C = Konstanta Kalibrasi Kamera (~250 untuk sensor digital standar)
    """
    # Rumus Fisika APEX
    lux = 250.0 * (aperture ** 2) / (exposure_time * iso) * (avg_brightness / 255.0)
    return float(lux)

def analyze_image_brightness(image_path):
    """Menganalisis intensitas cahaya (Lux) pada citra."""
    file_name = os.path.basename(image_path)
    if file_name not in IMAGE_METADATA:
        return None

    # Load citra menggunakan OpenCV
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Konversi ke Grayscale menggunakan standar Luma BT.601
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    rms_contrast = np.std(gray)

    # Ambil parameter kamera spesifik untuk gambar ini
    meta = IMAGE_METADATA[file_name]
    aperture = meta["aperture"]
    exposure_time = meta["exposure_time"]
    exposure_str = meta["exposure_str"]
    iso = meta["iso"]

    # Hitung Lux menggunakan rumus fisik APEX
    lux = calculate_exif_lux(avg_brightness, aperture, exposure_time, iso)

    # Klasifikasikan kategori lux ruangan
    if lux < 35:
        kategori = "Sangat Gelap (Minim Cahaya)"
    elif lux < 100:
        kategori = "Redup (Cahaya Kurang)"
    elif lux < 350:
        kategori = "Terang Normal (Cukup Cahaya - Standar Ruangan Kerja)"
    else:
        kategori = "Sangat Terang (Outdoor / Cahaya Berlebih)"

    return {
        "path": image_path,
        "avg_brightness": round(avg_brightness, 2),
        "rms_contrast": round(rms_contrast, 2),
        "estimated_lux": round(lux, 2),
        "kategori": kategori,
        "aperture": aperture,
        "exposure_time": exposure_str,
        "iso": iso,
        "device": meta["device"],
        "time": meta["time"],
        "description": meta["description"]
    }

def print_result_table(results):
    """Menampilkan tabel hasil analisis lux yang rapi dan menyimpannya ke berkas."""
    out = []
    out.append("=" * 135)
    out.append(f"{'NAMA BERKAS':<25} | {'KECERAHAN (0-255)':<18} | {'ESTIMASI LUX':<14} | {'KLASIFIKASI CAHAYA':<32} | {'ISO':<5} | {'WAKTU':<8} | {'PERANGKAT':<15}")
    out.append("=" * 135)
    for res in results:
        name = os.path.basename(res["path"])
        out.append(f"{name:<25} | {res['avg_brightness']:<18} | {res['estimated_lux']:<14} | {res['kategori']:<32} | {res['iso']:<5} | {res['time']:<8} | {res['device']:<15}")
    out.append("=" * 135)
    out.append("\n[ANALISIS DETIL METADATA & FISIKA APEX]:")
    for res in results:
        name = os.path.basename(res["path"])
        out.append(f"\n- Gambar: {name} ({res['description']})")
        out.append(f"  * Perangkat: {res['device']}")
        out.append(f"  * Waktu Pengambilan: {res['time']}")
        out.append(f"  * Rata-rata Kecerahan Piksel (Grayscale BT.601): {res['avg_brightness']} / 255.0")
        out.append(f"  * Kontras RMS: {res['rms_contrast']}")
        out.append(f"  * Parameter Kamera -> Aperture: f/{res['aperture']} | Shutter: {res['exposure_time']} | ISO: {res['iso']}")
        out.append(f"  * Rumus APEX Lux: 250.0 * ({res['aperture']}^2) / ({res['exposure_time']} * {res['iso']}) * ({res['avg_brightness']} / 255.0) = {res['estimated_lux']} Lux")
    
    output_str = "\n".join(out)
    print(output_str)
    
    # Simpan hasil ke info-lux.txt di direktori yang sama dengan skrip
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "info-lux.txt")
    with open(output_file, "w") as f:
        f.write(output_str)
    print(f"\n[INFO] Hasil analisis berhasil diperbarui di: {output_file}")

if __name__ == "__main__":
    # Gunakan jalur relatif terhadap lokasi berkas skrip
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "gambar-ruangan")
    images_to_check = [
        "cukup cahaya.jpeg", 
        "minim cahaya.jpeg", 
        "gelap.jpeg"
    ]
    
    results = []
    for img_name in images_to_check:
        full_path = os.path.join(base_dir, img_name)
        if os.path.exists(full_path):
            analysis = analyze_image_brightness(full_path)
            if analysis:
                results.append(analysis)
        else:
            print(f"[PERINGATAN] Berkas tidak ditemukan: {full_path}")
            
    if results:
        print_result_table(results)
    else:
        print("Tidak ada gambar yang berhasil dianalisis.")
