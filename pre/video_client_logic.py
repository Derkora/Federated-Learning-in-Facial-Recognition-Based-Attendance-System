import cv2
import requests
import torch
import time
import os
import json
from PIL import Image

SERVER_URL = "http://server-fl:8080" # Ubah ke URL server Anda
CLIENT_ID = "edge-terminal-1"
VIDEO_NAME = "sample_video.mp4" # Video target yang ingin diproses
SKIP_FRAMES = 10 # Lewati 10 frame (Hanya memproses 1 dari setiap 10 frame)

# Konfigurasi endpoint API
api_video_prefix = "/api/video"
api_video_stream = f"{api_video_prefix}/stream/{{video_name}}"
api_video_cache = f"{api_video_prefix}/cache/{{video_name}}"

# Pendeteksi dan Model AI Palsu/Placeholder (Sesuaikan dengan import model riil Anda)
class EdgeAIModelPlaceholder:
    def detect_face(self, img_pil):
        # Dalam implementasi riil Anda, panggil MTCNN:
        # face_tensor, box, prob = image_processor.detect_face(img_pil)
        # Di sini kita simulasikan koordinat deteksi wajah [x1, y1, x2, y2]
        time.sleep(0.01) # Simulasi latensi AI edge (10ms)
        return [120, 80, 280, 240], "Mahasiswa_5027221021", 0.95

def process_video_stream():
    """
    Logika Edge Client: 
    1. Cek apakah server sudah punya cache koordinat wajah untuk video ini.
    2. Jika ya, LEWATI pemrosesan AI (hemat CPU/GPU & RAM 100%).
    3. Jika tidak, ambil stream dari server, jalankan AI dengan Frame Skipping pada Resolusi Asli, 
       lalu kirim (upload) koordinat wajah tersebut ke cache server.
    """
    print(f"[EDGE CLIENT] Memeriksa cache koordinat wajah di server untuk: {VIDEO_NAME}")
    
    # 1. Hubungi server untuk cek cache
    try:
        res = requests.get(f"{SERVER_URL}{api_video_cache.format(video_name=VIDEO_NAME)}", timeout=5)
        if res.status_code == 200:
            cache_info = res.json()
            if cache_info.get("cached", False):
                print(f"[EDGE SUCCESS] Cache ditemukan di server! Client tidak perlu memproses AI lagi.")
                print(f"[EDGE SUCCESS] Jumlah data cache terdeteksi: {len(cache_info['data'])} frame.")
                return cache_info["data"]
    except Exception as e:
        print(f"[WARNING] Gagal memeriksa cache ke server: {e}. Terpaksa memproses secara lokal.")

    # 2. Jika cache kosong, kita buka stream byte-range dari server
    stream_url = f"{SERVER_URL}{api_video_stream.format(video_name=VIDEO_NAME)}"
    print(f"[EDGE CLIENT] Membuka aliran video stream server: {stream_url}")
    
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("[ERROR] Gagal membuka video stream server!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[EDGE CLIENT] Aliran video terhubung. FPS: {fps} | Total Frame: {total_frames}")

    # Inisialisasi pendeteksi
    ai_model = EdgeAIModelPlaceholder()
    detection_cache = []
    
    frame_idx = 0
    start_time = time.time()

    # 3. Baca stream frame-by-frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Hanya proses 1 dari setiap N frame untuk meringankan komputasi AI edge
        if frame_idx % SKIP_FRAMES != 0:
            continue
            
        # Menggunakan resolusi asli video untuk akurasi jarak jauh
        frame_resized = frame

        # Konversi ke PIL Image (sesuai format MTCNN/MobileFaceNet)
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Jalankan Model Deteksi AI pada resolusi asli
        box, label, confidence = ai_model.detect_face(img_pil)
        
        if box:
            x1, y1, x2, y2 = box
            # Simpan koordinat kotak wajah dan metadata waktu
            detection_cache.append({
                "frame": frame_idx,
                "seconds": round(frame_idx / fps, 3),
                "box": [x1, y1, x2, y2],
                "label": label,
                "confidence": float(confidence)
            })
            print(f" -> Frame {frame_idx}: Terdeteksi {label} (Conf: {confidence:.2f})")
            
        # Explicitly membuang frame lama dari memori (Stream-and-Discard)
        del frame
        del img_rgb
        del img_pil

    cap.release()
    duration = time.time() - start_time
    print(f"[EDGE PROCESS] Pemrosesan selesai. Waktu Pemrosesan: {duration:.2f} detik.")

    # 4. Kirim data koordinat wajah (cache) kembali ke server untuk disimpan
    print(f"[EDGE CLIENT] Mengirim {len(detection_cache)} data koordinat wajah ke cache server...")
    try:
        headers = {"Content-Type": "application/json"}
        res = requests.post(f"{SERVER_URL}{api_video_cache.format(video_name=VIDEO_NAME)}", json=detection_cache, headers=headers, timeout=10)
        if res.status_code == 200:
            print("[EDGE CLIENT SUCCESS] Hasil deteksi wajah berhasil disimpan di cache server!")
        else:
            print(f"[ERROR] Gagal menyimpan cache di server (Status: {res.status_code})")
    except Exception as e:
        print(f"[ERROR] Gagal menghubungi server untuk menyimpan cache: {e}")

    return detection_cache

if __name__ == "__main__":
    process_video_stream()
