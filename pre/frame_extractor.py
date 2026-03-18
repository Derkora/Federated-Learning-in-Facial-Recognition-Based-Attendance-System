import cv2
import os

# --- KONFIGURASI ---
VIDEO_SOURCE = 'videos/Kelas-C-24.mp4'  # Nama file video Komandan
OUTPUT_DIR = 'datasets/kelas-c-24'        # Nama folder hasil
SAMPLING_RATE = 0.17             

def run_extraction():
    # 1. Cek folder output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"[INFO] Folder '{OUTPUT_DIR}' dibuat.")

    # 2. Load video
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Video nggak ketemu.")
        return

    # Ambil info FPS video asli
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * SAMPLING_RATE)
    
    print(f"[INFO] Video terdeteksi: {fps} FPS")
    print(f"[INFO] Mengambil 1 frame setiap {SAMPLING_RATE} detik (Tiap {interval} frame).")

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Simpan frame berdasarkan interval
        if frame_idx % interval == 0:
            filename = f"frame_{frame_idx:06d}.jpg"
            save_path = os.path.join(OUTPUT_DIR, filename)
            
            # Simpan kualitas tinggi
            cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"\n[DONE] Selesai!")
    print(f"Total frame diproses: {frame_idx}")
    print(f"Total gambar bersih disimpan: {saved_idx}")

if __name__ == "__main__":
    run_extraction()