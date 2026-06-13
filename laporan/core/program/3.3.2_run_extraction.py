import os
import cv2

SAMPLING_RATE = 0.17             

def run_extraction():
    # Cek folder output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load video
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        return

    # Ambil info FPS video asli
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * SAMPLING_RATE)
    
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
            # Simpan gambar dengan kualitas JPEG 95%
            cv2.imwrite(
                save_path, frame, 
                [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            saved_idx += 1

        frame_idx += 1

    cap.release()
