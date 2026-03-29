import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm

def get_blur_score(image_path):
    """Menghitung skor ketajaman menggunakan Laplacian Variance."""
    img = cv2.imread(image_path)
    if img is None: return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_equalization(src_dir, dest_dir, limit=40):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    folders = [f for f in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, f))]

    print(f"mulai meratakan data ke {limit} foto terbaik...")

    for folder in tqdm(folders):
        src_folder_path = os.path.join(src_dir, folder)
        dest_folder_path = os.path.join(dest_dir, folder)
        images = [f for f in os.listdir(src_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        scored_images = []

        for img_name in images:
            path = os.path.join(src_folder_path, img_name)
            score = get_blur_score(path)
            scored_images.append((img_name, score))

        scored_images.sort(key=lambda x: x[1], reverse=True)
        top_images = scored_images[:limit]
        
        if not os.path.exists(dest_folder_path):
            os.makedirs(dest_folder_path)

        for img_name, score in top_images:
            shutil.copy(os.path.join(src_folder_path, img_name),
                        os.path.join(dest_folder_path, img_name))

    print(f"\nData sudah rata di folder: {dest_dir}")


if __name__ == "__main__":
    SOURCE_DATA = "datasets"
    DEST_DATA = "datasets_equalized"

    process_equalization(SOURCE_DATA, DEST_DATA, limit=50)