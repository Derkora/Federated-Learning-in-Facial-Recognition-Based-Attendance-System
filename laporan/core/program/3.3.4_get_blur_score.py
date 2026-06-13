import cv2

def get_blur_score(self, image_path):
    try:
        img = cv2.imread(image_path)
        if img is None: 
            return 0
        # Konversi gambar ke skala abu-abu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Variansi Laplacian untuk skor ketajaman
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception:
        return 0
