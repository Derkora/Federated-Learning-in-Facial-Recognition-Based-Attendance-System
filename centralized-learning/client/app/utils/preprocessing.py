import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from facenet_pytorch import MTCNN
from PIL import Image
import gc
import json
import time
from .logging import get_logger


DEVICE = torch.device('cpu')

# Landmark kanonik 96x112 Portrait (standar InsightFace/MobileFaceNet)
# Digunakan untuk warping/alignment wajah agar posisi mata, hidung, mulut konsisten.
CANONICAL_LANDMARKS = np.array([
    [30.2946, 51.6963],  # mata kiri
    [65.5318, 51.5014],  # mata kanan
    [48.0252, 71.7366],  # hidung
    [33.5493, 92.3655],  # mulut kiri
    [62.7299, 92.2041],  # mulut kanan
], dtype=np.float32)

class ImageProcessor:
    def __init__(self):
        # MTCNN dimuat secara malas (Lazy Load) untuk efisiensi RAM
        self._mtcnn = None
        self.logger = get_logger()
        
        # Normalisasi MobileFaceNet (Standard): (x - 127.5) / 128.0
        # 128/255 = 0.50196
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.50196, 0.50196, 0.50196])

    @property
    def mtcnn(self):
        if self._mtcnn is None:
            self.logger.info("Memuat detektor MTCNN ke RAM...")
            self._mtcnn = MTCNN(
                image_size=112, 
                margin=20, 
                keep_all=True, 
                device=DEVICE, 
                post_process=False 
            )
        return self._mtcnn

    def unload_detector(self):
        if self._mtcnn is not None:
            self.logger.info("Membersihkan detektor MTCNN dari RAM...")
            del self._mtcnn
            self._mtcnn = None
            
            gc.collect()

    def detect_face(self, img, save_path=None, keep_all=False):
        """
        Deteksi wajah dengan Landmark Alignment.
        Urutan: Deteksi -> Alignment -> Crop -> Resize 112x96 (Resolusi Asli).
        """
        try:
            # Gunakan resolusi asli untuk presensi real-life dan video
            img_detect = img
            scale = 1.0

            # Deteksi kotak dan landmark
            boxes, probs, landmarks = self.mtcnn.detect(img_detect, landmarks=True)
            
            if boxes is None or len(boxes) == 0:
                if keep_all:
                    return []
                return None, None, 0.0
            
            # Kembalikan koordinat jika terjadi penskalaan 
            if scale != 1.0:
                boxes = boxes / scale
                if landmarks is not None:
                    landmarks = landmarks / scale

            if not keep_all:
                face_img = None
                
                # Mencoba Landmark Alignment (Metode Paling Stabil)
                if landmarks is not None and landmarks[0] is not None:
                    try:
                        src = np.array(landmarks[0], dtype=np.float32)
                        # Hitung matriks transformasi affine parsial
                        M, _ = cv2.estimateAffinePartial2D(src, CANONICAL_LANDMARKS, method=cv2.LMEDS)
                        if M is not None:
                            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                            # Warping gambar ke dimensi target 96x112 (WxH)
                            aligned = cv2.warpAffine(img_cv, M, (96, 112), 
                                                     flags=cv2.INTER_LINEAR, 
                                                     borderMode=cv2.BORDER_REPLICATE)
                            face_img = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
                            # self.logger.success("Wajah berhasil disejajarkan menggunakan landmark (Alignment).")
                    except Exception as e:
                        self.logger.info(f"Gagal melakukan alignment: {e}")
                
                # Fallback: Bbox Crop (Jika landmark gagal)
                if face_img is None:
                    box = boxes[0]
                    margin = 20
                    x1, y1 = max(0, int(box[0] - margin/2)), max(0, int(box[1] - margin/2))
                    x2, y2 = min(img.width, int(box[2] + margin/2)), min(img.height, int(box[3] + margin/2))
                    face_img = img.crop((x1, y1, x2, y2)).resize((96, 112), Image.BILINEAR)
                    self.logger.warn("Deteksi wajah menggunakan fallback bbox crop.")

                if save_path and face_img:
                    face_img.save(save_path)
                    
                return face_img, boxes[0], probs[0]
            
            else:
                results = []
                for i, box in enumerate(boxes):
                    prob = probs[i]
                    if prob < 0.85:  # Filter out low probability detections to prevent false boxes
                        continue
                        
                    face_img = None
                    if landmarks is not None and landmarks[i] is not None:
                        try:
                            src = np.array(landmarks[i], dtype=np.float32)
                            M, _ = cv2.estimateAffinePartial2D(src, CANONICAL_LANDMARKS, method=cv2.LMEDS)
                            if M is not None:
                                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                                aligned = cv2.warpAffine(img_cv, M, (96, 112), 
                                                         flags=cv2.INTER_LINEAR, 
                                                         borderMode=cv2.BORDER_REPLICATE)
                                face_img = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
                        except Exception as e:
                            pass
                    
                    if face_img is None:
                        margin = 20
                        x1, y1 = max(0, int(box[0] - margin/2)), max(0, int(box[1] - margin/2))
                        x2, y2 = min(img.width, int(box[2] + margin/2)), min(img.height, int(box[3] + margin/2))
                        face_img = img.crop((x1, y1, x2, y2)).resize((96, 112), Image.BILINEAR)
                        
                    results.append((face_img, box, prob))
                return results
        except Exception as e:
            self.logger.error(f"Gagal deteksi wajah: {e}")
            return None, None, 0.0

    def prepare_for_model(self, face_data):
        """
        Menyiapkan data wajah untuk input model (Normalisasi akhir).
        Pastikan input sudah berukuran 112x96.
        """
        if face_data is None: return None

        if isinstance(face_data, torch.Tensor):
            # Jika input adalah tensor mentah [0, 255]
            if face_data.max() > 2.0:
                 face_data = face_data / 255.0
            face_img = TF.to_pil_image(face_data.clamp(0, 1))
        else:
            face_img = face_data

        # Pastikan ukuran tepat 112x96 (Portrait)
        if face_img.size != (96, 112):
            face_img = face_img.resize((96, 112), Image.BILINEAR)

        # Konversi ke Tensor [0, 1]
        face_tensor = TF.to_tensor(face_img)

        # Normalisasi ke [-1, 1] menggunakan standar MobileFaceNet
        normalized_tensor = self.normalize(face_tensor)
        
        return normalized_tensor.unsqueeze(0).to(DEVICE)

    def get_blur_score(self, image_path):
        """Hitung skor ketajaman menggunakan Laplacian Variance dengan optimasi RAM & CPU."""
        try:
            img = cv2.imread(image_path)
            if img is None: return 0
            
            # Optimasi RAM & CPU perangkat edge (RPi): resize gambar besar jika melebihi 640px
            h, w = img.shape[:2]
            max_size = 640
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Hitung variansi Laplacian dengan kedalaman CV_32F (presisi tunggal) demi efisiensi RAM/CPU
            return cv2.Laplacian(gray, cv2.CV_32F).var()
        except:
            return 0

    def select_best_faces(self, folder_path, n=50):
        if not os.path.exists(folder_path):
            return []
            
        final_cache_path = os.path.join(folder_path, ".selection_cache.json")
        scores_cache_path = os.path.join(folder_path, ".laplacian_scores.json")
        
        # Use previously cached selection results to save time
        if os.path.exists(final_cache_path):
            try:
                with open(final_cache_path, "r") as f:
                    cache_data = json.load(f)
                    if cache_data.get("n") == n:
                        return cache_data.get("filenames", [])
            except: pass

        all_imgs = sorted([
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        if not all_imgs: return []

        scored_map = {}
        # Load previously calculated Laplacian scores cache
        if os.path.exists(scores_cache_path):
            try:
                with open(scores_cache_path, "r") as f:
                    scored_map = json.load(f)
            except: pass

        needs_save = False
        total_imgs = len(all_imgs)
        # Evaluate sharpness for each new image
        for i, img_name in enumerate(all_imgs):
            if img_name in scored_map:
                continue
                
            img_path = os.path.join(folder_path, img_name)
            self.logger.info(f"Menguji ketajaman Laplacian gambar {i+1}/{total_imgs}: {img_name}")
            score = self.get_blur_score(img_path)
            scored_map[img_name] = score
            needs_save = True
            
            # Periodically save scores to ensure data persistence
            if i % 50 == 0 and needs_save:
                try:
                    with open(scores_cache_path, "w") as f:
                        json.dump(scored_map, f)
                except: pass
            
            if i % 10 == 0:
                gc.collect()

        if needs_save:
            try:
                with open(scores_cache_path, "w") as f:
                    json.dump(scored_map, f)
            except: pass

        # Sort photos and select N images with the highest scores
        scored_list = [(name, score) for name, score in scored_map.items()]
        scored_list.sort(key=lambda x: x[1], reverse=True)
        
        selected = [s[0] for s in scored_list[:n]]
        
        # Save the selected best image filenames to the cache file
        try:
            with open(final_cache_path, "w") as f:
                json.dump({"n": n, "filenames": selected}, f)
        except: pass

        gc.collect()
        return selected

image_processor = ImageProcessor()

