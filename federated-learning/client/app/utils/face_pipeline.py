import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, fixed_image_standardization
from .mobilefacenet import MobileFaceNet
import io
import torchvision.transforms as T
import cv2
import random
import base64
import os
import time

DEBUG_DIR = "app/static/debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

class FaceAnalysisPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[PIPELINE] Loading MTCNN on {self.device}...")
        self.mtcnn = MTCNN(
            image_size=112, 
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True, 
            device=self.device
        )
        
        print("[PIPELINE] Loading MobileFaceNet Backbone...")
        self.backbone = MobileFaceNet(embedding_size=128).to(self.device)
        self.backbone.eval()
        # AUGMENTASI (Disesuaikan untuk input 112x112)
        self.transforms = [
            T.RandomHorizontalFlip(p=1.0),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomRotation(degrees=10),
            # Simulasi Random Crop kecil lalu resize balik
            T.Compose([
                T.Resize(int(112 * 1.05)), 
                T.RandomCrop(112),
            ]),
            T.Compose([ 
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2)
            ])
        ]
        
    def _pil_to_base64(self, img_pil):
        """Konversi PIL Image ke string Base64"""
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _draw_bounding_box(self, img_pil, box):
        """Menggambar bounding box di gambar PIL dan mengembalikan base64"""
        if box is None:
            return self._pil_to_base64(img_pil)
            
        img_np = np.array(img_pil)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Koordinat MTCNN adalah [x1, y1, x2, y2]
        x1, y1, x2, y2 = [int(b) for b in box]
        cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2) # Warna Hijau
        
        img_result_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
        return self._pil_to_base64(img_result_pil)

    def _add_random_noise(self, img_pil):
        """Menambahkan Gaussian Noise (simulasi random noise)"""
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        mean = 0.0
        stddev = random.uniform(0.01, 0.05) 
        noise = np.random.normal(mean, stddev, img_np.shape)
        noisy_img_np = img_np + noise
        noisy_img_np = np.clip(noisy_img_np, 0.0, 1.0)
        return Image.fromarray((noisy_img_np * 255).astype(np.uint8))


    def _check_quality(self, img_pil):
        """Cek apakah gambar buram atau terlalu gelap/terang"""
        try:
            img_np = np.array(img_pil)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            if blur_score < 60: 
                return False, f"Blurry (Score: {blur_score:.1f})"
            mean_brightness = np.mean(gray)
            if mean_brightness < 30:
                return False, "Too Dark"
            if mean_brightness > 230:
                return False, "Too Bright"
            return True, "OK"
        except Exception as e:
            return True, "Quality Check Skip" 

    def _detect_and_crop(self, img_pil):
        """Deteksi wajah dan kembalikan Crop Wajah 112x112"""
        try:
            base_width = 640
            scale_factor = 1.0
            
            if img_pil.width > base_width:
                scale_factor = base_width / float(img_pil.width)
                h_size = int((float(img_pil.height) * scale_factor))
                img_pil = img_pil.resize((base_width, h_size), Image.Resampling.LANCZOS)
                print(f"[PIPELINE DEBUG] Downscaled input image to {img_pil.size} (Scale: {scale_factor:.2f})")

            is_good, msg = self._check_quality(img_pil)
            if not is_good:
                return None, None, f"Quality Reject: {msg}"

            boxes, probs = self.mtcnn.detect(img_pil)

            if boxes is None or len(boxes) == 0:
                return None, None, "No face"
            
            best_idx = np.argmax(probs)
            box = boxes[best_idx]
            prob = probs[best_idx]
            
            if prob < 0.60: 
                 return None, None, f"Low confidence ({prob:.2f})"

            # Crop wajah manual
            face_img_pil = img_pil.crop(box.astype(int))
            face_img_pil = face_img_pil.resize((112, 112))
            
            # Kembalikan koordinat box ke skala asli agar gambar di UI benar
            if scale_factor != 1.0:
                box = box / scale_factor
            
            return face_img_pil, box, "Success"
        except Exception as e:
            return None, None, str(e)

    def _get_embedding_from_face(self, face_img_pil, model=None):
        """Generate embedding dari gambar wajah 112x112"""
        try:
            # Tentukan model yang dipakai (internal atau external)
            net = model if model is not None else self.backbone
            
            # Cek device model
            device = next(net.parameters()).device
            
            # Konversi ke Tensor dan Normalisasi (fixed_image_standardization: (x - 127.5)/128.0)
            face_tensor = fixed_image_standardization(np.array(face_img_pil))
            face_tensor = torch.tensor(face_tensor).permute(2, 0, 1).float() # HWC -> CHW
            
            face_batch = face_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = net(face_batch)
            
            emb_numpy = embedding.cpu().numpy()[0]
            norm = np.linalg.norm(emb_numpy)
            if norm != 0:
                emb_numpy = emb_numpy / norm
            return emb_numpy
        except Exception as e:
            print(f"[EMBED ERROR] {e}")
            return None

    def _save_debug_image(self, img_pil, prefix="fail"):
        """Menyimpan gambar gagal untuk debug"""
        try:
            timestamp = int(time.time())
            filename = f"{prefix}_{timestamp}.jpg"
            path = os.path.join(DEBUG_DIR, filename)
            img_pil.save(path)
            print(f"[DEBUG] Saved failed image to {path}")
            return path
        except Exception as e:
            print(f"[DEBUG] Failed to save image: {e}")
            return None

    def process_live_frame(self, image_bytes, model=None):
        """
        Untuk endpoint /api/attendance/live (JSON Response).
        Mengembalikan embedding, box coords, dan pesan.
        """
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            # Tidak perlu copy jika hanya read
            
            # Detect -> Embed
            face_img_pil, box, msg = self._detect_and_crop(img)
            
            if face_img_pil is None:
                return None, None, msg

            emb = self._get_embedding_from_face(face_img_pil, model=model)
            return emb, box, msg
        except Exception as e:
            return None, None, str(e)

    def process_image(self, image_bytes, model=None):
        """
        Untuk endpoint /attendance. 
        Mengembalikan embedding, gambar asli dengan bounding box (b64), dan pesan
        """
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            original_img = img.copy() 
            
            # Flow Baru: Detect -> Embed
            face_img_pil, box, msg = self._detect_and_crop(original_img)
            
            drawn_img_b64 = self._draw_bounding_box(original_img, box)
            
            if face_img_pil is None:
                self._save_debug_image(original_img, prefix="attendance_fail")
                return None, drawn_img_b64, msg

            emb = self._get_embedding_from_face(face_img_pil, model=model)
            return emb, drawn_img_b64, msg
        except Exception as e:
            return None, None, str(e)

    def process_with_augmentation(self, image_bytes, filename="original"):
        """
        Untuk endpoint /register. 
        Mengembalikan list of dict: [{'emb': array, 'cropped_face_b64': str, 'original_img_with_bb_b64': str, 'source': str, 'status': str}, ...]
        """
        valid_data = []
        
        try:
            original_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return [], f"File Error: {e}", []
        
        # Deteksi Wajah di Gambar Asli
        face_img_pil, box, msg = self._detect_and_crop(original_img.copy())
        
        if face_img_pil is None:
             self._save_debug_image(original_img, prefix="register_original_fail")
             # Jika asli gagal, augmentasi juga tidak bisa jalan karena butuh wajahnya
             original_img_bb_b64 = self._draw_bounding_box(original_img.copy(), box)
             return [], f"Original Fail: {msg}", []

        # Simpan Data Asli (karena sudah detected)
        original_img_bb_b64 = self._draw_bounding_box(original_img.copy(), box)
        emb_original = self._get_embedding_from_face(face_img_pil)
        
        if emb_original is not None:
            valid_data.append({
                'emb': emb_original,
                'cropped_face_b64': self._pil_to_base64(face_img_pil),
                'original_img_with_bb_b64': original_img_bb_b64,
                'source': f"{filename} (Original)",
                'status': 'Success',
                'img_np': np.array(face_img_pil)
            })

        # Proses Augmentasi pada CROP WAJAH
        for i, transform in enumerate(self.transforms):
            aug_source = f"{filename} (Aug #{i+1})"
            try:
                # Augmentasi pada wajah yang sudah di-crop (112x112)
                aug_face_pil = transform(face_img_pil)
                
                is_noise_transform = (i == 3) # Index ke-3, tapi sesuaikan jika list berubah
                if is_noise_transform:
                    aug_face_pil = self._add_random_noise(aug_face_pil)
                    aug_source = f"{filename} (Aug #{i+1} - Noise)"
                
                # Tidak perlu detect lagi! Langsung embed.
                aug_emb = self._get_embedding_from_face(aug_face_pil)
                
                if aug_emb is not None:
                    valid_data.append({
                        'emb': aug_emb,
                        'cropped_face_b64': self._pil_to_base64(aug_face_pil),
                        'original_img_with_bb_b64': original_img_bb_b64, # Gambar bb pakai yang asli saja
                        'source': aug_source,
                        'status': 'Success',
                        'img_np': np.array(aug_face_pil)
                    })
                else:
                    valid_data.append({
                        'emb': None,
                        'cropped_face_b64': self._pil_to_base64(aug_face_pil),
                        'original_img_with_bb_b64': original_img_bb_b64,
                        'source': aug_source,
                        'status': 'Fail: Embedding Error'
                    })
            except Exception as e:
                print(f"[AUGMENT ERROR] Transform {i}: {e}")
                valid_data.append({
                    'emb': None,
                    'cropped_face_b64': None,
                    'original_img_with_bb_b64': None,
                    'source': aug_source,
                    'status': f"Error: {e}"
                })
        
        successful_data = [] # List of (emb, img_np)
        
        for d in valid_data:
            if d['emb'] is not None and 'img_np' in d:
                successful_data.append((d['emb'], d['img_np']))
        
        # Cek apakah setidaknya original_img berhasil
        if not [d for d in valid_data if d['source'].endswith('(Original)') and d['emb'] is not None]:
             return [], "Original Fail: No face detected in original image or poor quality.", valid_data
             
        print(f"[PIPELINE] Generated {len(successful_data)} successful samples from {filename}.")
        return successful_data, "Success", valid_data

face_pipeline = FaceAnalysisPipeline()