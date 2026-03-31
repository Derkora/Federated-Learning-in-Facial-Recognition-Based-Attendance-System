import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import time
from facenet_pytorch import MTCNN

# ─── Configuration ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "test_backbone.pth")
REGISTRY_PATH = os.path.join(BASE_DIR, "test_registry.pth")

# Architecture Import (reuse mobilefacenet from parent)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "uji-fl"))
from mobilefacenet import MobileFaceNet

DEVICE = torch.device("cpu") # Enforcement of pure CPU requirement
THRESHOLD = 0.42 # Consistent with main system hardening

# ─── Setup ───────────────────────────────────────────────────────────────────
def load_assets():
    print(f"[BOOT] Loading Assets on {DEVICE}...")
    
    # 1. Load Backbone
    backbone = MobileFaceNet().to(DEVICE)
    backbone.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    backbone.eval()

    # 2. Load Registry
    registry = torch.load(REGISTRY_PATH, map_location=DEVICE)
    
    # Convert registry to float tensors on device immediately
    torch_registry = {}
    for nrp, vec in registry.items():
        if isinstance(vec, np.ndarray):
            vec = torch.from_numpy(vec.copy())
        torch_registry[nrp] = vec.float().to(DEVICE).unsqueeze(0)
    
    # 3. Setup Detector
    detector = MTCNN(
        image_size=112, margin=20, keep_all=False, 
        device=DEVICE, post_process=False
    )
    
    print(f"[BOOT] Ready! {len(torch_registry)} identities in registry.")
    return backbone, torch_registry, detector

val_transform = transforms.Compose([
    transforms.Resize((112, 96)), # MobileFaceNet input
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ─── Main Camera Loop ────────────────────────────────────────────────────────
def run_camera():
    backbone, registry, detector = load_assets()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible.")
        return

    print("\n" + "="*50)
    print(" PRESS 'Q' TO QUIT ")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Mirror frame for intuitive use
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        # Convert to PIL for MTCNN
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # 1. Detection
        boxes, probs = detector.detect(img_pil)
        
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < 0.90: continue # Strict detection threshold
                
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # 2. Inference
                face_pil = img_pil.crop((x1, y1, x2, y2)).resize((96, 112), Image.BILINEAR)
                input_tensor = val_transform(face_pil).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    embedding = backbone(input_tensor)
                    embedding = F.normalize(embedding, p=2, dim=1)
                    
                    # Target-Search
                    best_match = "Unknown"
                    max_sim = -1.0
                    
                    for nrp, centroid in registry.items():
                        sim = F.linear(embedding, centroid).item()
                        if sim > max_sim:
                            max_sim = sim
                            best_match = nrp
                
                # 3. Visualization
                color = (0, 255, 0) if max_sim > THRESHOLD else (0, 0, 255)
                label = f"{best_match} ({max_sim:.2f})" if max_sim > THRESHOLD else f"Rejected ({max_sim:.2f})"
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Show Output
        cv2.imshow("Isolated FL Inference Test", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera()
