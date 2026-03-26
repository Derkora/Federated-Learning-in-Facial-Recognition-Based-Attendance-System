import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mobilefacenet import MobileFaceNet, ArcMarginProduct
from training import SimpleFaceDataset, DEVICE # Re-use components
import os

def test_client(client_name):
    print(f"\n{'='*10} TESTING {client_name.upper()} {'='*10}")
    
    processed_dir = f"processed_{client_name}"
    model_path = "global_model_final.pth"
    head_path = f"head_{client_name}.pth"

    if not os.path.exists(processed_dir):
        print(f"[ERROR] Folder {processed_dir} tidak ditemukan. Jalankan training.py dulu!")
        return
    
    if not os.path.exists(model_path) or not os.path.exists(head_path):
        print(f"[ERROR] Checkpoint {model_path} atau {head_path} tidak ditemukan!")
        return

    # 1. Load Dataset
    dataset = SimpleFaceDataset(processed_dir)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    num_classes = len(dataset.classes)

    # 2. Setup Model
    model = MobileFaceNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model.eval()

    head = ArcMarginProduct(128, num_classes).to(DEVICE)
    head.load_state_dict(torch.load(head_path, map_location=DEVICE), strict=False)
    head.eval()

    # 3. Evaluation Loop
    correct = 0
    total = 0
    
    print(f"[TEST] Menghitung akurasi Top-1 untuk {len(dataset)} gambar...")
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            # Forward
            embeddings = model(imgs)
            outputs = head(embeddings, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\n{'*'*30}")
    print(f"RESULT {client_name}:")
    print(f"Total Images: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'*'*30}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_client(sys.argv[1])
    else:
        # Test all clients in CONFIG
        from training import CLIENTS
        for c in CLIENTS:
            test_client(c['name'])