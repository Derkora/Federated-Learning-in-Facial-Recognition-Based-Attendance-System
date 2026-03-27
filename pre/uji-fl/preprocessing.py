import os, torch, random, shutil
from PIL import Image
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIENTS = [
    {"name": "client1", "path": "../../datasets/client1_data/students"},
    {"name": "client2", "path": "../../datasets/client2_data/students"}
]

def generate_and_lock_labels():
    all_nrps = set()
    for client in CLIENTS:
        if os.path.exists(client['path']):
            all_nrps.update([d for d in os.listdir(client['path']) if os.path.isdir(os.path.join(client['path'], d))])
    global_labels = sorted(list(all_nrps))
    torch.save(global_labels, "global_labels.pth")
    print(f"[OK] {len(global_labels)} NRP dikunci di 'global_labels.pth'")
    return global_labels

def run_preprocessing():
    labels = generate_and_lock_labels()
    mtcnn = MTCNN(image_size=112, margin=20, device=DEVICE)
    
    aug = transforms.Compose([
        transforms.ColorJitter(0.6, 0.6, 0.4),
        transforms.RandomRotation(30),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.7, scale=(0.02, 0.2)),
        transforms.ToPILImage()
    ])

    for client in CLIENTS:
        for nrp in os.listdir(client['path']):
            nrp_path = os.path.join(client['path'], nrp)
            
            clean_faces = []
            for img_name in sorted(os.listdir(nrp_path)):
                img = Image.open(os.path.join(nrp_path, img_name)).convert('RGB')
                face = mtcnn(img)
                if face is not None:
                    clean_faces.append(transforms.ToPILImage()(face * 0.5 + 0.5))
                if len(clean_faces) >= 20: break 

            if len(clean_faces) < 20:
                print(f"      [Peringatan] {nrp} cuma ada {len(clean_faces)} foto, akan tetap diproses.")

            train_save_path = os.path.join(f"data_{client['name']}/train", nrp)
            val_save_path = os.path.join(f"data_{client['name']}/val", nrp)
            os.makedirs(train_save_path, exist_ok=True)
            os.makedirs(val_save_path, exist_ok=True)

            random.shuffle(clean_faces)
            train_faces = clean_faces[:16]
            val_faces = clean_faces[16:]

            for i in range(200):
                face = train_faces[i % len(train_faces)]
                if i >= len(train_faces): face = aug(face)
                face.resize((96, 112)).save(os.path.join(train_save_path, f"{i:03d}.jpg"))
            
            for i, face in enumerate(val_faces):
                face.resize((96, 112)).save(os.path.join(val_save_path, f"{i:03d}.jpg"))
                
        print(f"[SUCCESS] {client['name']} Preprocessing (Max 20 Faces) Selesai.")

if __name__ == "__main__":
    run_preprocessing()