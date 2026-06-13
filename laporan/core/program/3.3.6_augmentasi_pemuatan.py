import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Konfigurasi pipa augmentasi citra
self.transform = transforms.Compose([
    transforms.Resize(
        (112, 96), 
        interpolation=InterpolationMode.BILINEAR
    ),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ColorJitter(
        brightness=0.5, contrast=0.5, 
        saturation=0.4, hue=0.1
    ),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomAutocontrast(p=0.2),
    transforms.RandomApply([
        transforms.GaussianBlur(
            kernel_size=(3, 3), sigma=(0.1, 2.0)
        )
    ], p=0.4),
    transforms.RandomAdjustSharpness(
        sharpness_factor=2, p=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5], 
        std=[0.50196, 0.50196, 0.50196]
    ),
    transforms.RandomErasing(p=0.1)
])

# Pemuatan sampel data latih secara dinamis
def __getitem__(self, idx):
    sample = self.samples[idx]
    label = sample['label']
    
    # Baca gambar jika tipe data berupa file citra
    if sample['type'] == "image":
        image = Image.open(sample['path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, False
    else:
        # Kembalikan data embedding memori global
        return sample['data'], label, True
