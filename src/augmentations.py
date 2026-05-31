# src/augmentations.py

from torchvision.transforms import v2
import torch

def get_transforms(is_train=True):
    if is_train:
        return v2.Compose([
            v2.Resize((256, 256), antialias=True),
            v2.RandomCrop(224),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Tập Dev/Test không được dùng Random nhiễu
        return v2.Compose([
            v2.Resize((256, 256), antialias=True),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])