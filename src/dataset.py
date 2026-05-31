# src/dataset.py

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from .config import SEQ_LENGTH, STRIDE, IMAGE_SIZE
from .augmentations import get_transforms

class DrowsinessSequenceDataset(Dataset):
    def __init__(self, root_dir, split="train", seq_len=SEQ_LENGTH, stride=STRIDE, transform=None):
        """
        root_dir: Đường dẫn tới dataset
        split: 'train', 'dev', hoặc 'test'
        """
        self.split_dir = os.path.join(root_dir, split)
        self.seq_len = seq_len
        self.transform = transform
        
        self.classes = {"0_alert": 0, "5_low_vigilant": 1, "10_drowsy": 2}
        self.sequences = []
        self.labels = []
        
        self._build_sequences(stride)
        
    def _build_sequences(self, stride):
        # Duyệt qua từng class
        for class_name, label in self.classes.items():
            class_path = os.path.join(self.split_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            # Lấy toàn bộ ảnh và sắp xếp theo thứ tự (RẤT QUAN TRỌNG)
            # Giả định ảnh của bạn có tên tuần tự như frame_001.jpg, frame_002.jpg
            image_paths = sorted(glob.glob(os.path.join(class_path, "*.jpg")))
            
            # Trượt cửa sổ gom frame
            for i in range(0, len(image_paths) - self.seq_len + 1, stride):
                seq = image_paths[i : i + self.seq_len]
                self.sequences.append(seq)
                self.labels.append(label)
                
        print(f"[{self.split_dir}] Đã tạo {len(self.sequences)} chuỗi ({self.seq_len} frames/chuỗi).")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_paths = self.sequences[idx]
        label = self.labels[idx]
        
        frames = []
        for path in seq_paths:
            # Đọc ảnh thành tensor [C, H, W]
            img = read_image(path)
            frames.append(img)
            
        # Stack lại thành tensor [T, C, H, W]
        video_tensor = torch.stack(frames) 
        
        if self.transform:
            video_tensor = self.transform(video_tensor)
            
        return video_tensor, label

# Hàm hỗ trợ tạo DataLoader chuẩn
def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    train_dataset = DrowsinessSequenceDataset(data_dir, split="train", transform=get_transforms(is_train=True))
    dev_dataset = DrowsinessSequenceDataset(data_dir, split="dev", transform=get_transforms(is_train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, dev_loader