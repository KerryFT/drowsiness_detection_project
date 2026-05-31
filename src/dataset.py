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
        
        # Cập nhật từ 3 class xuống 2 class theo thực tế của dataset
        self.classes = {"active": 0, "fatigue": 1}
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
# Cập nhật hàm get_dataloaders trong src/dataset.py

def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    # Khởi tạo 3 Dataset
    train_dataset = DrowsinessSequenceDataset(data_dir, split="train", transform=get_transforms(is_train=True))
    val_dataset = DrowsinessSequenceDataset(data_dir, split="val", transform=get_transforms(is_train=False))
    test_dataset = DrowsinessSequenceDataset(data_dir, split="test", transform=get_transforms(is_train=False))
    
    # Khởi tạo 3 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # Tập Val và Test không cần trộn (shuffle=False) để tiết kiệm tài nguyên
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader