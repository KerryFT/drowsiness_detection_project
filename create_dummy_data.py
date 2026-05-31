import os
import numpy as np
from PIL import Image

def create_dummy_dataset():
    base_dir = "./dataset"
    splits = ["train", "dev"]
    classes = ["0_alert", "5_low_vigilant", "10_drowsy"]
    
    for split in splits:
        for cls in classes:
            path = os.path.join(base_dir, split, cls)
            os.makedirs(path, exist_ok=True)
            
            # Tạo 10 ảnh dummy mỗi thư mục để đủ SEQ_LENGTH=8
            for i in range(10):
                # Tạo ảnh ngẫu nhiên 224x224 RGB
                img_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_data)
                img.save(os.path.join(path, f"frame_{i:03d}.jpg"))
    print("Da tao bo du lieu dummy thanh cong tai ./dataset!")

if __name__ == "__main__":
    create_dummy_dataset()
