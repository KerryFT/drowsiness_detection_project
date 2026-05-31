# test_dataset.py
import sys

# Cấu hình hiển thị tiếng Việt trên terminal Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from src.dataset import get_dataloaders
from src.config import DATA_DIR

def run_test():
    print("Khởi tạo DataLoader...")
    train_loader, dev_loader = get_dataloaders(DATA_DIR, batch_size=2, num_workers=0)
    
    # Rút ra 1 batch để kiểm tra
    videos, labels = next(iter(train_loader))
    
    print("\n--- BÁO CÁO NGHIỆM THU ---")
    print(f"1. Kích thước (Shape) của Input: {videos.shape}")
    print(f"   (Kỳ vọng: [2, 8, 3, 224, 224] -> [Batch, T, C, H, W])")
    print(f"2. Nhãn (Labels): {labels}")
    print(f"3. Kiểu dữ liệu Input: {videos.dtype}")
    print(f"4. Giá trị Max/Min của Tensor: {videos.max().item():.2f} / {videos.min().item():.2f} (Kỳ vọng đã Normalize qua số âm/dương)")

if __name__ == "__main__":
    # Để test không bị lỗi, hãy tạo vài folder dummy có vài ảnh .jpg trước khi chạy
    run_test()