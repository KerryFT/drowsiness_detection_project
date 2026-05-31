# src/config.py

import os

# Cấu hình Dataset
DATA_DIR = "./dataset"  # Đường dẫn tới thư mục chứa train/dev/test
SEQ_LENGTH = 8          # Số frames trong 1 chuỗi (T=8)
STRIDE = 4              # Bước nhảy khi gom frame (Overlap 50% để tăng lượng data)

# Cấu hình Ảnh
IMAGE_SIZE = 224
CHANNELS = 3

# Cấu hình Training
BATCH_SIZE = 16         # Hạ xuống nếu Colab báo hết VRAM
NUM_WORKERS = 4