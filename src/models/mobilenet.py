# src/models/mobilenet.py

import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def build_baseline_model(num_classes=3):
    # Load trọng số Pretrained cực nhẹ của PyTorch
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    
    # Can thiệp vào lớp Linear cuối cùng
    # Lấy số lượng input features của lớp cuối
    in_features = model.classifier[3].in_features
    
    # Thay thế bằng lớp mới với 3 đầu ra
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model