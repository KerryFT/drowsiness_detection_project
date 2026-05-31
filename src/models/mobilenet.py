# src/models/mobilenet.py

import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Đổi tham số num_classes=3 thành num_classes=2
def build_baseline_model(num_classes=2):
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    import torch.nn as nn
    
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    
    in_features = model.classifier[3].in_features
    # Lớp phân loại cuối cùng giờ sẽ trả về 2 xác suất (Active và Fatigue)
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model