# src/models/mobilenet.py

import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from .cbam import CBAMBlock
from .tsm import TemporalShift
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn as nn

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

# Cập nhật vào src/models/mobilenet.py

def build_proposed_model(num_classes=2, n_segment=8):
    # 1. Khởi tạo mô hình gốc
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    
    # 2. Xuyên vào phần lõi (features) để tích hợp TSM và CBAM
    for i, layer in enumerate(model.features):
        # Chỉ can thiệp vào các khối InvertedResidual (nơi trích xuất đặc trưng chính)
        if type(layer).__name__ == 'InvertedResidual':
            
            # Lấy số lượng channel đầu ra của khối này để cấu hình CBAM
            out_channels = layer.out_channels
            
            # Tạo một khối Sequential mới: [TSM -> InvertedResidual gốc -> CBAM]
            new_block = nn.Sequential(
                TemporalShift(n_segment=n_segment, n_div=8),
                layer,
                CBAMBlock(in_planes=out_channels, ratio=16, kernel_size=7)
            )
            
            # Thay thế khối gốc bằng khối đã độ chế
            model.features[i] = new_block

    # 3. Can thiệp lớp Classifier cuối cùng cho bài toán của chúng ta
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model