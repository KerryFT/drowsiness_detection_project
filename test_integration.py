# test_integration.py

import torch
from src.models.mobilenet import build_proposed_model

def run_test():
    print("Khởi tạo WACV/IoTAI Proposed Architecture...")
    # Khởi tạo mô hình với 2 class, T=8
    model = build_proposed_model(num_classes=2, n_segment=8)
    
    # Giả lập input từ DataLoader: Batch = 2, T = 8, Channels = 3, HxW = 224x224
    B, T, C, H, W = 2, 8, 3, 224, 224
    dummy_input = torch.randn(B, T, C, H, W)
    
    print(f"Kích thước Input gốc: {dummy_input.shape}")
    
    # Gộp B và T lại theo đúng luồng của file engine.py
    inputs_2d = dummy_input.view(B * T, C, H, W)
    
    # Chạy qua mạng nơ-ron
    outputs_2d = model(inputs_2d)
    
    # Tách lại thành [B, T, Classes] và lấy trung bình
    outputs_seq = outputs_2d.view(B, T, -1).mean(dim=1)
    
    print(f"Kích thước Output cuối cùng: {outputs_seq.shape}")
    
    if outputs_seq.shape == (B, 2):
        print("✅ THÀNH CÔNG: Dòng chảy dữ liệu qua TSM và CBAM hoàn hảo, không suy xuyển!")
        
        # In tổng số Parameters để kiểm chứng tiêu chí Lightweight
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📦 Tổng tham số (Parameters): {total_params / 1e6:.2f} Triệu")
    else:
        print("❌ THẤT BẠI: Kích thước ma trận bị vỡ!")

if __name__ == "__main__":
    run_test()