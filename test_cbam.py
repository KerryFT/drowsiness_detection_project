# test_cbam.py
import torch
from src.models.cbam import CBAMBlock

def run_test():
    # Giả lập một Tensor đi ra từ một lớp Conv của MobileNetV3
    # Shape: [Batch_size, Channels, Height, Width]
    dummy_input = torch.randn(8, 64, 56, 56) 
    
    # Khởi tạo CBAM cho 64 channels
    cbam = CBAMBlock(in_planes=64)
    
    # Cho tensor chạy qua CBAM
    output = cbam(dummy_input)
    
    print("Shape đầu vào:", dummy_input.shape)
    print("Shape đầu ra: ", output.shape)
    
    if dummy_input.shape == output.shape:
        print("✅ THÀNH CÔNG: Module CBAM không làm biến dạng kích thước Tensor!")
    else:
        print("❌ THẤT BẠI: Kích thước Tensor bị thay đổi!")

if __name__ == "__main__":
    run_test()