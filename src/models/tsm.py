# src/models/tsm.py

import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    def __init__(self, n_segment=8, n_div=8):
        """
        n_segment: Số lượng frame trong 1 chuỗi (T=8)
        n_div: Tỷ lệ dịch chuyển (8 nghĩa là dịch chuyển 1/8 số channel)
        """
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x):
        # Đầu vào x từ MobileNetV3 đang có dạng gộp: [B*T, C, H, W]
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        
        # Tách lại thành [B, T, C, H, W] để có trục thời gian mà dịch chuyển
        x = x.view(n_batch, self.n_segment, c, h, w)

        fold = c // self.fold_div
        out = torch.zeros_like(x)
        
        # Dịch chuyển sang trái (Vay mượn từ tương lai)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        
        # Dịch chuyển sang phải (Vay mượn từ quá khứ)
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]
        
        # Giữ nguyên phần còn lại của các channels
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        # Gộp lại thành [B*T, C, H, W] để trả về cho các lớp Conv2D xử lý tiếp
        return out.view(nt, c, h, w)