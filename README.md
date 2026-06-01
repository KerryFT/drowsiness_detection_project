# Lightweight Driver Drowsiness Detection using TSM & CBAM

Dự án phát hiện tài xế buồn ngủ và xao nhãng theo thời gian thực (Real-time), được thiết kế tối ưu cho các thiết bị Edge AI (CPU-only).

## 🚀 Điểm nổi bật (Key Features)
* **Kiến trúc đề xuất:** MobileNetV3-Small kết hợp **CBAM** (Spatial/Channel Attention) và **TSM** (Temporal Shift Module).
* **Siêu nhẹ (Lightweight):** Chỉ ~1.53 Triệu tham số (Parameters), không làm tăng chi phí tính toán (Zero-FLOPs addition).
* **Nhận thức thời gian (Temporal Awareness):** Khắc phục triệt để lỗi "Báo động giả" (False Positive) khi tài xế chớp mắt sinh lý nhờ cơ chế vay mượn chuỗi thời gian của TSM.

## 📊 Đánh giá mô hình (Ablation Study)
| Model | Params | Chạy Real-time | Test Macro-F1 | Hiện tượng Báo động giả |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (MobileNetV3 2D) | 1.50 M | Có | 98.22% | Rất cao (Khi chớp mắt) |
| **Proposed (TSM + CBAM)** | **1.53 M** | **Có** | **88.82%** | **Được khắc phục** |

## 🛠 Hướng dẫn Cài đặt & Chạy Demo

### 1. Cài đặt môi trường
Yêu cầu Python >= 3.8
```bash
git clone https://github.com/KerryFT/drowsiness_detection_project.git
cd drowsiness_detection_project
pip install -r requirements.txt

```

### 2. Tải Trọng số (Weights)

Tải 2 file `baseline_best.pt` và `proposed_best.pt` từ mục Releases và đặt vào thư mục `weights/`.

### 3. Chạy Inference bằng Webcam (Local)

Để chạy mô hình đề xuất (TSM + CBAM - Chống báo động giả):
```bash
python run_inference.py
```
Để chạy mô hình Baseline (Mạng tĩnh 2D - Dùng để so sánh):
```bash
python run_baseline.py
```

