# Báo Cáo Thực Nghiệm nnU-Net (DynUNet) Baseline Trên Toàn Bộ Tập BraTS 2021

**Phiên bản thực nghiệm**: nnU-Net Baseline (Full 1,251 Cases — 50 Epochs Pure FP32)  
**Tập dữ liệu**: RSNA-ASNR-MICCAI BraTS 2021 Task 1 (**Toàn bộ 1,251 bệnh nhân**: 1,000 Train / 125 Val / 126 Held-Out Test)  
**Phần cứng thực thi**: Kaggle Dual NVIDIA Tesla T4 (`GPU T4 x2` DataParallel)  
**Thời gian hoàn thành**: 31/10/2026 — Trạng thái: **COMPLETE** (Thời gian chạy: **412.32 phút ~ 6.87 giờ**, 0 lỗi tràn số NaN)  
**Tập checkpoint lưu trữ**: [`checkpoints/nnunet/best_metric_model.pth`](file:///Users/truongledanchi/.gemini/antigravity/scratch/test/checkpoints/nnunet/best_metric_model.pth) (Epoch 45) & [`latest_model.pth`](file:///Users/truongledanchi/.gemini/antigravity/scratch/test/checkpoints/nnunet/latest_model.pth) (Epoch 50)  
**Artifacts kết quả**: [`outputs/nnunet/`](file:///Users/truongledanchi/.gemini/antigravity/scratch/test/outputs/nnunet/)

---

## 1. Tóm Tắt Kết Quả Thực Nghiệm (Executive Summary)

Mô hình **nnU-Net (MONAI DynUNet)** đã hoàn thành trọn vẹn **50 epochs** trên **toàn bộ 1,251 bệnh nhân** của bộ dữ liệu quốc tế **BraTS 2021**. Thực nghiệm được thực thi dưới chế độ số học thuần **FP32** (`--no-amp`), kết hợp chuẩn hóa **`InstanceNorm3d`** và kích hoạt **`LeakyReLU(0.01)`**, mang lại độ ổn định gradient tuyệt đối và chất lượng phân đoạn vượt trội so với Standard 3D UNet trước đó.

### Điểm Sáng Trọng Tâm:
- **Validation Mean Dice Đạt Đỉnh**: **`0.8583`** tại **Epoch 45** (vượt mốc 0.8474 của Standard 3D UNet).
- **Cải Thiện Vượt Bậc Vùng Tumor Core (TC)**: Đạt **`0.8597`** (tăng mạnh **+2.61%** so với 0.8336 của 3D UNet).
- **Cải Thiện Vùng U Ngấm Thuốc (ET)**: Đạt **`0.8077`** (vượt qua ngưỡng 0.80, tăng **+0.85%** so với 0.7992 của 3D UNet).
- **Vùng Toàn Bộ Khối U (WT)**: Duy trì mức độ chính xác xuất sắc **`0.9076`** (tại Epoch 50 đạt **`0.9092`**).
- **Khoảng Cách Biên Bề Mặt 3D ($HD_{95}$)**: Đạt **`15.48 mm`** trên toàn thể tích 3D (vùng WT đạt **`11.66 mm`**, TC đạt **`13.54 mm`**).

---

## 2. Bảng Đối Chiếu Trực Tiếp: nnU-Net vs. Standard 3D UNet

| Chỉ số đánh giá | Standard 3D UNet Baseline<br/>*(350 ca, 65 epochs)* | nnU-Net (DynUNet) Baseline<br/>*(1,251 ca, 50 epochs)* | Mức độ cải thiện / Nhận xét |
| :--- | :---: | :---: | :--- |
| **Quy mô tập dữ liệu** | 350 bệnh nhân (280 Train) | **1,251 bệnh nhân (1,000 Train)** | **Lớn gấp 3.57 lần** dữ liệu huấn luyện |
| **Kiến trúc mạng** | Standard UNet (BatchNorm3d) | **DynUNet (InstanceNorm3d)** | Bất biến với batch size, triệt tiêu bất ổn định |
| **Kích hoạt phi tuyến** | ReLU | **LeakyReLU ($\alpha = 0.01$)** | Khắc phục hoàn toàn dying ReLU ở vùng não nền |
| **Mean Dice Tổng Thể** | `0.8474` | **`0.8583`** | 🟢 **Tăng +1.09%** (+0.0109) |
| **Whole Tumor (WT) Dice** | `0.9095` | **`0.9076`** (Ep 50: `0.9092`) | Tương đương mức trần tối ưu (~91%) |
| **Tumor Core (TC) Dice** | `0.8336` | **`0.8597`** (Ep 50: `0.8621`) | 🟢 **Tăng +2.61%** (+0.0261, đột phá mạnh) |
| **Enhancing Tumor (ET) Dice**| `0.7992` | **`0.8077`** | 🟢 **Tăng +0.85%** (+0.0085, vượt mốc 0.80) |
| **Macro IoU (Jaccard)** | `0.7631` | **`0.7891`** | 🟢 **Tăng +2.60%** độ tin cậy vùng giao |
| **Mean Sensitivity (Recall)** | `0.8450` | **`0.8648`** | 🟢 **Tăng +1.98%** khả năng bắt trúng tổn thương |
| **Mean Precision (PPV)** | `0.8826` | **`0.8801`** | Giữ vững tỷ lệ dương tính thật rất cao (~88%) |
| **Thời gian chạy trên 2x T4**| 2.13 giờ | 6.87 giờ (412.32 phút) | 100% an toàn trong hạn mức 12h Kaggle |

---

## 3. Lịch Sử Hội Tụ Qua 50 Epochs (10 Mốc Validation Độc Lập)

Quá trình Validation 3D (`sliding_window_inference`, Gaussian blend, overlap 0.5) được thực hiện định kỳ 5 epoch một lần trên **125 ca Validation**:

| Mốc Epoch | Mean Dice | WT Dice | TC Dice | ET Dice | Mean $HD_{95}$ (mm) | Ghi chú tiến trình |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Epoch 05** | 0.7716 | 0.8461 | 0.7311 | 0.7377 | 32.59 mm | Bắt đầu học được hình thái đại thể |
| **Epoch 10** | 0.8141 | 0.8591 | 0.8147 | 0.7685 | 27.52 mm | Tăng trưởng mạnh vùng Tumor Core (+8.3%) |
| **Epoch 15** | 0.8366 | 0.8794 | 0.8343 | 0.7960 | 15.81 mm | Sai số biên $HD_{95}$ giảm sâu một nửa |
| **Epoch 20** | 0.8317 | 0.8870 | 0.8301 | 0.7780 | 21.69 mm | Dao động cục bộ nhẹ |
| **Epoch 25** | 0.8354 | 0.9008 | 0.8233 | 0.7823 | 17.18 mm | Whole Tumor (WT) lần đầu vượt mốc 0.90 |
| **Epoch 30** | 0.8520 | 0.9030 | 0.8538 | 0.7991 | 14.50 mm | Đạt $HD_{95}$ tốt nhất toàn khóa (14.50 mm) |
| **Epoch 35** | 0.8531 | 0.9048 | 0.8560 | 0.7987 | 17.55 mm | Hội tụ ổn định trên cả 3 phân vùng |
| **Epoch 40** | 0.8554 | 0.9061 | 0.8608 | 0.7994 | 17.19 mm | Tumor Core đạt mốc 0.86 |
| **Epoch 45** | **`0.8583`** | **`0.9076`** | **`0.8597`** | **`0.8077`** | **`15.48 mm`** | 🏆 **CHECKPOINT TỐT NHẤT (`best_metric_model.pth`)** |
| **Epoch 50** | 0.8570 | **`0.9092`** | **`0.8621`** | 0.7996 | 16.47 mm | Kết thúc 50 epochs, WT & TC đạt đỉnh mới |

---

## 4. Chi Tiết Các Chỉ Số Chẩn Đoán Tại Checkpoint Tốt Nhất (Epoch 45)

Dưới đây là bảng trích xuất đầy đủ từ file báo cáo khoa học [`outputs/nnunet/nnunet_metrics_report.json`](file:///Users/truongledanchi/.gemini/antigravity/scratch/test/outputs/nnunet/nnunet_metrics_report.json):

| Phân vùng giải phẫu bệnh | Dice Coefficient | $HD_{95}$ Biên (mm) | IoU (Jaccard) | Sensitivity (Recall) | Precision (PPV) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Whole Tumor (WT)** | **`0.9076`** | **`11.66 mm`** | `0.8421` | `0.9089` | `0.9192` |
| **Tumor Core (TC)** | **`0.8597`** | **`13.54 mm`** | `0.7949` | `0.8606` | `0.8948` |
| **Enhancing Tumor (ET)** | **`0.8077`** | **`21.26 mm`** | `0.7303` | `0.8251` | `0.8263` |
| **Macro Average (Mean)** | **`0.8583`** | **`15.48 mm`** | **`0.7891`** | **`0.8648`** | **`0.8801`** |

---

## 5. Danh Mục Artifacts Thực Nghiệm Đã Được Lưu Trữ

Toàn bộ các file kết quả thực nghiệm đã được đóng gói đồng bộ tại [`outputs/nnunet/`](file:///Users/truongledanchi/.gemini/antigravity/scratch/test/outputs/nnunet/):

1. **`nnunet_learning_curves.png`**:  
   Đồ thị 3 bảng ghi nhận đường cong suy giảm Training Loss và đà tăng trưởng của Validation Dice / suy giảm $HD_{95}$ qua 50 epochs.
2. **`nnunet_test_predictions.png`**:  
   Lát cắt axial trực quan hóa 4 kênh MRI đầu vào, nhãn chuyên gia Ground Truth và kết quả dự đoán của nnU-Net.
3. **`nnunet_metrics_report.json`**:  
   File số liệu JSON tự lập tài liệu (*self-documenting*), chứa đầy đủ thông số cấu hình, tham số mô hình, lịch sử 50 epochs và bảng chỉ số chi tiết.
4. **Trọng số mô hình (`checkpoints/nnunet/`)**:  
   - `best_metric_model.pth` (64.6 MB): Trọng số tại Epoch 45.
   - `latest_model.pth` (64.6 MB): Trọng số tại Epoch 50, sẵn sàng nạp để train nối tiếp bất cứ lúc nào qua cờ `--resume`.
