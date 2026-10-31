# Báo Cáo Thực Nghiệm Standard 3D UNet Baseline trên BraTS 2021 (Full Dataset)
**Phiên bản thực nghiệm**: Version 4 (Full 1,251 Patients, 65 Epochs Pure FP32)  
**Tập dữ liệu**: RSNA-ASNR-MICCAI BraTS 2021 Task 1 (Toàn bộ 1,251 bệnh nhân: 1,000 Train / 125 Val / 126 Held-Out Test)  
**Phần cứng thực thi**: Kaggle Dual NVIDIA Tesla T4 (`GPU T4 x2` DataParallel)  
**Thời gian hoàn thành**: 31/10/2026 — Trạng thái: **COMPLETE** (Thời gian chạy: 8.76 giờ, 0 lỗi tràn số NaN)

---

## 1. Tóm Tắt Kết Quả Cốt Lõi (Executive Summary)

Thực nghiệm baseline độc lập cho kiến trúc **Standard 3D UNet** trên toàn bộ tập dữ liệu ung thư não **BraTS 2021 (1,251 bệnh nhân)** đã được chạy trọn vẹn **65 epochs** dưới cơ chế số học thuần **FP32** (`amp=False`), giải quyết triệt để vấn đề tràn số `NaN` ở các tầng `BatchNorm3d` từng gặp ở phiên bản FP16 AMP.

Mô hình đạt điểm Validation tốt nhất tại **Epoch 65** với **Validation Mean Dice = 0.8562** và **Validation Mean $HD_{95}$ = 13.00 mm**. Trên tập kiểm thử độc lập hoàn toàn chưa từng thấy (**Held-Out Test Set** gồm 126 bệnh nhân, chia độc lập cấp bệnh nhân):

- **Mean Dice**: **0.8767 ± 0.1403** (Trung vị / Median: **0.9324**, Cao nhất: **0.9733**)
- **Mean $HD_{95}$**: **13.58 ± 34.17 mm** (Trung vị / Median: **3.02 mm**, Thấp nhất: **1.00 mm**)
- **Whole Tumor (WT)**: Dice = **0.9161 ± 0.0814** (Median: **0.9427**) | $HD_{95}$ = **9.17 mm** (Median: **3.08 mm**)
- **Tumor Core (TC)**: Dice = **0.8755 ± 0.1993** (Median: **0.9526**) | $HD_{95}$ = **14.04 mm** (Median: **2.12 mm**)
- **Enhancing Tumor (ET)**: Dice = **0.8385 ± 0.2115** (Median: **0.9086**) | $HD_{95}$ = **17.52 mm** (Median: **1.41 mm**)
- **Macro IoU (Jaccard)**: **0.8110** | **Sensitivity**: **0.8861** | **Precision**: **0.8893**

So với phiên bản huấn luyện trên mẫu 350 ca trước đó, việc mở rộng lên toàn bộ 1,251 ca giúp **Mean Dice tăng thêm +2.93%**, điểm trung vị **Median Dice tăng từ 0.8886 lên 0.9324**, và tỷ lệ ca đạt Dice $\ge 0.90$ tăng vọt từ 45.7% lên **68.3% (86/126 bệnh nhân)**.

---

## 2. Thiết Lập Hệ Thống & Tham Số Huấn Luyện (Experimental Setup)

### 2.1. Phân chia tập dữ liệu (Patient-Level Split)
Phân chia nghiêm ngặt ở cấp độ mã bệnh nhân (`BraTS2021_xxxxx`) theo tỷ lệ chuẩn 80/10/10:

| Phân vùng | Số ca bệnh nhân | Tỷ lệ (%) | Mục đích sử dụng |
|---|:---:|:---:|---|
| **Training Set** | **1,000** | 79.9% | Tối ưu hóa trọng số mô hình qua gradient descent (250 batches/epoch) |
| **Validation Set** | **125** | 10.0% | Đánh giá định kỳ mỗi 13 epochs, chọn checkpoint tốt nhất |
| **Held-Out Test Set** | **126** | 10.1% | Đánh giá độc lập 1 lần duy nhất bằng checkpoint tốt nhất |
| **Tổng cộng** | **1,251** | 100.0% | Trọn vẹn toàn bộ tập dữ liệu thi đấu quốc tế BraTS 2021 |

### 2.2. Kiến trúc mô hình & Siêu tham số
- **Kiến trúc**: Standard 3D Residual UNet (MONAI `UNet`)
  - Spatial dimensions: 3 (khối thể tích không gian 3 chiều)
  - Số kênh đầu vào: 4 modalities đồng đăng ký (FLAIR, T1, T1ce, T2)
  - Số kênh đầu ra: 3 nhãn phân đoạn đa nhãn độc lập (TC, WT, ET)
  - Cấu trúc kênh đặc trưng: `(16, 32, 64, 128, 256)` với 4 tầng downsampling (strides 2)
  - Residual units: 2 units mỗi tầng; Chuẩn hóa: `BatchNorm3d`; Dropout: 0.1
  - Tổng số tham số có thể huấn luyện: **4,812,512 tham số** (Kích thước mô hình: **18.36 MB**)
- **Hàm mất mát (Loss Function)**: `SoftDiceBCELoss`
  $$\mathcal{L} = 0.5 \times \mathcal{L}_{\text{SoftDice}} + 0.5 \times \mathcal{L}_{\text{BCE}}$$
- **Tối ưu hóa & Lịch trình học**:
  - AdamW ($lr = 1.0 \times 10^{-3}$, weight decay = $1.0 \times 10^{-5}$)
  - `CosineAnnealingLR` ($T_{\max} = 65$, $\eta_{\min} = 1.0 \times 10^{-5}$)
  - Gradient clipping: `clip_grad_norm_(max_norm=1.0)`
- **Kích thước patch & Inference Protocol**:
  - ROI Size: $96 \times 96 \times 96$ voxels
  - Batch size huấn luyện: 4 (chia đều 2 GPU T4 qua `torch.nn.DataParallel`, sw_batch_size = 4)
  - Inference Protocol: `sliding_window_inference` kích thước $96^3$, overlap = 0.5, hòa trộn Gaussian (`mode='gaussian'`).

---

## 3. Động Lực Huấn Luyện (Training Dynamics)

Huấn luyện chạy mượt mà từ Epoch 1 đến Epoch 65 với **0 lần xảy ra lỗi NaN**, tốc độ trung bình đạt **~464 giây/epoch** trên 2 GPU T4 (250 batches/epoch).

### Tiến trình điểm Validation Dice qua 5 mốc:
- **Epoch 13**: Mean Dice = **0.8156** (TC: 0.7904, WT: 0.8872, ET: 0.7691) | Mean $HD_{95}$ = 20.01 mm -> Saved Best Model
- **Epoch 26**: Mean Dice = **0.8375** (TC: 0.8229, WT: 0.8981, ET: 0.7915) | Mean $HD_{95}$ = 21.95 mm -> Saved Best Model
- **Epoch 39**: Mean Dice = **0.8423** (TC: 0.8312, WT: 0.8998, ET: 0.7959) | Mean $HD_{95}$ = 15.80 mm -> Saved Best Model
- **Epoch 52**: Mean Dice = **0.8550** (TC: 0.8492, WT: 0.9068, ET: 0.8090) | Mean $HD_{95}$ = 12.90 mm -> Saved Best Model
- **Epoch 65**: Mean Dice = **0.8562** (TC: 0.8501, WT: 0.9079, ET: 0.8106) | Mean $HD_{95}$ = **13.00 mm** -> Saved Best Model (Final)

Checkpoint tại **Epoch 65** đạt chỉ số cao nhất và được tự động nạp để đánh giá tập Test.

---

## 4. Kết Quả Chi Tiết Trên Tập Held-Out Test (126 Bệnh Nhân)

### 4.1. Bảng chỉ số tổng hợp theo từng vùng u
| Vùng giải phẫu bệnh | Dice Score | $HD_{95}$ (mm) | IoU (Jaccard) | Sensitivity (Recall) | Precision (PPV) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Whole Tumor (WT)** | **0.9161** | **9.17** | **0.8533** | **0.9363** | **0.9032** |
| **Tumor Core (TC)** | **0.8755** | **14.04** | **0.8176** | **0.8770** | **0.9045** |
| **Enhancing Tumor (ET)** | **0.8385** | **17.52** | **0.7619** | **0.8449** | **0.8602** |
| **Macro Average (Mean)** | **0.8767** | **13.58** | **0.8110** | **0.8861** | **0.8893** |

### 4.2. Phân bố thống kê trên 126 ca bệnh nhân (Mean, Std, Median, Min, Max)
| Chỉ số | Trung bình (Mean) | Độ lệch chuẩn (Std) | Trung vị (Median) | Giá trị nhỏ nhất (Min) | Giá trị lớn nhất (Max) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Dice Mean** | **0.8767** | 0.1403 | **0.9324** | 0.2463 | 0.9733 |
| **Dice WT** | **0.9161** | 0.0814 | **0.9427** | 0.2948 | 0.9802 |
| **Dice TC** | **0.8755** | 0.1993 | **0.9526** | 0.0000 | 0.9878 |
| **Dice ET** | **0.8385** | 0.2115 | **0.9086** | 0.0000 | 1.0000 |
| **$HD_{95}$ Mean (mm)** | **13.58** | 34.17 | **3.02** | 1.00 | 251.90 |
| **$HD_{95}$ WT (mm)** | **9.17** | 15.42 | **3.08** | 1.00 | 79.45 |
| **$HD_{95}$ TC (mm)** | **14.04** | 48.73 | **2.12** | 1.00 | 373.13 |
| **$HD_{95}$ ET (mm)** | **17.52** | 66.17 | **1.41** | 0.00 | 373.13 |

> [!NOTE]
> **Nhận xét quan trọng về chất lượng tổng thể**:
> - Điểm **Median Dice đạt 0.9324**, vượt trội so với Mean Dice (0.8767).
> - Sai số biên trung vị **Median $HD_{95}$ chỉ là 3.02 mm** (đặc biệt vùng ET chỉ lệch **1.41 mm**).
> - Có tới **86/126 ca bệnh nhân (68.3%)** đạt Dice Mean $\ge 0.90$, và **102/126 ca (81.0%)** đạt Dice Mean $\ge 0.85$.
> - Điểm số bị kéo nhẹ bởi ~10% ca dị biệt (u dạng nang rỗng mỏng hoặc u không bắt thuốc ngấm thuốc).

---

## 5. Artifacts & File Kết Quả Đính Kèm

Tất cả kết quả thô, biểu đồ và trọng số của thực nghiệm Version 4 được đồng bộ trực tiếp vào repository:

- **Báo cáo chỉ số đầy đủ (JSON)**: [`outputs/unet_baseline/baseline_metrics_report.json`](outputs/unet_baseline/baseline_metrics_report.json) (bảng tra cứu chi tiết từng voxel, từng metric của toàn bộ 126 bệnh nhân test).
- **Đồ thị học tập (3 panels PNG)**: [`outputs/unet_baseline/baseline_learning_curves.png`](outputs/unet_baseline/baseline_learning_curves.png) (Train Loss, Validation Dice, Validation $HD_{95}$).
- **Ảnh dự đoán phân đoạn thực tế (PNG)**: [`outputs/unet_baseline/baseline_test_predictions.png`](outputs/unet_baseline/baseline_test_predictions.png) (Đối chiếu 4 modalities, Ground Truth và Prediction trên 3 mặt cắt Axial, Coronal, Sagittal).
- **Notebook Kaggle tái lập**: [`kaggle_unet_baseline_full.ipynb`](kaggle_unet_baseline_full.ipynb).
- **Script dòng lệnh độc lập**: [`scripts/run_unet_baseline.py`](scripts/run_unet_baseline.py).
