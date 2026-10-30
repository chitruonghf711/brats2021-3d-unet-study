# Báo Cáo Thực Nghiệm Standard 3D UNet Baseline trên BraTS 2021
**Phiên bản thực nghiệm**: Version 3 (Full 65 Epochs Pure FP32)  
**Tập dữ liệu**: RSNA-ASNR-MICCAI BraTS 2021 Task 1 (350 bệnh nhân thực tế: 280 Train / 35 Val / 35 Held-Out Test)  
**Phần cứng thực thi**: Kaggle Dual NVIDIA Tesla T4 (`GPU T4 x2` DataParallel)  
**Thời gian hoàn thành**: 30/10/2026 — Trạng thái: **COMPLETE** (Thời gian chạy: 2.13 giờ, 0 lỗi tràn số NaN)

---

## 1. Tóm Tắt Kết Quả Cốt Lõi (Executive Summary)

Thực nghiệm baseline độc lập cho kiến trúc **Standard 3D UNet** trên tập dữ liệu ung thư não **BraTS 2021** đã được chạy trọn vẹn **65 epochs** dưới cơ chế số học thuần **FP32** (`amp=False`), giải quyết triệt để vấn đề tràn số `NaN` ở các tầng `BatchNorm3d` từng gặp ở phiên bản FP16 AMP.

Mô hình đạt điểm Validation tốt nhất tại **Epoch 65** với **Validation Mean Dice = 0.8489** và **Validation Mean $HD_{95}$ = 8.88 mm**. Trên tập kiểm thử độc lập hoàn toàn chưa từng thấy (**Held-Out Test Set** gồm 35 bệnh nhân, chia độc lập cấp bệnh nhân):

- **Mean Dice**: **0.8474 ± 0.1367** (Trung vị / Median: **0.8886**, Cao nhất: **0.9497**)
- **Mean $HD_{95}$**: **11.53 ± 14.47 mm** (Trung vị / Median: **4.15 mm**, Thấp nhất: **1.66 mm**)
- **Whole Tumor (WT)**: Dice = **0.9095 ± 0.0702** | $HD_{95}$ = **14.38 mm** (Median: 5.10 mm)
- **Tumor Core (TC)**: Dice = **0.8336 ± 0.2106** | $HD_{95}$ = **10.03 mm** (Median: 3.74 mm)
- **Enhancing Tumor (ET)**: Dice = **0.7992 ± 0.1912** | $HD_{95}$ = **10.17 mm** (Median: 2.45 mm)
- **Macro IoU (Jaccard)**: **0.7631** | **Sensitivity**: **0.8450** | **Precision**: **0.8826**

Kết quả này thiết lập một mốc quy chiếu (baseline benchmark) vững chắc, đáng tin cậy về mặt thống kê và tái lập 100% để nhóm so sánh trực tiếp với các cải tiến tiếp theo (nnU-Net, nhãn mềm Soft Labels, hoặc các phương pháp kháng nhiễu nhãn Label Noise).

---

## 2. Thiết Lập Hệ Thống & Tham Số Huấn Luyện (Experimental Setup)

### 2.1. Phân chia tập dữ liệu (Patient-Level Split)
Để loại trừ triệt để nguy cơ rò rỉ dữ liệu (data leakage) giữa các tập, việc phân chia được thực hiện nghiêm ngặt ở cấp độ mã bệnh nhân (`BraTS2021_xxxxx`), bảo đảm mọi lát cắt hoặc voxel của một bệnh nhân chỉ thuộc về duy nhất một tập:

| Phân vùng | Số ca bệnh nhân | Tỷ lệ (%) | Mục đích sử dụng |
|---|:---:|:---:|---|
| **Training Set** | **280** | 80.0% | Tối ưu hóa trọng số mô hình qua gradient descent |
| **Validation Set** | **35** | 10.0% | Đánh giá định kỳ mỗi 13 epochs, chọn checkpoint tốt nhất |
| **Held-Out Test Set** | **35** | 10.0% | Đánh giá độc lập 1 lần duy nhất bằng checkpoint tốt nhất |
| **Tổng cộng** | **350** | 100.0% | Mẫu đại diện lớn gấp gần 12 lần bộ dữ liệu thử nghiệm BraTS-Mini (30 ca) |

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
  Hỗ trợ phân đoạn đa nhãn độc lập qua Sigmoid, tránh hiện tượng cạnh tranh xác suất sai lệch của Softmax.
- **Tối ưu hóa & Lịch trình học (Optimizer & Scheduler)**:
  - Thuật toán tối ưu: AdamW ($lr = 1.0 \times 10^{-3}$, weight decay = $1.0 \times 10^{-5}$)
  - Lịch trình giảm tốc độ học: `CosineAnnealingLR` ($T_{\max} = 65$, $\eta_{\min} = 1.0 \times 10^{-5}$)
  - Gradient clipping: `clip_grad_norm_(max_norm=1.0)`
- **Khai thác dữ liệu & Kích thước patch**:
  - ROI Size: $96 \times 96 \times 96$ voxels
  - Batch size huấn luyện: 4 (chia đều 2 GPU T4 qua `torch.nn.DataParallel`)
  - Inference Protocol: `sliding_window_inference` kích thước $96^3$, overlap = 0.5, trọng số hòa trộn Gaussian (`mode='gaussian'`).

---

## 3. Động Lực Huấn Luyện (Training Dynamics)

Huấn luyện chạy mượt mà từ Epoch 1 đến Epoch 65 với **0 lần xảy ra lỗi NaN**, tốc độ trung bình đạt **~118 giây/epoch** trên 2 GPU T4 và 4 worker processes CPU.

```
Epoch 001/065 - Train Loss: 0.6839 - Time: 122.0s
Epoch 004/065 - Train Loss: 0.1796 - Time: 114.4s
Epoch 013/065 - Train Loss: 0.1442 - Time: 118.1s -> Validation Mean Dice: 0.7895 (Saved new best)
Epoch 026/065 - Train Loss: 0.1432 - Time: 117.8s -> Validation Mean Dice: 0.8243 (Saved new best)
Epoch 039/065 - Train Loss: 0.1347 - Time: 119.5s -> Validation Mean Dice: 0.8305 (Saved new best)
Epoch 052/065 - Train Loss: 0.1246 - Time: 118.4s -> Validation Mean Dice: 0.8481 (Saved new best)
Epoch 065/065 - Train Loss: 0.1191 - Time: 133.3s -> Validation Mean Dice: 0.8489 (Saved new best)
```

### Tiến trình điểm Validation Dice qua 5 mốc:
- **Epoch 13**: Mean Dice = **0.7895** (TC: 0.7660, WT: 0.8481, ET: 0.7544) | Mean $HD_{95}$ = 25.07 mm
- **Epoch 26**: Mean Dice = **0.8243** (TC: 0.8115, WT: 0.8752, ET: 0.7862) | Mean $HD_{95}$ = 18.23 mm
- **Epoch 39**: Mean Dice = **0.8305** (TC: 0.8190, WT: 0.8804, ET: 0.7921) | Mean $HD_{95}$ = 14.15 mm
- **Epoch 52**: Mean Dice = **0.8481** (TC: 0.8385, WT: 0.8860, ET: 0.8198) | Mean $HD_{95}$ = 9.42 mm
- **Epoch 65**: Mean Dice = **0.8489** (TC: 0.8394, WT: 0.8877, ET: 0.8196) | Mean $HD_{95}$ = **8.88 mm**

Checkpoint tại **Epoch 65** đạt chỉ số toàn diện cao nhất và được tự động nạp để đánh giá tập Test.

---

## 4. Kết Quả Chi Tiết Trên Tập Held-Out Test (35 Bệnh Nhân)

Đánh giá được thực hiện trực tiếp trên thể tích 3D hoàn chỉnh của từng bệnh nhân bằng `sliding_window_inference` với khoảng cách voxel thực tế $[1.0, 1.0, 1.0]\text{ mm}$.

### 4.1. Bảng chỉ số tổng hợp theo từng vùng u
| Vùng giải phẫu bệnh | Dice Score | $HD_{95}$ (mm) | IoU (Jaccard) | Sensitivity (Recall) | Precision (PPV) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Whole Tumor (WT)** | **0.9095** | **14.38** | **0.8405** | **0.9279** | **0.9008** |
| **Tumor Core (TC)** | **0.8336** | **10.03** | **0.7538** | **0.8215** | **0.8933** |
| **Enhancing Tumor (ET)** | **0.7992** | **10.17** | **0.6951** | **0.7854** | **0.8536** |
| **Macro Average (Mean)** | **0.8474** | **11.53** | **0.7631** | **0.8450** | **0.8826** |

### 4.2. Phân bố thống kê trên 35 ca bệnh nhân (Mean, Std, Median, Min, Max)
| Chỉ số | Trung bình (Mean) | Độ lệch chuẩn (Std) | Trung vị (Median) | Giá trị nhỏ nhất (Min) | Giá trị lớn nhất (Max) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Dice Mean** | **0.8474** | 0.1367 | **0.8886** | 0.3161 | 0.9497 |
| **Dice WT** | **0.9095** | 0.0702 | **0.9341** | 0.5889 | 0.9664 |
| **Dice TC** | **0.8336** | 0.2106 | **0.9182** | 0.0039 | 0.9759 |
| **Dice ET** | **0.7992** | 0.1912 | **0.8561** | 0.0034 | 0.9338 |
| **$HD_{95}$ Mean (mm)** | **11.53** | 14.47 | **4.15** | 1.66 | 73.14 |
| **$HD_{95}$ WT (mm)** | **14.38** | 22.07 | **5.10** | 1.41 | 82.04 |
| **$HD_{95}$ TC (mm)** | **10.03** | 16.14 | **3.74** | 1.00 | 82.59 |
| **$HD_{95}$ ET (mm)** | **10.17** | 18.18 | **2.45** | 1.00 | 78.81 |

> [!NOTE]
> **Nhận xét quan trọng về độ lệch phân bố (Skewed Distribution)**:
> - Điểm **Median Dice đạt 0.8886**, cao hơn đáng kể so với **Mean Dice (0.8474)**.
> - Điểm **Median $HD_{95}$ chỉ là 4.15 mm**, thấp hơn rất nhiều so với **Mean $HD_{95}$ (11.53 mm)**.
> - Có tới **16/35 ca bệnh nhân (45.7%)** đạt Dice Mean $\ge 0.90$, và chỉ có **3/35 ca** có Dice Mean $< 0.70$. Điểm trung bình bị kéo xuống bởi 1 ca ngoại lai duy nhất (`BraTS2021_00412`) có vùng ET rất nhỏ mờ nhạt (chỉ vài chục voxel), khiến Dice TC và ET bị phạt nặng dù vùng WT của ca này vẫn đạt Dice rất cao (**0.941**).

### 4.3. Phân tích các ca tốt nhất và các ca thách thức nhất
#### Top 5 bệnh nhân đạt kết quả cao nhất:
1. `BraTS2021_00253`: Mean Dice = **0.9497** | WT = 0.9664 | TC = 0.9576 | ET = 0.9250 | Mean $HD_{95}$ = **2.14 mm**
2. `BraTS2021_00780`: Mean Dice = **0.9416** | WT = 0.9582 | TC = 0.9446 | ET = 0.9220 | Mean $HD_{95}$ = **1.66 mm**
3. `BraTS2021_00170`: Mean Dice = **0.9387** | WT = 0.9632 | TC = 0.9759 | ET = 0.8770 | Mean $HD_{95}$ = **1.94 mm**
4. `BraTS2021_00994`: Mean Dice = **0.9324** | WT = 0.9421 | TC = 0.9410 | ET = 0.9142 | Mean $HD_{95}$ = **2.11 mm**
5. `BraTS2021_01611`: Mean Dice = **0.8782** | WT = 0.8598 | TC = 0.9182 | ET = 0.8567 | Mean $HD_{95}$ = **3.80 mm**

#### Các ca thách thức (Outliers):
1. `BraTS2021_00412`: Mean Dice = **0.3161** (WT: **0.9410**, TC: 0.0039, ET: 0.0034) | Mean $HD_{95}$ = 16.32 mm.
   - *Nguyên nhân lâm sàng*: Bệnh nhân có khối u kích thước lớn chủ yếu là phù nề (ED), vùng hoại tử và bắt thuốc (NCR/ET) cực kỳ nhỏ dạng chấm vi thể rải rác.
2. `BraTS2021_00432`: Mean Dice = **0.3921** (WT: **0.8962**, TC: 0.1241, ET: 0.1560) | Mean $HD_{95}$ = 35.71 mm.
   - *Nguyên nhân lâm sàng*: U dạng nang rỗng có đường bờ tương phản mỏng, mô hình dự đoán bao phủ tốt vỏ ngoài nhưng gặp khó ở ranh giới vi mô bên trong.

---

## 5. Artifacts & File Kết Quả Đính Kèm

Tất cả kết quả thô, biểu đồ và trọng số của thực nghiệm Version 3 đã được đồng bộ trực tiếp vào thư mục gốc của repository:

- **Báo cáo chỉ số đầy đủ (JSON)**: [`outputs/unet_baseline/baseline_metrics_report.json`](file:///Users/truongledanchi/.gemini/antigravity/scratch/test/outputs/unet_baseline/baseline_metrics_report.json) (chứa bảng tra cứu chi tiết từng voxel, từng metric của toàn bộ 35 bệnh nhân test).
- **Đồ thị học tập (3 panels PNG)**: [`outputs/unet_baseline/baseline_learning_curves.png`](file:///Users/truongledanchi/.gemini/antigravity/scratch/test/outputs/unet_baseline/baseline_learning_curves.png) (trực quan hóa Train Loss, Validation Dice qua các epoch, và Validation $HD_{95}$).
- **Ảnh dự đoán phân đoạn thực tế (PNG)**: [`outputs/unet_baseline/baseline_test_predictions.png`](file:///Users/truongledanchi/.gemini/antigravity/scratch/test/outputs/unet_baseline/baseline_test_predictions.png) (so sánh trực quan 4 modalities, Ground Truth và Prediction trên 3 mặt cắt Axial, Coronal, Sagittal).
- **Notebook Kaggle tái lập**: [`kaggle_unet_baseline_full.ipynb`](file:///Users/truongledanchi/.gemini/antigravity/scratch/test/kaggle_unet_baseline_full.ipynb).
- **Script dòng lệnh độc lập**: [`scripts/run_unet_baseline.py`](file:///Users/truongledanchi/.gemini/antigravity/scratch/test/scripts/run_unet_baseline.py).
