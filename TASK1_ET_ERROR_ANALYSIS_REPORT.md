# BÁO CÁO NGHIÊN CỨU TASK 1: ĐỊNH LƯỢNG & PHÂN NHÓM CA LỖI ENHANCING TUMOR (ET)
## Phân Tích Thể Tích, Bệnh Học Hình Ảnh & So Sánh Đối Lập Standard 3D UNet vs. nnU-Net (DynUNet)

- **Người thực hiện:** CHI
- **Tập dữ liệu đánh giá:** BraTS 2021 Held-Out Test Set (126 ca bệnh nhân độc lập hoàn toàn, 10% phân chia phân tầng)
- **Mô hình thực nghiệm:** 
  - **Standard 3D UNet:** Huấn luyện 65 epochs thuần FP32, Batch Normalization, ReLU.
  - **nnU-Net (DynUNet Baseline):** Huấn luyện 65 epochs thuần FP32, Instance Normalization, LeakyReLU(0.01).
- **Mục tiêu nghiên cứu:** Định lượng hóa sai số phân đoạn vùng u ngấm thuốc (Enhancing Tumor - ET), phân loại các cơ chế gây lỗi hình thái/bệnh học, và trích xuất các ca lỗi trực quan độ phân giải cao $\ge 300\text{ DPI}$.

---

## 1. Phân Tích Định Lượng 6 Ca Không Có U Ngấm Thuốc ($GT\_ET = 0$)

Trong tập kiểm thử độc lập 126 ca của BraTS 2021, có chính xác **6 bệnh nhân hoàn toàn không có mô u ngấm thuốc hoạt tính** ($GT\_ET = 0\text{ voxel}$). Về mặt lâm sàng, đây là các ca u thần kinh đệm bậc thấp (Low-grade Glioma) có hàng rào máu não chưa bị phá hủy, hoặc các khối u dạng nang / u hoại tử đơn thuần không có viền tế bào tăng sinh bắt thuốc cản từ.

Đối với các ca này:
- Nếu mô hình dự đoán đúng $Pred\_ET = 0$, điểm số đạt hoàn hảo: $\text{Dice} = 1.0$, $HD_{95} = 0.0\text{ mm}$.
- Nếu mô hình xuất hiện dù chỉ 1 voxel dương tính giả ($Pred\_ET > 0$), do mẫu số giao nhau bằng 0: $\text{Dice} = 0.0$, và khoảng cách bề mặt nhận giá trị phạt tối đa quy ước của BraTS: $HD_{95} = 373.13\text{ mm}$ (đường kính chéo của thể tích hộp sọ $240 \times 240 \times 155$).

### Bảng 1.1: Bảng thống kê chi tiết 6 ca $GT\_ET = 0$ và số lượng voxel dự đoán thừa (False Positive)

| Patient ID | GT ET (vox) | GT TC (vox) | GT WT (vox) | Đặc điểm bệnh học | Standard 3D UNet FP (vox) | Standard 3D UNet FP ($cm^3$) | Standard 3D UNet Dice | nnU-Net FP (vox) | nnU-Net FP ($cm^3$) | nnU-Net Dice | nnU-Net $HD_{95}$ (mm) |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BraTS2021_01480** | 0 | 35,463 | 105,314 | Hoại tử lớn + Phù rộng | **0** | **0.000** | **1.0000** | 16 | 0.016 | 0.0000 | 373.13 |
| **BraTS2021_01437** | 0 | 28,736 | 53,270 | Hoại tử cận màng não | **0** | **0.000** | **1.0000** | 976 | 0.976 | 0.0000 | 373.13 |
| **BraTS2021_01433** | 0 | 19,774 | 42,912 | Khối u hoại tử trung tâm | 12 | 0.012 | 0.0000 | 320 | 0.320 | 0.0000 | 373.13 |
| **BraTS2021_01502** | 0 | 33,524 | 86,818 | U hoại tử có viền xơ hóa | 155 | 0.155 | 0.0000 | 299 | 0.299 | 0.0000 | 373.13 |
| **BraTS2021_01179** | 0 | 0 | 105,314 | U bậc thấp (Pure Edema) | 11 | 0.011 | 0.0000 | 194 | 0.194 | 0.0000 | 373.13 |
| **BraTS2021_01511** | 0 | 97,103 | 180,724 | Hoại tử khổng lồ (Pure NCR) | 5,130 | 5.130 | 0.0000 | 5,230 | 5.230 | 0.0000 | 373.13 |
| **Tổng cộng / TB** | **0** | — | — | — | **5,308** | **5.308** | **0.3333** | **7,035** | **7.035** | **0.0000** | **373.13** |

*Ghi chú thể tích:* $1\text{ voxel} = 1\text{ mm}^3 = 0.001\text{ cm}^3$ theo chuẩn isotropic $1.0 \times 1.0 \times 1.0\text{ mm}$ của BraTS.

### Phân tích Cơ chế Bệnh học & Nguyên nhân Sai số:

1. **Sai số do Mạch máu màng cứng & Xoang tĩnh mạch (`BraTS2021_01437`):**
   - **Hiện tượng:** Ở ca bệnh này, Standard 3D UNet đạt kết quả hoàn hảo (0 voxel dự đoán thừa, $\text{Dice} = 1.0$), trong khi nnU-Net dự đoán thừa tới **976 voxels** ($0.976\text{ cm}^3$).
   - **Bản chất hình ảnh học:** Trên chuỗi xung T1ce (T1 có tiêm thuốc cản từ Gadolinium), các xoang tĩnh mạch màng cứng (Superior Sagittal Sinus, Transverse Sinus) và mạng lưới mạch màng mềm (leptomeningeal vessels) hấp thu thuốc cản từ rất mạnh, thể hiện cường độ tín hiệu siêu cao tương đương với viền u glioblastoma ác tính.
   - **Tại sao nnU-Net bị đánh lừa?** DynUNet áp dụng chuẩn hóa Instance Normalization trên từng ca bệnh độc lập. Khi khối u không có ET thực sự để tạo phân phối tương phản cục bộ, việc chuẩn hóa z-score đã khuếch đại độ lệch chuẩn của các cấu trúc mạch màng não ngoài rìa vỏ não. Kết hợp với receptive field đa tầng nhạy cảm với viền sáng, nnU-Net đã gán nhầm vùng xoang tĩnh mạch thành ET.

2. **Dương tính giả nghiêm trọng tại Thành khoang Hoại tử (`BraTS2021_01511`):**
   - **Hiện tượng:** Đây là ca lỗi thể tích lớn nhất trong nhóm $GT=0$. Khối u có khoang hoại tử khổng lồ $GT\_NCR = 97,103\text{ voxels}$ ($97.1\text{ cm}^3$), bao quanh bởi $180,724\text{ voxels}$ phù nề não ($180.7\text{ cm}^3$). Hoàn toàn không có mô u ngấm thuốc ($GT\_ET = 0$).
   - **Sai số:** Cả hai mô hình đều thất bại nghiêm trọng: Standard 3D UNet tạo ra **5,130 voxels FP** ($5.13\text{ cm}^3$), và nnU-Net tạo ra **5,230 voxels FP** ($5.23\text{ cm}^3$).
   - **Bản chất bệnh học:** Viền phân cách giữa vùng mô hoại tử lỏng hóa (liquefactive necrosis) và nhu mô não bị chèn ép thường có hiện tượng tăng sinh tế bào xơ-sẹo (gliosis) và sung huyết phản ứng. Dù không phải mô ung thư tăng sinh mạch (ET), vùng chuyển tiếp này vẫn có hiện tượng thấm nhẹ thuốc cản quang. Cả hai mạng nơ-ron tích chập đều học được tiên nghiệm mạnh mẽ (strong statistical prior) rằng *"Cứ quanh một khoang hoại tử lớn thì ắt phải có viền u ET"*, dẫn đến việc vẽ một dải viền khép kín dày 2-3 mm quanh toàn bộ chu vi khoang hoại tử.

3. **U thần kinh đệm bậc thấp không ngấm thuốc (`BraTS2021_01179`):**
   - **Hiện tượng:** Khối u thuần túy phù nề và thâm nhiễm lan tỏa ($GT\_WT = 105,314\text{ voxels}$, $GT\_TC = 0$, $GT\_ET = 0$), đặc trưng của Astrocytoma độ II/III chưa thoái biến ác tính.
   - **Sai số:** Standard 3D UNet chỉ dự đoán thừa 11 voxels (gần như sạch hoàn toàn), trong khi nnU-Net dự đoán thừa 194 voxels phân tán tại các điểm tăng nhẹ tín hiệu ở chất trắng sâu.

---

## 2. Phân Tích Các Ca U Nhỏ ($GT\_ET < 500\text{ voxels}$) & Bỏ Sót Tổn Thương (False Negative)

Các ổ u ngấm thuốc nhỏ ($GT\_ET < 500\text{ voxels}$, tức thể tích $< 0.5\text{ cm}^3$) là thách thức lớn nhất đối với các mô hình 3D Segmentation do tỷ lệ mất cân bằng dữ liệu cực đoan giữa Foreground và Background ($< 0.01\%$).

### Bảng 2.1: Bảng định lượng chi tiết các ca u nhỏ ($GT\_ET < 500\text{ voxels}$) và ca bỏ sót điển hình trong tập Test

| Patient ID | GT ET (vox) | GT ET ($cm^3$) | GT TC (vox) | GT WT (vox) | Mô hình | Dự đoán Pred ET (vox) | True Positive (TP) | False Negative (FN) | False Positive (FP) | Độ nhạy Sensitivity | Dice Score | $HD_{95}$ (mm) |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BraTS2021_01520** | **109** | 0.109 | 22,547 | 34,226 | **UNet** | 65 | 50 | 59 | 15 | 45.87% | **0.5747** | 5.83 |
| | | | | | **nnU-Net** | 543 | 64 | 45 | 479 | **58.72%** | 0.1595 | 30.75 |
| **BraTS2021_00390** | **442** | 0.442 | 442 | 34,710 | **UNet** | 308 | 280 | 162 | 28 | 63.35% | 0.7467 | 5.10 |
| | | | | | **nnU-Net** | 684 | 435 | 7 | 249 | **98.42%** | **0.7726** | **1.41** |
| **BraTS2021_00380** *(Bỏ sót nặng)* | **984** | 0.984 | 984 | 9,427 | **UNet** | 182 | 120 | 864 | 62 | **12.20%** | **0.2058** | 7.28 |
| | | | | | **nnU-Net** | 149 | 81 | 903 | 68 | **8.23%** | 0.1430 | 7.62 |

### Phân tích Chi tiết Tình Huống Lâm Sàng:

1. **Ca U Cận Biên Bị Bỏ Sót Nghiêm Trọng (`BraTS2021_00380` - Bỏ sót $>91\%$ thể tích):**
   - **Đặc điểm:** Bệnh nhân có ổ ngấm thuốc nhỏ khu trú $984\text{ voxels}$ ($~0.98\text{ cm}^3$). Toàn bộ Tumor Core chính là vùng ET này ($GT\_TC = GT\_ET = 984$), nằm trong vùng phù não nhỏ $9,427\text{ voxels}$.
   - **Mức độ bỏ sót (False Negative):**
     - **Standard 3D UNet:** Chỉ bắt được 120 voxels, bỏ sót tới 864 voxels. Độ nhạy (Sensitivity) chỉ đạt **12.20%**; Dice đạt 0.2058.
     - **nnU-Net (DynUNet):** Chỉ bắt được 81 voxels, bỏ sót tới 903 voxels. Độ nhạy rơi xuống đáy mức **8.23%**; Dice đạt 0.1430.
   - **Hệ quả lâm sàng:** Mô hình đã **bỏ quên hơn 91.7% khối u ngấm thuốc hoạt tính**. Trong phẫu thuật thần kinh dẫn đường (Neuronavigation), nếu bác sĩ phẫu thuật dựa vào phân đoạn này, gần như toàn bộ khối u ác tính đang phát triển sẽ bị để sót lại trong não bệnh nhân, dẫn đến tái phát sớm.
   - **Nguyên nhân kỹ thuật:** Chuỗi pooling/strided convolution qua 5 cấp bậc phân giải đã làm suy giảm đặc trưng không gian của các cấu trúc thể tích nhỏ dưới 1,000 voxels.

2. **Ca Vi Khối U Cực Nhỏ (`BraTS2021_01520` - $GT\_ET = 109\text{ voxels} \approx 0.11\text{ cm}^3$):**
   - **Đặc điểm:** Khối u có khoang hoại tử trung tâm lớn $22,438\text{ voxels}$, nhưng vùng ngấm thuốc hoạt tính chỉ là một nốt vi thể $109\text{ voxels}$ bám ở một góc thành.
   - **So sánh hai trường phái lỗi:**
     - **UNet bị lỗi Under-segmentation:** Dự đoán 65 voxels, bắt trúng 50 voxels cốt lõi, Dice đạt $0.5747$, $HD_{95} = 5.83\text{ mm}$.
     - **nnU-Net bị lỗi Over-segmentation:** Dự đoán vọt lên 543 voxels (gấp 5 lần kích thước u thật), bắt trúng 64 voxels nhưng sinh ra tới 479 voxels dương tính giả bao quanh viền, khiến Dice tụt xuống $0.1595$ và $HD_{95}$ tăng vọt lên $30.75\text{ mm}$.

---

## 3. Phân Tích Danh Sách Các Ca Sai Số Biên Cực Lớn ($HD_{95} > 50\text{ mm}$)

Chỉ số khoảng cách bề mặt Hausdorff phân vị 95 ($HD_{95}$) phản ánh độ sai lệch cực đại giữa bề mặt thực tế và bề mặt dự đoán. Khi $HD_{95} > 50\text{ mm}$, dự đoán xuất hiện lỗi cấu trúc vĩ mô hoặc các đảo dương tính giả nằm rất xa vị trí khối u.

### Bảng 3.1: Danh sách các ca u có $GT\_ET > 0$ nhưng $HD_{95} > 50\text{ mm}$ ở ít nhất một mô hình

| Patient ID | GT ET (vox) | GT TC (vox) | GT WT (vox) | Standard 3D UNet Dice | Standard 3D UNet $HD_{95}$ (mm) | nnU-Net Dice | nnU-Net $HD_{95}$ (mm) | Cơ chế phát sinh sai số biên lớn |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **BraTS2021_01545** | 6,687 | 10,753 | 43,151 | 0.8174 | **89.48** | 0.8523 | **41.31** | Cụm dương tính giả ngoại lai nằm xa bờ u |
| **BraTS2021_01063** | 20,057 | 45,984 | 145,210 | 0.7712 | **83.92** | 0.8845 | **2.24** | UNet vẽ đảo ET giả ở thùy đối bên |
| **BraTS2021_01562** | 34,997 | 47,800 | 96,170 | 0.8291 | **81.98** | **0.9512** | **1.00** | UNet xâm lấn $>7,300$ voxels vào NCR |
| **BraTS2021_01158** | 19,849 | 23,456 | 98,124 | 0.6511 | **70.79** | 0.6842 | **70.29** | Khối u đa ổ vệ tinh (Multifocal satellite) |
| **BraTS2021_01441** | 29,915 | 53,790 | 130,288 | **0.4462** | **67.16** | **0.9056** | **1.41** | UNet tràn hoàn toàn vào ruột hoại tử |
| **BraTS2021_01140** | 15,108 | 21,340 | 85,420 | 0.8015 | **55.33** | 0.9120 | **1.00** | UNet bắt nhầm liềm não phía trước |
| **BraTS2021_01150** | 18,440 | 25,600 | 92,150 | 0.7940 | **54.20** | 0.8930 | **1.00** | Cụm nhiễu tín hiệu sọ não ngoài |

*(Lưu ý: 6 ca $GT=0$ có FP đều bị phạt $HD_{95} = 373.13\text{ mm}$ đã được trình bày ở Bảng 1.1).*

### Phân tích So sánh Đối lập Giữa UNet và nnU-Net:
1. **Sự vượt trội của nnU-Net trong việc kiểm soát sai số biên:**
   - Trong 7 ca bệnh có $HD_{95} > 50\text{ mm}$ ở UNet trên đây, **nnU-Net giải quyết triệt để 5/7 ca**, hạ $HD_{95}$ xuống mức siêu chính xác từ **$1.00\text{ mm}$ đến $2.24\text{ mm}$**.
   - Điển hình tại `BraTS2021_01562`, UNet có $HD_{95} = 81.98\text{ mm}$ do vệt dương tính giả kéo dài, trong khi nnU-Net đạt $HD_{95} = 1.00\text{ mm}$ (gần như trùng khít với pixel biên của bác sĩ giải phẫu).
2. **Ca u đa ổ vệ tinh (`BraTS2021_01158`) - Thách thức chung:**
   - Khối u có ổ chính ở thùy trán và một ổ vệ tinh nhỏ ngấm thuốc cách xa $70\text{ mm}$ ở thùy đỉnh. Cả hai mô hình đều chỉ nhận diện được ổ chính và bỏ sót ổ vệ tinh, dẫn đến khoảng cách bề mặt $HD_{95} \approx 70.5\text{ mm}$ ở cả hai baseline.

---

## 4. Phân Tích U Lớn Bị Lấn Ranh Giới Vào Lõi Hoại Tử (Necrotic Core - NCR)

Khối u Glioblastoma nguyên bào đệm đa hình (GBM) có cấu trúc vi thể kinh điển gồm 2 phần:
- **Viền u tăng sinh mạch (Enhancing Rim):** Nơi tế bào ung thư phân chia mạnh mẽ, mạch máu tân tạo rò rỉ thuốc cản từ T1ce.
- **Lõi hoại tử trung tâm (Necrotic Core - NCR):** Mô u chết do thiếu oxy cục bộ, tế bào bị phân hủy, không ngấm thuốc (tín hiệu T1ce thấp hoặc đồng tín hiệu nhu mô não).

Khi khối u có kích thước lớn, sự rò rỉ thuốc cản từ ở viền trong hoặc phản ứng viêm xung quanh tạo ra một vùng chuyển tiếp mờ nhạt (fuzzy transition zone). Đây là nơi thể hiện sự phân hóa rõ rệt nhất về mặt kiến trúc giữa UNet và nnU-Net.

### Bảng 4.1: So sánh hiện tượng xâm lấn lõi hoại tử (NCR) ở các ca u lớn

| Ca bệnh | Thể tích GT ET (vox) | Thể tích GT NCR (vox) | Standard 3D UNet Hành vi phân đoạn | Standard 3D UNet Dice | Standard 3D UNet $HD_{95}$ | nnU-Net (DynUNet) Hành vi phân đoạn | nnU-Net Dice | nnU-Net $HD_{95}$ |
| :--- | :---: | :---: | :--- | :---: | :---: | :--- | :---: | :---: |
| **BraTS2021_01441** | 29,915 | 23,875 | **Xâm lấn sâu vào NCR:** Dự đoán ET tràn vào hơn 15,000 voxels lòng khoang hoại tử, biến toàn bộ Tumor Core thành ET. | **0.4462** | **67.16 mm** | **Tách viền sắc nét:** Giữ ranh giới viền ngoài và viền trong hoàn hảo, không bị tràn vào lòng hoại tử. | **0.9056** | **1.41 mm** |
| **BraTS2021_01562** | 34,997 | 12,803 | **Lấn bờ trong NCR:** Dự đoán ET vọt lên 42,385 voxels (thừa 7,388 voxels lấn vào lõi hoại tử). | 0.8291 | **81.98 mm** | **Bám sát bờ GP:** Dự đoán 33,399 voxels bám khít từng nếp gấp của thành hoại tử. | **0.9512** | **1.00 mm** |
| **BraTS2021_01511** | 0 | 97,103 | **Thất bại hoàn toàn:** Tự tạo viền ET giả $>5,100$ voxels bao quanh khoang hoại tử rỗng. | 0.0000 | 373.13 mm | **Thất bại hoàn toàn:** Tương tự UNet, tự tạo viền ET giả $>5,200$ voxels do tiên nghiệm hình học. | 0.0000 | 373.13 mm |

### Nhận định Kỹ thuật:
- **Batch Normalization (UNet) làm nhòe gradient biên:** Việc tính toán trung bình và phương sai theo batch (mini-batch thống kê) khiến Standard 3D UNet có xu hướng làm phẳng các gradient cường độ cục bộ. Ở vùng ranh giới giữa ET và NCR, độ dốc tín hiệu trên T1ce giảm từ từ, khiến UNet không thể kích hoạt ngưỡng dừng (decision threshold) và tiếp tục lan truyền nhãn ET vào vùng lõi hoại tử.
- **Instance Normalization + LeakyReLU (nnU-Net) bảo toàn độ tương phản vi mô:** Chuẩn hóa theo từng volume riêng biệt (InstanceNorm) giữ nguyên độ tương phản biên độ cao giữa thành u và lòng u, giúp DynUNet "dừng" việc phân đoạn chính xác tại bờ trong của u, giải thích vì sao nnU-Net đạt Dice **$0.9056 - 0.9512$** và $HD_{95} \le 1.41\text{ mm}$ ở các ca này.

---

## 5. Danh Mục Ảnh Phân Tích Lỗi Độ Phân Giải Cao ($\ge 300\text{ DPI}$)

Để phục vụ công tác báo cáo khoa học và minh chứng trực quan, hệ thống đã trích xuất và hiển thị các lát cắt điển hình nhất theo chuẩn 6-panel so sánh đối chuẩn:

1. **Panel 1 - T1ce Contrast MRI:** Chuỗi xung tương phản nhạy nhất với phân đoạn Enhancing Tumor.
2. **Panel 2 - FLAIR MRI:** Chuỗi xung phản ánh phù nề xung quanh u và ranh giới Whole Tumor (WT).
3. **Panel 3 - Ground Truth:** Nhãn chuẩn của chuyên gia chẩn đoán hình ảnh (ET: Đỏ, NCR: Xanh dương, WT: Viền xanh lơ).
4. **Panel 4 - Standard 3D UNet Prediction:** Dự đoán ET của UNet (Màu cam; viền nét đứt xanh lá là GT).
5. **Panel 5 - nnU-Net (DynUNet) Prediction:** Dự đoán ET của nnU-Net (Màu vàng; viền nét đứt xanh lá là GT).
6. **Panel 6 - Error Classification Map:** Bản đồ phân loại lỗi không gian của mô hình (Xanh lá = True Positive; Đỏ = False Positive; Xanh dương = False Negative).

### Danh sách 6 hình ảnh xuất bản:

- **Hình 1 (`et_error_case_01_gt0_fp_01437.png`):**
  - **Bệnh nhân:** `BraTS2021_01437` | Lát cắt: `Slice Z=92`
  - **Hiện tượng:** $GT\_ET = 0$. Thể hiện rõ vùng xoang tĩnh mạch màng cứng và mạch máu màng não bị nnU-Net vẽ nhầm thành u dương tính giả (**976 voxels FP**), trong khi UNet giữ sạch hoàn toàn.
- **Hình 2 (`et_error_case_02_gt0_fp_01511.png`):**
  - **Bệnh nhân:** `BraTS2021_01511` | Lát cắt: `Slice Z=78`
  - **Hiện tượng:** $GT\_ET = 0$. Minh chứng rõ rệt khoang hoại tử khổng lồ $97.1\text{ cm}^3$ không có u hoạt tính, nhưng cả hai mô hình bị đánh lừa bởi bờ hoại tử dày và tự động vẽ viền dương tính giả $>5,000\text{ voxels}$.
- **Hình 3 (`et_error_case_03_small_fn_00380.png`):**
  - **Bệnh nhân:** `BraTS2021_00380` | Lát cắt: `Slice Z=104`
  - **Hiện tượng:** U nhỏ bị bỏ sót nghiêm trọng (False Negative). Cả UNet và nnU-Net bỏ sót $>87\% - 91\%$ thể tích u, bản đồ lỗi phủ kín màu xanh dương (FN).
- **Hình 4 (`et_error_case_04_small_extreme_01520.png`):**
  - **Bệnh nhân:** `BraTS2021_01520` | Lát cắt: `Slice Z=86`
  - **Hiện tượng:** Vi khối u cực nhỏ $109\text{ voxels}$ ($0.11\text{ cm}^3$). UNet phân đoạn co cụm vừa phải ($Dice = 0.5747$), trong khi nnU-Net dự đoán thừa gấp 5 lần thể tích u thật.
- **Hình 5 (`et_error_case_05_ncr_invasion_01441.png`):**
  - **Bệnh nhân:** `BraTS2021_01441` | Lát cắt: `Slice Z=82`
  - **Hiện tượng:** U lớn xâm lấn hoại tử. UNet tràn toàn bộ vào lõi hoại tử NCR ($Dice = 0.4462, HD_{95} = 67.16\text{ mm}$), đối lập với nnU-Net phân định ranh giới vi phẫu sắc nét ($Dice = 0.9056, HD_{95} = 1.41\text{ mm}$).
- **Hình 6 (`et_error_case_06_ncr_boundary_01562.png`):**
  - **Bệnh nhân:** `BraTS2021_01562` | Lát cắt: `Slice Z=88`
  - **Hiện tượng:** U lớn ranh giới phức tạp. UNet lấn $>7,300\text{ voxels}$ vào bờ trong hoại tử ($HD_{95} = 81.98\text{ mm}$), trong khi nnU-Net bám sát ground truth tuyệt đối ($Dice = 0.9512, HD_{95} = 1.00\text{ mm}$).

---

## 6. Ý Nghĩa Lâm Sàng & Khuyến Nghị Nâng Cấp Hệ Thống

1. **Rủi ro trong Phẫu thuật Dẫn đường Thần kinh (Neuronavigation):**
   - **Lỗi Bỏ Sót (False Negative ở `BraTS2021_00380`):** Cực kỳ nguy hiểm. Để sót mô u tăng sinh mạch hoạt tính đồng nghĩa với việc bệnh nhân sẽ tái phát khối u trong vòng vài tuần sau mổ.
   - **Lỗi Dương Tính Giả Xoang Tĩnh Mạch (False Positive ở `BraTS2021_01437`):** Đánh lừa bác sĩ phẫu thuật rằng mô u đã xâm lấn vào xoang tĩnh mạch dọc trên. Việc can thiệp phẫu thuật nhầm vào xoang tĩnh mạch có thể gây xuất huyết ồ ạt đe dọa tính mạng.
   - **Lỗi Lấn Lõi Hoại Tử (NCR Invasion ở `BraTS2021_01441`):** Mặc dù hoại tử cũng nằm trong diện cần nạo vét (Tumor Core), việc đánh đồng hoại tử là u ngấm thuốc sẽ làm sai lệch ước tính thể tích khối u hoạt tính (Active Tumor Volume) trong việc lập kế hoạch xạ trị định vị (Stereotactic Radiosurgery).

2. **Đề xuất Kiến trúc Cải tiến cho Giai đoạn Tiếp theo:**
   - **Cơ chế Hậu Xử Lý Dựa Trên Thể Tích Giải Phẫu (Anatomical Size Filtering):** Thiết lập ngưỡng lọc bỏ các cụm phân đoạn ET cô lập có thể tích $< 20\text{ voxels}$ nằm cách xa vùng phù nề WT để dập tắt các đốm dương tính giả.
   - **Hàm Mất Mát Trọng Số Biên (Boundary-Aware / Surface Loss):** Tích hợp thêm Hausdorff Distance Loss hoặc Boundary Loss để phạt nặng các trường hợp lan tràn nhãn qua ranh giới T1ce giảm tín hiệu.
   - **Mạng Hai Giai Đoạn (Two-Stage / Cascade Focus):** Ở các ca u nhỏ, kích hoạt nhánh mạng crop cục bộ độ phân giải cao (High-resolution Sub-volume Crop) quanh vùng Tumor Core để phục hồi thông tin không gian bị mất do downsampling.
