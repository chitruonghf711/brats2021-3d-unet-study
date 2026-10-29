# Nghiên Cứu Khảo Sát 3D UNet Với Soft Label & Label Noise Trên BraTS 2021

Tài liệu này tổng hợp toàn bộ phần công việc thiết kế, mở rộng mã nguồn, chuẩn bị dữ liệu và thực nghiệm đối sánh **3D UNet** phục vụ hướng nghiên cứu **Nhiễu nhãn (Label Noise)**, **Nhãn mềm liên tục (Soft Labels)** và **Ước lượng độ bất định / Hiệu chuẩn xác suất (Uncertainty & Calibration)** trên bài toán phân đoạn khối u não MRI 3D đa phương thức (**BraTS 2021**).

> - Mô tả chi tiết các bộ dữ liệu đã chạy, lý do lựa chọn và cơ chế phân nhánh giám sát: [`DATASETS.md`](./DATASETS.md).
> - Tài liệu giải thích chi tiết luồng chạy từng hàm và biến đổi kích thước tensor: [`CODE_FLOW.md`](./CODE_FLOW.md).
> - Báo cáo phân tích chuyên sâu kết quả thực nghiệm (định lượng, định tính & chứng minh toán học): [`EXPERIMENTAL_REPORT.md`](./EXPERIMENTAL_REPORT.md).

---

## 1. Động Lực Khoa Học & Mục Tiêu Nghiên Cứu

### 1.1. Tại sao cần nghiên cứu về Label Noise & Soft Labels trong ảnh MRI u não 3D?
Trong bài toán phân đoạn ảnh y tế 3D (đặc biệt là khối u thần kinh đệm *glioma* trên MRI sọ não), việc nghiên cứu **Label Noise (nhiễu nhãn)** và **Soft Labels (nhãn mềm)** xuất phát từ **3 thách thức thực tiễn và toán học cốt lõi**:

1. **Trong lâm sàng u não, "Nhãn chuẩn tuyệt đối" (*Absolute Ground Truth*) gần như không tồn tại**:
   - Khác với ảnh tự nhiên có ranh giới vật lý sắc nét giữa các vật thể, tế bào u thần kinh đệm xâm lấn lan tỏa dọc theo các bó sợi chất trắng khiến tín hiệu MRI chuyển tiếp mờ dần từ lõi u ra nhu mô não lành, cộng hưởng với hiệu ứng trung bình thể tích (*partial volume effect*) trong mỗi voxel `1 mm³`.
   - Theo công bố nền tảng của BraTS (*Menze et al., IEEE TMI 2015*), khi nhiều bác sĩ X-quang thần kinh độc lập khoanh vùng cùng một khối u, **mức độ đồng thuận Dice giữa các chuyên gia chỉ đạt khoảng `74% – 85%`** (riêng vùng u ngấm thuốc nhỏ còn thấp hơn). Nghĩa là ngay cả nhãn chuyên gia cũng luôn hàm chứa `15% – 25%` phương sai bất đồng nội tại (*inter-observer variability*).
2. **Khoảng cách lớn giữa dữ liệu thi đấu quốc tế (*Challenge*) và dữ liệu bệnh viện thực tế**:
   - Một ca nhãn sạch trong BraTS 2021 đòi hỏi tổ hợp mô hình AI khởi tạo nháp + bác sĩ khoanh thủ công tỉ mỉ trên 4 chuỗi MRI + chuyên gia cao cấp (> 10–15 năm kinh nghiệm) kiểm duyệt vòng lặp.
   - Ngược lại, khi thu thập dữ liệu huấn luyện thực tế tại các bệnh viện (hoặc trích xuất từ hệ thống lưu trữ xạ trị PACS, dữ liệu đa trung tâm, nhãn bán tự động), mỗi ca bệnh thường **chỉ do 1 bác sĩ nội trú/kỹ thuật viên khoanh nhanh dưới áp lực thời gian** và thiếu hội đồng duyệt chéo $\rightarrow$ sai lệch ranh giới, khoanh nhầm tổn thương giả u hoặc bỏ sót ổ u nhỏ là điều tất yếu.
3. **Mạng 3D UNet rất dễ "học vẹt nhãn sai" (*Noise Memorization*) và "quá tự tin" (*Overconfidence*)**:
   - Mạng CNN 3D dung lượng lớn (hàng triệu tham số) khi huấn luyện với nhãn cứng `{0, 1}` và hàm mất mát Cross-Entropy tiêu chuẩn sẽ chịu gradient phạt bùng nổ $-\log(\hat{p}) \to \infty$ tại các voxel bị gán nhãn sai.
   - Để giảm hàm mất mát, mạng buộc phải bẻ cong ranh giới quyết định để **học thuộc lòng cả vết vẽ trượt tay của người gán nhãn**, đồng thời đẩy xác suất đầu ra về mức cực đoan (`0.99` hoặc `0.01`) ngay tại vùng biên mơ hồ — gây rủi ro lớn khi bác sĩ phẫu thuật thần kinh hoặc xạ trị dùng bản đồ dự đoán đó để lên kế hoạch can thiệp.

### 1.2. Ba câu hỏi nghiên cứu trọng tâm của hệ thống
Hệ thống mở rộng trong repository này giải quyết trọn vẹn 3 câu hỏi cốt lõi:
1. **Mô phỏng nhiễu lâm sàng trung thực**: Làm thế nào tạo ra nhiễu nhãn 3D tĩnh theo từng bệnh nhân (lệch nét cọ biên hình thái học + nhầm ổ giả u + bỏ sót vùng u — xem giải thích chi tiết tại [`DATASETS.md` Mục 3.2](./DATASETS.md)) có kiểm soát định lượng mà vẫn giữ đúng cấu trúc bao hàm giải phẫu $\text{ET} \subseteq \text{TC} \subseteq \text{WT}$?
2. **Khai thác Soft Label & Loss kháng nhiễu**: Các biểu diễn nhãn mềm liên tục (`Gaussian Boundary`, `Signed Distance Function - SDF`, `Label Smoothing`) và các hàm mất mát kháng nhiễu (`SoftDiceBCE`, `Symmetric Cross-Entropy - SCE`, `Generalized Cross-Entropy - GCE`) giúp **3D UNet** chống lại hiện tượng học vẹt nhãn sai ra sao?
3. **Đánh giá hiệu chuẩn & bất định**: Mô hình dự đoán xác suất đáng tin cậy đến mức nào trên tập bệnh nhân kiểm thử nhãn sạch (đo qua **Dice**, **Brier Score**, **ECE** và **Bản đồ Entropy 3D**)?

---

## 2. Tổng Hợp Các Hạng Mục Đã Thực Hiện

### 2.1. Chuẩn bị bộ dữ liệu thực tế `BraTS-Mini` (`scripts/prepare_brats_mini.py`)
- Xây dựng cơ chế **HTTP Streaming Extraction** trích xuất trực tiếp đúng $N$ ca bệnh nhân thực tế từ kho dữ liệu **BraTS 2021** mà không cần tải toàn bộ gói lưu trữ nặng ~13 GB.
- Đã tải và xác thực hoàn chỉnh **30 ca bệnh nhân BraTS 2021 thực tế** (~285 MB tại `dataset/BraTS2021_Training_Data`), mỗi ca gồm đủ 4 chuỗi MRI (`flair`, `t1`, `t1ce`, `t2`) và nhãn chuẩn (`seg`) kích thước `(240, 240, 155)`.
- Tích hợp kiểm tra dung lượng ổ cứng an toàn trước khi tải và chế độ dự phòng `--mode synthetic` sinh dữ liệu khối cầu lồng nhau 3D khi chạy ngoại tuyến.

### 2.2. Module tạo Nhiễu Nhãn & Nhãn Mềm 3D (`src/noise_soft_labels.py`)
- **Mô phỏng Label Noise tĩnh theo từng bệnh nhân (`corrupt_label_3d`, `AddSyntheticLabelNoised`)**:
  - Sử dụng hàm băm tất định `MD5(base_seed:patient_id)` giúp mỗi bệnh nhân trong tập Train mang một bản nhãn lỗi cố định xuyên suốt mọi epoch (phản ánh đúng bản chất dữ liệu gán nhãn bẩn thực tế mà không tốn thêm dung lượng lưu file NIfTI).
  - Hỗ trợ 3 chế độ nhiễu:
    - `'boundary'`: Phép giãn/co hình thái học 3D tách chiều tốc độ cao $O(r)$ kết hợp trường ngẫu nhiên mượt 3D (mô phỏng bác sĩ vẽ lẹm ra ngoài hoặc hụt vào trong).
    - `'flip'`: Chèn ngẫu nhiên các khối ellipsoid 3D giả khối u trong nhu mô não (*False Positives*) và khoét khuyết vùng u (*False Negatives*).
    - `'mixed'`: Kết hợp đồng thời nhiễu ranh giới và nhiễu vùng, tự động cưỡng bức thứ bậc giải phẫu BraTS ($\text{ET} \subseteq \text{TC} \subseteq \text{WT}$).
- **Chuyển đổi sang Soft Label liên tục $[0, 1]$ (`generate_soft_label_3d`, `ConvertHardToSoftLabeld`)**:
  - `'label_smoothing'`: Làm mềm đồng nhất $y_{\text{soft}} = (1 - \alpha)y + 0.5\alpha$.
  - `'gaussian_boundary'`: Làm mờ biên u bằng tích chập Gaussian 3D tách chiều theo độ lệch chuẩn $\sigma$.
  - `'sdf'`: Biến đổi khoảng cách Euclid có dấu $\text{SDF}(x) = d_{\text{trong}}(x) - d_{\text{ngoài}}(x)$ qua hàm Sigmoid $y_{\text{soft}}(x) = \sigma(\text{SDF}(x)/\tau)$, gán xác suất chuyển tiếp mượt quanh `0.5` ngay tại ranh giới thâm nhiễm của khối u.

![Minh họa Nhiễu nhãn & Nhãn mềm trên BraTS 2021](./outputs/softlabel_study/label_noise_and_soft_labels_demo.png)

### 2.3. Mở rộng Hàm Loss & Bộ Đo Hiệu Chuẩn (`src/loss_metric.py`)
- **4 hàm Loss mới chuyên biệt cho Soft Label & Label Noise**:
  - `SoftDiceLoss` (`soft_dice`): Công thức Dice liên tục với mẫu số chuẩn bậc hai $\sum p_i^2 + \sum y_i^2$, đạt cực tiểu toàn cục đúng tại $p_i = y_i \in [0, 1]$, bảo toàn hiệu chuẩn xác suất của nhãn mềm.
  - `SoftDiceBCELoss` (`soft_dice_bce`): Kết hợp `SoftDiceLoss` và `BCEWithLogitsLoss` tính độc lập trên từng kênh (`TC`, `WT`, `ET`), khắc phục triệt để xung đột Softmax cạnh tranh ngang giữa các vùng u chồng lấn.
  - `DiceSymmetricBCELoss` (`sce`): Kết hợp chiều thuận BCE và chiều nghịch Reverse-BCE (Wang et al., ICCV 2019) cùng `SoftDiceLoss` để ngăn mạng học thuộc lòng các voxel gán nhãn sai.
  - `DiceGeneralizedCELoss` (`gce`): Hàm mất mát kháng nhiễu tổng quát Box-Cox $\frac{1 - p_t^q}{q}$ (Zhang & Sabuncu, NeurIPS 2018), tự động giảm trọng số gradient của các mẫu nhiễu cực đoan.
- **Đo lường Hiệu chuẩn & Độ bất định (`compute_calibration_stats`, `BraTSMetric(track_calibration=True)`)**:
  - Bổ sung **Brier Score** $\mathbb{E}[(p - y)^2]$, **Expected Calibration Error (ECE)** 10-bin và **Entropy dự đoán 3D** bên cạnh bộ chỉ số **Dice (TC, WT, ET, Mean)**.

### 2.4. Cấu hình Presets, Pipeline, Kiểm thử & Công cụ Thực thi
- **Cấu hình & Pipeline** (`src/config.py`, `src/dataset.py`, `src/trainer.py`, `src/utils.py`):
  - Bổ sung hàm tạo cấu hình nhanh `get_softlabel_study_config()` cho 6 kịch bản chuẩn: `clean`, `noisy_baseline`, `noisy_soft_sdf`, `noisy_soft_gaussian`, `noisy_sce`, `noisy_gce`.
  - Đảm bảo nguyên tắc đánh giá nghiêm ngặt: **nhiễu và nhãn mềm chỉ áp dụng trên tập Train; tập Validation và Held-out Test luôn giữ nguyên nhãn sạch (Clean Ground Truth)**.
- **Kiểm thử tự động (`tests/test_noise_softlabel.py`)**:
  - Thêm 7 bài kiểm thử chuyên sâu (tính đơn điệu của nhiễu theo `noise_ratio`, tính tất định theo `patient_id`, tính bao hàm `ET <= TC <= WT`, miền giá trị nhãn mềm, gradient các hàm loss mới và chỉ số hiệu chuẩn). Toàn bộ **22/22 bài test** trong `tests/` đều vượt qua 100%.
- **Script & Notebook nghiên cứu**:
  - [`scripts/run_noise_softlabel_study.py`](./scripts/run_noise_softlabel_study.py): Script chạy tự động benchmark đa nhánh và xuất báo cáo JSON + biểu đồ.
  - [`train_softlabel_noise_study.ipynb`](./train_softlabel_noise_study.ipynb): Notebook tương tác chạy được trên cả môi trường Local (Mac MPS / Windows) và Google Colab.

---

## 3. Kết Quả Thực Nghiệm Đối Sánh Ban Đầu (Pilot Study)

### 3.1. Thiết lập thực nghiệm
- **Tập dữ liệu**: 20 ca bệnh nhân thực tế từ `BraTS-Mini`, chia chặt chẽ ở cấp bệnh nhân theo tỷ lệ `80% Train (16 ca) / 10% Val (2 ca) / 10% Held-out Test (2 ca)`.
- **Kiến trúc & Huấn luyện**: Standard **3D UNet** (`4,812,512` tham số, `18.36 MB`), kích thước patch `80×80×80`, `12 epochs`, tối ưu `AdamW (lr=1e-3)` + `CosineAnnealingLR`, chạy trên GPU Apple Silicon (`mps`).
- **Định lượng mức suy giảm nhãn huấn luyện theo tỷ lệ nhiễu hỗn hợp $\eta$** (đo trên `BraTS2021_00000`):

| Mức nhiễu ($\eta$) | Label Dice WT | Label Dice TC | Label Dice ET | Label Dice Mean | Tỷ lệ sai khác Voxel |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.10` | `0.9413` | `0.9487` | `0.8862` | `0.9254` | `0.189%` |
| `0.20` | `0.8943` | `0.9034` | `0.7863` | `0.8613` | `0.357%` |
| **`0.35` (Dùng trong Study)** | **`0.7602`** | **`0.7570`** | **`0.5545`** | **`0.6906`** | **`0.888%`** |
| `0.50` | `0.6784` | `0.6596` | `0.3898` | `0.5760` | `1.265%` |

### 3.2. Kết quả đối sánh 4 nhánh trên nhãn sạch độc lập
Số liệu chi tiết lưu tại [`outputs/softlabel_study/benchmark_summary.json`](./outputs/softlabel_study/benchmark_summary.json):

| Nhánh Thực Nghiệm | Nhiễu Train | Xử lý Nhãn & Hàm Loss | Best Val WT Dice | Best Val Mean Dice | Test WT Dice | Test Mean Dice | Test Brier $\downarrow$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. `clean`** | Không (`0.0`) | Hard GT + `dice_ce` | `0.2771` | `0.1146` | `0.0836` | `0.0308` | `0.2463` |
| **2. `noisy_baseline`** | Mixed (`0.35`) | Noisy Hard + `dice_ce` | `0.2303` | `0.0982` | `0.0704` | `0.0274` | `0.2449` |
| **3. `noisy_soft_sdf`** | Mixed (`0.35`) | **SDF Soft Label + `soft_dice_bce`** | `0.7085` | `0.3299` | `0.3963` | `0.1631` | **`0.0936`** |
| **4. `noisy_sce`** | Mixed (`0.35`) | **Label Smoothing + `sce`** | **`0.7351`** | **`0.3481`** | **`0.4308`** | **`0.1788`** | `0.1073` |

![Biểu đồ đối sánh kết quả 4 nhánh thực nghiệm](./outputs/softlabel_study/study_comparison.png)

### 3.3. Các nhận xét & phát hiện chính
1. **Nhiễu nhãn làm suy giảm rõ rệt mô hình dùng loss chuẩn (`clean` $\rightarrow$ `noisy_baseline`)**: Khi dùng cùng hàm loss `dice_ce`, nhiễu nhãn $\eta = 0.35$ làm giảm **14.3% Mean Dice** trên tập Validation (`0.1146` $\rightarrow$ `0.0982`) và giảm **11.0% Mean Dice** trên tập Held-out Test.
2. **Nhãn mềm SDF (`noisy_soft_sdf`) tối ưu hóa hiệu chuẩn xác suất và phản ánh đúng bất định biên u**:
   - Đạt **Brier Score thấp nhất (`0.0936`)**, giảm 62% sai số xác suất so với baseline `dice_ce`, đồng thời đẩy Validation WT Dice lên **`0.7085`**.
   - Bản đồ bất định Entropy bên dưới cho thấy mô hình tự tin cao ở lõi khối u (Entropy $\approx 0$, màu tối) và tập trung dải bất định cao (màu vàng sáng) bám sát ranh giới thâm nhiễm của khối u.
3. **Loss kháng nhiễu đối xứng `sce` + Label Smoothing (`noisy_sce`) đạt độ chính xác phân đoạn cao nhất**: Đạt **Val WT Dice `0.7351`** (`Val Mean Dice 0.3481`) và **Held-out Test WT Dice `0.4308`** ngay cả khi học trên nhãn huấn luyện bị làm hỏng 31% (`Mean Label Dice = 0.6906`).
4. **Phát hiện giới hạn toán học của `DiceCELoss(sigmoid=True)` mặc định trong MONAI cho BraTS**: Với đầu ra 3 kênh chồng lấn (`TC`, `WT`, `ET`), nhánh Cross-Entropy mặc định của `DiceCELoss` áp dụng chuẩn hóa **Softmax ngang qua 3 kênh**, gây triệt tiêu gradient ở các vùng giao nhau (nơi cả 3 kênh cùng bằng `1`). Chuyển sang BCE độc lập từng kênh kết hợp `SoftDiceLoss` chuẩn $L_2$ (`soft_dice_bce` / `sce`) giúp tốc độ hội tụ tăng vọt.

![Bản đồ dự đoán và độ bất định biên u của mô hình SDF Soft Label](./outputs/softlabel_study/uncertainty_noisy_soft_sdf.png)

---

## 4. Cấu Trúc Các File Mới & Đã Nâng Cấp

```text
.
├── dataset/
│   └── BraTS2021_Training_Data/          # 30 ca MRI 3D BraTS 2021 thực tế đã tải sẵn
├── scripts/
│   ├── prepare_brats_mini.py             # Script stream BraTS-Mini hoặc tạo dữ liệu giả lập
│   └── run_noise_softlabel_study.py      # Script chạy tự động thực nghiệm đối sánh đa nhánh
├── src/
│   ├── noise_soft_labels.py              # [MỚI] Module tạo Label Noise 3D & Soft Labels [0, 1]
│   ├── loss_metric.py                    # [NÂNG CẤP] Thêm SoftDice, SoftDiceBCE, SCE, GCE, ECE, Brier
│   ├── config.py                         # [NÂNG CẤP] Thêm tham số noise/soft-label & study presets
│   ├── dataset.py                        # [NÂNG CẤP] Tích hợp biến đổi nhiễu/nhãn mềm vào tập Train
│   ├── trainer.py                        # [NÂNG CẤP] Theo dõi thêm chỉ số hiệu chuẩn ECE & Brier
│   └── utils.py                          # [NÂNG CẤP] Vẽ mặt cắt nhãn mềm & bản đồ bất định Entropy
├── tests/
│   └── test_noise_softlabel.py           # [MỚI] Kiểm thử đơn vị & tích hợp cho Noise / Soft Label
├── outputs/
│   └── softlabel_study/                  # Biểu đồ EDA, Demo nhãn mềm, Uncertainty & JSON tổng kết
├── train_softlabel_noise_study.ipynb     # [MỚI] Notebook tương tác nghiên cứu Soft Label & Noise
├── DATASETS.md                           # [MỚI] Đặc tả 3 tầng dataset, quy trình chuyên gia & ý nghĩa mô phỏng nhiễu
├── EXPERIMENTAL_REPORT.md                # [MỚI] Báo cáo phân tích định lượng, định tính & toán học thực nghiệm
├── CODE_FLOW.md                          # [MỚI] Tài liệu sơ đồ kiến trúc & giải thích luồng code
└── README_SOFTLABEL_NOISE.md             # [MỚI] Báo cáo tổng hợp công việc & hướng dẫn tái lập
```

---

## 5. Hướng Dẫn Chạy Lại & Mở Rộng Thực Nghiệm

### Bước 1: Kích hoạt môi trường ảo đã cài đặt sẵn
```bash
source .env/bin/activate
```

### Bước 2: Chuẩn bị / Mở rộng số ca dữ liệu `BraTS-Mini`
```bash
# Mặc định đã có sẵn 30 ca; có thể tăng lên 50 ca nếu ổ cứng còn trống:
python scripts/prepare_brats_mini.py --num-cases 30 --mode stream
```

### Bước 3: Chạy bộ kiểm thử tự động (22 bài test)
```bash
pytest tests/ -v
```

### Bước 4: Chạy thực nghiệm khảo sát qua dòng lệnh (CLI)
```bash
# Chạy đối sánh mặc định 4 nhánh (clean, noisy_baseline, noisy_soft_sdf, noisy_sce)
python -u scripts/run_noise_softlabel_study.py \
    --max-samples 20 \
    --epochs 12 \
    --val-interval 4 \
    --roi-size 80 \
    --noise-ratio 0.35

# Hoặc thử nghiệm thêm Gaussian Soft Label và Generalized Cross-Entropy (GCE) với 30 ca & 30 epochs:
python -u scripts/run_noise_softlabel_study.py \
    --max-samples 30 \
    --epochs 30 \
    --val-interval 5 \
    --roi-size 96 \
    --noise-ratio 0.30 \
    --presets clean noisy_baseline noisy_soft_gaussian noisy_soft_sdf noisy_sce noisy_gce
```

### Bước 5: Chạy trên Jupyter Notebook / Google Colab
Mở file [`train_softlabel_noise_study.ipynb`](./train_softlabel_noise_study.ipynb) để tương tác trực tiếp với từng bước biến đổi nhiễu, điều chỉnh tham số làm mềm ($\sigma$, $\alpha$) và quan sát bản đồ bất định trên từng lát cắt MRI.

