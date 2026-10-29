# BraTS 2021 3D Brain Tumor Segmentation Baselines: UNet & nnU-Net

This repository contains two complete, production-ready baseline pipelines for 3D multi-modal brain tumor segmentation on the **BraTS 2021** (Brain Tumor Segmentation Challenge 2021) dataset using **MONAI** and **PyTorch**:
1. **Standard 3D UNet Baseline** (`monai.networks.nets.UNet`)
2. **nnU-Net Baseline** (`monai.networks.nets.DynUNet` - the official MONAI implementation of nnU-Net)

Both baselines follow a modular software design with clean Python modules (`src/`), comprehensive smoke test suites (`tests/`), and dual-environment branching notebooks with zero hardcoded paths.

---

## 1. Overview of Baselines

### Input Modalities
Each patient volume consists of 4 co-registered and skull-stripped MRI sequences:
- **FLAIR**: Fluid Attenuated Inversion Recovery
- **T1**: T1-weighted
- **T1ce**: T1-weighted Contrast Enhanced (Gadolinium)
- **T2**: T2-weighted

The 4 modalities are stacked as 4 input channels: shape `(4, H, W, D)`.

### Target Sub-regions (BraTS Standard)
BraTS 2021 raw labels are:
- `0`: Background
- `1`: Necrotic Tumor Core (NCR)
- `2`: Peritumoral Edematous Tissue (ED)
- `4`: GD-Enhancing Tumor (ET)

Following the standard MICCAI BraTS competition evaluation, raw labels are converted into 3 overlapping binary targets:
- **Whole Tumor (WT)**: Labels 1 + 2 + 4 (entire tumor extent)
- **Tumor Core (TC)**: Labels 1 + 4 (necrotic core + enhancing tumor)
- **Enhancing Tumor (ET)**: Label 4 (actively enhancing tumor region)

Each channel is predicted independently using Sigmoid activation.

### Patient-Level Dataset Splitting (80% / 10% / 10%)
To prevent data leakage across slices, all 1,251 BraTS 2021 cases are split strictly at the **patient level** (`BraTS2021_xxxxx`):
- **80% Training Set (~1,000 cases)**: Used to compute gradients and update network parameters.
- **10% Validation Set (~125 cases)**: Evaluated during training to tune hyperparameters and save the best checkpoint (`best_metric_model.pth`).
- **10% Held-Out Test Set (~126 cases)**: Kept completely separate during training and hyperparameter tuning, evaluated only once for the final unbiased performance benchmark.

### Baseline Comparison

| Feature | Standard 3D UNet Baseline | nnU-Net Baseline (`DynUNet`) |
| :--- | :--- | :--- |
| **Model Class** | `monai.networks.nets.UNet` | `monai.networks.nets.DynUNet` |
| **Normalization** | Batch Normalization (`BatchNorm`) | Instance Normalization (`InstanceNorm`) |
| **Activation** | ReLU | LeakyReLU (`negative_slope = 0.01`) |
| **Encoder Stages** | 5 levels `(16, 32, 64, 128, 256)` | 5 dynamic stages `(16, 32, 64, 128, 256)` |
| **Downsampling** | Stride 2 at all levels | Stage-wise matched strides & kernels |
| **Loss Function** | `DiceCELoss(sigmoid=True)` | `DiceCELoss(sigmoid=True)` |
| **Optimizer** | `AdamW` (lr=1e-4, weight_decay=1e-5) | `AdamW` or `SGD` (Nesterov momentum=0.99) |
| **Scheduler** | Cosine Annealing | Cosine Annealing / PolyLR |
| **Evaluation** | Mean Dice over TC, WT, ET | Mean Dice over TC, WT, ET |
| **Inference** | Sliding Window (Gaussian overlap 0.5) | Sliding Window (Gaussian overlap 0.5) |
| **Notebook** | `train_baseline.ipynb` | `train_nnunet_baseline.ipynb` |

---

## 2. Directory Structure

```
Segmentation/
|-- dataset/
|   `-- BraTS2021_Training_Data/       # BraTS patient folders (BraTS2021_00000, ...)
|-- src/
|   |-- __init__.py
|   |-- config.py                      # BaselineConfig dataclass, device auto-detection (CUDA/MPS/CPU)
|   |-- eda.py                         # Dataset scanning, statistics, and slice visualization
|   |-- dataset.py                     # MONAI transforms, dataset split, and DataLoaders
|   |-- model.py                       # UNet and nnU-Net (DynUNet) factories, parameter counters
|   |-- loss_metric.py                 # DiceCELoss and BraTSMetric (TC, WT, ET, Mean Dice)
|   |-- trainer.py                     # BraTSTrainer with AMP and sliding window validation
|   `-- utils.py                       # Seed, checkpointing, and prediction visualization
|-- tests/
|   |-- __init__.py
|   |-- test_smoke.py                  # Smoke tests for standard UNet pipeline
|   `-- test_smoke_nnunet.py           # Smoke tests for nnU-Net (DynUNet) pipeline
|-- checkpoints/                       # Saved model checkpoints
|   `-- nnunet/                        # Dedicated directory for nnU-Net checkpoints
|-- outputs/                           # Generated EDA plots, learning curves, and prediction visuals
|   `-- nnunet/                        # Dedicated directory for nnU-Net figures
|-- train_baseline.ipynb               # Master notebook for 3D UNet baseline
|-- train_nnunet_baseline.ipynb        # Master notebook for nnU-Net baseline
|-- requirements.txt                   # Project dependencies
`-- README.md                          # Documentation
```

---

## 3. How to Reproduce on Windows

### Prerequisites
- Python 3.10, 3.11, or 3.12 (64-bit)
- Git
- NVIDIA GPU with CUDA drivers (optional, falls back to CPU automatically)

### Step 1: Clone or Navigate to the Repository
```powershell
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```
*(Or navigate into your local repository root folder)*

### Step 2: Set Up Virtual Environment
```powershell
# Create virtual environment
python -m venv .env

# Activate virtual environment
.\.env\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

### Step 3: Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Dataset Configuration (Optional)
The pipeline automatically locates `dataset/BraTS2021_Training_Data` relative to the repository.
If your dataset is stored in an external location, set the environment variable:
```powershell
$env:BRATS_DATA_DIR = "path\to\BraTS2021_Training_Data"
```

### Step 5: Run Smoke Tests
Execute the smoke test suites to verify that all modules, transformations, model forward passes, loss functions, and sliding window inference run with zero errors:
```powershell
# 1. Run standard UNet smoke tests (9 tests)
python tests/test_smoke.py

# 2. Run nnU-Net (DynUNet) smoke tests (6 tests)
python tests/test_smoke_nnunet.py
```
Both test suites will execute and display passing status in under 10 seconds.

### Step 6: Launch Baseline Notebooks
```powershell
jupyter lab
```
Open either `train_baseline.ipynb` (UNet) or `train_nnunet_baseline.ipynb` (nnU-Net).

---

## 4. How to Reproduce on macOS (Apple Silicon / Intel)

### Prerequisites
- macOS 12 Monterey or newer
- Python 3.10, 3.11, or 3.12 (installed via Homebrew or official installer)
- Apple Silicon (M1/M2/M3/M4) models automatically utilize **Metal Performance Shaders (`mps`)** GPU acceleration.

### Step 1: Clone or Navigate to the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```
*(Or navigate into your local repository root folder)*

### Step 2: Set Up Virtual Environment
```bash
# Create virtual environment
python3 -m venv .env

# Activate virtual environment
source .env/bin/activate
```

### Step 3: Install Dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Verify Device Acceleration
The configuration module in `src/config.py` automatically detects Apple Silicon hardware:
```bash
python -c "import torch; print('Device:', 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')"
```
If running on an Apple Silicon Mac, this will print `Device: mps`. On Intel Macs, it will fall back to `cpu`.

### Step 5: Dataset Configuration (Optional)
If your dataset is located outside the repository workspace, export the environment variable:
```bash
export BRATS_DATA_DIR="path/to/BraTS2021_Training_Data"
```

### Step 6: Run Smoke Tests
```bash
# Run standard UNet smoke tests
python tests/test_smoke.py

# Run nnU-Net (DynUNet) smoke tests
python tests/test_smoke_nnunet.py
```

### Step 7: Launch Baseline Notebooks
```bash
jupyter lab
```
Open `train_baseline.ipynb` or `train_nnunet_baseline.ipynb`.

---

## 5. Training on Google Colab with GPU

Both notebooks are equipped with automatic environment detection and zero hardcoded paths:

1. **Upload the Project to Google Drive**:
   Upload the repository folder (or clone via GitHub) into your Google Drive.
2. **Open the Notebook in Colab**:
   Open either `train_baseline.ipynb` or `train_nnunet_baseline.ipynb`.
3. **Select GPU Hardware Accelerator**:
   Navigate to `Runtime` -> `Change runtime type` -> Select `T4 GPU`, `V100`, or `A100`.
4. **Run Cells Sequentially**:
   - **Section 1 (Colab Branch)**: Automatically installs `monai`, `nibabel` and mounts Google Drive.
   - **Section 2 (Dynamic Project Root)**: Automatically discovers the repository root across Google Drive or Colab storage and adds `src` to `sys.path`.
   - **Section 3 (Dynamic Dataset Discovery)**: Dynamically scans Google Drive and Colab storage for the BraTS dataset without hardcoding paths. You can also specify `CUSTOM_DATASET_PATH` if needed.
   - **Section 4 to 6**: Runs Exploratory Data Analysis, sets up DataLoaders with patient-level 80/10/10 split, instantiates the architecture, and performs a 1-batch sanity check.
   - **Section 7 to 8**: Executes `trainer.fit(...)`. Checkpoints are automatically saved based on highest validation Mean Dice score.
   - **Section 9 to 10**: Plots training loss and validation Dice curves across epochs, and produces qualitative visual comparisons on validation cases.
   - **Section 11**: Evaluates the best trained model on the held-out 10% test set (~126 cases) and reports final unbiased benchmark Dice metrics.
