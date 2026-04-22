# HandEffTrans: Transformer-Based Fusion for Lung & Colon Cancer Classification

> **Transformer-based fusion of handcrafted and deep features with incremental learning on the LC25000 histopathology dataset.**

---

## 📌 Overview

**HandEffTrans** is a 4-phase pipeline for classifying lung and colon cancer histopathology images into **5 classes** using a hybrid feature fusion approach:

1. **Phase 1** — Classical handcrafted feature extraction (LBP, GLCM, Color, Wavelet, Morphological)
2. **Phase 2** — Deep feature extraction using partially fine-tuned **EfficientNetB0**
3. **Phase 3** — Feature fusion via a **Transformer encoder** (Multi-Head Self-Attention)
4. **Phase 4** — **Incremental learning** with 5-fold cross-validation

---

## ✨ Key Contributions

- Hybrid fusion of handcrafted + deep features using Transformer encoder
- 4-stage incremental learning strategy to reduce catastrophic forgetting
- EfficientNetB0 fine-tuning with partial layer freezing
- Achieves ~99.8% accuracy on LC25000 dataset with strong generalization
- End-to-end reproducible pipeline with cross-validation support

---

## 🗂️ Dataset

**LC25000 Lung and Colon Histopathological Image Dataset**

| Class | Type |
|-------|------|
| `colon_aca` | Colon Adenocarcinoma |
| `colon_n` | Colon Benign (Normal) |
| `lung_aca` | Lung Adenocarcinoma |
| `lung_n` | Lung Benign (Normal) |
| `lung_scc` | Lung Squamous Cell Carcinoma |

- Image size: **224 × 224** (resized)
- Total images: **25,000** (5,000 per class)

> Download the dataset from [Kaggle – LC25000](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

---

## 🏗️ Project Structure

```
project/
│
├── main.py                        # All 4 phases in one file
│
├── results/
│   ├── csv/
│   │   ├── handcrafted_features.csv   # Phase 1 output
│   │   ├── deep_features.csv          # Phase 2 output
│   │   ├── fold_results.csv           # Phase 4 per-fold metrics
│   │   └── incremental_metrics.csv    # Phase 4 stage-wise metrics
│   │
│   ├── graphs/
│   │   ├── phase1_feature_distribution.png
│   │   ├── fold_N_results.png         # Accuracy, Loss, ROC, Confusion Matrix
│   │   └── fold_N_incremental.png     # Kappa & Accuracy vs stage
│   │
│   ├── models/
│   │   └── fold_N_best.weights.h5     # Best weights per fold
│   │
│   ├── reports/
│   │   └── fold_N_report.txt          # Per-fold classification report
│   │
│   └── summary/
│       ├── cv_summary.csv             # 5-fold cross-validation summary
│       └── cv_summary.png             # Summary bar chart
```

---

## ⚙️ Requirements

```bash
pip install opencv-python numpy pandas scikit-learn scikit-image
pip install tensorflow pywavelets matplotlib seaborn tqdm
```

> Tested with Python 3.9+, TensorFlow 2.10+

---

## 🚀 Usage

### Step 1 — Update Dataset Path

In `main.py`, set your local path:

```python
DATASET_PATH = r"path/to/LC25000/lung_colon_image_set"
BASE_OUT     = r"path/to/results"
```

### Step 2 — Run Phase 1 (Handcrafted Features)

```bash
python main.py
```

Extracts LBP, GLCM, Color (RGB + HSV), Wavelet (DWT), and Morphological features.  
Output: `results/csv/handcrafted_features.csv` + feature distribution plot.

### Step 3 — Run Phase 2 (Deep Features)

Runs automatically after Phase 1.  
Fine-tunes last 20 layers of EfficientNetB0 on up to 200 images/class.  
Output: `results/csv/deep_features.csv`

### Step 4 — Run Phase 3 + 4 (Fusion + Training)

```bash
# Run a single fold
python main.py --fold 1

# Run all 5 folds
python main.py --fold all

# Show cross-fold summary (after running folds)
python main.py --summary
```

---

## 🧠 Model Architecture — HandEffTrans

```
Handcrafted Features (62-dim)        Deep Features (1280-dim)
         │                                    │
    Dense(1024) + BN                    Dense(1024) + BN
    Reshape(8 × 128)                    Reshape(8 × 128)
         └──────────────┬───────────────────┘
                        │  Concatenate (16 × 128)
                        │
              ┌─────────▼──────────┐
              │  Transformer Block  │ × 4
              │  (8-head Attention) │
              └─────────┬──────────┘
                        │
              GlobalAveragePooling1D
                        │
              Dense(1024) + BN + Dropout(0.4)
              Dense(512)  + BN + Dropout(0.3)
                        │
              Dense(5, softmax)
```

---

## 📊 Results

All experimental outputs (graphs, reports, models) are available here:

👉 https://github.com/Parth-here/Lung_Colon_Cancer_Results

## 📈 Sample Output

<p align="center">
  <img src="https://raw.githubusercontent.com/Parth-here/Lung_Colon_Cancer_Results/main/graphs/fold_5_results.png" width="500"/>
</p>

## 📊 Per-Fold Test Accuracy (5-Fold CV)

| Fold | Accuracy | Precision | Recall | F1 Score | Kappa |
|------|----------|-----------|--------|----------|--------|
| 1    | 99.66%   | 0.9966    | 0.9966 | 0.9966   | 0.9958 |
| 2    | 99.92%   | 0.9992    | 0.9992 | 0.9992   | 0.9990 |
| 3    | 99.88%   | 0.9988    | 0.9988 | 0.9988   | 0.9985 |
| 4    | 99.78%   | 0.9978    | 0.9978 | 0.9978   | 0.9972 |
| 5    | 99.86%   | 0.9986    | 0.9986 | 0.9986   | 0.9982 |
| AVG  | 99.82%   | 0.9982    | 0.9982 | 0.9982   | 0.9977 |

> Fill in your actual values after running `--summary`. Each fold also generates a per-class classification report in `results/reports/fold_N_report.txt`.

### Generated Plots (per fold)

| Plot | Description |
|------|-------------|
| `fold_N_results.png` | Training/Validation Accuracy, Loss, ROC Curves, Confusion Matrix |
| `fold_N_incremental.png` | Kappa & Accuracy across HandEffTrans-0 → HandEffTrans-3 stages |
| `phase1_feature_distribution.png` | Mean feature values per class for all 5 feature groups |
| `cv_summary.png` | Cross-fold bar chart (Accuracy, F1, Kappa, Sensitivity) |

---

## 🔬 Feature Details

| Feature | Method | Dimensionality |
|---------|--------|----------------|
| LBP | Local Binary Pattern (uniform, r=1, n=8) | 10 |
| GLCM | Gray-Level Co-occurrence Matrix (4 angles) | 16 |
| Color | RGB + HSV mean, std, max, min per channel | 24 |
| Wavelet | Haar DWT (mean + std of 4 subbands) | 8 |
| Morphological | Largest region: area, eccentricity, solidity, extent | 4 |
| **Total** | | **62** |

---

## 🔄 Incremental Learning Strategy

Training is split into **4 stages**, each using 1,000 images/class:

| Stage | Name | Learning Rate | Loss | Epochs |
|-------|------|--------------|------|--------|
| 1 | HandEffTrans-0 (Initial) | 3e-4 | Cross-Entropy | 20 |
| 2 | HandEffTrans-1 | 1e-4 | CE + Label Smoothing (0.05) | 10 |
| 3 | HandEffTrans-2 | 5e-5 | CE + Label Smoothing (0.05) | 10 |
| 4 | HandEffTrans-3 | 2e-5 | CE + Label Smoothing (0.05) | 10 |

Label smoothing in later stages reduces catastrophic forgetting.

---

## 📝 Notes

- **Run order matters**: Phase 1 and Phase 2 must complete before Phase 3+4 (their CSVs are required).
- **Label alignment**: Both CSVs must have images in the same order. The code asserts this.
- **GPU recommended**: EfficientNetB0 fine-tuning and Transformer training are significantly faster on GPU.
- **Reproducibility**: Random seed is fixed at `42` for NumPy and TensorFlow.

---

## 📄 Citation

If you use this code, please cite the LC25000 dataset:

> Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. *Lung and Colon Cancer Histopathological Image Dataset (LC25000)*. arXiv:1912.12378v1 [eess.IV], 2019.

---

- **Project**: HandEffTrans — Handcrafted + EfficientNet + Transformer Fusion
- **Dataset**: LC25000 (Lung & Colon Histopathology)
- **Framework**: TensorFlow / Keras, Scikit-learn, OpenCV
