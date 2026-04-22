# =============================================================================
# PHASE 1 - HANDCRAFTED FEATURE EXTRACTION
# Run: python phase1.py
# =============================================================================

import os
import cv2
import numpy as np
import pandas as pd
import pywt
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import label, regionprops
from sklearn.preprocessing import LabelEncoder
import glob

import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# PATHS
# =============================================================================
DATASET_PATH = r"D:\_6th sem\seminar\LC25000\lung_colon_image_set"
BASE_OUT     = r"D:\_6th sem\seminar\results"
CSV_DIR      = os.path.join(BASE_OUT, "csv")
GRAPH_DIR    = os.path.join(BASE_OUT, "graphs")
OUTPUT_CSV   = os.path.join(CSV_DIR,  "handcrafted_features.csv")
IMG_SIZE     = (224, 224)

CLASS_NAMES = [
    "colon_image_sets/colon_aca",
    "colon_image_sets/colon_n",
    "lung_image_sets/lung_aca",
    "lung_image_sets/lung_n",
    "lung_image_sets/lung_scc"
]

for d in [CSV_DIR, GRAPH_DIR]:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# FEATURE EXTRACTORS
# =============================================================================

def extract_lbp_features(gray_image, num_points=8, radius=1):
    lbp = local_binary_pattern(gray_image, num_points, radius, method='uniform')
    n_bins = num_points + 2
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                                range=(0, n_bins), density=True)
    return lbp_hist
#Compares each pixel with neighbors
#Captures micro-texture patterns

def extract_glcm_features(gray_image):
    gray_uint8   = (gray_image * 255).astype(np.uint8) \
                   if gray_image.max() <= 1.0 else gray_image.astype(np.uint8)
    gray_reduced = (gray_uint8 // 16).astype(np.uint8)
    distances    = [1]
    angles       = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray_reduced, distances=distances, angles=angles,
                        levels=16, symmetric=True, normed=True)
    contrast      = graycoprops(glcm, 'contrast').ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    energy        = graycoprops(glcm, 'energy').ravel()
    correlation   = graycoprops(glcm, 'correlation').ravel()
    return np.concatenate([contrast, dissimilarity, energy, correlation])
#Captures macro-texture patterns

def extract_color_features(rgb_image):
    features = []
    for c in range(3):
        ch = rgb_image[:, :, c].astype(np.float32)
        features += [np.mean(ch), np.std(ch), np.max(ch), np.min(ch)]
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    for c in range(3):
        ch = hsv[:, :, c].astype(np.float32)
        features += [np.mean(ch), np.std(ch), np.max(ch), np.min(ch)]
    return np.array(features)
#Extracts color features from RGB and HSV images

def extract_wavelet_features(gray_image):
    coeffs        = pywt.dwt2(gray_image, 'haar')
    cA, (cH, cV, cD) = coeffs
    features = []
    for sb in [cA, cH, cV, cD]:
        features += [np.mean(np.abs(sb)), np.std(sb)]
    return np.array(features)
#Extracts wavelet features from grayscale image

def extract_morphological_features(gray_image):
    gray_uint8 = (gray_image * 255).astype(np.uint8) \
                 if gray_image.max() <= 1.0 else gray_image.astype(np.uint8)
    _, binary  = cv2.threshold(gray_uint8, 127, 255, cv2.THRESH_BINARY)
    labeled    = label(binary)
    regions    = regionprops(labeled)
    if len(regions) == 0:
        return np.zeros(4)
    lr = max(regions, key=lambda r: r.area)
    return np.array([lr.area, lr.eccentricity, lr.solidity, lr.extent])
#Extracts morphological features from grayscale image

def extract_all_features(image_path):
    img_bgr  = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read: {image_path}")
    img_bgr  = cv2.resize(img_bgr, IMG_SIZE)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

    lbp   = extract_lbp_features(img_gray)
    glcm  = extract_glcm_features(img_gray)
    color = extract_color_features(img_rgb)
    wav   = extract_wavelet_features(img_gray)
    morph = extract_morphological_features(img_gray)
    return np.concatenate([lbp, glcm, color, wav, morph])
#Extracts all features from image

# =============================================================================
# PROCESS DATASET
# =============================================================================

def process_dataset():
    all_features, all_labels = [], []

    print("\n" + "="*60)
    print("  PHASE 1 - HANDCRAFTED FEATURE EXTRACTION")
    print("="*60)

    for class_name in CLASS_NAMES:
        folder = os.path.join(DATASET_PATH, class_name)
        if not os.path.exists(folder):
            print(f"  [WARNING] Not found: {folder}")
            continue

        short_name = os.path.basename(class_name)
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))]
        print(f"\n  {short_name:<20s}  {len(files)} images")

        for f in tqdm(files, desc=f"  {short_name}", ncols=70):
            try:
                feat = extract_all_features(os.path.join(folder, f))
                all_features.append(feat)
                all_labels.append(short_name)
            except Exception as e:
                print(f"  [SKIP] {f}: {e}")

    col_names = (
        [f"lbp_{i}"   for i in range(10)] +
        [f"glcm_{i}"  for i in range(16)] +
        [f"color_{i}" for i in range(24)] +
        [f"wav_{i}"   for i in range(8)]  +
        ["morph_area", "morph_eccentricity", "morph_solidity", "morph_extent"]
    )

    df = pd.DataFrame(all_features, columns=col_names)
    df.insert(0, "label", all_labels)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  Saved -> {OUTPUT_CSV}")
    print(f"  Images processed : {len(all_labels)}")
    print(f"  Features per image : {len(col_names)}")
    return df

#Visualizes the distribution of handcrafted features
# =============================================================================
# VISUALIZE
# =============================================================================

def visualize_features(df):
    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df['label'])
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    feature_groups = {
        "LBP (Texture)"  : [c for c in df.columns if c.startswith("lbp")],
        "GLCM (Texture)" : [c for c in df.columns if c.startswith("glcm")],
        "Color RGB/HSV"  : [c for c in df.columns if c.startswith("color")],
        "Wavelet (DWT)"  : [c for c in df.columns if c.startswith("wav")],
        "Morphological"  : [c for c in df.columns if c.startswith("morph")],
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Phase 1 - Handcrafted Feature Distribution",
                 fontsize=14, fontweight='bold')

    for ax, (name, cols) in zip(axes.ravel(), feature_groups.items()):
        means = df.groupby('label')[cols].mean()
        for i, (cls, row) in enumerate(means.iterrows()):
            ax.plot(row.values, label=cls, color=colors[i], lw=1.5)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Mean Value")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    ax = axes[1][2]
    counts = df['label'].value_counts()
    ax.bar(counts.index, counts.values, color=colors, edgecolor='black')
    ax.set_title("Class Distribution", fontweight='bold')
    ax.set_ylabel("Images")
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(GRAPH_DIR, "phase1_feature_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved -> {out}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    df = process_dataset()
    visualize_features(df)
    print("\n  PHASE 1 COMPLETE")
    print(f"  CSV  -> {OUTPUT_CSV}")
    print(f"  Plot -> {GRAPH_DIR}")





# =============================================================================
# PHASE 2 - DEEP FEATURE EXTRACTION WITH PARTIAL FINE-TUNING
# Run: python phase2.py
# =============================================================================

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings("ignore")

DATASET_PATH = r"D:\_6th sem\seminar\LC25000\lung_colon_image_set"
BASE_DIR     = r"D:\_6th sem\seminar\results"
CSV_DIR      = os.path.join(BASE_DIR, "csv")
OUTPUT_CSV   = os.path.join(CSV_DIR, "deep_features.csv")
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
FINETUNE_EPOCHS = 5

CLASS_NAMES = [
    "colon_image_sets/colon_aca",
    "colon_image_sets/colon_n",
    "lung_image_sets/lung_aca",
    "lung_image_sets/lung_n",
    "lung_image_sets/lung_scc"
]

os.makedirs(CSV_DIR, exist_ok=True)


# =============================================================================
# STEP 1: BUILD MODEL - FREEZE ALL, THEN UNFREEZE LAST 20 LAYERS
# =============================================================================

def build_feature_extractor():
    base = EfficientNetB0(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))

    # Stage 1: freeze entire base
    for layer in base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    model = Model(inputs=base.input, outputs=x)
    return base, model


def unfreeze_and_finetune(base, model, X_sample, y_sample):
    # Stage 2: unfreeze last 20 layers for domain adaptation
    for layer in base.layers[-20:]:
        layer.trainable = True

    # Attach a small classification head for fine-tuning
    x      = model.output
    x      = Dense(128, activation='relu')(x)
    x      = Dropout(0.3)(x)
    out    = Dense(len(CLASS_NAMES), activation='softmax')(x)
    ft_model = Model(inputs=model.input, outputs=out)

    ft_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(patience=2, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=1, verbose=0)
    ]

    print("\n  [Phase 2] Fine-tuning last 20 EfficientNetB0 layers ...")
    ft_model.fit(
        X_sample, y_sample,
        epochs=FINETUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Return feature extractor (without the classification head)
    feature_model = Model(inputs=ft_model.input,
                          outputs=ft_model.layers[-3].output)
    return feature_model


# =============================================================================
# STEP 2: LOAD A SUBSET FOR FINE-TUNING WITH AUGMENTATION
# =============================================================================

def load_sample_for_finetuning():
    print("\n  [Phase 2] Loading sample images for fine-tuning ...")
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.1,
        preprocessing_function=preprocess_input
    )

    X, y = [], []
    for cls_idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(DATASET_PATH, cls)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Use up to 200 images per class for fine-tuning
        files = files[:200]
        for f in files:
            img = cv2.imread(os.path.join(folder, f))
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(cls_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # Apply one pass of augmentation to double the training set
    X_aug, y_aug = [], []
    for i in range(len(X)):
        aug = datagen.random_transform(X[i])
        X_aug.append(preprocess_input(aug))
        y_aug.append(y[i])

    X_orig = preprocess_input(X.copy())
    X_all  = np.concatenate([X_orig, np.array(X_aug)], axis=0)
    y_all  = np.concatenate([y, np.array(y_aug)], axis=0)

    print(f"  Fine-tune set: {len(X_all)} images across {len(CLASS_NAMES)} classes")
    return X_all, y_all


# =============================================================================
# STEP 3: EXTRACT DEEP FEATURES USING FINE-TUNED MODEL
# =============================================================================

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)


def extract_deep_features():
    print("\n" + "="*60)
    print("  PHASE 2 - DEEP FEATURE EXTRACTION (PARTIAL FINE-TUNING)")
    print("="*60)

    base, frozen_model = build_feature_extractor()
    print(f"  EfficientNetB0 loaded  |  total layers: {len(base.layers)}")
    print(f"  Stage 1: all layers frozen")

    X_ft, y_ft     = load_sample_for_finetuning()
    feature_model  = unfreeze_and_finetune(base, frozen_model, X_ft, y_ft)
    print(f"\n  Stage 2: last 20 layers fine-tuned on histopathology data")

    features, labels = [], []
    print("\n  Extracting deep features from all images ...")

    for cls in CLASS_NAMES:
        folder = os.path.join(DATASET_PATH, cls)
        files  = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in tqdm(files, desc=f"  {os.path.basename(cls)}", ncols=70):
            path = os.path.join(folder, f)
            try:
                x    = preprocess_image(path)
                feat = feature_model.predict(x, verbose=0).flatten()
                features.append(feat)
                labels.append(os.path.basename(cls))
            except Exception as e:
                print(f"  [SKIP] {f}: {e}")

    df = pd.DataFrame(features)
    df.insert(0, "label", labels)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  Saved -> {OUTPUT_CSV}")
    print(f"  Images processed : {len(labels)}")
    print(f"  Feature dims     : {len(features[0]) if features else 0}")
    print("\n  PHASE 2 COMPLETE")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    extract_deep_features()



# =============================================================================
# PHASE 3 + PHASE 4  — Transformer Fusion + Incremental Learning

# Usage:
#   python phase34.py --fold 1       # run fold 1 only
#   python phase34.py --fold all     # run all 5 folds
#   python phase34.py --summary      # cross-fold summary
# =============================================================================

import os, sys, re, csv, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Reshape, Concatenate,
    MultiHeadAttention, LayerNormalization, Add,
    BatchNormalization, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report,
    cohen_kappa_score
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# OUTPUT PATHS
# =============================================================================
BASE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CSV_DIR     = os.path.join(BASE_DIR, "csv")
GRAPH_DIR   = os.path.join(BASE_DIR, "graphs")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
REPORT_DIR  = os.path.join(BASE_DIR, "reports")
SUMMARY_DIR = os.path.join(BASE_DIR, "summary")

for _d in [CSV_DIR, GRAPH_DIR, MODEL_DIR, REPORT_DIR, SUMMARY_DIR]:
    os.makedirs(_d, exist_ok=True)


# =============================================================================
# HYPER-PARAMETERS
# =============================================================================

IMGS_PER_CLASS_PER_STAGE = 1000   # images per class per training stage

N_STAGES           = 4            # 1 initial + 3 incremental
N_SPLITS           = 5            # 5-fold cross-validation 
INITIAL_EPOCHS     = 20          
INCREMENTAL_EPOCHS = 10           
BATCH_SIZE         = 32

# Adaptive LR per stage — reduces in later stages to limit catastrophic forgetting
STAGE_LR = [3e-4, 1e-4, 5e-5, 2e-5]   # Stage 1 → Stage 4

# Transformer architecture  (Fig. 3)
TOKEN_DIM = 128          # dimension of each token
N_TOKENS  = 8            # number of tokens per branch
N_HEADS   = 8            # multi-head attention heads
N_BLOCKS  = 4            # transformer encoder blocks
PROJ_DIM  = N_TOKENS * TOKEN_DIM   # = 1024


# =============================================================================
# CLI
# =============================================================================
parser = argparse.ArgumentParser(
    description="HandEffTrans: Transformer Fusion + Incremental Learning (Paper-aligned)")
parser.add_argument("--fold",    type=str, default=None,
                    help="Fold to run: 1-5 or 'all'")
parser.add_argument("--summary", action="store_true",
                    help="Print cross-fold summary from saved reports")
args = parser.parse_args()


# =============================================================================
# SUMMARY MODE
# =============================================================================
if args.summary:
    def run_summary():
        files = sorted(f for f in os.listdir(REPORT_DIR)
                       if "_report" in f and f.endswith(".txt"))
        if not files:
            print("  No reports found. Run folds first."); return

        def _ex(pat, txt):
            m = re.search(pat, txt); return float(m.group(1)) if m else 0.0

        rows = []
        for fname in files:
            txt = open(os.path.join(REPORT_DIR, fname),
                       encoding="utf-8", errors="replace").read()
            rows.append({
                'fold'       : int(_ex(r'HandEffTrans — Fold (\d+)', txt)),
                'test_acc'   : _ex(r'Test Accuracy\s*:\s*([\d.]+)%', txt) / 100,
                'precision'  : _ex(r'Precision\s*:\s*([\d.]+)', txt),
                'recall'     : _ex(r'Recall\s*:\s*([\d.]+)', txt),
                'f1'         : _ex(r'F1 Score\s*:\s*([\d.]+)', txt),
                'kappa'      : _ex(r'Kappa\s*:\s*([\d.]+)', txt),
                'sensitivity': _ex(r'Sensitivity\s*:\s*([\d.]+)', txt),
                'specificity': _ex(r'Specificity\s*:\s*([\d.]+)', txt),
            })
        rows.sort(key=lambda r: r['fold'])

        keys = ['test_acc', 'precision', 'recall', 'f1',
                'kappa', 'sensitivity', 'specificity']
        avg  = {k: sum(r[k] for r in rows) / len(rows) for k in keys}

        W = 108
        print("\n" + "=" * W)
        print("  HandEffTrans — 5-Fold Cross-Validation Summary")
        print("=" * W)
        print(f"  {'Fold':<6}"
              + "".join(f"{k.replace('_',' ').title():>13}" for k in keys))
        print("-" * W)
        for r in rows:
            print(f"  {r['fold']:<6}"
                  + "".join(f"{r[k]:>13.4f}" for k in keys))
        print("-" * W)
        print(f"  {'AVG':<6}"
              + "".join(f"{avg[k]:>13.4f}" for k in keys))
        print("=" * W)

        # Save CSV
        csv_path = os.path.join(SUMMARY_DIR, "cv_summary.csv")
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=['fold'] + keys)
            w.writeheader(); w.writerows(rows)
            w.writerow({'fold': 'AVG', **{k: round(avg[k], 4) for k in keys}})
        print(f"\n  Summary CSV  → {csv_path}")

        # Bar chart
        x, width = range(len(rows)), 0.2
        fig, ax = plt.subplots(figsize=(13, 5))
        for i, (lbl, col) in enumerate([
                ('Test Acc', 'test_acc'), ('F1', 'f1'),
                ('Kappa', 'kappa'), ('Sensitivity', 'sensitivity')]):
            offsets = [xi + (i - 1.5) * width for xi in x]
            ax.bar(offsets, [r[col] for r in rows], width, label=lbl)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"Fold {r['fold']}" for r in rows])
        ax.set_ylim(0.85, 1.005)
        ax.set_ylabel('Score')
        ax.set_title('HandEffTrans — 5-Fold Cross-Validation Summary')
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        chart_path = os.path.join(SUMMARY_DIR, "cv_summary.png")
        plt.savefig(chart_path, dpi=150); plt.close()
        print(f"  Summary chart → {chart_path}\n")

    run_summary()
    sys.exit(0)


# =============================================================================
# PHASE 3 — LOAD FEATURES
# =============================================================================
print("\n" + "=" * 60)
print("  PHASE 3 — Feature Fusion")
print("  Loading pre-computed handcrafted and deep feature CSVs")
print("=" * 60)

hand_csv = os.path.join(CSV_DIR, "handcrafted_features.csv")
deep_csv = os.path.join(CSV_DIR, "deep_features.csv")

if not os.path.exists(hand_csv):
    raise FileNotFoundError(
        f"Handcrafted features not found: {hand_csv}\n"
        "  → Run Phase 1 (classical feature extraction) first.")
if not os.path.exists(deep_csv):
    raise FileNotFoundError(
        f"Deep features not found: {deep_csv}\n"
        "  → Run Phase 2 (EfficientNetB0 feature extraction) first.")

df_hand = pd.read_csv(hand_csv)
df_deep = pd.read_csv(deep_csv)

# Cheking order of both CSV 
assert list(df_hand['label']) == list(df_deep['label']), (
    "Label mismatch between handcrafted and deep feature CSVs. "
    "Ensure both were produced from the same sorted image list.")

labels      = df_hand['label'].values
Xh_raw      = df_hand.drop(columns=['label']).values.astype(np.float32)
Xd_raw      = df_deep.drop(columns=['label']).values.astype(np.float32)

le          = LabelEncoder()
y           = le.fit_transform(labels)
NUM_CLASSES = len(le.classes_)
CLASS_NAMES = list(le.classes_)

print(f"  Samples          : {len(y)}")
print(f"  Classes ({NUM_CLASSES})       : {CLASS_NAMES}")
print(f"  Handcrafted dim  : {Xh_raw.shape[1]}")
print(f"  Deep dim         : {Xd_raw.shape[1]}")
print("  PHASE 3 COMPLETE\n")


# =============================================================================
# PHASE 4 — TRAINING
# =============================================================================

def make_stage_batches(Xh, Xd, y_int,
                       n_stages=N_STAGES,
                       imgs_per_cls=IMGS_PER_CLASS_PER_STAGE):
    """
    splits the data into n_stages class-balanced batches.
    Returns list of n_stages tuples: (Xh_batch, Xd_batch, y_int_batch).
    """
    classes = np.unique(y_int)
    needed  = n_stages * imgs_per_cls
    stage_idx = [[] for _ in range(n_stages)]

    for cls in classes:
        idx = np.where(y_int == cls)[0]
        # Fall back to sampling with replacement if data is scarce
        if len(idx) < needed:
            extra = needed - len(idx)
            idx   = np.concatenate([idx,
                        np.random.choice(idx, extra, replace=True)])
        perm = np.random.permutation(len(idx))[:needed]
        idx  = idx[perm]
        for s in range(n_stages):
            stage_idx[s].append(idx[s * imgs_per_cls: (s + 1) * imgs_per_cls])

    batches = []
    for s in range(n_stages):
        idx = np.concatenate(stage_idx[s])
        idx = idx[np.random.permutation(len(idx))]   # interleave classes
        batches.append((Xh[idx], Xd[idx], y_int[idx]))
    return batches


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity and Specificity (macro)
# ─────────────────────────────────────────────────────────────────────────────
def sensitivity_specificity(y_true, y_pred):
    """Compute macro-averaged sensitivity and specificity from confusion matrix."""
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    sens, spec = [], []
    for i in range(NUM_CLASSES):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FP - FN
        sens.append(TP / (TP + FN + 1e-8))
        spec.append(TN / (TN + FP + 1e-8))
    return float(np.mean(sens)), float(np.mean(spec))


# ─────────────────────────────────────────────────────────────────────────────
# Full metrics dictionary
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    sens, spec = sensitivity_specificity(y_true, y_pred)
    return dict(
        acc         = float(accuracy_score(y_true, y_pred)),
        precision   = float(precision_score(y_true, y_pred,
                            average='weighted', zero_division=0)),
        recall      = float(recall_score(y_true, y_pred,
                            average='weighted', zero_division=0)),
        f1          = float(f1_score(y_true, y_pred,
                            average='weighted', zero_division=0)),
        kappa       = float(cohen_kappa_score(y_true, y_pred)),
        sensitivity = sens,
        specificity = spec,
    )


# =============================================================================
# TRANSFORMER FUSION MODEL  (HandEffTrans)
#
# Architecture :
#   Handcrafted branch : Dense(PROJ_DIM) → BN(batch normalization) → Reshape(N_TOKENS, TOKEN_DIM)
#   Deep branch        : Dense(PROJ_DIM) → BN(batch normalization) → Reshape(N_TOKENS, TOKEN_DIM)
#   Fusion             : Concatenate (2·N_TOKENS tokens)
#                        → N_BLOCKS × TransformerBlock
#                        → GlobalAveragePooling
#                        → Dense(1024) → BN → Dropout(0.4)
#                        → Dense(512)  → BN → Dropout(0.3)
#                        → Dense(NUM_CLASSES, softmax) 
# =============================================================================
def transformer_block(x):
    # Multi-Head Self-Attention  (learns inter-feature dependencies)
    attn = MultiHeadAttention(
        num_heads=N_HEADS, key_dim=TOKEN_DIM, dropout=0.1)(x, x)
    x    = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))
    # Feed-Forward Network
    ffn  = Dense(TOKEN_DIM * 4, activation='gelu')(x)
    ffn  = Dropout(0.1)(ffn)
    ffn  = Dense(TOKEN_DIM)(ffn)
    return LayerNormalization(epsilon=1e-6)(Add()([x, ffn]))


def build_model(xh_dim, xd_dim, loss_fn, lr=3e-4):
    # ── Inputs
    h_in = Input(shape=(xh_dim,), name='handcrafted')
    d_in = Input(shape=(xd_dim,),  name='deep')

    # ── Handcrafted feature branch
    h = Dense(PROJ_DIM, activation='gelu')(h_in)
    h = BatchNormalization()(h)
    h = Reshape((N_TOKENS, TOKEN_DIM))(h)

    # ── Deep feature branch
    d = Dense(PROJ_DIM, activation='gelu')(d_in)
    d = BatchNormalization()(d)
    d = Reshape((N_TOKENS, TOKEN_DIM))(d)

    # ── Transformer fusion 
    x = Concatenate(axis=1)([h, d])   # shape: (2·N_TOKENS, TOKEN_DIM)
    for _ in range(N_BLOCKS):
        x = transformer_block(x)

    # ── Classification head  (Eq. 8: y = softmax(W·x + b))
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(1024, activation='gelu')(x)
    x   = BatchNormalization()(x)
    x   = Dropout(0.4)(x)
    x   = Dense(512,  activation='gelu')(x)
    x   = BatchNormalization()(x)
    x   = Dropout(0.3)(x)
    out = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model([h_in, d_in], out, name='HandEffTrans')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=['accuracy']
    )
    return model


# =============================================================================
# SAVE TEXT REPORT
# =============================================================================
def save_report(fold, y_true, y_pred, best_train_acc, incr_history):
    """Write per-fold result report (mirrors Tables 3-13 of the paper)."""
    m = compute_metrics(y_true, y_pred)

    lines = [
        f"HandEffTrans — Fold {fold} Results",
        "=" * 55,
        f"Test Accuracy  : {m['acc'] * 100:.2f}%",
        f"Best Train Acc : {best_train_acc * 100:.2f}%",
        f"Precision      : {m['precision']:.4f}",
        f"Recall         : {m['recall']:.4f}",
        f"F1 Score       : {m['f1']:.4f}",
        f"Kappa          : {m['kappa']:.4f}",
        f"Sensitivity    : {m['sensitivity']:.4f}",
        f"Specificity    : {m['specificity']:.4f}",
        "",
        "Per-class Classification Report:",
        classification_report(y_true, y_pred,
                              target_names=CLASS_NAMES, digits=4),
        "Incremental Stage Progress:",
    ]
    for row in incr_history:
        lines.append(
            f"  {row['label']:<30}  "
            f"Acc={row['acc'] * 100:.2f}%  "
            f"F1={row['f1']:.4f}  "
            f"Kappa={row['kappa']:.4f}")

    path = os.path.join(REPORT_DIR, f"fold_{fold}_report.txt")
    open(path, "w", encoding="utf-8").write("\n".join(lines))

    print(f"\n  [Fold {fold}] FINAL TEST  "
          f"Acc={m['acc'] * 100:.2f}%  "
          f"Precision={m['precision']:.4f}  "
          f"Recall={m['recall']:.4f}  "
          f"F1={m['f1']:.4f}  "
          f"Kappa={m['kappa']:.4f}")
    return {'fold': fold, **m}


# =============================================================================
# PLOTS 
# =============================================================================
def plot_fold_results(fold, tr_acc, val_acc, tr_loss, val_loss,
                      boundaries, y_te, y_pred, y_prob):
    """
    4-panel figure per fold:
      (a) Training & Validation Accuracy  (Fig. 4a / 5a)
      (b) Training & Validation Loss      (Fig. 4b / 5b)
      (c) ROC Curves per class            (Fig. 4c / 5c)
      (d) Confusion Matrix                (Fig. 4d / 5d)
    """
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(f"Fold {fold} — HandEffTrans",
                 fontsize=13, fontweight='bold')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax_acc, ax_los, ax_roc, ax_cm = [fig.add_subplot(gs[r, c])
                                      for r in range(2) for c in range(2)]
    ep = list(range(1, len(tr_acc) + 1))

    def _stage_vlines(ax):
        for idx, (ep_end, lbl) in enumerate(boundaries):
            ax.axvline(ep_end, color='gray', ls='--', lw=0.9, alpha=0.8,
                       label=(lbl if idx <= 1 else '_nolegend_'))

    # (a) Accuracy curve
    ax_acc.plot(ep, tr_acc,  color='#2980B9', lw=1.8, label='Training Accuracy')
    ax_acc.plot(ep, val_acc, color='#E74C3C', lw=1.8, label='Validation Accuracy')
    _stage_vlines(ax_acc)
    ax_acc.set_title(f'Fold {fold} - Training and Validation Accuracy')
    ax_acc.set_xlabel('Epoch'); ax_acc.set_ylabel('Accuracy')
    ax_acc.legend(fontsize=9); ax_acc.grid(alpha=0.3)

    # (b) Loss curve
    ax_los.plot(ep, tr_loss,  color='#2980B9', lw=1.8, label='Training Loss')
    ax_los.plot(ep, val_loss, color='#E74C3C', lw=1.8, label='Validation Loss')
    _stage_vlines(ax_los)
    ax_los.set_title(f'Fold {fold} - Training and Validation Loss')
    ax_los.set_xlabel('Epoch'); ax_los.set_ylabel('Loss')
    ax_los.legend(fontsize=9); ax_los.grid(alpha=0.3)

    # (c) ROC curves — one per class
    y_bin = label_binarize(y_te, classes=list(range(NUM_CLASSES)))
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax_roc.plot(fpr, tpr, lw=1.8,
                    label=f"ROC curve of class {name} (area = {auc(fpr, tpr):.4f})")
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=0.9)
    ax_roc.set_title(f'Fold {fold} - Receiver Operating Characteristic')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend(fontsize=7, loc='lower right'); ax_roc.grid(alpha=0.3)

    # (d) Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax_cm, linewidths=0.5, annot_kws={"size": 10})
    ax_cm.set_title(f'Fold {fold} - Confusion Matrix')
    ax_cm.set_xlabel('Predicted label')
    ax_cm.set_ylabel('True label')
    ax_cm.tick_params(axis='x', rotation=20)

    plt.savefig(os.path.join(GRAPH_DIR, f"fold_{fold}_results.png"),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_incremental_progress(fold, incr_rows):
    """
    Kappa value & test accuracy vs incremental training stages (Fig. 6 style).
    Shows model improvement across HandEffTrans-0 → HandEffTrans-3.
    """
    x = list(range(len(incr_rows)))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, [r['acc']   for r in incr_rows], 'o-', lw=2, label='Val Accuracy')
    ax.plot(x, [r['f1']    for r in incr_rows], 's-', lw=2, label='F1 Score')
    ax.plot(x, [r['kappa'] for r in incr_rows], '^-', lw=2, label='Kappa Value')
    ax.set_xticks(x)
    ax.set_xticklabels([r['label'] for r in incr_rows], rotation=10)
    ax.set_ylim(0.85, 1.005)
    ax.set_ylabel('Score')
    ax.set_title(f'Fold {fold} — Kappa Value & Accuracy vs Incremental Training Stage')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, f"fold_{fold}_incremental.png"), dpi=150)
    plt.close()


# =============================================================================
# 5-FOLD CROSS-VALIDATION TRAINING LOOP
# =============================================================================
print("\n" + "=" * 60)
print("  PHASE 4 — 5-Fold Cross-Validation + Incremental Training")
print("  Eq. 9  : θ₀   = argmin J(θ, X_initial, Y_initial)")
print("  Eq. 10 : θₜ₊₁ = θₜ + α ∇J(θₜ, X_new, Y_new)")
print("  Eq. 11 : 5-fold partitioning and fold-wise evaluation")
print("=" * 60)

# Build all 5 fold splits  
kf         = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
all_splits = list(kf.split(Xh_raw, y))

if args.fold is None:
    print("\nUsage:")
    print("  --fold 1      run fold 1 only")
    print("  --fold all    run all 5 folds")
    print("  --summary     show cross-fold summary\n")
    sys.exit(0)

if args.fold.lower() == 'all':
    folds_to_run = list(range(1, N_SPLITS + 1))
else:
    f = int(args.fold)
    if f not in range(1, N_SPLITS + 1):
        print(f"  Invalid fold '{f}'. Choose 1-{N_SPLITS}.")
        sys.exit(1)
    folds_to_run = [f]

print(f"\n  Folds to run       : {folds_to_run}")
print(f"  Stages per fold    : {N_STAGES}  (1 initial + {N_STAGES-1} incremental)")
print(f"  Imgs/class/stage   : {IMGS_PER_CLASS_PER_STAGE}")
print(f"  Classes            : {NUM_CLASSES}  {CLASS_NAMES}\n")

all_fold_rows = []
all_incr_rows = []

# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP — one iteration per fold
# ──────────────────────────────────────────────────────────────────────────────
for fold in folds_to_run:

    print(f"\n{'=' * 55}")
    print(f"  FOLD {fold} / {N_SPLITS}")
    print(f"{'=' * 55}")

    # ── Split: this fold → validation/test; others → training ────────────────
    _, test_idx  = all_splits[fold - 1]
    train_idx    = np.concatenate(
        [all_splits[i][0] for i in range(N_SPLITS) if i != fold - 1])
    train_idx    = np.setdiff1d(train_idx, test_idx)   # safety overlap check

    print(f"  Train samples : {len(train_idx)}"
          f"   |   Test samples : {len(test_idx)}")

    # ── Feature normalization: fit ONLY on training data ─────────────────────
    sc_h = StandardScaler().fit(Xh_raw[train_idx])
    sc_d = StandardScaler().fit(Xd_raw[train_idx])

    Xh_tr = sc_h.transform(Xh_raw[train_idx]).astype(np.float32)
    Xd_tr = sc_d.transform(Xd_raw[train_idx]).astype(np.float32)
    y_tr  = y[train_idx]

    Xh_te = sc_h.transform(Xh_raw[test_idx]).astype(np.float32)
    Xd_te = sc_d.transform(Xd_raw[test_idx]).astype(np.float32)
    y_te  = y[test_idx]

    # ── Class weights (handles any minor imbalance within training fold) ──────
    cw = dict(enumerate(
        compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)))

    # ── Build N_STAGES class-balanced incremental batches ────────────────────
    #    Paper Table 2: 1,000 images per class per stage
    batches = make_stage_batches(Xh_tr, Xd_tr, y_tr)

    # ── Loss functions ────────────────────────────────────────────────────────
    #    Initial stage : standard categorical cross-entropy  (Eq. 9)
    #    Incremental   : label smoothing 0.05 (anti-forgetting, paper §4)
    init_loss = tf.keras.losses.CategoricalCrossentropy()
    incr_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)

    # ── Fresh model for this fold  (Eq. 9: θ₀ = argmin J) ───────────────────
    model = build_model(Xh_raw.shape[1], Xd_raw.shape[1],
                        loss_fn=init_loss, lr=STAGE_LR[0])

    tr_acc, val_acc   = [], []
    tr_loss, val_loss = [], []
    boundaries        = []
    fold_incr         = []
    best_val_acc      = -1.0
    best_w_path       = os.path.join(MODEL_DIR, f"fold_{fold}_best.weights.h5")

    # =========================================================================
    # STAGE 1 — INITIAL TRAINING  (HandEffTrans-0)
    # =========================================================================
    Xh_b, Xd_b, y_b = batches[0]
    print(f"\n  [Stage 1 / HandEffTrans-0 / Initial Training]")
    print(f"  Train: {len(y_b)} samples  |  "
          f"Val (fold {fold}): {len(y_te)} samples  |  "
          f"lr={STAGE_LR[0]:.0e}  |  Epochs: {INITIAL_EPOCHS}")

    hist = model.fit(
        [Xh_b, Xd_b],
        to_categorical(y_b, NUM_CLASSES),
        validation_data=(
            [Xh_te, Xd_te],
            to_categorical(y_te, NUM_CLASSES)
        ),
        epochs=INITIAL_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            # Monitor val_accuracy for early stopping (paper: evaluate after each stage)
            EarlyStopping(monitor='val_accuracy', patience=15,
                          restore_best_weights=True, verbose=0),
            # Adaptive LR reduction within a stage (additional stabilisation)
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                              patience=6, min_lr=1e-7, verbose=1),
        ],
        class_weight=cw,
        verbose=1
    )
    tr_acc   += hist.history['accuracy']
    val_acc  += hist.history['val_accuracy']
    tr_loss  += hist.history['loss']
    val_loss += hist.history['val_loss']
    boundaries.append((len(tr_acc), 'Stage 1 / Initial'))

    # Save best weights after initial training  (paper: "Save Model Parameters")
    cur_val = max(hist.history['val_accuracy'])
    if cur_val > best_val_acc:
        best_val_acc = cur_val
        model.save_weights(best_w_path)

    # Evaluate on validation set  (paper: "Evaluate on Validation Set for Current Fold")
    y_pred_v = np.argmax(model.predict([Xh_te, Xd_te], verbose=0), axis=1)
    m_v = compute_metrics(y_te, y_pred_v)
    fold_incr.append({**m_v, 'label': 'HandEffTrans-0 (Initial)'})
    all_incr_rows.append({'fold': fold, 'stage': 0, **m_v})
    print(f"  → Val Acc={m_v['acc']*100:.2f}%  "
          f"Precision={m_v['precision']:.4f}  "
          f"Recall={m_v['recall']:.4f}  "
          f"F1={m_v['f1']:.4f}  "
          f"Kappa={m_v['kappa']:.4f}")

    # =========================================================================
    # STAGES 2-4 — INCREMENTAL TRAINING  (HandEffTrans-1, 2, 3)
    # =========================================================================
    for i in range(1, N_STAGES):         # i = 1, 2, 3

        # Recompile with decayed LR + label smoothing  (Eq. 10 / anti-forgetting)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=STAGE_LR[i]),
            loss=incr_loss,
            metrics=['accuracy']
        )

        Xh_b, Xd_b, y_b = batches[i]
        print(f"\n  [Stage {i+1} / HandEffTrans-{i} / Increment {i}]")
        print(f"  Train: {len(y_b)} samples  |  "
              f"Val (fold {fold}): {len(y_te)} samples  |  "
              f"lr={STAGE_LR[i]:.0e}  |  Epochs: {INCREMENTAL_EPOCHS}")

        hist = model.fit(
            [Xh_b, Xd_b],
            to_categorical(y_b, NUM_CLASSES),
            validation_data=(
                [Xh_te, Xd_te],
                to_categorical(y_te, NUM_CLASSES)
            ),
            epochs=INCREMENTAL_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[
                EarlyStopping(monitor='val_accuracy', patience=10,
                              restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                  patience=4, min_lr=1e-8, verbose=1),
            ],
            class_weight=cw,
            verbose=1
        )
        tr_acc   += hist.history['accuracy']
        val_acc  += hist.history['val_accuracy']
        tr_loss  += hist.history['loss']
        val_loss += hist.history['val_loss']
        boundaries.append((len(tr_acc), f'Stage {i+1} / Incr {i}'))

        cur_val = max(hist.history['val_accuracy'])
        if cur_val > best_val_acc:
            best_val_acc = cur_val
            model.save_weights(best_w_path)
            print(f"  → [NEW BEST] val_acc = {best_val_acc:.4f}  — weights saved.")

        # Evaluate on validation set after each increment  (paper: "Calculate Metrics")
        y_pred_v = np.argmax(model.predict([Xh_te, Xd_te], verbose=0), axis=1)
        m_v = compute_metrics(y_te, y_pred_v)
        fold_incr.append({**m_v, 'label': f'HandEffTrans-{i}'})
        all_incr_rows.append({'fold': fold, 'stage': i, **m_v})
        print(f"  → Val Acc={m_v['acc']*100:.2f}%  "
              f"Precision={m_v['precision']:.4f}  "
              f"Recall={m_v['recall']:.4f}  "
              f"F1={m_v['f1']:.4f}  "
              f"Kappa={m_v['kappa']:.4f}")

    # ── Restore best weights across all 4 stages ─────────────────────────────
    model.load_weights(best_w_path)
    print(f"\n  Restored best weights  (best val_acc = {best_val_acc:.4f})")

    # ── Final test evaluation on this fold's designated set ───────────────────
    #    Paper: "Store Results: Accuracy, Precision, Recall"
    y_prob = model.predict([Xh_te, Xd_te], verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    row = save_report(fold, y_te, y_pred, max(tr_acc), fold_incr)
    all_fold_rows.append(row)

    # ── Generate plots (Figs. 4 & 5 style) ───────────────────────────────────
    plot_fold_results(fold, tr_acc, val_acc, tr_loss, val_loss,
                      boundaries, y_te, y_pred, y_prob)
    plot_incremental_progress(fold, fold_incr)

    print(f"\n  Fold {fold} complete.")


# =============================================================================
# SAVE CSVs
# =============================================================================
# Incremental metrics per fold × stage
incr_path = os.path.join(CSV_DIR, "incremental_metrics.csv")
incr_df   = pd.DataFrame(all_incr_rows)
if os.path.exists(incr_path):
    old = pd.read_csv(incr_path)
    for r in all_incr_rows:
        old = old[~((old['fold'] == r['fold']) & (old['stage'] == r['stage']))]
    incr_df = pd.concat([old, incr_df], ignore_index=True)
incr_df.sort_values(['fold', 'stage']).to_csv(incr_path, index=False)

# Final per-fold test results
res_path = os.path.join(CSV_DIR, "fold_results.csv")
res_df   = pd.DataFrame(all_fold_rows)
if os.path.exists(res_path):
    old = pd.read_csv(res_path)
    for r in all_fold_rows:
        old = old[old['fold'] != r['fold']]
    res_df = pd.concat([old, res_df], ignore_index=True)
res_df.sort_values('fold').to_csv(res_path, index=False)

print(f"\n  Results CSV      → {res_path}")
print(f"  Incremental CSV  → {incr_path}")
print(f"  Graphs           → {GRAPH_DIR}")
print(f"  Reports          → {REPORT_DIR}")
print(f"  Models           → {MODEL_DIR}")
print("\n  PHASE 3 + PHASE 4 COMPLETE")
print(f"  → Run: python {os.path.basename(__file__)} --summary")