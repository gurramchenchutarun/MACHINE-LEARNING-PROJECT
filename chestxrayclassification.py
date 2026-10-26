# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1 — INSTALL PACKAGES                                   ║
# ╚══════════════════════════════════════════════════════════════╝
import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

for pkg in ['opencv-python-headless', 'scikit-image', 'scikit-learn', 'matplotlib', 'seaborn', 'Pillow']:
    pip_install(pkg)

print('✅ All packages installed.')
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2 — IMPORTS & GLOBAL CONSTANTS                         ║
# ╚══════════════════════════════════════════════════════════════╝
import os, re, sys, gc, random, hashlib, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import VGG19, VGG16
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

SEED = 42
os.environ['PYTHONHASHSEED']       = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
LR_PHASE1     = 1e-4
LR_PHASE2     = 1e-5
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10
DROPOUT_RATE  = 0.4
L2_LAMBDA     = 0.001

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# [MODIFIED]: Decreased crop width and height per professor's feedback
CROP_TOP    = 0.08
CROP_BOTTOM = 0.08
CROP_LEFT   = 0.14
CROP_RIGHT  = 0.14

SAVE_DIR = '/content/ieee_outputs'
for sub in ['', '/metrics', '/models', '/augmentation_vis', '/pixel_comparisons', '/gradcam_comparisons']:
    os.makedirs(SAVE_DIR + sub, exist_ok=True)

print(f'✅ Crop  top={CROP_TOP}  bot={CROP_BOTTOM}  left={CROP_LEFT}  right={CROP_RIGHT}')
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3 — DATASET SETUP (Option A — Kaggle API)              ║
# ╚══════════════════════════════════════════════════════════════╝
from google.colab import files
from pathlib import Path

print("Please upload your kaggle.json file below:")
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!pip install -q kaggle
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip -q -o chest-xray-pneumonia.zip -d /content/

DATA_ROOT = Path('/content/chest_xray')
assert DATA_ROOT.exists(), f'❌ DATA_ROOT not found at {DATA_ROOT}.'
print(f'✅ DATA_ROOT found: {DATA_ROOT}')
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4 — LOAD ALL FILES INTO DATAFRAME                      ║
# ╚══════════════════════════════════════════════════════════════╝
records = []
for subset in ['train', 'test', 'val']:
    for class_name in ['NORMAL', 'PNEUMONIA']:
        folder = DATA_ROOT / subset / class_name
        if not folder.exists(): continue
        for fp in sorted(folder.glob('*')):
            if fp.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                records.append({
                    'path'      : str(fp),
                    'label'     : 0 if class_name == 'NORMAL' else 1,
                    'label_name': class_name,
                    'orig_split': subset,
                    'filename'  : fp.name,
                })

df_all = pd.DataFrame(records)
print(f'📂 Total images found : {len(df_all):,}')
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5 — MD5 DEDUPLICATION                                  ║
# ╚══════════════════════════════════════════════════════════════╝
def compute_md5(filepath):
    h = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

md5s = []
for _, row in df_all.iterrows():
    try: md5s.append(compute_md5(row['path']))
    except Exception: md5s.append(None)

df_all['md5'] = md5s
df_all = df_all[df_all['md5'].notna()].copy()

df_uniq = df_all.drop_duplicates(subset='md5', keep='first').reset_index(drop=True)
print(f'📊 Before dedup: {len(df_all):,}  |  After: {len(df_uniq):,}')
df_uniq[['path','label','label_name','md5']].to_csv(f'{SAVE_DIR}/metrics/md5_manifest.csv', index=False)
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6 — PATIENT-ID EXTRACTION                              ║
# ╚══════════════════════════════════════════════════════════════╝
def extract_patient_id(filename: str, label_name: str) -> str:
    stem = Path(filename).stem
    if label_name == 'NORMAL':
        m = re.match(r'^(IM-\d+)', stem, re.IGNORECASE)
        if m: return m.group(1).upper()
        parts = stem.rsplit('-', 1)
        return (parts[0] if len(parts) == 2 else stem).upper()
    else:
        m = re.match(r'^(person\d+)', stem, re.IGNORECASE)
        if m: return m.group(1).lower()
        return stem.split('_')[0].lower()

df_uniq['patient_id'] = df_uniq.apply(lambda r: extract_patient_id(r['filename'], r['label_name']), axis=1)
print(f'✅ Unique patients: {df_uniq["patient_id"].nunique():,}')
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 7 — PATIENT-AWARE 70 / 15 / 15 SPLIT                  ║
# ╚══════════════════════════════════════════════════════════════╝
normal_pts    = set(df_uniq[df_uniq['label']==0]['patient_id'].unique())
pneumonia_pts = set(df_uniq[df_uniq['label']==1]['patient_id'].unique())
shared_pts    = normal_pts & pneumonia_pts

shared_list = sorted(shared_pts)
random.seed(SEED)
random.shuffle(shared_list)
n_s  = len(shared_list)
s_tr = set(shared_list[:int(n_s * 0.70)])
s_va = set(shared_list[int(n_s * 0.70): int(n_s * 0.85)])
s_te = set(shared_list[int(n_s * 0.85):])

only_n = sorted(normal_pts - shared_pts)
n_tr, tmp = train_test_split(only_n, test_size=0.30, random_state=SEED)
n_va, n_te = train_test_split(tmp,   test_size=0.50, random_state=SEED)

only_p = sorted(pneumonia_pts - shared_pts)
p_tr, tmp = train_test_split(only_p, test_size=0.30, random_state=SEED)
p_va, p_te = train_test_split(tmp,   test_size=0.50, random_state=SEED)

train_pts = set(n_tr) | set(p_tr) | s_tr
val_pts   = set(n_va) | set(p_va) | s_va
test_pts  = set(n_te) | set(p_te) | s_te

df_tr_raw = df_uniq[df_uniq['patient_id'].isin(train_pts)].copy()
df_va_raw = df_uniq[df_uniq['patient_id'].isin(val_pts)].copy()
df_te_raw = df_uniq[df_uniq['patient_id'].isin(test_pts)].copy()

def balance(df, seed=SEED):
    d0, d1 = df[df['label'] == 0], df[df['label'] == 1]
    n = min(len(d0), len(d1))
    return pd.concat([d0.sample(n, random_state=seed), d1.sample(n, random_state=seed)]).sample(frac=1, random_state=seed).reset_index(drop=True)

df_train, df_val, df_test = balance(df_tr_raw), balance(df_va_raw), balance(df_te_raw)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Patient-Aware Split — Class Distribution', fontsize=13, fontweight='bold')
clrs = ['#4CAF50', '#F44336']
for ax, (title, dff) in zip(axes, [
    (f'Train 70%\n(n={len(df_train)})', df_train),
    (f'Val 15%\n(n={len(df_val)})',     df_val),
    (f'Test 15%\n(n={len(df_test)})',   df_test),
]):
    cnt = dff['label_name'].value_counts()
    ax.pie(cnt, labels=cnt.index, autopct='%1.1f%%', colors=clrs, startangle=90)
    ax.set_title(title)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/metrics/split_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 8 — PREPROCESSING FUNCTIONS                            ║
# ╚══════════════════════════════════════════════════════════════╝
def safe_load_bgr(img_path: str) -> np.ndarray:
    img = cv2.imread(str(img_path))
    if img is None:
        arr = np.array(Image.open(str(img_path)).convert('RGB'))
        img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def crop_hospital_marks(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    t = int(h * CROP_TOP);    b = int(h * (1 - CROP_BOTTOM))
    l = int(w * CROP_LEFT);   r = int(w * (1 - CROP_RIGHT))
    return img_bgr[t:b, l:r]

def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([cl.apply(l), a, b]), cv2.COLOR_LAB2BGR)

def apply_gaussian(img_bgr: np.ndarray, ksize=(3, 3), sigma=0.5) -> np.ndarray:
    return cv2.GaussianBlur(img_bgr, ksize, sigma)

def denorm_for_display(img_norm: np.ndarray) -> np.ndarray:
    return np.clip(img_norm * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)

def preprocess_image(img_path: str, target_size: tuple = IMG_SIZE) -> np.ndarray:
    img = safe_load_bgr(img_path)
    img = crop_hospital_marks(img)
    img = apply_clahe(img)
    img = apply_gaussian(img)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (img - IMAGENET_MEAN) / IMAGENET_STD
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 9 — PREPROCESSING PIPELINE VISUALISATION               ║
# ╚══════════════════════════════════════════════════════════════╝
sample_paths  = [df_test[df_test['label'] == 0].iloc[0]['path'], df_test[df_test['label'] == 1].iloc[0]['path']]
sample_labels = ['Normal', 'Pneumonia']

fig = plt.figure(figsize=(24, 14))
fig.suptitle(f'Preprocessing Pipeline Comparison\nCrop: {int(CROP_TOP*100)}% Top/Bottom | {int(CROP_LEFT*100)}% Left/Right', fontsize=13, fontweight='bold', y=1.01)

step_names = ['① RAW Original', '② After Crop', '③ CLAHE Enhanced', '④ Final 224×224']
outer_gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.55)

for row_idx, (path, lbl) in enumerate(zip(sample_paths, sample_labels)):
    raw      = safe_load_bgr(path)
    raw_rgb  = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    cropped  = crop_hospital_marks(raw)
    crop_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    clahe_im = apply_clahe(cropped)
    cla_rgb  = cv2.cvtColor(clahe_im, cv2.COLOR_BGR2RGB)
    final_n  = preprocess_image(path)
    final_d  = denorm_for_display(final_n)

    steps_display = [raw_rgb, crop_rgb, cla_rgb, final_d]
    steps_hist = [raw_rgb.astype(np.float32) / 255.0, crop_rgb.astype(np.float32) / 255.0, cla_rgb.astype(np.float32) / 255.0, final_d]

    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer_gs[row_idx], hspace=0.08, wspace=0.08, height_ratios=[3, 1])

    for col, (img_show, img_hist, sname) in enumerate(zip(steps_display, steps_hist, step_names)):
        ax_img = fig.add_subplot(inner_gs[0, col])
        ax_img.imshow(img_show); ax_img.set_xticks([]); ax_img.set_yticks([])
        ax_img.set_title(f'{sname}\n[{lbl}]' if col == 0 else sname, fontsize=8.5, fontweight='bold')

        ax_hist = fig.add_subplot(inner_gs[1, col])
        ax_hist.hist(img_hist.ravel(), bins=64, color='steelblue', alpha=0.85, edgecolor='none')
        ax_hist.set_xlim(0, 1); ax_hist.set_yticks([])

plt.savefig(f'{SAVE_DIR}/pixel_comparisons/preprocessing_comparison.png', dpi=180, bbox_inches='tight')
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 11 — AUGMENTATION VISUALISATION                        ║
# ╚══════════════════════════════════════════════════════════════╝
vis_samples = [df_train.iloc[0], df_train.iloc[1], df_train.iloc[2]]
N_AUG = 3

fig, axes = plt.subplots(len(vis_samples), 1 + N_AUG, figsize=(5 * (1 + N_AUG), 5 * len(vis_samples)))
fig.suptitle('Data Augmentation Visualisation', fontsize=13, fontweight='bold')

for row_idx, sample_row in enumerate(vis_samples):
    orig = preprocess_image(sample_row['path'])
    axes[row_idx, 0].imshow(denorm_for_display(orig)); axes[row_idx, 0].set_xticks([]); axes[row_idx, 0].set_yticks([])

    for aug_idx in range(N_AUG):
        single_gen = ChestXrayGenerator(df_train, batch_size=1, augment=True, shuffle=False, seed=SEED + row_idx * 100 + aug_idx * 7)
        aug_img = single_gen._augment(orig.copy())
        axes[row_idx, 1 + aug_idx].imshow(denorm_for_display(aug_img)); axes[row_idx, 1 + aug_idx].set_xticks([]); axes[row_idx, 1 + aug_idx].set_yticks([])
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 12 — BUILD MODELS                                      ║
# ╚══════════════════════════════════════════════════════════════╝
def build_vgg19(input_shape=(224, 224, 3)):
    base = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base.layers: layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(L2_LAMBDA), name='dense_512')(x)
    x = layers.Dropout(DROPOUT_RATE, seed=SEED)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(L2_LAMBDA), name='dense_256')(x)
    x = layers.Dropout(DROPOUT_RATE, seed=SEED)(x)
    out = layers.Dense(1, activation='sigmoid', name='output')(x)
    return Model(inputs=base.input, outputs=out, name='VGG19_Pneumonia'), base

def build_vgg16(input_shape=(224, 224, 3)):
    base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base.layers: layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(L2_LAMBDA), name='dense_512')(x)
    x = layers.Dropout(DROPOUT_RATE, seed=SEED)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(L2_LAMBDA), name='dense_256')(x)
    x = layers.Dropout(DROPOUT_RATE, seed=SEED)(x)
    out = layers.Dense(1, activation='sigmoid', name='output')(x)
    return Model(inputs=base.input, outputs=out, name='VGG16_Pneumonia'), base

vgg19_model, vgg19_base = build_vgg19()
vgg16_model, vgg16_base = build_vgg16()

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/augmentation_vis/augmentation_comparison.png', dpi=180, bbox_inches='tight')
plt.show()
plt.show()
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 13 — CALLBACKS HELPER                                  ║
# ╚══════════════════════════════════════════════════════════════╝
def get_metrics():
    return ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]

def get_callbacks(model_name: str, phase: int):
    ckpt = f'{SAVE_DIR}/models/{model_name}_phase{phase}_best.weights.h5'
    return [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(filepath=ckpt, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0),
    ]
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 14 — TRAIN VGG19  Phase 1                              ║
# ╚══════════════════════════════════════════════════════════════╝
vgg19_model.compile(optimizer=Adam(LR_PHASE1), loss='binary_crossentropy', metrics=get_metrics())
print('📌 VGG19 Phase 1 — training dense head only...')
hist_vgg19_p1 = vgg19_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE1, callbacks=get_callbacks('vgg19', 1))
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 15 — TRAIN VGG19  Phase 2                              ║
# ╚══════════════════════════════════════════════════════════════╝
for layer in vgg19_base.layers:
    layer.trainable = True if layer.name == 'block5_conv4' else False
for layer in vgg19_model.layers:
    if any(k in layer.name for k in ['dense', 'dropout', 'output']): layer.trainable = True

vgg19_model.compile(optimizer=Adam(LR_PHASE2), loss='binary_crossentropy', metrics=get_metrics())
print('📌 VGG19 Phase 2 — fine-tuning...')
hist_vgg19_p2 = vgg19_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE2, callbacks=get_callbacks('vgg19', 2))
vgg19_model.save(f'{SAVE_DIR}/models/vgg19_final.keras')
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 16 — TRAIN VGG16  Phase 1 & 2                          ║
# ╚══════════════════════════════════════════════════════════════╝
vgg16_model.compile(optimizer=Adam(LR_PHASE1), loss='binary_crossentropy', metrics=get_metrics())
print('📌 VGG16 Phase 1 — training dense head only...')
hist_vgg16_p1 = vgg16_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE1, callbacks=get_callbacks('vgg16', 1))

for layer in vgg16_base.layers:
    layer.trainable = True if layer.name == 'block5_conv3' else False
for layer in vgg16_model.layers:
    if any(k in layer.name for k in ['dense', 'dropout', 'output']): layer.trainable = True

vgg16_model.compile(optimizer=Adam(LR_PHASE2), loss='binary_crossentropy', metrics=get_metrics())
print('📌 VGG16 Phase 2 — fine-tuning...')
hist_vgg16_p2 = vgg16_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_PHASE2, callbacks=get_callbacks('vgg16', 2))
vgg16_model.save(f'{SAVE_DIR}/models/vgg16_final.keras')
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 17 — TRAINING HISTORY PLOTS                            ║
# ╚══════════════════════════════════════════════════════════════╝
def merge_hist(h1, h2):
    return {key: h1.history[key] + h2.history[key] for key in h1.history}

hist19, hist16 = merge_hist(hist_vgg19_p1, hist_vgg19_p2), merge_hist(hist_vgg16_p1, hist_vgg16_p2)
plot_metrics, plot_titles = ['loss', 'accuracy', 'precision', 'recall', 'auc'], ['Loss', 'Accuracy', 'Precision', 'Recall', 'AUC-ROC']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
for i, (metric, title) in enumerate(zip(plot_metrics, plot_titles)):
    ax = axes.flatten()[i]
    e19, e16 = range(1, len(hist19[metric]) + 1), range(1, len(hist16[metric]) + 1)
    ax.plot(e19, hist19[metric], color='#1565C0', lw=2, label='VGG19 Train')
    ax.plot(e19, hist19[f'val_{metric}'], color='#42A5F5', lw=2, ls='--', label='VGG19 Val')
    ax.plot(e16, hist16[metric], color='#B71C1C', lw=2, label='VGG16 Train')
    ax.plot(e16, hist16[f'val_{metric}'], color='#EF9A9A', lw=2, ls='--', label='VGG16 Val')
    ax.axvline(x=EPOCHS_PHASE1, color='gray', ls=':', lw=1.5)
    ax.set_title(title); ax.legend()

axes.flatten()[-1].axis('off')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/metrics/training_history.png', dpi=180, bbox_inches='tight')
plt.show()
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 17b — [IEEE] OVERFITTING DIAGNOSTIC                    ║
# ╚══════════════════════════════════════════════════════════════╝
def overfitting_report(hist_dict, model_name):
    rows = []
    for m in ['loss', 'accuracy', 'precision', 'recall', 'auc']:
        tr_final, val_final = hist_dict[m][-1], hist_dict[f'val_{m}'][-1]
        gap = abs(tr_final - val_final)
        flag = '⚠️  OVERFIT CHECK' if (m == 'loss' and gap > 0.15) or (m == 'accuracy' and gap > 0.10) else '✅ OK'
        rows.append({'Metric': m.capitalize(), 'Train': tr_final, 'Val': val_final, 'Gap': gap, 'Status': flag})
    df_ov = pd.DataFrame(rows)
    print(f'\n=== OVERFITTING DIAGNOSTIC — {model_name} ===')
    print(df_ov.to_string(index=False))
    return df_ov

ov19_df = overfitting_report(hist19, 'VGG19')
ov16_df = overfitting_report(hist16, 'VGG16')
ov19_df.to_csv(f'{SAVE_DIR}/metrics/overfitting_diagnostic_vgg19.csv', index=False)
ov16_df.to_csv(f'{SAVE_DIR}/metrics/overfitting_diagnostic_vgg16.csv', index=False)
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 18 — EVALUATION ON TEST SET                            ║
# ╚══════════════════════════════════════════════════════════════╝
def evaluate_model(model, df_te, model_name):
    print(f'\n🔍 Evaluating {model_name} on {len(df_te)} images...')
    gen = ChestXrayGenerator(df_te, augment=False, shuffle=False)
    y_prob_list = []
    for i in range(len(gen)):
        X_batch, _ = gen[i]
        preds = model.predict(X_batch, verbose=0)
        y_prob_list.extend(preds[:, 0].tolist())

    y_true = df_te['label'].values
    y_prob = np.array(y_prob_list[:len(y_true)], dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(int)

    res = {
        'accuracy' : accuracy_score(y_true, y_pred) * 100,
        'precision': precision_score(y_true, y_pred, zero_division=0) * 100,
        'recall'   : recall_score(y_true, y_pred, zero_division=0) * 100,
        'f1'       : f1_score(y_true, y_pred, zero_division=0) * 100,
        'auc'      : roc_auc_score(y_true, y_prob) * 100,
    }
    return y_true, y_prob, y_pred, res

y_true19, y_prob19, y_pred19, met19 = evaluate_model(vgg19_model, df_test, 'VGG19')
y_true16, y_prob16, y_pred16, met16 = evaluate_model(vgg16_model, df_test, 'VGG16')

# [MODIFIED] Stock Market Trend Chart & Radar Chart
def plot_stock_trend_comparison(met19, met16, save_dir):
    labels = [l.capitalize() for l in met19.keys()]
    v19, v16 = list(met19.values()), list(met16.values())
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, v19, marker='o', lw=3, color='#1565C0', label='VGG19')
    ax.fill_between(x, v19, color='#1565C0', alpha=0.15)
    ax.plot(x, v16, marker='s', lw=3, color='#B71C1C', label='VGG16')
    ax.fill_between(x, v16, color='#B71C1C', alpha=0.15)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontweight='bold', fontsize=11)
    ax.set_ylim(max(0, min(min(v19), min(v16)) - 5), 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_title('Metric Trend Comparison (Stock Market Area Chart)', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right')
    plt.savefig(f'{save_dir}/metrics/trend_comparison.png', dpi=180, bbox_inches='tight'); plt.show()

def plot_radar_comparison(met19, met16, save_dir):
    labels = list(met19.keys())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    v19_vals = list(met19.values()) + [list(met19.values())[0]]
    v16_vals = list(met16.values()) + [list(met16.values())[0]]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], [l.capitalize() for l in labels], color='grey', size=12, fontweight='bold')
    ax.set_rlabel_position(30); plt.yticks([80, 85, 90, 95, 100], ["80","85","90","95","100"], color="grey", size=10)
    plt.ylim(75, 100)
    ax.plot(angles, v19_vals, linewidth=2, linestyle='solid', label='VGG19', color='#1565C0')
    ax.fill(angles, v19_vals, '#1565C0', alpha=0.25)
    ax.plot(angles, v16_vals, linewidth=2, linestyle='solid', label='VGG16', color='#B71C1C')
    ax.fill(angles, v16_vals, '#B71C1C', alpha=0.25)
    plt.title('VGG19 vs VGG16 Performance (Radar Chart)', size=15, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(f'{save_dir}/metrics/radar_comparison.png', dpi=180, bbox_inches='tight'); plt.show()

plot_stock_trend_comparison(met19, met16, SAVE_DIR)
plot_radar_comparison(met19, met16, SAVE_DIR)
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 19 — CONFUSION MATRICES & ROC CURVES                   ║
# ╚══════════════════════════════════════════════════════════════╝
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle('Confusion Matrices & ROC Curves — VGG19 vs VGG16', fontsize=13, fontweight='bold')
class_names = ['Normal', 'Pneumonia']

for idx, (mname, y_tr, y_pr, y_pb, clr, cmap) in enumerate([
    ('VGG19', y_true19, y_pred19, y_prob19, '#1565C0', 'Blues'),
    ('VGG16', y_true16, y_pred16, y_prob16, '#B71C1C', 'Reds'),
]):
    cm = confusion_matrix(y_tr, y_pr)
    ax_cm = axes[idx * 2]
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names, annot_kws={'size': 14}, ax=ax_cm)
    ax_cm.set_title(f'{mname} — Confusion Matrix', fontsize=11, fontweight='bold')

    fpr, tpr, _ = roc_curve(y_tr, y_pb)
    ax_roc = axes[idx * 2 + 1]
    ax_roc.plot(fpr, tpr, color=clr, lw=2.5, label=f'{mname} (AUC={roc_auc_score(y_tr, y_pb):.3f})')
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random classifier')
    ax_roc.fill_between(fpr, tpr, alpha=0.10, color=clr)
    ax_roc.set_title(f'{mname} — ROC Curve', fontsize=11, fontweight='bold')
    ax_roc.legend(fontsize=9); ax_roc.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/metrics/confusion_roc.png', dpi=180, bbox_inches='tight')
plt.show()
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 20 — GRADCAM ENGINE                                    ║
# ╚══════════════════════════════════════════════════════════════╝
LAST_CONV_VGG19 = 'block5_conv4'
LAST_CONV_VGG16 = 'block5_conv3'

def get_gradcam(model, img_arr, last_conv_name):
    grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[model.get_layer(last_conv_name).output, model.output])
    tensor = tf.cast(img_arr[np.newaxis, ...], tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(tensor)
        conv_out, preds = grad_model(tensor)
        loss = preds[:, 0]

    grads      = tape.gradient(loss, conv_out)
    pool_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap    = tf.reduce_sum(conv_out[0] * pool_grads, axis=-1)
    heatmap    = tf.nn.relu(heatmap).numpy()

    vmin, vmax = heatmap.min(), heatmap.max()
    return ((heatmap - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)

def overlay_cam(img_norm, heatmap, alpha=0.45):
    img_u8 = (denorm_for_display(img_norm) * 255).astype(np.uint8)
    hm_rsz = cv2.resize(heatmap, (img_u8.shape[1], img_u8.shape[0]))
    hm_rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * hm_rsz), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_u8, 1 - alpha, hm_rgb, alpha, 0)
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 21 — GRADCAM COMPARISON FIGURE                         ║
# ╚══════════════════════════════════════════════════════════════╝
gcam_paths  = [df_test[df_test['label'] == 0].iloc[0]['path'], df_test[df_test['label'] == 1].iloc[0]['path']]
gcam_labels = ['Normal (label=0)', 'Pneumonia (label=1)']

fig, axes = plt.subplots(2, 4, figsize=(26, 13))
fig.suptitle('GradCAM: VGG19 vs VGG16 on Test Images\n', fontsize=13, fontweight='bold')
col_hdrs = ['RAW Original', 'Preprocessed\n(de-norm)', 'VGG19 GradCAM', 'VGG16 GradCAM']
for col, hdr in enumerate(col_hdrs): axes[0, col].set_title(hdr, fontsize=11, fontweight='bold', pad=8)

for row, (path, lbl) in enumerate(zip(gcam_paths, gcam_labels)):
    proc = preprocess_image(path)
    hm19, hm16 = get_gradcam(vgg19_model, proc, LAST_CONV_VGG19), get_gradcam(vgg16_model, proc, LAST_CONV_VGG16)

    imgs_lbls = [
        (cv2.cvtColor(safe_load_bgr(path), cv2.COLOR_BGR2RGB), lbl),
        (denorm_for_display(proc), 'CLAHE + crop + de-norm'),
        (overlay_cam(proc, hm19), f'VGG19  P(pneumonia)={float(vgg19_model.predict(proc[np.newaxis], verbose=0)[0][0]):.3f}'),
        (overlay_cam(proc, hm16), f'VGG16  P(pneumonia)={float(vgg16_model.predict(proc[np.newaxis], verbose=0)[0][0]):.3f}'),
    ]
    for col, (img_show, sub) in enumerate(imgs_lbls):
        ax = axes[row, col]
        ax.imshow(img_show); ax.set_xlabel(sub, fontsize=9, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/gradcam_comparisons/gradcam_vgg19_vs_vgg16.png', dpi=180, bbox_inches='tight')
plt.show()

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 21 — GRADCAM COMPARISON FIGURE                         ║
# ╚══════════════════════════════════════════════════════════════╝
gcam_paths  = [df_test[df_test['label'] == 0].iloc[0]['path'], df_test[df_test['label'] == 1].iloc[0]['path']]
gcam_labels = ['Normal (label=0)', 'Pneumonia (label=1)']

fig, axes = plt.subplots(2, 4, figsize=(26, 13))
fig.suptitle('GradCAM: VGG19 vs VGG16 on Test Images\n', fontsize=13, fontweight='bold')
col_hdrs = ['RAW Original', 'Preprocessed\n(de-norm)', 'VGG19 GradCAM', 'VGG16 GradCAM']
for col, hdr in enumerate(col_hdrs): axes[0, col].set_title(hdr, fontsize=11, fontweight='bold', pad=8)

for row, (path, lbl) in enumerate(zip(gcam_paths, gcam_labels)):
    proc = preprocess_image(path)
    hm19, hm16 = get_gradcam(vgg19_model, proc, LAST_CONV_VGG19), get_gradcam(vgg16_model, proc, LAST_CONV_VGG16)

    imgs_lbls = [
        (cv2.cvtColor(safe_load_bgr(path), cv2.COLOR_BGR2RGB), lbl),
        (denorm_for_display(proc), 'CLAHE + crop + de-norm'),
        (overlay_cam(proc, hm19), f'VGG19  P(pneumonia)={float(vgg19_model.predict(proc[np.newaxis], verbose=0)[0][0]):.3f}'),
        (overlay_cam(proc, hm16), f'VGG16  P(pneumonia)={float(vgg16_model.predict(proc[np.newaxis], verbose=0)[0][0]):.3f}'),
    ]
    for col, (img_show, sub) in enumerate(imgs_lbls):
        ax = axes[row, col]
        ax.imshow(img_show); ax.set_xlabel(sub, fontsize=9, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/gradcam_comparisons/gradcam_vgg19_vs_vgg16.png', dpi=180, bbox_inches='tight')
plt.show()

# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 22 — LUNG SEGMENTATION + LOCALISATION TREND            ║
# ╚══════════════════════════════════════════════════════════════╝
def segment_lung(img_norm: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor((denorm_for_display(img_norm) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # [FIXED]: We use THRESH_BINARY_INV. Since lungs are dark, this turns the dark lungs into 255 (1)
    # and turns the bright spine/heart into 0.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern), cv2.MORPH_OPEN, kern)
    return mask

def localisation_scores(heatmap: np.ndarray, lung_mask_u8: np.ndarray) -> dict:
    hm = cv2.resize(heatmap, (lung_mask_u8.shape[1], lung_mask_u8.shape[0]))
    gm, lm = (hm >= np.percentile(hm, 80)).astype(np.uint8), (lung_mask_u8 > 127).astype(np.uint8)
    py, px = np.unravel_index(np.argmax(hm), hm.shape)
    return {'iou': (gm & lm).sum() / ((gm | lm).sum() + 1e-8), 'ecr': (hm * lm).sum() / (hm.sum() + 1e-8), 'pointing_hit': int(lm[py, px] == 1)}

score_df = df_test[df_test['label'] == 1].sample(min(50, int((df_test['label'] == 1).sum())), random_state=SEED).reset_index(drop=True)
sc19_list, sc16_list = [], []

for _, row in score_df.iterrows():
    proc, lung = preprocess_image(row['path']), segment_lung(preprocess_image(row['path']))
    sc19_list.append(localisation_scores(get_gradcam(vgg19_model, proc, LAST_CONV_VGG19), lung))
    sc16_list.append(localisation_scores(get_gradcam(vgg16_model, proc, LAST_CONV_VGG16), lung))

sc19_df, sc16_df = pd.DataFrame(sc19_list), pd.DataFrame(sc16_list)

# Stock Trend chart for Localisation
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('GradCAM Localisation Quality — VGG19 vs VGG16 (Stock Trend)', fontsize=12, fontweight='bold')

keys = ['IoU (%)', 'ECR (%)', 'Pointing-Game (%)']
v19_loc = [sc19_df['iou'].mean() * 100, sc19_df['ecr'].mean() * 100, sc19_df['pointing_hit'].mean() * 100]
v16_loc = [sc16_df['iou'].mean() * 100, sc16_df['ecr'].mean() * 100, sc16_df['pointing_hit'].mean() * 100]
x = np.arange(len(keys))

ax.plot(x, v19_loc, marker='o', lw=3, color='#1565C0', label='VGG19')
ax.fill_between(x, v19_loc, color='#1565C0', alpha=0.15)
ax.plot(x, v16_loc, marker='s', lw=3, color='#B71C1C', label='VGG16')
ax.fill_between(x, v16_loc, color='#B71C1C', alpha=0.15)

ax.set_xticks(x); ax.set_xticklabels(keys, fontweight='bold', fontsize=11)
ax.set_ylim(0, 105); ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.set_ylabel('Score (%)', fontweight='bold'); ax.legend(loc='lower right')
plt.savefig(f'{SAVE_DIR}/gradcam_comparisons/localisation_trend.png', dpi=180, bbox_inches='tight'); plt.show()
