
from __future__ import annotations

import os
import re
import gc
import csv
import glob
import json
import random
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import convnext as keras_convnext

# -------------------- Runtime knobs (keep XLA/autotune conservative) --------------------
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=1 --xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("TF_USE_CUDNN_AUTOTUNE", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Disable graph compilation explicitly; we rely on eager + tf.function
tf.config.optimizer.set_jit(False)

# ============================= CONFIG ==============================================
parser = argparse.ArgumentParser(description="Train a ConvNeXt probe with custom dataset path.")
parser.add_argument("--data_root", type=str, required=True,
                    help="Path to the root directory of sorted image folders.")
args = parser.parse_args()
DATA_ROOT = args.data_root



# Default backbone (sweep disabled by default) — lock to e5
SSL_BACKBONE_PATH = "./ssl_ckpts/convnext_tiny_ssl_e5.keras"

IMG_SIZE = 256
BATCH_SIZE = 12
VAL_SPLIT = 0.20
SEED = 1337

# tf.data pipeline
MAP_PARALLEL_CALLS = 2
PREFETCH_TRAIN = 1
PREFETCH_EVAL = 1
CACHE_EVAL_IN_RAM = False

# Training schedule
EPOCHS_LP = 15
EPOCHS_FT = 12
EPOCHS_FT_PHASE2 = 8
UNFREEZE_LAST_N_LAYERS = 120

AUG = True
USE_MIXED_PRECISION = True

# Class weighting
USE_CLASS_WEIGHTS = True
FOCUS_CLASS_NAME = "multiple_head_echoes"
FOCUS_CLASS_BUMP = 1.55
SECOND_FOCUS_NAME = "head_echo_and_unrelated_coh_scatter"
SECOND_FOCUS_BUMP = 1.10

# Loss smoothing
LABEL_SMOOTHING_LP = 0.0
LABEL_SMOOTHING_FT_BASE = 0.03
LABEL_SMOOTHING_FT_FOR_FOCUS = 0.005

# Test‑time augmentation
TTA_FLIP = False
TTA_SHIFT_PX = 3

# EMA
USE_EMA = True
EMA_DECAY = 0.9995
EMA_TRACK_LAST_N = None  # e.g., 256 to track only last-N vars

# Optional logit nudging at inference
ENABLE_LOGIT_TUNING = True
LOGIT_TUNE_GRID = [0.00, 0.03, 0.06, 0.09, 0.12]

# Grouped split (avoid leakage)
GROUP_BY = "ancestor"  # {"regex", "ancestor"}
REGEX = r"(\d{8}T\d{6})"
ANCESTOR_LEVELS_UP = 2

# -------- SSL checkpoint sweep (disabled by default) --------
SWEEP_SSL_BACKBONES = False
SWEEP_GLOB = "./ssl_ckpts/convnext_tiny_ssl_e*.keras"
SWEEP_INCLUDE_FINAL = True
SWEEP_LP_EPOCHS = 4
SWEEP_SELECT_BY = "macro_f1"  # {"macro_f1", "acc"}
SWEEP_SAVE_CSV = "ssl_sweep_leaderboard.csv"

SWEEP_BATCH_SIZE = 8
SWEEP_DISABLE_AUG = True
SWEEP_USE_TTA = False
SWEEP_USE_CACHE = False
SWEEP_PREFETCH = 1
SWEEP_MAX_TRAIN_STEPS = 100
SWEEP_MAX_VAL_STEPS = 60
SWEEP_TRAIN_FRAC = 0.75
SWEEP_VAL_FRAC = 1.0
SWEEP_MAX_CANDIDATES = 8
CLEAR_SESSION_BETWEEN = True
# =====================================================================================

# --------- Seed & precision ---------
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if USE_MIXED_PRECISION:
    try:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
        print("Using mixed precision: mixed_float16")
    except Exception as e:  # pragma: no cover
        print("Mixed precision unavailable; using float32:", e)

# --------- GPU memory growth (first visible GPU) ---------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:  # pragma: no cover
        print("Could not set GPU memory growth:", e)

# --------- Enumerate files per class ---------
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
root = Path(DATA_ROOT)
class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
assert class_names, f"No class folders found in {DATA_ROOT}"
name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(class_names)}

paths: List[str] = []
labels_l: List[int] = []
for cname in class_names:
    files = sorted(
        p for p in (root / cname).rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED
    )
    for p in files:
        paths.append(str(p))
        labels_l.append(name_to_idx[cname])

paths_np = np.array(paths)
labels_np = np.array(labels_l, np.int32)
print(f"Found {len(paths_np)} files across {len(class_names)} classes.")
print("Classes:", class_names)

# --------- Grouping for leakage‑robust split ---------

def group_key(p: str) -> str:
    if GROUP_BY == "regex":
        m = re.search(REGEX, os.path.basename(p))
        if m:
            return m.group(1)
    if GROUP_BY == "ancestor":
        parts = Path(p).parts
        return "/".join(parts[-(ANCESTOR_LEVELS_UP + 1) : -1])
    return Path(p).stem[:12]

rng = np.random.default_rng(SEED)
train_idx: List[int] = []
val_idx: List[int] = []
for c in range(len(class_names)):
    idx_c = np.where(labels_np == c)[0]
    n_c = len(idx_c)
    desired_val = int(round(n_c * VAL_SPLIT))
    target_val = max(1, min(desired_val, n_c - 1)) if n_c > 1 else 1

    buckets: Dict[str, List[int]] = defaultdict(list)
    for i in idx_c:
        buckets[group_key(paths_np[i])].append(i)
    groups = list(buckets.keys())
    rng.shuffle(groups)

    if len(groups) <= 1:
        idx_list = idx_c.copy()
        rng.shuffle(idx_list)
        val_take = max(1, min(target_val, n_c - 1)) if n_c > 1 else 1
        val_idx.extend(idx_list[:val_take].tolist())
        train_idx.extend(idx_list[val_take:].tolist())
    else:
        taken = 0
        val_groups: List[str] = []
        for g in groups:
            if taken < target_val:
                val_idx.extend(buckets[g])
                taken += len(buckets[g])
                val_groups.append(g)
            else:
                train_idx.extend(buckets[g])
        has_train = any(labels_np[i] == c for i in train_idx)
        if n_c > 1 and not has_train and val_groups:
            g = val_groups.pop()
            moved = set(buckets[g])
            val_idx = [i for i in val_idx if i not in moved]
            train_idx.extend(list(moved))

train_idx_np = np.array(train_idx, dtype=np.int64)
val_idx_np = np.array(val_idx, dtype=np.int64)


def _counts_of(idxs: Iterable[int]) -> Dict[str, int]:
    idxs = np.asarray(list(idxs), dtype=np.int64)
    if idxs.size == 0:
        return {cname: 0 for cname in class_names}
    vals, cnts = np.unique(labels_np[idxs], return_counts=True)
    m = {class_names[int(v)]: int(c) for v, c in zip(vals, cnts)}
    for cname in class_names:
        m.setdefault(cname, 0)
    return m

print("Train counts:", _counts_of(train_idx_np))
print("Val counts:  ", _counts_of(val_idx_np))

train_paths = paths_np[train_idx_np]
train_labels = labels_np[train_idx_np]
val_paths = paths_np[val_idx_np]
val_labels = labels_np[val_idx_np]
num_classes = len(class_names)

# Persist class names for downstream consumers
os.makedirs("ssl_ckpts", exist_ok=True)
with open("ssl_ckpts/class_names.json", "w") as f:
    json.dump(class_names, f)

# --------- Preprocess layer (ConvNeXt default) ---------

@tf.keras.utils.register_keras_serializable(package="custom")
class ConvNeXtPreprocess(tf.keras.layers.Layer):
    def call(self, x: tf.Tensor) -> tf.Tensor:  # expects float32 in [0, 255]
        return keras_convnext.preprocess_input(x)


# -------------------------------- tf.data --------------------------------

def _decode_resize_factory(image_size: int):
    # Decode → resize in float32 for quality → store as float16 to reduce RAM.
    def _decode_resize(path: tf.Tensor) -> tf.Tensor:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, 3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize_with_pad(img, image_size, image_size, antialias=True)
        return tf.cast(img * 255.0, tf.float16)

    return _decode_resize


def _to_one_hot(y: tf.Tensor, n: int) -> tf.Tensor:
    return tf.one_hot(y, n, dtype=tf.float32)


def make_ds(
    x_paths: np.ndarray,
    y_labels: np.ndarray,
    image_size: int,
    batch_size: int,
    training: bool,
    drop_last: bool = False,
    cache_eval: bool = False,
    prefetch_buf: int = 1,
    num_parallel: int = 2,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x_paths, y_labels))
    if training and len(x_paths) > 0:
        ds = ds.shuffle(len(x_paths), seed=SEED, reshuffle_each_iteration=True)
    dec = _decode_resize_factory(image_size)
    ds = ds.map(lambda p, y: (dec(p), _to_one_hot(y, num_classes)), num_parallel_calls=num_parallel)
    if not training and cache_eval:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=drop_last)
    return ds.prefetch(prefetch_buf)


# --------- Class weights as sample weights ---------
class_weight_vec: tf.Tensor | None = None
if USE_CLASS_WEIGHTS:
    counts = np.bincount(train_labels, minlength=num_classes)
    total = counts.sum()
    cw = {i: float(total / (num_classes * max(1, counts[i]))) for i in range(num_classes)}
    if FOCUS_CLASS_NAME in name_to_idx:
        cw[name_to_idx[FOCUS_CLASS_NAME]] *= FOCUS_CLASS_BUMP
    if SECOND_FOCUS_NAME in name_to_idx:
        cw[name_to_idx[SECOND_FOCUS_NAME]] *= SECOND_FOCUS_BUMP
    print("Class weights (with bumps):", cw)
    class_weight_vec = tf.constant([cw[i] for i in range(num_classes)], tf.float32)


def attach_sample_weights(ds: tf.data.Dataset) -> tf.data.Dataset:
    if class_weight_vec is None:
        return ds

    def _add_w(x, y):
        w = tf.reduce_sum(y * class_weight_vec, axis=-1)
        return x, y, w

    return ds.map(_add_w, num_parallel_calls=MAP_PARALLEL_CALLS).prefetch(PREFETCH_TRAIN)


# -------------------------------- Metrics --------------------------------

def _softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    y_pred = np.argmax(probs, axis=-1)
    acc = float((y_pred == y_true).mean())
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()
    tp = cm.diagonal().astype(np.float64)
    fp = (cm.sum(axis=0) - tp).astype(np.float64)
    fn = (cm.sum(axis=1) - tp).astype(np.float64)
    prec = tp / np.clip(tp + fp, 1, None)
    rec = tp / np.clip(tp + fn, 1, None)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    macro_f1 = float(np.mean(f1))
    return acc, macro_f1


# --------------------------- Model build & inference ---------------------------

def build_model_with_backbone(
    backbone_path: str, image_size_hint: int, aug_enabled: bool
) -> Tuple[Model, Model, int]:
    if not os.path.exists(backbone_path):
        raise FileNotFoundError(f"Backbone not found at '{backbone_path}'.")
    base: Model = tf.keras.models.load_model(backbone_path, compile=False)
    base._name = "backbone"

    img_h, img_w = base.inputs[0].shape[1], base.inputs[0].shape[2]
    img_size_local = image_size_hint
    try:
        if img_h and img_w and (
            int(img_h) != image_size_hint or int(img_w) != image_size_hint
        ):
            print(
                f"Backbone expects {(int(img_h), int(img_w))}; overriding IMG_SIZE"
                f" {image_size_hint}→{int(img_h)}"
            )
            img_size_local = int(img_h)
    except Exception as e:  # pragma: no cover
        print("Could not read backbone input shape; using IMG_SIZE:", image_size_hint, "-", e)

    # Keep model input float32; cast happens inside infer fn.
    inp = layers.Input((img_size_local, img_size_local, 3), name="input_image", dtype="float32")
    x = inp

    if aug_enabled:
        x = tf.keras.Sequential(
            [
                layers.RandomRotation(0.03),
                layers.RandomZoom(0.05),
                layers.RandomContrast(0.10),
                layers.RandomBrightness(0.08),
                layers.RandomTranslation(0.01, 0.01),
            ],
            name="aug",
        )(x)

    x = ConvNeXtPreprocess(name="preprocess_convnext")(x)

    base.trainable = False
    x = base(x, training=False)
    if isinstance(x, (list, tuple)):
        x = x[0]
    elif isinstance(x, dict):
        x = next(iter(x.values()))

    if len(x.shape) == 4:
        x = layers.GlobalAveragePooling2D()(x)
    elif len(x.shape) > 2:
        x = layers.Flatten()(x)

    x = layers.LayerNormalization(epsilon=1e-6, dtype="float32")(x)
    x = layers.Dense(384, activation="gelu", dtype="float32", kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.Dropout(0.30)(x)

    # Keep final dense in float32 when using mixed precision for numerical stability.
    dense_dtype = (
        "float32" if tf.keras.mixed_precision.global_policy().compute_dtype == "float16" else None
    )
    out = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        name="head",
        dtype=dense_dtype,
    )(x)

    model = Model(inp, out, name="convnext_probe")
    return model, base, int(img_size_local)


def build_infer_fn(model_obj: Model, img_size_local: int):
    # Accepts float16 input, casts to float32 inside the function.
    @tf.function(
        input_signature=[tf.TensorSpec([None, img_size_local, img_size_local, 3], tf.float16)],
        jit_compile=False,
        reduce_retracing=True,
    )
    def _infer(x: tf.Tensor) -> tf.Tensor:
        x = tf.cast(x, tf.float32)
        return tf.cast(model_obj(x, training=False), tf.float32)

    return _infer


def _pad_crop_shift(x: tf.Tensor, dy: int, dx: int) -> tf.Tensor:
    H = tf.shape(x)[1]
    W = tf.shape(x)[2]
    pad_t = tf.maximum(dy, 0)
    pad_b = tf.maximum(-dy, 0)
    pad_l = tf.maximum(dx, 0)
    pad_r = tf.maximum(-dx, 0)
    xpad = tf.pad(x, [[0, 0], [pad_t, pad_b], [pad_l, pad_r], [0, 0]], mode="REFLECT")
    return tf.image.crop_to_bounding_box(
        xpad, offset_height=pad_b, offset_width=pad_r, target_height=H, target_width=W
    )


def predict_with_tta(infer_fn, x: tf.Tensor, flip: bool = TTA_FLIP, shift_px: int = TTA_SHIFT_PX) -> tf.Tensor:
    preds = [infer_fn(x)]
    if flip:
        preds.append(infer_fn(tf.image.flip_up_down(x)))
    if isinstance(shift_px, int) and shift_px > 0:
        preds.extend(
            [
                infer_fn(_pad_crop_shift(x, shift_px, 0)),
                infer_fn(_pad_crop_shift(x, -shift_px, 0)),
                infer_fn(_pad_crop_shift(x, 0, shift_px)),
                infer_fn(_pad_crop_shift(x, 0, -shift_px)),
            ]
        )
    return tf.add_n(preds) / float(len(preds))


def collect_probs(model_obj: Model, val_ds_full: tf.data.Dataset, img_size_local: int, use_tta: bool = True):
    infer = build_infer_fn(model_obj, img_size_local)
    ys, ps = [], []
    for bx, by in val_ds_full:
        probs = predict_with_tta(infer, bx) if use_tta else infer(bx)
        ys.append(tf.argmax(by, axis=-1).numpy())
        ps.append(probs.numpy())
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


# --------------------------- Evaluation helpers ---------------------------

def eval_from_probs(
    y_true: np.ndarray, probs: np.ndarray, tag: str = "", boost_vec: np.ndarray | None = None
) -> float:
    probs_use = probs
    if boost_vec is not None:
        logits = np.log(np.clip(probs, 1e-8, 1.0)) + boost_vec
        probs_use = _softmax_np(logits)

    y_pred = np.argmax(probs_use, axis=-1)
    acc = float((y_pred == y_true).mean())

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()
    tp = cm.diagonal().astype(np.float64)
    fp = (cm.sum(axis=0) - tp).astype(np.float64)
    fn = (cm.sum(axis=1) - tp).astype(np.float64)

    prec = tp / np.clip(tp + fp, 1, None)
    rec = tp / np.clip(tp + fn, 1, None)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)

    print(f"\n[{tag}] Val accuracy: {acc:.4f}")
    print(f"\n[{tag}] Confusion matrix (rows=true, cols=pred):\n", cm)
    print(f"\n[{tag}] Per-class metrics:")
    for i, name in enumerate(class_names):
        print(f"{name:>40s}:  precision={prec[i]:.3f}  recall={rec[i]:.3f}  f1={f1[i]:.3f}")
    return acc


def export_predictions_csv_from_probs(
    img_paths: np.ndarray, probs: np.ndarray, out_csv: str = "preds.csv", boost_vec: np.ndarray | None = None
) -> None:
    probs_use = probs
    if boost_vec is not None:
        logits = np.log(np.clip(probs, 1e-8, 1.0)) + boost_vec
        probs_use = _softmax_np(logits)
    pred_idx = np.argmax(probs_use, axis=-1)
    conf = np.max(probs_use, axis=-1)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "pred", "conf"])
        w.writeheader()
        for i in range(len(img_paths)):
            w.writerow({
                "path": img_paths[i],
                "pred": class_names[int(pred_idx[i])],
                "conf": f"{float(conf[i]):.4f}",
            })
    print(f"Wrote {len(img_paths)} rows to {out_csv}")


# --------------------------- Subsets for sweep ---------------------------

def make_subset(paths_np: np.ndarray, labels_np: np.ndarray, frac: float, seed: int):
    if frac >= 0.999:
        return paths_np, labels_np
    n = len(paths_np)
    m = max(1, int(round(n * frac)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=m, replace=False)
    return paths_np[idx], labels_np[idx]


# -------------------------------- Runner --------------------------------

def run_experiment(backbone_path: str, image_size_hint: int, sweep_mode: bool = False) -> Dict[str, object]:
    aug_enabled = (AUG if not sweep_mode else (AUG and not SWEEP_DISABLE_AUG))
    model, base, img_size_local = build_model_with_backbone(backbone_path, image_size_hint, aug_enabled)

    # Datasets
    if sweep_mode:
        bsz = SWEEP_BATCH_SIZE
        cache_eval = SWEEP_USE_CACHE
        prefetch_tr = SWEEP_PREFETCH
        prefetch_ev = SWEEP_PREFETCH
        parallel = 1
        tr_p, tr_y = make_subset(train_paths, train_labels, SWEEP_TRAIN_FRAC, SEED)
        va_p, va_y = make_subset(val_paths, val_labels, SWEEP_VAL_FRAC, SEED + 1)
    else:
        bsz = BATCH_SIZE
        cache_eval = CACHE_EVAL_IN_RAM
        prefetch_tr = PREFETCH_TRAIN
        prefetch_ev = PREFETCH_EVAL
        parallel = MAP_PARALLEL_CALLS
        tr_p, tr_y = train_paths, train_labels
        va_p, va_y = val_paths, val_labels

    train_ds = make_ds(tr_p, tr_y, img_size_local, bsz, training=True, drop_last=True, cache_eval=False,
                       prefetch_buf=prefetch_tr, num_parallel=parallel)
    val_ds_eval = make_ds(va_p, va_y, img_size_local, bsz, training=False, drop_last=True, cache_eval=cache_eval,
                          prefetch_buf=prefetch_ev, num_parallel=parallel)
    val_ds_full = make_ds(va_p, va_y, img_size_local, bsz, training=False, drop_last=False, cache_eval=cache_eval,
                          prefetch_buf=prefetch_ev, num_parallel=parallel)

    train_ds_w = attach_sample_weights(train_ds)
    val_ds_w = attach_sample_weights(val_ds_full)

    # ----- Linear probe -----
    loss_lp = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING_LP)
    try:
        opt_lp = tf.keras.optimizers.AdamW(1e-3, weight_decay=1e-5)
    except Exception:
        opt_lp = tf.keras.optimizers.Adam(1e-3)

    model.compile(optimizer=opt_lp, loss=loss_lp, metrics=["accuracy"], jit_compile=False)

    if sweep_mode:
        print(f"\n— SWEEP linear probe for '{backbone_path}' ({SWEEP_LP_EPOCHS} epochs) —")
        steps_train = int(tf.data.experimental.cardinality(train_ds_w).numpy())
        steps_val = int(tf.data.experimental.cardinality(val_ds_w).numpy())
        steps_train = min(steps_train, SWEEP_MAX_TRAIN_STEPS) if steps_train > 0 else 1
        steps_val = min(steps_val, SWEEP_MAX_VAL_STEPS) if steps_val > 0 else 1

        model.fit(
            train_ds_w,
            validation_data=val_ds_w,
            epochs=SWEEP_LP_EPOCHS,
            steps_per_epoch=steps_train,
            validation_steps=steps_val,
            verbose=0,
        )
        y_eval, probs_eval = collect_probs(model, val_ds_full, img_size_local, use_tta=SWEEP_USE_TTA)
        acc, macro_f1 = metrics_from_probs(y_eval, probs_eval)

        if CLEAR_SESSION_BETWEEN:
            del model, base, train_ds, val_ds_eval, val_ds_full, train_ds_w, val_ds_w
            tf.keras.backend.clear_session()
            gc.collect()
        return {"path": backbone_path, "img_size": img_size_local, "acc": acc, "macro_f1": macro_f1}

    ckpt_lp = tf.keras.callbacks.ModelCheckpoint("linear_probe.keras", save_best_only=True,
                                                 monitor="val_accuracy", mode="max")
    early = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5,
                                             restore_best_weights=True)

    print("\n— Training linear probe (frozen backbone) —")
    model.fit(train_ds_w, validation_data=val_ds_w, epochs=EPOCHS_LP, callbacks=[ckpt_lp, early])

    # ----- Fine‑tuning: Phase 1 (last‑N layers) -----
    base.trainable = True
    if hasattr(base, "layers") and UNFREEZE_LAST_N_LAYERS > 0:
        for l in base.layers[:-UNFREEZE_LAST_N_LAYERS]:
            l.trainable = False

    steps_per_epoch = max(1, int(tf.data.experimental.cardinality(train_ds).numpy()))
    total_steps = steps_per_epoch * EPOCHS_FT
    schedule1 = tf.keras.optimizers.schedules.CosineDecay(7e-5, total_steps)
    try:
        opt_ft1 = tf.keras.optimizers.AdamW(learning_rate=schedule1, weight_decay=8e-6)
    except Exception:
        opt_ft1 = tf.keras.optimizers.Adam(learning_rate=7e-5)

    focus_idx = name_to_idx.get(FOCUS_CLASS_NAME, -1)

    def per_class_smoothed_ce(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Lower smoothing for the focus class to preserve peak confidence.
        true_idx = tf.argmax(y_true, axis=-1, output_type=tf.int32)
        s_focus = tf.constant(LABEL_SMOOTHING_FT_FOR_FOCUS, tf.float32)
        s_base = tf.constant(LABEL_SMOOTHING_FT_BASE, tf.float32)
        if focus_idx >= 0:
            s = tf.where(tf.equal(true_idx, focus_idx), s_focus, s_base)
        else:
            s = tf.fill(tf.shape(true_idx), s_base)
        s = tf.expand_dims(s, -1)
        y_smooth = y_true * (1.0 - s) + s / tf.cast(num_classes, tf.float32)
        return tf.keras.losses.categorical_crossentropy(y_smooth, y_pred)

    model.compile(optimizer=opt_ft1, loss=per_class_smoothed_ce, metrics=["accuracy"], jit_compile=False)
    ckpt_ft = tf.keras.callbacks.ModelCheckpoint("finetuned.keras", save_best_only=True,
                                                 monitor="val_accuracy", mode="max")

    print("\n— Fine‑tuning last layers (Phase 1) —")
    model.fit(train_ds_w, validation_data=val_ds_w, epochs=EPOCHS_FT, callbacks=[ckpt_ft, early])

    # ----- Fine‑tuning: Phase 2 (unfreeze all, lower LR) -----
    for l in base.layers:
        l.trainable = True

    total_steps2 = steps_per_epoch * EPOCHS_FT_PHASE2
    schedule2 = tf.keras.optimizers.schedules.CosineDecay(2.0e-5, total_steps2)
    try:
        opt_ft2 = tf.keras.optimizers.AdamW(learning_rate=schedule2, weight_decay=9e-6)
    except Exception:
        opt_ft2 = tf.keras.optimizers.Adam(learning_rate=2.0e-5)

    model.compile(optimizer=opt_ft2, loss=per_class_smoothed_ce, metrics=["accuracy"], jit_compile=False)

    class EMAWeights(tf.keras.callbacks.Callback):
        """Exponential Moving Average over a subset of trainable vars."""

        def __init__(self, model: Model, decay: float = EMA_DECAY, track_last_n: int | None = None):
            super().__init__()
            self.decay = decay
            w = list(model.trainable_weights)
            if track_last_n is not None and track_last_n < len(w):
                w = w[-int(track_last_n) :]
            self.vars = w
            self.ema = [v.numpy().copy() for v in self.vars]

        def on_train_batch_end(self, batch, logs=None):  # noqa: ARG002
            for i, v in enumerate(self.vars):
                self.ema[i] = self.decay * self.ema[i] + (1.0 - self.decay) * v.numpy()

        def apply_to(self, model: Model) -> None:
            w = list(model.trainable_weights)
            if EMA_TRACK_LAST_N is not None and EMA_TRACK_LAST_N < len(w):
                w = w[-int(EMA_TRACK_LAST_N) :]
            for v, ew in zip(w, self.ema):
                v.assign(ew)

    ema_cb = EMAWeights(model, EMA_DECAY, track_last_n=EMA_TRACK_LAST_N) if USE_EMA else None

    print("\n— Fine‑tuning all layers (Phase 2) —")
    callbacks = [ckpt_ft, tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=4,
                                                           restore_best_weights=True)]
    if USE_EMA:
        callbacks.insert(0, ema_cb)
    if EPOCHS_FT_PHASE2 > 0:
        model.fit(train_ds_w, validation_data=val_ds_w, epochs=EPOCHS_FT_PHASE2, callbacks=callbacks)

    if USE_EMA and ema_cb is not None and EPOCHS_FT_PHASE2 > 0:
        print("\n— Saving EMA‑averaged weights —")
        ema_cb.apply_to(model)
        model.save("finetuned_ema.keras")

    # --------- Best checkpoint eval + CSV export ---------
    print("\n— Evaluating best checkpoint(s) —")
    custom_objects = {"ConvNeXtPreprocess": ConvNeXtPreprocess}

    def _load_any(path: str) -> Model:
        return tf.keras.models.load_model(path, compile=False, custom_objects=custom_objects)

    best_model = _load_any("finetuned.keras")
    best_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=per_class_smoothed_ce,
                       metrics=["accuracy"], jit_compile=False)

    y_eval, probs_eval = collect_probs(best_model, val_ds_full, img_size_local, use_tta=True)

    boost_vec_base = np.zeros((num_classes,), dtype=np.float32)
    selected_boost = 0.0
    if ENABLE_LOGIT_TUNING and FOCUS_CLASS_NAME in name_to_idx:
        fi = name_to_idx[FOCUS_CLASS_NAME]
        best_acc = -1.0
        best_boost = 0.0
        for b in LOGIT_TUNE_GRID:
            vec = boost_vec_base.copy()
            vec[fi] = b
            acc_try = eval_from_probs(y_eval, probs_eval, tag=f"BEST+TTA (boost={b:.2f})", boost_vec=vec)
            if acc_try > best_acc:
                best_acc, best_boost = acc_try, b
        selected_boost = best_boost
        print(f"\nSelected logit boost for '{FOCUS_CLASS_NAME}': {best_boost:.2f} (val acc={best_acc*100:.2f}%)")
    else:
        eval_from_probs(y_eval, probs_eval, tag="BEST")

    boost_vec_np = None
    if ENABLE_LOGIT_TUNING and FOCUS_CLASS_NAME in name_to_idx and selected_boost > 0:
        boost_vec_np = boost_vec_base.copy()
        boost_vec_np[name_to_idx[FOCUS_CLASS_NAME]] = selected_boost

    export_predictions_csv_from_probs(val_paths, probs_eval, out_csv="preds.csv", boost_vec=boost_vec_np)

    if USE_EMA and os.path.exists("finetuned_ema.keras"):
        print("\n— Evaluating EMA checkpoint —")
        ema_model = tf.keras.models.load_model(
            "finetuned_ema.keras", compile=False, custom_objects={"ConvNeXtPreprocess": ConvNeXtPreprocess}
        )
        ema_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=per_class_smoothed_ce,
                          metrics=["accuracy"], jit_compile=False)
        y_eval_ema, probs_eval_ema = collect_probs(ema_model, val_ds_full, img_size_local, use_tta=True)
        eval_from_probs(y_eval_ema, probs_eval_ema, tag="EMA+TTA")

    print("\nDone. Saved 'linear_probe.keras', 'finetuned.keras' and optionally 'finetuned_ema.keras'.")
    acc_best, macro_f1_best = metrics_from_probs(y_eval, probs_eval)
    return {"path": backbone_path, "img_size": img_size_local, "acc": acc_best, "macro_f1": macro_f1_best}


# --------------------------- SSL SWEEP utilities ---------------------------

def parse_epoch_from_filename(p: str) -> int:
    m = re.search(r"_e(\d+)\.keras$", os.path.basename(p))
    return int(m.group(1)) if m else -1


def gather_sweep_candidates() -> List[str]:
    cands: List[str] = []
    if SWEEP_SSL_BACKBONES:
        cands.extend(glob.glob(SWEEP_GLOB))
        if SWEEP_INCLUDE_FINAL and os.path.exists(SSL_BACKBONE_PATH):
            cands.append(SSL_BACKBONE_PATH)
    if not cands and os.path.exists(SSL_BACKBONE_PATH):
        cands = [SSL_BACKBONE_PATH]
    cands = sorted(set(cands), key=lambda p: (parse_epoch_from_filename(p), p))
    if SWEEP_MAX_CANDIDATES and len(cands) > SWEEP_MAX_CANDIDATES:
        idxs = np.linspace(0, len(cands) - 1, SWEEP_MAX_CANDIDATES).round().astype(int)
        cands = [cands[i] for i in idxs]
    return cands


def print_leaderboard(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("\nNo sweep results to display.")
        return
    key = "macro_f1" if SWEEP_SELECT_BY.lower() == "macro_f1" else "acc"
    rows_sorted = sorted(rows, key=lambda r: r[key], reverse=True)
    print("\nSSL checkpoint leaderboard (linear‑probe sweep):")
    print(f"{'rank':<4} {'epoch':>5} {'acc':>8} {'macroF1':>9}  {'img':>4}  path")
    for i, r in enumerate(rows_sorted, 1):
        ep = parse_epoch_from_filename(r["path"])  # type: ignore[index]
        print(f"{i:<4} {ep:>5} {r['acc']:>8.4f} {r['macro_f1']:>9.4f}  {r['img_size']:>4}  {r['path']}")
    if SWEEP_SAVE_CSV:
        with open(SWEEP_SAVE_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["rank", "epoch", "acc", "macro_f1", "img_size", "path"])
            w.writeheader()
            for i, r in enumerate(rows_sorted, 1):
                w.writerow(
                    {
                        "rank": i,
                        "epoch": parse_epoch_from_filename(r["path"]),
                        "acc": f"{r['acc']:.6f}",
                        "macro_f1": f"{r['macro_f1']:.6f}",
                        "img_size": r["img_size"],
                        "path": r["path"],
                    }
                )
        print(f"(Saved leaderboard to {SWEEP_SAVE_CSV})")


# --------------------------------- MAIN ---------------------------------
if __name__ == "__main__":
    print("WARNING: TF logs before absl::InitializeLog() go to STDERR")
    print("Using mixed precision: mixed_float16" if USE_MIXED_PRECISION else "Using float32")

    candidates = gather_sweep_candidates()
    print("\nBackbone(s) to use:")
    for p in candidates:
        print(" -", p)

    if SWEEP_SSL_BACKBONES and len(candidates) > 1:
        sweep_rows = []
        for p in candidates:
            res = run_experiment(p, IMG_SIZE, sweep_mode=True)
            sweep_rows.append(res)
        print_leaderboard(sweep_rows)
        key = "macro_f1" if SWEEP_SELECT_BY.lower() == "macro_f1" else "acc"
        best = sorted(sweep_rows, key=lambda r: r[key], reverse=True)[0]
        chosen = best["path"]
        print(f"\nChosen backbone for full training (by {key}): {chosen}")
        _ = run_experiment(chosen, IMG_SIZE, sweep_mode=False)
    else:
        chosen = candidates[0] if candidates else SSL_BACKBONE_PATH
        print(f"\nRunning full training with backbone: {chosen}")
        _ = run_experiment(chosen, IMG_SIZE, sweep_mode=False)
