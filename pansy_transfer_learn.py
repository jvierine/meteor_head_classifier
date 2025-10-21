#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PANSY Radar Data Classification — Fixed & Improved (TF 2.10 compatible)

Key fixes vs your previous script
---------------------------------
1) **Evaluation crash fixed**: removed erroneous `tf.one_hot(..., tf.float32)` usage
   in tf.data maps. We now use **sparse integer labels everywhere** with
   `SparseCategoricalCrossentropy` + `SparseTopKCategoricalAccuracy`.
2) **Preprocessing correctness**: single, explicit EfficientNetB3 preprocessing
   ([-1, 1]) done exactly once per image via `preprocess_input`.
3) **Stable TTA**: deterministic 8× transforms (identity/flips/rotations/transpose)
   performed outside the model (keeps BatchNorm in inference mode) and averaged.
4) **Safer mixed precision**: global `mixed_float16` policy with final Dense
   forced to `float32` for numerically stable softmax/loss.
5) **Learning-rate schedule & stages**: feature-extract → partial unfreeze
   ("tail") → full unfreeze, with progressively smaller LRs.
6) **Metrics**: prints accuracy, top‑3 accuracy, confusion matrix, and
   balanced accuracy (macro recall) on the test set.

Tested under: TensorFlow 2.10.x on Windows (RTX 40xx); no TensorFlow Addons required.

Usage
-----
python pansy_transfer_learn.py \
  --data_dir C:/Users/ragav/meteor_head_classifier/data1/pansy/sorted_images \
  --img_size 300 --batch_size 32 --seed 1337

The script writes best weights and logs under ./output/ .
"""

import os
import sys
import math
import json
import random
import argparse
from collections import Counter, defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

AUTOTUNE = tf.data.AUTOTUNE

# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int = 1337):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def enable_mixed_precision():
    try:
        mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled (float16).")
    except Exception as e:
        print("Mixed precision not enabled:", e)


def configure_gpus():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s), memory growth enabled")
        except Exception as e:
            print("Could not set GPU memory growth:", e)
    else:
        print("No GPU found. Running on CPU.")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Dataset building
# ---------------------------

def list_images(data_dir):
    """Return (paths, labels, class_names).

    Expects directory structure:
    data_dir/
        class_0/
            *.png|jpg
        class_1/
            ...
    """
    class_names = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])
    if not class_names:
        raise RuntimeError(f"No class subdirectories found under: {data_dir}")

    paths, labels = [], []
    for idx, cname in enumerate(class_names):
        cdir = os.path.join(data_dir, cname)
        for root, _, files in os.walk(cdir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    paths.append(os.path.join(root, f))
                    labels.append(idx)
    return np.array(paths), np.array(labels, dtype=np.int32), class_names


def stratified_split(paths, labels, seed=1337, train_ratio=0.72, val_ratio=0.18):
    """Deterministic stratified split into train/val/test."""
    rng = np.random.RandomState(seed)

    idx_by_class = defaultdict(list)
    for i, y in enumerate(labels):
        idx_by_class[int(y)].append(i)

    tr_idx, va_idx, te_idx = [], [], []
    for y, idxs in idx_by_class.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = int(round(n * train_ratio))
        n_va = int(round(n * val_ratio))
        n_te = n - n_tr - n_va
        tr_idx.extend(idxs[:n_tr])
        va_idx.extend(idxs[n_tr:n_tr + n_va])
        te_idx.extend(idxs[n_tr + n_va:])

    def subset(indexes):
        return paths[indexes], labels[indexes]

    return subset(tr_idx), subset(va_idx), subset(te_idx)


def load_image(path, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size], method='bilinear')
    img = tf.cast(img, tf.float32)  # 0..255
    # EfficientNet preprocess brings to [-1, 1]
    img = effnet_preprocess(img)
    return img


def make_ds(paths, labels, img_size, batch_size, shuffle=False, cache=False):
    paths = tf.convert_to_tensor(paths)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map(p, y):
        x = load_image(p, img_size)
        return x, y

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ---------------------------
# Model
# ---------------------------

def build_model(img_size, num_classes):
    inp = layers.Input(shape=(img_size, img_size, 3))

    # Light in-model augmentation (applies in training only).
    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),  # ~±5°
        layers.RandomContrast(0.1),
    ], name="aug")

    x = aug(inp)
    base = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=x)
    base.trainable = False  # feature extraction first

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    # IMPORTANT: force float32 for numerically stable softmax under mixed precision
    out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs=inp, outputs=out, name="EfficientNetB3_meteor_classifier")
    return model, base


# ---------------------------
# Class weights (effective number of samples)
# ---------------------------

def effective_number_class_weights(train_labels, beta=0.9999):
    counts = Counter(train_labels.tolist())
    classes = sorted(counts.keys())
    weights = {}
    for c in classes:
        n = counts[c]
        eff_num = (1.0 - math.pow(beta, n)) / (1.0 - beta)
        weights[c] = 1.0 / eff_num
    # Normalize so avg weight ~1
    mean_w = np.mean(list(weights.values()))
    for c in weights:
        weights[c] = float(weights[c] / mean_w)
    return weights, counts


# ---------------------------
# Training helpers
# ---------------------------

def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top3_acc'),
        ],
    )


def stage_train(model, base, train_ds, val_ds, out_path, lr, epochs, unfreeze_from=None):
    if unfreeze_from is not None:
        # Unfreeze last N layers (or by name). Here we unfreeze from given index of base.layers
        for layer in base.layers[unfreeze_from:]:
            layer.trainable = True
        # Keep earlier layers frozen
        for layer in base.layers[:unfreeze_from]:
            layer.trainable = False
        print(f"Unfreezing from base layer idx {unfreeze_from} / {len(base.layers)} (tail)")
    else:
        # Feature extraction stage (base frozen)
        base.trainable = False

    compile_model(model, lr=lr)

    ckpt = ModelCheckpoint(
        out_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
    early = EarlyStopping(monitor='val_loss', patience=max(3, epochs // 4), restore_best_weights=True, verbose=1)
    # Quiet LR reducer for plateaus
    reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[ckpt, early, reduce],
        verbose=1,
    )
    return history


# ---------------------------
# Deterministic TTA transforms
# ---------------------------

def tta_transforms(x):
    """Return list of 8 deterministic transforms of x (image batch)."""
    # x is a float32 tensor in [-1, 1]
    out = [x]
    out.append(tf.image.flip_left_right(x))
    out.append(tf.image.flip_up_down(x))
    out.append(tf.image.flip_left_right(tf.image.flip_up_down(x)))
    out.append(tf.image.rot90(x, k=1))
    out.append(tf.image.rot90(x, k=2))
    out.append(tf.image.rot90(x, k=3))
    out.append(tf.image.transpose(x))
    return out


def predict_with_tta(model, ds, tta=8):
    """Average predictions over 8 deterministic transforms."""
    assert tta in (1, 2, 4, 8), "tta must be one of {1,2,4,8}"
    # Gather batches
    preds_sum = None
    labels_all = []

    for batch_x, batch_y in ds:
        labels_all.append(batch_y.numpy())
        # Build transforms
        t_list = tta_transforms(batch_x)
        t_list = t_list[:tta]  # use first tta transforms
        batch_pred_sum = None
        for t_img in t_list:
            p = model.predict(t_img, verbose=0)
            batch_pred_sum = p if batch_pred_sum is None else (batch_pred_sum + p)
        batch_pred_sum /= float(len(t_list))
        preds_sum = batch_pred_sum if preds_sum is None else np.vstack([preds_sum, batch_pred_sum])

    y_true = np.concatenate(labels_all, axis=0)
    y_pred = preds_sum  # probabilities
    return y_true, y_pred


# ---------------------------
# Metrics / Evaluation
# ---------------------------

def topk_acc(y_true, y_prob, k=3):
    topk = np.argpartition(-y_prob, kth=k-1, axis=1)[:, :k]
    hit = np.any(topk == y_true.reshape(-1, 1), axis=1)
    return float(np.mean(hit))


def confusion_matrix(y_true, y_pred, num_classes):
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes, dtype=tf.int32).numpy()
    return cm


def balanced_accuracy(cm):
    # Macro recall = mean(diagonal / row_sum)
    recalls = []
    for i in range(cm.shape[0]):
        denom = cm[i].sum()
        recalls.append(cm[i, i] / denom if denom > 0 else 0.0)
    return float(np.mean(recalls))


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to root of class folders')
    parser.add_argument('--img_size', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--out_dir', type=str, default='./output')
    parser.add_argument('--tta', type=int, default=8, choices=[1,2,4,8])
    args = parser.parse_args()

    print("\n===== PANSY Radar Data Classification (fixed) =====")
    print("Data:", args.data_dir)

    # Env setup
    set_seed(args.seed)
    enable_mixed_precision()
    configure_gpus()

    out_models = os.path.join(args.out_dir, 'models')
    out_misc = os.path.join(args.out_dir, 'misc')
    ensure_dir(out_models)
    ensure_dir(out_misc)

    # Load & split
    paths, labels, class_names = list_images(args.data_dir)
    print("Class names:", class_names)

    (tr_p, tr_y), (va_p, va_y), (te_p, te_y) = stratified_split(paths, labels, seed=args.seed)
    print(f"Split sizes -> Train: {len(tr_p)}, Val: {len(va_p)}, Test: {len(te_p)}")
    print("Train counts:", Counter(tr_y.tolist()))
    print("Val counts:", Counter(va_y.tolist()))
    print("Test counts:", Counter(te_y.tolist()))

    # Class weights
    cls_weights, train_counts = effective_number_class_weights(tr_y)
    print("Class weights (effective-number):", cls_weights)

    # Datasets
    train_ds = make_ds(tr_p, tr_y, args.img_size, args.batch_size, shuffle=True, cache=False)
    val_ds = make_ds(va_p, va_y, args.img_size, args.batch_size, shuffle=False, cache=False)
    test_ds = make_ds(te_p, te_y, args.img_size, args.batch_size, shuffle=False, cache=False)

    # Build model
    model, base = build_model(args.img_size, num_classes=len(class_names))
    model.summary()

    # ---------- Stage 1: Feature Extraction ----------
    print("\n===== Feature Extraction =====")
    fe_path = os.path.join(out_models, 'best_feature.weights.h5')
    stage_train(
        model, base,
        train_ds, val_ds,
        out_path=fe_path,
        lr=1e-3,
        epochs=12,
        unfreeze_from=None,
    )

    # ---------- Stage 2: Fine-tune (tail layers) ----------
    print("\n===== Fine-tune (tail layers) =====")
    # Unfreeze last ~20% of base layers
    tail_from = int(len(base.layers) * 0.8)
    ft_tail_path = os.path.join(out_models, 'best_ft_tail.weights.h5')
    stage_train(
        model, base,
        train_ds, val_ds,
        out_path=ft_tail_path,
        lr=1e-4,
        epochs=25,
        unfreeze_from=tail_from,
    )

    # ---------- Stage 3: Fine-tune (all layers) ----------
    print("\n===== Fine-tune (all layers) =====")
    for layer in base.layers:
        layer.trainable = True
    ft_all_path = os.path.join(out_models, 'best_ft_all.weights.h5')
    stage_train(
        model, base,
        train_ds, val_ds,
        out_path=ft_all_path,
        lr=1e-5,
        epochs=10,
        unfreeze_from=len(base.layers),  # essentially all
    )

    # Save final full model (SavedModel + weights)
    final_model_dir = os.path.join(args.out_dir, 'final_model')
    ensure_dir(final_model_dir)
    try:
        model.save(final_model_dir)
    except Exception as e:
        print("WARNING: SavedModel export failed:", e)

    # ---------- Evaluation (TTA) ----------
    print("\n===== Evaluation (Test + TTA={}) =====".format(args.tta))
    # Load best-all weights if present
    if os.path.exists(ft_all_path):
        try:
            model.load_weights(ft_all_path)
        except Exception as e:
            print("Could not load best_ft_all weights:", e)

    y_true, y_prob = predict_with_tta(model, test_ds, tta=args.tta)
    y_pred = np.argmax(y_prob, axis=1).astype(np.int32)

    acc = (y_pred == y_true).mean()
    t3 = topk_acc(y_true, y_prob, k=3)
    cm = confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    bacc = balanced_accuracy(cm)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Test top-3 accuracy: {t3:.4f}")
    print(f"Balanced accuracy: {bacc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

    # Save predictions and metrics
    np.savez(os.path.join(out_misc, 'test_preds_tta.npz'),
             y_true=y_true, y_prob=y_prob, y_pred=y_pred,
             class_names=np.array(class_names))

    with open(os.path.join(out_misc, 'metrics.json'), 'w') as f:
        json.dump({
            'accuracy': float(acc),
            'top3_accuracy': float(t3),
            'balanced_accuracy': float(bacc),
            'class_names': class_names,
            'train_counts': {str(k): int(v) for k,v in train_counts.items()},
        }, f, indent=2)


if __name__ == '__main__':
    main()
