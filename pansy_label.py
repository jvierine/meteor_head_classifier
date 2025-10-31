# pansy_label_autoload.py

# ---- Make TF quiet & disable costly autotune/JIT BEFORE importing TF ----
import os
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=1 --xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("TF_USE_CUDNN_AUTOTUNE", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Try OpenCV; fall back to matplotlib if no GUI
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

import matplotlib.pyplot as plt

# ---------- Helpers ----------

def to_wsl_path(p: str) -> str:
    """Convert Windows 'C:\\...' to WSL '/mnt/c/...' when running on Linux."""
    if not p:
        return p
    if os.name == "posix" and re.match(r"^[A-Za-z]:\\", p):
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return p

def ensure_dir_exists(path_str: str, what: str = "Directory"):
    if not os.path.isdir(path_str):
        raise FileNotFoundError(f"{what} not found: {path_str}")

def gui_available_for_opencv() -> bool:
    if not OPENCV_AVAILABLE:
        return False
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

def list_dir(path_str: str):
    try:
        entries = sorted(os.listdir(path_str))
        print(f"\nContents of {path_str}:")
        for e in entries:
            print(" -", e)
    except Exception as e:
        print(f"[WARN] Could not list {path_str}: {e}")

# ====== Register custom objects needed to load your model ======
from tensorflow.keras.applications import convnext as keras_convnext

@tf.keras.utils.register_keras_serializable(package="custom")
class ConvNeXtPreprocess(tf.keras.layers.Layer):
    # Expects float32 in [0..255]
    def call(self, x):
        return keras_convnext.preprocess_input(x)

# Shim for Lambda(function='preprocess_input') serialized in the model
from tensorflow.keras.applications.convnext import preprocess_input as _convnext_preprocess_input
import builtins

@tf.keras.utils.register_keras_serializable(name="preprocess_input")
def preprocess_input(x):
    return _convnext_preprocess_input(x)

@tf.keras.utils.register_keras_serializable(name="function")
def function(x):
    # some serializers recorded registered_name="function"
    return _convnext_preprocess_input(x)

# Also expose on builtins (models may reference builtins.preprocess_input/function)
builtins.preprocess_input = preprocess_input
builtins.function = function

CUSTOM_OBJECTS = {
    "ConvNeXtPreprocess": ConvNeXtPreprocess,
    "custom>ConvNeXtPreprocess": ConvNeXtPreprocess,
    "preprocess_input": preprocess_input,
    "function": function,
    "builtins.preprocess_input": preprocess_input,
    "builtins.function": function,
}

def try_load_model_any(path_candidate: str):
    """Load a Keras model from a file or SavedModel directory."""
    if os.path.isfile(path_candidate):
        print(f"Attempting to load model file: {path_candidate}")
        return load_model(path_candidate, compile=False, custom_objects=CUSTOM_OBJECTS, safe_mode=False)

    if os.path.isdir(path_candidate):
        if os.path.isfile(os.path.join(path_candidate, "saved_model.pb")) or \
           os.path.isfile(os.path.join(path_candidate, "keras_metadata.pb")):
            print(f"Attempting to load SavedModel directory: {path_candidate}")
            return load_model(path_candidate, compile=False, custom_objects=CUSTOM_OBJECTS, safe_mode=False)

    raise FileNotFoundError(f"No model found at: {path_candidate}")

def smart_find_and_load_model(preferred_path: str):
    """Try to load model from various file extensions and locations."""
    import glob
    preferred_path = to_wsl_path(preferred_path)
    parent = os.path.dirname(preferred_path) or "."
    stem = Path(preferred_path).stem

    if os.path.exists(preferred_path):
        return try_load_model_any(preferred_path)

    print(f"[INFO] Preferred model not found: {preferred_path}")
    list_dir(parent)

    candidates = [
        os.path.join(parent, stem + ext)
        for ext in (".keras", ".h5", ".hdf5")
    ] + [
        os.path.join(parent, stem),
        os.path.join(parent, "saved_model"),
        os.path.join(parent, "model"),
    ]

    candidates += sorted(glob.glob(os.path.join(parent, "*.keras")))
    candidates += sorted(glob.glob(os.path.join(parent, "*.h5")))
    candidates += sorted(glob.glob(os.path.join(parent, "*.hdf5")))

    tried = []
    for c in candidates:
        if os.path.exists(c):
            try:
                return try_load_model_any(c)
            except Exception as e:
                tried.append((c, str(e)))

    msg = "[ERROR] Could not locate/load a model. Tried:\n"
    for c, err in tried:
        msg += f"  - {c}  --> {err}\n"
    raise FileNotFoundError(msg)

# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="Label images with a Keras ConvNeXt model (no dataset needed).")
    p.add_argument("--predict", required=True, help="Path to folder with images to label.")
    p.add_argument("--model",   required=True, help="Path to model (.keras/.h5) or SavedModel directory.")
    p.add_argument("--classes", required=True, help="Path to class_names.json (from training).")
    return p.parse_args()

# ---------- Main ----------

def main():
    args = parse_args()

    PREDICT_DIR = to_wsl_path(args.predict)
    MODEL_PATH  = to_wsl_path(args.model)
    CLASS_FILE  = to_wsl_path(args.classes)

    print(f"Predict dir:  {PREDICT_DIR}")
    print(f"Model path:   {MODEL_PATH}")
    print(f"Class file:   {CLASS_FILE}")

    ensure_dir_exists(PREDICT_DIR, "Predict dir")
    if not os.path.isfile(CLASS_FILE):
        raise FileNotFoundError(f"class_names.json not found: {CLASS_FILE}")

    # Load class names from JSON only (no dataset usage)
    with open(CLASS_FILE, "r") as f:
        names = json.load(f)
    if isinstance(names, dict):
        items = sorted(((int(k), v) for k, v in names.items()), key=lambda x: x[0])
        class_names = [v for _, v in items]
    elif isinstance(names, list):
        class_names = names
    else:
        raise ValueError("class_names.json must be a list or a dict of index->name")
    print(f"Loaded class names: {class_names}")

    # Load model
    model = smart_find_and_load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # Check model input
    in_shape = model.inputs[0].shape
    h, w = int(in_shape[1]), int(in_shape[2])
    assert h is not None and h == w, f"Unexpected model input shape: {in_shape}"
    MODEL_IMG_SIZE = (h, w)
    print(f"Model expects input size: {MODEL_IMG_SIZE}")

    # Sanity check: output dims match class count
    out_dim = int(model.outputs[0].shape[-1])
    if out_dim != len(class_names):
        raise ValueError(f"Model output dim ({out_dim}) != number of classes ({len(class_names)}).")

    # Collect images
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [str(p) for p in Path(PREDICT_DIR).glob("*") if p.suffix.lower() in IMG_EXTS]
    if not files:
        print(f"[INFO] No images in: {PREDICT_DIR}")
        list_dir(PREDICT_DIR)
        raise FileNotFoundError(f"No images found in: {PREDICT_DIR}")

    np.random.shuffle(files)
    print(f"Found {len(files)} files to label.")

    def predict_image_path(img_path: str) -> str:
        # Resize to the model's actual expected size; keep 0..255 float
        img = image.load_img(img_path, target_size=MODEL_IMG_SIZE)
        arr = image.img_to_array(img).astype("float32")
        arr = np.expand_dims(arr, axis=0)  # DO NOT divide by 255 (preprocess layer handles it)
        preds = model.predict(arr, verbose=0)

        # Convert logits->probs if needed
        if preds.ndim == 2:
            probs = tf.nn.softmax(preds, axis=1).numpy()[0]
        else:
            probs = preds[0]

        top_idx = int(np.argmax(probs))
        # Optional: print per-class probabilities for debugging
        print("Probabilities:", {class_names[i]: float(probs[i]) for i in range(len(class_names))})
        return class_names[top_idx]

    def show_image_cv2(img_path: str, window_title: str = "Label Image", scale: float = 2.0):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Could not read image with cv2: {img_path}")
            return
        if scale != 1.0:
            h0, w0 = img.shape[:2]
            img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))
        cv2.imshow(window_title, img)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if chr(key) == "x":
            print("Exiting on 'x' key.")
            sys.exit(0)

    def show_image_matplotlib(img_path: str, title: str = "Label Image"):
        if OPENCV_AVAILABLE:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[WARN] Could not read image: {img_path}")
                return
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            pil_img = image.load_img(img_path)
            img_rgb = np.array(pil_img)

        plt.figure()
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis("off")
        plt.show(block=True)
        choice = input("Press 'n' for next or 'x' to exit: ").strip().lower()
        if choice == "x":
            print("Exiting on 'x'.")
            sys.exit(0)

    for f in files:
        print(f"\nImage: {f}")
        pred = predict_image_path(f)
        print(f"Predicted class: {pred}")
        title = f"Predicted: {pred}"
        if gui_available_for_opencv():
            show_image_cv2(f, window_title=title)
        else:
            show_image_matplotlib(f, title=title)

    print("Done.")

if __name__ == "__main__":
    main()
