import os, sys, argparse, json, shutil, pathlib
# --- Disable XLA/JIT & expensive autotune BEFORE importing TF ---
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=1 --xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("TF_USE_CUDNN_AUTOTUNE", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

tf.config.optimizer.set_jit(False)

# Try to enable GPU memory growth
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e:
        print("GPU memory growth not set:", e)

# ===== ConvNeXt preprocess layer (must match training) =====
from tensorflow.keras.applications import convnext as keras_convnext

@tf.keras.utils.register_keras_serializable(package="custom")
class ConvNeXtPreprocess(tf.keras.layers.Layer):
    def call(self, x):  # expects float32 in [0..255]
        return keras_convnext.preprocess_input(x)

# --- Shim for models saved with Lambda(function='preprocess_input') ---
from tensorflow.keras.applications.convnext import preprocess_input as _convnext_preprocess_input

@tf.keras.utils.register_keras_serializable(package="custom", name="preprocess_input")
def preprocess_input(x):
    """Shim for models saved with Lambda(function='preprocess_input')."""
    return _convnext_preprocess_input(x)

globals()["function"] = preprocess_input  # needed for legacy builtins.function references

# ------------ utils ------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

def list_images(root: str):
    """List all image files (non-recursive)."""
    p = pathlib.Path(root)
    return [str(q) for q in p.glob("*") if q.is_file() and q.suffix.lower() in IMG_EXTS]

def safe_makedirs(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def load_class_names(class_file: str, model_out_dim: int):
    with open(class_file, "r") as f:
        names = json.load(f)
    if isinstance(names, dict):
        items = sorted(((int(k), v) for k, v in names.items()), key=lambda x: x[0])
        names = [v for _, v in items]
    assert len(names) == model_out_dim, f"class_file has {len(names)} names; model has {model_out_dim} outputs"
    return names

# ---------- image I/O ----------
def build_decoder(image_size: int):
    @tf.function(reduce_retracing=True, jit_compile=False)
    def _decode(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)  # uint8
        img = tf.image.convert_image_dtype(img, tf.float32)                    # [0,1]
        img = tf.image.resize_with_pad(img, image_size, image_size, antialias=True)
        img = tf.cast(img * 255.0, tf.float16)                                 # keep fp16 in pipeline
        return img
    return _decode

# ---------- inference ----------
def build_infer_fn(model, img_size: int):
    @tf.function(
        input_signature=[tf.TensorSpec([None, img_size, img_size, 3], tf.float16)],
        reduce_retracing=True, jit_compile=False
    )
    def infer(x):
        x32 = tf.cast(x, tf.float32)   # model expects float32 0..255; ConvNeXtPreprocess handles normalization
        y = model(x32, training=False)
        return tf.cast(y, tf.float32)
    return infer

# ---------- dataset ----------
def make_ds(paths, batch, image_size):
    decode = build_decoder(image_size)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    ds = ds.map(lambda p: (p, decode(p)), num_parallel_calls=2)
    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(1)
    return ds

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder of unsorted images.")
    ap.add_argument("--out_dir",   required=True, help="Output root where class folders will be created.")
    ap.add_argument("--model",     required=True, help="Path to .keras/.h5 or SavedModel directory.")
    ap.add_argument("--class_file", required=True, help="class_names.json mapping used at training time.")
    ap.add_argument("--batch", type=int, default=32, help="Inference batch size (perf only).")
    ap.add_argument("--min_conf", type=float, default=0.55, help="Min softmax prob to assign; else goes to _low_conf.")
    ap.add_argument("--move", action="store_true", help="Move files instead of copy (destructive).")
    args = ap.parse_args()

    # Load model (map missing function names)
    custom_objects = {
        "ConvNeXtPreprocess": ConvNeXtPreprocess,
        "custom>ConvNeXtPreprocess": ConvNeXtPreprocess,
        "preprocess_input": preprocess_input,
        "function": preprocess_input,
    }

    model = tf.keras.models.load_model(
        args.model,
        compile=False,
        custom_objects=custom_objects,
        safe_mode=False,
    )
    model.trainable = False

    in_shape = model.inputs[0].shape
    h, w = int(in_shape[1]), int(in_shape[2])
    assert h == w and h is not None, f"Unexpected model input shape {in_shape}"
    print(f"Loaded {args.model} | expects input {h}x{w}", end="")

    out_dim = int(model.outputs[0].shape[-1])
    classes = load_class_names(args.class_file, out_dim)
    print(f" | classes={classes}")

    paths = list_images(args.input_dir)
    if not paths:
        print("No images found. Exiting.")
        return
    print(f"Found {len(paths)} images to sort.")

    ds = make_ds(paths, args.batch, h)
    infer = build_infer_fn(model, h)

    # Prepare output dirs
    for c in classes:
        safe_makedirs(os.path.join(args.out_dir, c))
    low_conf_dir = os.path.join(args.out_dir, "_low_conf")
    safe_makedirs(low_conf_dir)

    moved, copied, total = 0, 0, len(paths)
    processed = 0

    for batch_paths, batch_imgs in ds:
        logits = infer(batch_imgs).numpy()
        probs = tf.nn.softmax(logits, axis=1).numpy()
        pred_idx = probs.argmax(axis=1)
        conf = probs.max(axis=1)

        for i in range(len(pred_idx)):
            processed += 1
            p = batch_paths[i].numpy().decode("utf-8")
            dst_dir = os.path.join(args.out_dir, classes[int(pred_idx[i])]) if conf[i] >= args.min_conf else low_conf_dir
            safe_makedirs(dst_dir)
            dst_path = os.path.join(dst_dir, os.path.basename(p))
            try:
                if args.move:
                    shutil.move(p, dst_path); moved += 1
                else:
                    base, ext = os.path.splitext(dst_path)
                    k, final = 1, dst_path
                    while os.path.exists(final):
                        final = f"{base}_{k}{ext}"; k += 1
                    shutil.copy2(p, final); copied += 1
            except Exception as e:
                print(f"[WARN] failed to {'move' if args.move else 'copy'} {p} -> {dst_dir}: {e}")

            if processed % 100 == 0 or processed == total:
                print(f"Progress: {processed}/{total} images processed")

    print(f"Done. {'Moved' if args.move else 'Copied'}: {moved if args.move else copied} files.")

if __name__ == "__main__":
    main()
