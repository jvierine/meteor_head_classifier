import os, sys, argparse, json, shutil, pathlib
# --- Hard-disable XLA/JIT & expensive autotune BEFORE importing TF ---
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
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception as e: print("GPU memory growth not set:", e)

# ===== ConvNeXt preprocess layer (must match training) =====
from tensorflow.keras.applications import convnext as keras_convnext

@tf.keras.utils.register_keras_serializable(package="custom")
class ConvNeXtPreprocess(tf.keras.layers.Layer):
    def call(self, x):  # expects float32 in [0..255]
        return keras_convnext.preprocess_input(x)

# ------------ utils ------------
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".gif",".tif",".tiff"}

def list_images(root: str, recursive: bool):
    p = pathlib.Path(root)
    it = (q for q in (p.rglob("*") if recursive else p.glob("*"))
          if q.is_file() and q.suffix.lower() in IMG_EXTS)
    return [str(q) for q in it]

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

# ---------- TTA helpers ----------
@tf.function(reduce_retracing=True, jit_compile=False)
def _pad_crop_shift(x, dy, dx):
    H = tf.shape(x)[1]; W = tf.shape(x)[2]
    pad_t = tf.maximum(dy, 0); pad_b = tf.maximum(-dy, 0)
    pad_l = tf.maximum(dx, 0); pad_r = tf.maximum(-dx, 0)
    xpad = tf.pad(x, [[0,0],[pad_t,pad_b],[pad_l,pad_r],[0,0]], mode="REFLECT")
    return tf.image.crop_to_bounding_box(xpad, offset_height=pad_b, offset_width=pad_r,
                                         target_height=H, target_width=W)

def build_infer_fn(model, img_size: int, flip: bool, shift_px: int):
    @tf.function(
        input_signature=[tf.TensorSpec([None, img_size, img_size, 3], tf.float16)],
        reduce_retracing=True, jit_compile=False
    )
    def infer(x):
        x32 = tf.cast(x, tf.float32)
        y = model(x32, training=False)
        return tf.cast(y, tf.float32)

    def predict_with_tta(x):
        preds = [infer(x)]
        if flip:
            preds.append(infer(tf.image.flip_up_down(x)))
        if isinstance(shift_px, int) and shift_px > 0:
            preds.append(infer(_pad_crop_shift(x,  shift_px,  0)))
            preds.append(infer(_pad_crop_shift(x, -shift_px,  0)))
            preds.append(infer(_pad_crop_shift(x,  0,  shift_px)))
            preds.append(infer(_pad_crop_shift(x,  0, -shift_px)))
        return tf.add_n(preds) / float(len(preds))
    return infer, predict_with_tta

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
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir",   required=True)
    ap.add_argument("--model",     default="finetuned_ema.keras")
    ap.add_argument("--class_file", default="ssl_ckpts/class_names.json")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tta_shift", type=int, default=3)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--min_conf", type=float, default=0.55)
    ap.add_argument("--move", action="store_true", help="move files instead of copy")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    # Load model (provide custom layer mapping)
    custom_objects = {
        "ConvNeXtPreprocess": ConvNeXtPreprocess,
        "custom>ConvNeXtPreprocess": ConvNeXtPreprocess,  # just in case
    }
    model = tf.keras.models.load_model(args.model, compile=False, custom_objects=custom_objects)
    model.trainable = False

    # Expected input size
    in_shape = model.inputs[0].shape
    h, w = int(in_shape[1]), int(in_shape[2])
    assert h == w and h is not None, f"Unexpected model input shape {in_shape}"
    print(f"Loaded {args.model} | expects input {h}x{w}", end="")

    # Class names
    out_dim = int(model.outputs[0].shape[-1])
    classes = load_class_names(args.class_file, out_dim)
    print(f" | classes={classes}")

    # Collect images
    paths = list_images(args.input_dir, args.recursive)
    if not paths:
        print("No images found. Exiting.")
        return
    print(f"Found {len(paths)} images to sort.")

    # Build ds & infer fns
    ds = make_ds(paths, args.batch, h)
    infer, predict_with_tta = build_infer_fn(model, h, args.flip, args.tta_shift)

    # Output dirs
    for c in classes: safe_makedirs(os.path.join(args.out_dir, c))
    low_conf_dir = os.path.join(args.out_dir, "_low_conf"); safe_makedirs(low_conf_dir)

    # Run
    moved, copied = 0, 0
    for batch_paths, batch_imgs in ds:
        probs = predict_with_tta(batch_imgs).numpy() if (args.flip or args.tta_shift > 0) else infer(batch_imgs).numpy()
        pred_idx = probs.argmax(axis=1)
        conf = probs.max(axis=1)

        for i in range(len(pred_idx)):
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

    print(f"Done. {'Moved' if args.move else 'Copied'}: {moved if args.move else copied} files.")

if __name__ == "__main__":
    main()
