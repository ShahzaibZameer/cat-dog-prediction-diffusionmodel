"""
Cat / Dog / Neither classifier (TensorFlow / Keras, MobileNetV2 transfer learning)

Three classes:  cat, dog, neither   ("neither" = anything that is not a cat or dog)

Folders:
    dataset/train/cat, dog, neither     REAL training images (neither: your own real photos)
    dataset/val/cat,   dog, neither     REAL validation images (never mixed with generated ones)
    dataset_sd/train/cat, dog, neither  Stable Diffusion images from sd_generate.py (optional)

Anything that is not a cat or dog is reported as "Out of distribution". Two things trigger it:
    1. the model itself says 'neither' or is unsure
    2. the photo looks too different from all known cats/dogs (fingerprint distance check,
       saved after training in ood_reference.npz)

Usage:
    python cat_dog_cnn_v4.py train --sd          # real + Stable Diffusion images (recommended)
    python cat_dog_cnn_v4.py train               # real images only (needs real 'neither' photos)
    python cat_dog_cnn_v4.py predict image.jpg
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras import layers, models, callbacks

IMG_SIZE = (160, 160)
BATCH = 16
EPOCHS = 30
FINE_TUNE_EPOCHS = 15
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
SD_DIR = "dataset_sd/train"
MODEL_PATH = "cat_dog_neither_model.keras"
CONF_THRESHOLD = 0.70
OOD_FILE = "ood_reference.npz"   # saved after training, used by predict
OOD_KEEP = 0.95                  # keep 95% of real val cats/dogs; lower = stricter
CLASS_NAMES = ["cat", "dog", "neither"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT)])


def ensure_dirs(dirs):
    for d in dirs:
        for c in CLASS_NAMES:
            os.makedirs(os.path.join(d, c), exist_ok=True)


# ---------------------------------------------------------------- data
def load_data(use_sd):
    dirs = [TRAIN_DIR, VAL_DIR] + ([SD_DIR] if use_sd else [])
    ensure_dirs(dirs)

    kw = dict(image_size=IMG_SIZE, label_mode="int", class_names=CLASS_NAMES, batch_size=None)
    train = tf.keras.utils.image_dataset_from_directory(TRAIN_DIR, shuffle=True, seed=42, **kw)
    if use_sd:
        sd = tf.keras.utils.image_dataset_from_directory(SD_DIR, shuffle=True, seed=42, **kw)
        train = train.concatenate(sd)
    train_ds = train.shuffle(3000, seed=42).batch(BATCH).prefetch(tf.data.AUTOTUNE)

    # Validation = REAL images only, so the score is honest
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int",
        class_names=CLASS_NAMES, shuffle=False).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def class_weights(use_sd):
    dirs = [TRAIN_DIR] + ([SD_DIR] if use_sd else [])
    counts = [sum(count_images(os.path.join(d, c)) for d in dirs) for c in CLASS_NAMES]
    print("Train counts:", dict(zip(CLASS_NAMES, counts)))
    for c, n in zip(CLASS_NAMES, counts):
        if n == 0:
            raise SystemExit(
                f"No training images for '{c}'. For 'neither' run sd_generate.py and use "
                f"--sd, or add real photos to {TRAIN_DIR}/neither")
    total = sum(counts)
    weights = {i: total / (len(counts) * n) for i, n in enumerate(counts)}
    print("Class weights:", {CLASS_NAMES[i]: round(w, 2) for i, w in weights.items()})
    return weights


augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.1, 0.1),
], name="augmentation")


# --------------------------------------------------------------- model
def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return models.Model(inputs, outputs, name="mobilenet_3class"), base


# ---------------------------------------------------------- evaluation
def report(model, val_ds):
    y_true, y_pred = [], []
    for x, y in val_ds:
        p = model.predict(x, verbose=0)
        y_pred += np.argmax(p, axis=1).tolist()
        y_true += y.numpy().astype(int).tolist()
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print(f"\nValidation accuracy: {(y_true == y_pred).mean():.4f}  ({len(y_true)} real images)")

    n = len(CLASS_NAMES)
    print("Confusion matrix (rows = actual, columns = predicted)")
    print(f"{'':>10}" + "".join(f"{c:>9}" for c in CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
        row = [int(((y_true == i) & (y_pred == j)).sum()) for j in range(n)]
        print(f"{name:>10}" + "".join(f"{v:>9}" for v in row))

    print("\nPer-class recall (how many real images of each class were found):")
    for i, name in enumerate(CLASS_NAMES):
        total = int((y_true == i).sum())
        if total == 0:
            print(f"  {name:>8}: no real validation images - NOT TESTED")
        else:
            print(f"  {name:>8}: {int(((y_true == i) & (y_pred == i)).sum())}/{total}")


# ------------------------------------------- out-of-distribution check
# Idea: after training, every image becomes a 1280-number "fingerprint" (the layer before the
# final answer). Real cat/dog photos have fingerprints that sit close to each other. A new photo
# whose fingerprint is far from ALL known cat/dog fingerprints is called out-of-distribution,
# even if the classifier never saw that kind of object. This catches unknown things that the
# 'neither' class did not cover.
def feature_extractor(model):
    gap = next(l for l in model.layers if isinstance(l, layers.GlobalAveragePooling2D))
    return tf.keras.Model(model.input, gap.output)


def normalize(e):
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


def max_similarity(query, ref):
    """Cosine similarity of each query to its closest reference fingerprint (1.0 = identical)."""
    return (normalize(query) @ ref.T).max(axis=1)


def embed_folder(folder, fm, model):
    ds = tf.keras.utils.image_dataset_from_directory(
        folder, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int",
        class_names=CLASS_NAMES, shuffle=False)
    embs, labels, probs = [], [], []
    for x, y in ds:
        embs.append(fm.predict(x, verbose=0))
        probs.append(model.predict(x, verbose=0))
        labels.append(y.numpy())
    return np.concatenate(embs), np.concatenate(labels), np.concatenate(probs)


def decide(probs, sim, threshold):
    """Returns (is_cat_or_dog, label, confidence, reason)."""
    idx = int(np.argmax(probs))
    label, conf = CLASS_NAMES[idx], float(probs[idx])
    if label == "neither":
        return False, label, conf, f"model says 'neither' ({conf:.0%})"
    if conf < CONF_THRESHOLD:
        return False, label, conf, f"model is unsure (confidence {conf:.0%})"
    if sim is not None and sim < threshold:
        return False, label, conf, (
            f"far from all known cats/dogs (similarity {sim:.2f}, needs {threshold:.2f})")
    return True, label, conf, ""


def build_ood(model, use_sd):
    ensure_dirs([TRAIN_DIR, VAL_DIR] + ([SD_DIR] if use_sd else []))
    fm = feature_extractor(model)

    refs = []
    for d in [TRAIN_DIR] + ([SD_DIR] if use_sd else []):
        e, y, _ = embed_folder(d, fm, model)
        refs.append(e[y < 2])                     # only cat and dog fingerprints
    ref = normalize(np.concatenate(refs)).astype(np.float32)

    ve, vy, vp = embed_folder(VAL_DIR, fm, model)
    sims = max_similarity(ve, ref)
    threshold = float(np.quantile(sims[vy < 2], 1 - OOD_KEEP))
    np.savez(OOD_FILE, ref=ref, threshold=threshold)
    print(f"\nOut-of-distribution reference saved: {OOD_FILE} "
          f"({len(ref)} cat/dog fingerprints, similarity threshold {threshold:.3f})")

    accepted = np.array([decide(vp[i], sims[i], threshold)[0] for i in range(len(vy))])
    pets, neither = vy < 2, vy == 2
    print("Out-of-distribution check on REAL validation images:")
    print(f"  real cats/dogs accepted as cat/dog : {int(accepted[pets].sum())}/{int(pets.sum())}"
          f"   (about {1 - OOD_KEEP:.0%} are rejected on purpose by the threshold)")
    if neither.sum() == 0:
        print("  real 'neither' flagged             : NOT TESTED (no real val/neither photos)")
    else:
        print(f"  real 'neither' flagged as out-of-distribution: "
              f"{int((~accepted[neither]).sum())}/{int(neither.sum())}")


# --------------------------------------------------------------- train
def train(use_sd):
    if use_sd and sum(count_images(os.path.join(SD_DIR, c)) for c in CLASS_NAMES) == 0:
        raise SystemExit(f"No images in '{SD_DIR}'. Run: python sd_generate.py")
    print("Mode:", "real + Stable Diffusion images" if use_sd else "real images only")

    # Stop early with a clear message if the real dataset is missing or not unzipped properly
    ensure_dirs([TRAIN_DIR, VAL_DIR])
    for split in (TRAIN_DIR, VAL_DIR):
        counts = {c: count_images(os.path.join(split, c)) for c in CLASS_NAMES}
        print(f"Real images in {split}: {counts}")
        for c in ("cat", "dog"):
            if counts[c] == 0:
                raise SystemExit(
                    f"\nSTOP: no REAL images found in {split}/{c}.\n"
                    f"Unzip your dataset so that {TRAIN_DIR}/cat, {TRAIN_DIR}/dog, "
                    f"{VAL_DIR}/cat and {VAL_DIR}/dog exist and contain images, then run again.")

    train_ds, val_ds = load_data(use_sd)
    cw = class_weights(use_sd)
    n_val_neither = count_images(os.path.join(VAL_DIR, "neither"))
    if n_val_neither < 10:
        print(f"WARNING: only {n_val_neither} real photos in dataset/val/neither, so the "
              "'neither' class and the out-of-distribution check cannot be tested properly. "
              "Add at least 40 real non-cat/dog photos (half go to val).")

    model, base = build_model()

    def cbs():
        return [callbacks.EarlyStopping(patience=6, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(factor=0.5, patience=2)]

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              class_weight=cw, callbacks=cbs())

    print("\n--- Fine-tuning ---")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
              class_weight=cw, callbacks=cbs())

    report(model, val_ds)
    model.save(MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")
    build_ood(model, use_sd)


# ------------------------------------------------------------- predict
def predict(image_path, path, ood_path=OOD_FILE):
    model = tf.keras.models.load_model(path)
    # Same preprocessing as training: upright photo, RGB, bilinear resize (same as the
    # dataset loader). Keras load_img would use 'nearest' resize by default.
    img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    arr = tf.image.resize(np.asarray(img, dtype=np.float32), IMG_SIZE, method="bilinear").numpy()
    arr = np.expand_dims(arr, 0)

    probs = model.predict(arr, verbose=0)[0]
    sim, threshold = None, 0.0
    if os.path.exists(ood_path):
        data = np.load(ood_path)
        fm = feature_extractor(model)
        sim = float(max_similarity(fm.predict(arr, verbose=0), data["ref"])[0])
        threshold = float(data["threshold"])
    else:
        print(f"(note: {ood_path} not found - similarity check skipped)")

    is_pet, label, conf, reason = decide(probs, sim, threshold)
    detail = " | ".join(f"{n} {p:.1%}" for n, p in zip(CLASS_NAMES, probs))
    sim_txt = f" | similarity {sim:.2f}" if sim is not None else ""

    if is_pet:
        print(f"Cat/Dog: Yes | Prediction: {label} (confidence {conf:.2%})   [{detail}{sim_txt}]")
    else:
        print(f"Cat/Dog: No | Out of distribution - {reason}   [{detail}{sim_txt}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--sd", action="store_true", help="add Stable Diffusion images to training")
    p = sub.add_parser("predict")
    p.add_argument("image")
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--ood", default=OOD_FILE, help="out-of-distribution reference file")
    args = parser.parse_args()

    if args.cmd == "train":
        train(args.sd)
    else:
        predict(args.image, args.model, args.ood)
