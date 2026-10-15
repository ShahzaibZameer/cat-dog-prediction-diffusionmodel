"""
Streamlit app: Cat / Dog detector with out-of-distribution check

Files needed in the SAME folder as this app.py:
    cat_dog_neither_model.keras     (made by cat_dog_cnn_v4.py)
    ood_reference.npz               (made by cat_dog_cnn_v4.py)

Run on your computer:
    pip install streamlit tensorflow pillow numpy
    streamlit run app.py
"""

import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# These must match the training script (cat_dog_cnn_v4.py)
IMG_SIZE = (160, 160)
CLASS_NAMES = ["cat", "dog", "neither"]
CONF_THRESHOLD = 0.70
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "cat_dog_neither_model.keras"
)
OOD_PATH = os.path.join(
    BASE_DIR, "ood_reference.npz"
)

st.set_page_config(page_title="Cat / Dog Detector", page_icon="🐾", layout="centered")


#@st.cache_resource(show_spinner="Loading model...")
#def load_assets():
# model = tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    try:
        model = tf.keras.models.load_model(
            MODEL_PATH, compile=False
        )
    except Exception as e:
        st.error(f"Model loading error: {type(e).__name__}: {e}")
        st.stop()

    gap = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
    )
    feature_model = tf.keras.Model(model.input, gap.output)
    data = np.load(OOD_PATH)

    return model, feature_model, data["ref"], float(data["threshold"])

def preprocess(img):
    """Same steps as training: RGB, bilinear resize to 160x160, pixel values 0-255."""
    arr = np.asarray(img, dtype=np.float32)
    arr = tf.image.resize(arr, IMG_SIZE, method="bilinear").numpy()
    return np.expand_dims(arr, 0)


def analyse(img, model, feature_model, ref, threshold):
    arr = preprocess(img)
    probs = model.predict(arr, verbose=0)[0]

    emb = feature_model.predict(arr, verbose=0)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sim = float((emb @ ref.T).max())             # closeness to the nearest known cat/dog

    idx = int(np.argmax(probs))
    label, conf = CLASS_NAMES[idx], float(probs[idx])

    if label == "neither":
        return False, label, conf, probs, sim, f"The model says this looks like something else ({conf:.0%})."
    if conf < CONF_THRESHOLD:
        return False, label, conf, probs, sim, f"The model is not sure (confidence {conf:.0%})."
    if sim < threshold:
        return False, label, conf, probs, sim, (
            "This photo looks too different from all the cats and dogs the model knows "
            f"(similarity {sim:.2f}, needs at least {threshold:.2f}).")
    return True, label, conf, probs, sim, ""


# ------------------------------------------------------------------ page
st.title("🐾 Cat / Dog Detector")
st.write("Upload a photo. The model says if it is a **cat**, a **dog**, or "
         "**out of distribution** (not a cat or dog).")

st.write("App folder:", BASE_DIR)
st.write("Model exists:", os.path.isfile(MODEL_PATH))
st.write("OOD file exists:", os.path.isfile(OOD_PATH))

if not os.path.isfile(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not os.path.isfile(OOD_PATH):
    st.error(f"OOD file not found: {OOD_PATH}")
    st.stop()


model, feature_model, ref, threshold = load_assets()

with st.sidebar:
    st.header("About")
    st.write("A MobileNetV2 model trained on cat and dog photos, plus a check that rejects "
             "photos that look unlike anything it knows.")
    st.write(f"Needs at least **{CONF_THRESHOLD:.0%}** confidence and a similarity of "
             f"**{threshold:.2f}** to known cats/dogs.")
    st.caption("No model is perfect. Very blurry, dark, or unusual photos can be wrong, "
               "and a few real cats/dogs may be rejected on purpose.")

files = st.file_uploader("Upload one or more photos",
                         type=["jpg", "jpeg", "png", "jfif", "webp", "bmp"],
                         accept_multiple_files=True)

for f in files or []:
    st.divider()
    try:
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    except Exception as e:
        st.warning(f"Could not open {f.name}: {e}")
        continue

    is_pet, label, conf, probs, sim, reason = analyse(img, model, feature_model, ref, threshold)

    left, right = st.columns([1, 1])
    with left:
        st.image(img, caption=f.name, use_container_width=True)
    with right:
        if is_pet:
            st.success(f"**Cat/Dog: Yes**\n\nPrediction: **{label}** ({conf:.1%} confident)")
        else:
            st.error(f"**Cat/Dog: No**\n\n**Out of distribution**\n\n{reason}")
        for name, p in zip(CLASS_NAMES, probs):
            st.write(f"{name}: {p:.1%}")
            st.progress(float(p))
        with st.expander("Details"):
            st.write(f"Similarity to known cats/dogs: **{sim:.3f}** (threshold {threshold:.3f})")
