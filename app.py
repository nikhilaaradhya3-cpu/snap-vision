"""
SnapVision — Live Deployment App
AI GPU Summer Internship Program · Summer 2026 · NVIDIA
Team: Nikhil D P, Rabiya Banu, Amit Suresh Balekundaragi, Shaili Ramjanm Goswami

Run after training the model in SnapVision_CNN_Training.ipynb (which produces
snapvision_model.keras and class_names.json in the same folder):

    streamlit run app.py
"""

import json

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

MODEL_PATH = "snapvision_model.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (32, 32)

st.set_page_config(page_title="SnapVision", page_icon="📷", layout="centered")


@st.cache_resource
def load_model_and_classes():
    model = keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, class_names, image: Image.Image):
    batch = preprocess(image)
    probs = model.predict(batch, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    return class_names[top_idx], float(probs[top_idx]), probs


st.title("📷 SnapVision")
st.caption("Real-Time CNN Image Classifier — AI GPU Summer Internship Program, Summer 2026, NVIDIA")

try:
    model, class_names = load_model_and_classes()
except Exception:
    st.error(
        "Couldn't find a trained model. Run every cell in "
        "**SnapVision_CNN_Training.ipynb** first — its last cell saves "
        "`snapvision_model.keras` and `class_names.json` into this folder."
    )
    st.stop()

tab_upload, tab_camera = st.tabs(["Upload an image", "Use webcam"])

image = None
with tab_upload:
    uploaded = st.file_uploader("Upload a photo (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded)

with tab_camera:
    snapshot = st.camera_input("Take a photo")
    if snapshot is not None:
        image = Image.open(snapshot)

if image is not None:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Input", use_container_width=True)

    label, confidence, probs = predict(model, class_names, image)

    with col2:
        st.metric("Prediction", label.upper(), f"{confidence*100:.1f}% confidence")
        st.bar_chart(
            {name: float(p) for name, p in zip(class_names, probs)},
            height=280,
        )
else:
    st.info("Upload an image or take a photo above to classify it — CIFAR-10 classes only: "
             + ", ".join(class_names) + ".")

st.divider()
st.caption(
    "Model: SnapVision CNN (Conv2D → BatchNorm → MaxPooling → Dropout → Dense → Softmax), "
    "trained on CIFAR-10 with GPU-accelerated CUDA/cuDNN training."
)
