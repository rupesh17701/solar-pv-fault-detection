"""Streamlit UI for the solar panel fault-detection CNN.

Same model and preprocessing as app/main.py (the FastAPI service), just a
different front end — this one is what Streamlit Community Cloud runs
directly (`streamlit run app/streamlit_app.py`).
"""
import os

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "app", "model", "solar_fault_classifier.h5")

st.set_page_config(page_title="Solar PV Fault Detector", page_icon="☀️")


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"No trained model found at {MODEL_PATH}. Run app/train.py first.")
        st.stop()
    return load_model(MODEL_PATH)


def predict(model, image: Image.Image):
    img = image.convert("RGB").resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = float(model.predict(arr, verbose=0)[0][0])
    label = "Defect" if pred > 0.5 else "Clean"
    confidence = round(pred if label == "Defect" else 1 - pred, 4)
    return label, confidence


st.title("Solar Panel Fault Detector")
st.write(
    "Upload a photo of a solar panel to check whether it is clean or shows "
    "a fault (dust, bird drop, snow, physical or electrical damage)."
)

model = get_model()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    label, confidence = predict(model, image)

    if label == "Clean":
        st.success(f"**Clean** — confidence {confidence}")
    else:
        st.error(f"**Defect** — confidence {confidence}")
