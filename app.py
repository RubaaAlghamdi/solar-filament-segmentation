import io
import json
from pathlib import Path

import cv2
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "run_config.json").read_text(encoding="utf-8"))
@st.cache_resource
def get_model():
    return load_model(ROOT / CONFIG["checkpoint"], compile=False)


def segment(image):
    size = CONFIG["input_size"]
    preprocessing = CONFIG["preprocessing"]
    clahe_settings = preprocessing["clahe"]

    gray = image.convert("L")
    resized = gray.resize((size, size), Image.Resampling.BILINEAR)
    clahe = cv2.createCLAHE(
        clipLimit=clahe_settings["clip_limit"],
        tileGridSize=tuple(clahe_settings["tile_grid_size"]),
    )
    pixels = clahe.apply(np.asarray(resized))
    batch = (pixels.astype(np.float32) / preprocessing["normalize_divisor"])[None, ..., None]
    probabilities = get_model()(batch, training=False).numpy()[0, ..., 0]

    binary = (probabilities >= CONFIG["threshold"]).astype(np.uint8) * 255
    mask = Image.fromarray(binary).resize(gray.size, Image.Resampling.NEAREST)
    original = image.convert("RGB")
    red = Image.new("RGB", original.size, (255, 25, 25))
    highlighted = Image.blend(original, red, alpha=0.76)
    overlay = Image.composite(highlighted, original, mask)
    return original, mask, overlay


st.set_page_config(page_title="Solar Filament Segmentation", layout="wide")
st.title("Solar Filament Segmentation")

with st.form("solar_image"):
    uploaded = st.file_uploader("Upload a full-disk H-alpha solar image", type=["png", "jpg", "jpeg"])
    submitted = st.form_submit_button("Detect filaments", type="primary")

if submitted:
    st.session_state.pop("prediction", None)
    if uploaded is None:
        st.warning("Upload a solar image first.")
    else:
        try:
            with Image.open(uploaded) as source:
                image = source.copy()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            st.error("This image could not be opened. Try a valid PNG or JPEG.")
        else:
            with st.spinner("Detecting filaments…"):
                st.session_state["prediction"] = segment(image)

if "prediction" in st.session_state:
    original, mask, overlay = st.session_state["prediction"]
    for column, image, label in zip(
        st.columns(3),
        (original, mask, overlay),
        ("Original image", "Predicted mask", "Filament overlay"),
    ):
        column.subheader(label)
        column.image(image, width="stretch")
    mask_png = io.BytesIO()
    mask.save(mask_png, format="PNG")
    st.download_button("Download mask", mask_png.getvalue(), "filament-mask.png", "image/png")
