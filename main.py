"""
🌿 Plant Disease Classifier
A CNN-based web application to identify plant diseases from leaf images.
Built with Streamlit and TensorFlow.
"""

import os
import json
import gdown
from PIL import Image


import numpy as np
import tensorflow as tf
import streamlit as st


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(WORKING_DIR, "trained_model")
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_prediction_model.h5")
CLASS_INDICES_PATH = os.path.join(WORKING_DIR, "class_indices.json")

# Google Drive file ID for the trained model
MODEL_GDRIVE_ID = "1cDDNR6eYnMyGvX7dYkjNfu6fQLU6V6cr"


# ──────────────────────────────────────────────
# Model Download & Loading
# ──────────────────────────────────────────────
def download_model():
    """Download the trained model from Google Drive if it doesn't exist locally."""
    if os.path.exists(MODEL_PATH):
        if os.path.getsize(MODEL_PATH) > 100_000_000:  # Must be > 100MB
            return True
        else:
            # The file is likely a corrupted HTML page from a previous failed download
            os.remove(MODEL_PATH)

    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        with st.spinner("⬇️ Downloading the trained model (547 MB)... This only happens once."):
            gdown.download(url=f"https://drive.google.com/uc?id={MODEL_GDRIVE_ID}", output=MODEL_PATH, quiet=False)
        
        # Verify it downloaded the actual model and not a tiny HTML warning page
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 100_000_000:
            os.remove(MODEL_PATH)
            raise Exception("Downloaded file is too small (likely a Google Drive warning page).")
            
        return os.path.exists(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.info(
            "**Manual download:** Go to "
            f"[Google Drive](https://drive.google.com/file/d/{MODEL_GDRIVE_ID}/view) "
            f"→ download the file → place it at `{MODEL_PATH}`"
        )
        return False



@st.cache_resource
def load_model():
    """Load the Keras model (cached so it only loads once per session)."""
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_indices():
    """Load the class-name mapping from JSON."""
    if not os.path.exists(CLASS_INDICES_PATH):
        return None
    with open(CLASS_INDICES_PATH, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────────
# Image Preprocessing & Prediction
# ──────────────────────────────────────────────
def load_and_preprocess_image(image_file, target_size=(224, 224)):
    """Load an image, convert to RGB, resize, and normalize to [0, 1]."""
    img = Image.open(image_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim
    img_array = img_array.astype("float32") / 255.0
    return img_array


def predict_image_class(model, image_file, class_indices):
    """Run inference and return the human-readable class name."""
    preprocessed_img = load_and_preprocess_image(image_file)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


def format_disease_name(raw_name: str) -> str:
    """Convert 'Apple___Black_rot' → 'Apple — Black Rot'."""
    parts = raw_name.split("___")
    plant = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip().title() if len(parts) > 1 else ""
    if disease.lower() == "healthy":
        return f"✅ {plant} — Healthy"
    return f"⚠️ {plant} — {disease}"


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15, 32, 39, 0.95);
        border-right: 1px solid rgba(76, 175, 80, 0.3);
    }

    /* File uploader area */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(76, 175, 80, 0.5);
        border-radius: 16px;
        padding: 1rem;
        transition: border-color 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(76, 175, 80, 1);
    }

    /* Success / result box */
    .result-box {
        background: rgba(76, 175, 80, 0.15);
        border: 1px solid rgba(76, 175, 80, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.3rem;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.5);
    }

    /* Image container */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* Divider */
    hr {
        border-color: rgba(76, 175, 80, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("🌿 About")
    st.markdown(
        """
        This app uses a **Convolutional Neural Network** trained on the 
        [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) 
        dataset to detect **38 classes** of plant diseases across 14 crop species.

        ---

        **How to use:**
        1. 📸 Upload a clear photo of a plant leaf.
        2. 🔍 Click **Classify Disease**.
        3. 📋 View the prediction result.

        ---

        **Supported Plants:**  
        Apple · Blueberry · Cherry · Corn · Grape · Orange · Peach · 
        Pepper · Potato · Raspberry · Soybean · Squash · Strawberry · Tomato
        """
    )
    st.divider()
    st.caption("Built with ❤️ using Streamlit & TensorFlow")


# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────
st.title("🌱 Plant Disease Classifier")
st.markdown("Upload an image of a plant leaf to identify potential diseases using AI.")

# Attempt to download model if missing or corrupted
download_model()

# Load model and class indices
model = load_model()
class_indices = load_class_indices()

if model is None:
    st.error("⚠️ **Model Not Available**")
    st.warning(
        f"The model file could not be found or downloaded.\n\n"
        f"Please manually place the `.h5` file at:\n`{MODEL_PATH}`"
    )
    st.stop()

if class_indices is None:
    st.error("⚠️ **Class Indices File Missing**")
    st.warning(f"Please ensure `class_indices.json` exists at `{CLASS_INDICES_PATH}`.")
    st.stop()

st.divider()

# File uploader
uploaded_image = st.file_uploader(
    "Choose an image…", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="📷 Uploaded Leaf Image", use_column_width=True)

    # Centered classify button
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        classify_clicked = st.button(
            "🔍 Classify Disease", use_container_width=True
        )

    if classify_clicked:
        with st.spinner("🧠 Analyzing image…"):
            try:
                prediction = predict_image_class(model, uploaded_image, class_indices)
                formatted = format_disease_name(prediction)
                st.markdown(
                    f'<div class="result-box">{formatted}</div>',
                    unsafe_allow_html=True,
                )
                st.balloons()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    # Placeholder when no image is uploaded
    st.markdown(
        """
        <div style="
            border: 2px dashed rgba(76,175,80,0.4);
            border-radius: 16px;
            padding: 3rem 1rem;
            text-align: center;
            color: rgba(255,255,255,0.5);
            margin-top: 1rem;
        ">
            <p style="font-size: 3rem; margin: 0;">📤</p>
            <p style="font-size: 1.1rem;">Drag & drop or click to upload a leaf image</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
