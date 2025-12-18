import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import DenseNet121

# -------------------------------
# Load Model Safely
# -------------------------------
@st.cache_resource  # يحفظ الموديل في الكاش لتسريع إعادة التحميل
def load_xray_model():
    return load_model("WaAI_Xray_Model.keras", custom_objects={"DenseNet121": DenseNet121})

model = load_xray_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="WaAI X-ray", layout="centered")
st.title("🩻 WaAI Chest X-ray Classifier")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Open and display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)[0][0]

    # Display result
    st.subheader("🧪 Result")
    if prediction < 0.5:
        st.success(f"🟢 NORMAL — Confidence: {(1 - prediction) * 100:.2f}%")
    else:
        st.error(f"🔴 PNEUMONIA — Confidence: {prediction * 100:.2f}%")

    st.info("⚠️ This is an AI-assisted prediction, not a medical diagnosis.")
