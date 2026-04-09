import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("keras.cnn")

st.title("Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((128, 128))  # match training size
    img = np.array(img) / 255.0
    img = np.reshape(img, (1, 128, 128, 3))

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.success("DOG")
    else:
        st.success("CAT")

    st.write(f"Confidence: {prediction:.2f}")