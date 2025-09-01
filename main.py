import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

from PIL import image

def load_model():
    model = MobileNetV2(weights="imagenet")
    return model


def preprocess_image(image):
    img = cv2.resize(img, (224, 224))
    img = np.array(image)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)
    return img

def classify_image(model, image):

    try:
        processed_image = preprocess_image(image)
        preds = model.predict(processed_image)
        decode_predicts = decode_predictions(preds, top=3)[0]
        
        return decode_predicts
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Image Classifier", page_icon="👨🏻‍💻", Layout="centered")

    st.write("AI Image Classifier")
    st.write("Upload an image to classify and let it tell you what it is")

    @st.cache_resource
    def load_cache_model():
        return load_model()
    # to utilize the same model already loaded in memory

    model = load_cache_model()
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = st.image(uploaded_file, caption="Uploaded Image", use_container_width=True )

        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Analyzing image..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(image)

        if predictions is not None:
            st.subheader("Predictions")
            for _, label, score in predictions:
                st.write(f"{label}: {score:.2f}%")


if __name__ == "__main__":
    main()