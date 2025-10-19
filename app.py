import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Title and Description
st.title("MNIST Handwritten Digit Classifier")
st.write("Upload a handwritten digit image (0–9) to classify it using the trained CNN model.")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_model.h5")
    return model

model = load_model()
st.success("Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader("Choose an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = image.convert("L")  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors (MNIST digits are white on black)
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions)

    # Display results
    st.subheader(f"Predicted Digit: {predicted_label}")
    st.write(f"Prediction Confidence: {confidence * 100:.2f}%")

    # Probability chart
    st.bar_chart(predictions[0])

else:
    st.info("Please upload an image of a handwritten digit to continue.")
