import streamlit as st
import tensorflow as tf
from data import load_and_preprocess_data
from model import build_model
import numpy as np

def load_model():
    model = build_model()
    model.load_weights('model_weights.weights.h5')  # Load the saved weights
    return model

def preprocess_image(image_file):
    # Read and preprocess the image
    image = tf.image.decode_image(image_file.read(), channels=1)
    image = tf.image.resize(image, [28, 28])  # Resize to model input size
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize the image
    return image

def main():
    st.title("Real-Time Handwritten Digit Recognition")

    # Load the pre-trained model
    model = load_model()

    # Allow user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type="png")

    if uploaded_file is not None:
        # Preprocess the image
        image = preprocess_image(uploaded_file)
        
        # Convert TensorFlow tensor to NumPy array for visualization
        image_np = image.numpy().squeeze()  # Remove batch dimension and convert to NumPy array

        # Display the image
        st.image(image_np, caption='Uploaded Image.', use_column_width=True)

        # Make a prediction
        predictions = model.predict(image)
        predicted_class = tf.argmax(predictions[0])
        confidence = tf.reduce_max(predictions[0])

        # Show the prediction result
        st.write(f"Predicted Digit: {predicted_class.numpy()}")
        st.write(f"Confidence: {confidence.numpy():.2f}")

if __name__ == "__main__":
    main()