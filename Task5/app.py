import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. Page Configuration and Title Setup
st.set_page_config(page_title="Image Classification App", layout="centered")
st.title("🖼️ Image Classification Deployment App")
st.write("Upload an image below to get an instant model prediction.")

# 2. Cache the model to ensure fast loading times across runs
@st.cache_resource
def load_my_model():
    # Adjust path if your weights are saved inside a subfolder structure
    return tf.keras.models.load_model("models/cnn_model.h5")

try:
    model = load_my_model()
    st.success("🤖 CNN Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model weights. Error: {e}")

# 3. File Uploader UI Component
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image to the user
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image Preview", use_container_width=True)
    
    st.write("🔄 Processing and generating prediction...")
    
    # 4. Image Preprocessing (Matches your training configuration)
    # Resize to 224x224 as required by your input layer shape
    resized_image = image.resize((224, 224))
    
    # Convert image pixels to a numpy array
    img_array = np.array(resized_image)
    
    # Handle grayscale or RGBA conversions to clean RGB channels
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    elif len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
        
    # Scale pixel intensities to [0, 1] matching your training generator steps
    img_array = img_array / 255.0
    
    # Expand dimensions to add the batch size channel (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 5. Run Model Inference
    predictions = model.predict(img_batch)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    confidence_score = float(np.max(predictions)) * 100
    
    # Map class indices back to original string folder labels
   # Change this list sequence to match your dataset folder sorting perfectly
    class_labels = ["Cardboard", "Glass", "Metal/Batteries", "Paper", "Plastic"]
    predicted_label = class_labels[predicted_class_idx]
    
    # 6. Display Final Performance Metrics
    st.markdown("---")
    st.subheader("🎯 Prediction Results")
    st.metric(label="Predicted Category", value=predicted_label)
    st.metric(label="Prediction Confidence", value=f"{confidence_score:.2f}%")
