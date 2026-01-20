import streamlit as st
import os
import numpy as np
import pickle
from numpy.linalg import norm
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors

# ------------------------------
# Set up page
st.set_page_config(
    page_title="Fashion Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛍️ Fashion Recommendation System")
st.markdown("""
Welcome to the **Fashion Recommendation System**!  
Upload an image of a fashion item, and this app will recommend similar items based on our advanced machine learning model.  
Enjoy discovering your next favorite style! 😊
""")

# ------------------------------
# Current directory
current_dir = os.path.dirname(__file__)

# Load embeddings and filenames
feature_list = np.array(pickle.load(open(os.path.join(current_dir, 'embeddings.pkl'), 'rb')))
filenames = pickle.load(open(os.path.join(current_dir, 'filenames.pkl'), 'rb'))  # should contain 'images/1541.jpg' etc

# Load model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

# ------------------------------
# Functions
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list, n=5):
    neighbors = NearestNeighbors(n_neighbors=n+1, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices[0][1:]  # skip the uploaded image itself

def save_uploaded_file(uploaded_file):
    uploads_dir = os.path.join(current_dir, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    uploaded_path = os.path.join(uploads_dir, uploaded_file.name)
    with open(uploaded_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_path

# ------------------------------
# Sidebar
st.sidebar.header("Navigation")
st.sidebar.markdown("""
- **About**: Learn how the system works.
- **Contact**: Get in touch for queries or feedback.
""")
st.sidebar.markdown("### Contact")
st.sidebar.info("""
For any queries or feedback, feel free to reach out:  
📧 **[anubhavpatwal2929@gmail.com](mailto:anubhavpatwal2929@gmail.com)**
""")

# ------------------------------
# File upload section
st.markdown("### Upload Your Fashion Image")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"],
    key="user_upload"
)

if uploaded_file:
    uploaded_path = save_uploaded_file(uploaded_file)
    display_img = Image.open(uploaded_path)
    st.image(display_img, caption="Uploaded Image", use_column_width=True)

    st.markdown("### Recommendations")
    with st.spinner("Analyzing your image and fetching recommendations..."):
        features = extract_features(uploaded_path, model)
        indices = recommend(features, feature_list, n=5)
        st.success("Here are your recommendations:")

    st.markdown("#### Similar Items:")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            recommended_img_path = os.path.join(current_dir, filenames[indices[i]])
            if os.path.exists(recommended_img_path):
                rec_img = Image.open(recommended_img_path)
                st.image(rec_img, use_column_width=True)
            else:
                st.warning(f"Image not found: {filenames[indices[i]]}")

# ------------------------------
# Footer
st.markdown("---")
st.markdown("""
**Developed by Anubhav Patwal (https://github.com/dai-29)**  
Powered by Python, TensorFlow, and Streamlit.
""")
