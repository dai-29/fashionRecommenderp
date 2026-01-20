import streamlit as st
import os
import numpy as np
import pickle
from numpy.linalg import norm
import tensorflow
from PIL import Image
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

# Load pre-trained model and data
feature_list = np.array(pickle.load(open(os.path.join(current_dir, 'embeddings.pkl'), 'rb')))
filenames = pickle.load(open(os.path.join(current_dir, 'filenames.pkl'), 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

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

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

def save_uploaded_file(uploaded_file):
    try:
        uploads_dir = os.path.join(current_dir, "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        uploaded_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(uploaded_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return uploaded_path
    except Exception as e:
        print(e)
        return None

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

if uploaded_file is not None:
    uploaded_path = save_uploaded_file(uploaded_file)
    if uploaded_path:
        display_img = Image.open(uploaded_path)
        st.image(display_img, caption="Uploaded Image", use_column_width=True)

        # Feature extraction and recommendation
        st.markdown("### Recommendations")
        with st.spinner("Analyzing your image and fetching recommendations..."):
            features = extract_features(uploaded_path, model)
            indices = recommend(features, feature_list)
            st.success("Here are your recommendations:")

        # Display recommendations in a grid
        st.markdown("#### Similar Items:")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                img_path = os.path.join(current_dir, filenames[indices[0][i]])
                st.image(img_path, use_column_width=True)
    else:
        st.error("Error uploading your image. Please try again.")

# ------------------------------
# Footer
st.markdown("---")
st.markdown("""
**Developed by Anubhav Patwal (https://github.com/dai-29)**  
Powered by Python, TensorFlow, and Streamlit.
""")
