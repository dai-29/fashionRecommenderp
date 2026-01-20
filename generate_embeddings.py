import os
import pickle
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Model

# ------------------------------
# Load ResNet50 pre-trained model + GlobalMaxPooling
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = GlobalMaxPooling2D()(base_model.output)
model = Model(inputs=base_model.input, outputs=x)
# ------------------------------

# Folder containing images
img_folder = 'images'

# Get all image filenames
filenames = [os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith(('.jpg','.jpeg','.png'))]

# Extract features for all images
feature_list = []
print("⏳ Extracting features from images...")
for file in tqdm(filenames):
    img = image.load_img(file, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(expanded)
    result = model.predict(preprocessed).flatten()
    normalized = result / norm(result)
    feature_list.append(normalized)

# Save embeddings and filenames
pickle.dump(np.array(feature_list), open('embeddings.pkl','wb'))
pickle.dump(filenames, open('filenames.pkl','wb'))

print("✅ Real embeddings generated and saved!")
