import os
os.system('pip install torch torchvision pillow numpy scikit-learn streamlit')

import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load Pretrained Model (ResNet50)
@st.cache_resource
def load_model():
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
    model.eval()
    return model

model = load_model()

# Define Image Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract Feature Vector
def extract_features(image):
    try:
        image = image.convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = model(image)
        return features.squeeze().numpy().flatten().astype(np.float32)
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def compute_similarity(img1, img2):
    feat1 = extract_features(img1)
    feat2 = extract_features(img2)
    
    if feat1 is None or feat2 is None:
        return None

    # Compute Cosine Similarity
    similarity = cosine_similarity([feat1], [feat2])[0][0]
    similarity_percentage = round(similarity * 100, 2)
    return similarity_percentage

# Streamlit UI
st.title('Image Similarity Checker')
st.write('Upload one original image and multiple generated images to calculate similarity.')

uploaded_file1 = st.file_uploader("Choose the original image", type=["jpg", "jpeg", "png"])
uploaded_files2 = st.file_uploader("Choose generated images (You can upload multiple images)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_file1 and uploaded_files2:
    image1 = Image.open(uploaded_file1)
    images2 = [Image.open(file) for file in uploaded_files2]

    st.image(image1, caption="Original Image", width=300)
    st.image(images2, caption=[f"Generated Image {i+1}" for i in range(len(images2))], width=300)

    for i, img2 in enumerate(images2):
        with st.spinner(f"Calculating similarity for Image {i+1}..."):
            similarity_score = compute_similarity(image1, img2)
            if similarity_score is not None:
                st.write(f"**Similarity Score for Image {i+1}:** {similarity_score}%")
            else:
                st.write(f"**Failed to calculate similarity for Image {i+1}**")
