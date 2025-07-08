# streamlit_image_matcher_auto.py

import streamlit as st
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import io

# Load Deep Model
@st.cache_resource
def load_model():
    try:
        model = resnet50(pretrained=True)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model:
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Deep Feature Extraction
def extract_features(image_file_or_path):
    try:
        if isinstance(image_file_or_path, str):
            img = Image.open(image_file_or_path).convert("RGB")
        else:
            img = Image.open(image_file_or_path).convert("RGB")
        
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = model(tensor).squeeze().numpy()
        return features
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

# Create ZIP with Custom Names
def create_zip_from_matches(matched_images, ref_files_dict, custom_name="match"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, (file_name, score) in enumerate(matched_images):
            if file_name in ref_files_dict:
                file_data = ref_files_dict[file_name]
                extension = file_name.rsplit('.', 1)[1] if '.' in file_name else 'jpg'
                new_filename = f"{custom_name}_{i+1}.{extension}"
                zipf.writestr(new_filename, file_data)
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Extract Images from ZIP folder
def extract_images_from_zip(zip_file):
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            image_files = []
            file_data_dict = {}
            
            for file_info in zip_ref.filelist:
                if not file_info.is_dir():
                    filename = file_info.filename
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        try:
                            file_data = zip_ref.read(filename)
                            file_obj = io.BytesIO(file_data)
                            file_obj.name = filename.split('/')[-1]
                            image_files.append(file_obj)
                            file_data_dict[file_obj.name] = file_data
                        except Exception as e:
                            st.warning(f"Could not read {filename}: {e}")
                            continue
            
            return image_files, file_data_dict
    except Exception as e:
        st.error(f"Error extracting ZIP file: {e}")
        return [], {}

# Main Application
def main():
    st.title("PixelMatch")
    
    with st.expander("ℹ️ How to Use This Tool"):
        st.markdown("""
        **Welcome to PixelMatch!**  
        This AI-powered tool helps you find **visually similar images** in your reference library, perfect for cataloging, e-commerce, and creative teams.

        **📌 How it works:**
        - Upload your **Query Image** (the image you want to find similar matches for).
        - Upload your **Reference Images** (your internal database/library) either as:
        - Multiple individual images
        - A single ZIP folder containing multiple images
        - Optionally, preview your reference images before matching.

        **⚡ Matching Process:**
        - The AI model extracts deep features from your images.
        - It compares your query image with all reference images.
        - It shows **matches with 80%+ similarity**, ranked by similarity score.

        **✅ Features:**
        - Fully supports batch processing via ZIP upload.
        - Displays clear **similarity percentages** for all matches.
        - Lets you set a **custom base name** for matched images.
        - Packages all matched images into a **downloadable ZIP** with new naming.

        **💡 Tips:**
        - Use high-resolution images for best results.
        - Make sure your reference library has clear, well-lit images.
        - Try different query images if you get no matches.

        Ready to find your perfect matches? 🚀
        """)

    # Query Image Section
    st.subheader("🎯 Query Image")
    uploaded_query = st.file_uploader("Upload the image to find matches for", type=['jpg', 'jpeg', 'png'])
    
    # Show query image preview
    if uploaded_query:
        st.markdown("**Query Image Preview**")
        st.image(uploaded_query, use_column_width=True, caption="Your Query Image")
    
    # Reference Images Section (now below query image)
    st.subheader("📚 Reference Images")
    upload_method = st.radio("Choose upload method:", ["Multiple Images", "ZIP Folder"], horizontal=True)
    
    if upload_method == "Multiple Images":
        uploaded_refs = st.file_uploader("Upload reference images", 
                                        type=['jpg', 'jpeg', 'png'], 
                                        accept_multiple_files=True)
        ref_files_dict = {}
    else:
        uploaded_zip = st.file_uploader("Upload ZIP folder", type=['zip'])
        if uploaded_zip:
            uploaded_refs, ref_files_dict = extract_images_from_zip(uploaded_zip)
            if uploaded_refs:
                st.success(f"✅ Extracted {len(uploaded_refs)} images")
            else:
                uploaded_refs = []
        else:
            uploaded_refs = []
            ref_files_dict = {}
    
    # Preview reference images button
    if 'uploaded_refs' in locals() and uploaded_refs:
        if st.button(f"👁️ Preview Reference Images ({len(uploaded_refs)} files)", use_container_width=True):
            st.session_state.show_refs = not st.session_state.get('show_refs', False)
    
    # Show reference images preview if button clicked
    if uploaded_refs and st.session_state.get('show_refs', False):
        st.markdown("---")
        st.markdown("**📖 Reference Images Preview**")
        
        # Show in grid
        cols_per_row = 4
        num_images = len(uploaded_refs)
        
        for i in range(0, num_images, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < num_images:
                    with cols[j]:
                        ref_img = uploaded_refs[i + j]
                        img_name = ref_img.name if hasattr(ref_img, 'name') else f"Image {i+j+1}"
                        st.image(ref_img, caption=img_name, use_column_width=True)
    
    # Processing Section
    if uploaded_query and uploaded_refs:
        st.markdown("---")
        
        # Custom naming section
        st.subheader("🏷️ Set Download Name")
        
        custom_name = st.text_input("Enter base name for matched images:", 
                                   value="", 
                                   placeholder="e.g., model, person, object...")
        
        # Only show process button when name is entered
        if custom_name.strip():
            if st.button("🚀 Start Processing & Find Matches", type="primary"):
                st.session_state.processing = True
                st.session_state.custom_name = custom_name.strip()
        else:
            st.info("💡 Enter a name above to start processing")
        
        # Processing and Results
        if st.session_state.get('processing', False):
            with st.spinner("🔄 AI is analyzing images..."):
                # Extract query features
                query_features = extract_features(uploaded_query)
                
                if query_features is None:
                    st.error("Failed to extract features from query image")
                    st.session_state.processing = False
                    return
                
                # Process reference images
                similarities = []
                
                # Populate ref_files_dict for individual files
                if upload_method == "Multiple Images":
                    ref_files_dict = {}
                    for ref_file in uploaded_refs:
                        ref_file.seek(0)
                        ref_files_dict[ref_file.name] = ref_file.read()
                        ref_file.seek(0)
                
                # Calculate similarities
                progress_bar = st.progress(0)
                for i, ref_file in enumerate(uploaded_refs):
                    file_name = ref_file.name if hasattr(ref_file, 'name') else f"image_{i+1}.jpg"
                    ref_features = extract_features(ref_file)
                    
                    if ref_features is not None:
                        similarity = cosine_similarity([query_features], [ref_features])[0][0]
                        similarities.append((file_name, similarity))
                    
                    progress_bar.progress((i + 1) / len(uploaded_refs))
                
                progress_bar.empty()
                
                # Filter matches above 0.8 threshold
                matched_images = [(name, score) for name, score in similarities if score >= 0.8]
                matched_images = sorted(matched_images, key=lambda x: x[1], reverse=True)
                
                # Store results in session state
                st.session_state.matched_images = matched_images
                st.session_state.ref_files_dict = ref_files_dict
                st.session_state.processing = False
                st.session_state.results_ready = True
        
        # Show Results
        if st.session_state.get('results_ready', False):
            matched_images = st.session_state.get('matched_images', [])
            ref_files_dict = st.session_state.get('ref_files_dict', {})
            custom_name = st.session_state.get('custom_name', 'match')
            
            st.markdown("---")
            
            if matched_images:
                st.subheader(f"🎯 Found {len(matched_images)} High-Quality Matches (≥80% similarity)")
                
                # Show matches in a beautiful grid
                cols_per_row = 3
                for i in range(0, len(matched_images), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(matched_images):
                            file_name, score = matched_images[i + j]
                            with cols[j]:
                                st.markdown(f"**Match {i+j+1}**")
                                st.markdown(f"Similarity: {score:.1%}")
                                st.markdown(f"Will be: {custom_name}_{i+j+1}")
                                
                                # Find and show the reference image
                                for ref_img in uploaded_refs:
                                    ref_name = ref_img.name if hasattr(ref_img, 'name') else f"image_{uploaded_refs.index(ref_img)+1}"
                                    if ref_name == file_name:
                                        st.image(ref_img, use_column_width=True)
                                        break
                
                # Download Section
                st.subheader("📦 Download Your Matches")
                st.markdown("All matched images will be renamed and packaged for you!")
                
                # Create and offer download
                zip_data = create_zip_from_matches(matched_images, ref_files_dict, custom_name)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label=f"📥 Download {len(matched_images)} matches as {custom_name}_*.zip",
                        data=zip_data,
                        file_name=f"{custom_name}_matches_{len(matched_images)}_images.zip",
                        mime="application/zip",
                        type="primary"
                    )
                
                st.success(f"✅ Ready to download {len(matched_images)} high-quality matches!")
                
            else:
                st.subheader("❌ No matches found above 80% similarity")
                st.info("Try uploading different reference images or a different query image.")
    
    elif uploaded_query or uploaded_refs:
        st.info("📋 Upload both query and reference images to start matching")

if __name__ == "__main__":
    if model is None:
        st.error("Model failed to load. Please check PyTorch installation.")
    else:
        main()