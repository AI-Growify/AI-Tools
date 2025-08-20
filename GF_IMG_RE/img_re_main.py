import os
import io, zipfile
import ssl, warnings
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rembg import remove
from ultralytics import YOLO
from multiprocessing import Pool
from functools import partial
import pickle
import warnings
warnings.filterwarnings("ignore",message=".*ScriptRunContext.*")


warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"

def preprocess_uploaded_image(img, max_dim=2048):
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        return img.convert("RGB")


def optimize_image(img, max_size_kb):
        buf = io.BytesIO()
        q = 95
        img.save(buf, "JPEG", quality=q, optimize=True, progressive=True)
        while buf.tell()/1024 > max_size_kb and q > 10:
            buf.seek(0); buf.truncate()
            q -= 5
            img.save(buf, "JPEG", quality=q, optimize=True, progressive=True)
        buf.seek(0)
        return buf


def enhanced_subject_detection_for_mp(img_array):
        """Modified version for multiprocessing that uses numpy array instead of PIL Image"""
        # Load model inside the function for multiprocessing
        local_model = YOLO("yolov8n-seg.pt")
        
        # Convert numpy array to opencv format
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        results = local_model.predict(img_cv, classes=0, verbose=False)
        
        for r in results:
            if hasattr(r, 'masks') and r.masks is not None:
                masks = r.masks.xy
                if masks:
                    largest_mask = max(masks, key=lambda m: cv2.contourArea(m.astype(np.int32)))
                    x, y, w, h = cv2.boundingRect(largest_mask.astype(np.int32))
                    return (x, y, x + w, y + h)
        
        # Fallback to rembg if YOLO doesn't detect
        img_pil = Image.fromarray(img_array)
        bg_removed = remove(img_pil, post_process_mask=True)
        alpha = bg_removed.split()[-1]
        bbox = alpha.getbbox()
        if bbox:
            dx = int((bbox[2] - bbox[0]) * 0.05)
            dy = int((bbox[3] - bbox[1]) * 0.05)
            x0, y0 = max(0, bbox[0] - dx), max(0, bbox[1] - dy)
            x1 = min(img_pil.width, bbox[2] + dx)
            y1 = min(img_pil.height, bbox[3] + dy)
            return (x0, y0, x1, y1)
        return None


def smart_resize_preserve_background(image, bbox, target_size, top_space=0, bottom_space=0):
        img_w, img_h = image.size
        target_w, target_h = target_size
        target_ratio = target_w / target_h

        x0, y0, x1, y1 = bbox
        y0, y1 = max(0, y0 - top_space), min(img_h, y1 + bottom_space)
        box_w, box_h = x1 - x0, y1 - y0
        box_cx, box_cy = (x0 + x1) // 2, (y0 + y1) // 2

        if (box_w / box_h) < target_ratio:
            new_box_w = int(box_h * target_ratio)
            new_box_h = box_h
        else:
            new_box_w = box_w
            new_box_h = int(box_w / target_ratio)

        margin_w, margin_h = int(new_box_w * 0.1), int(new_box_h * 0.1)
        new_box_w += margin_w
        new_box_h += margin_h

        left = max(0, box_cx - new_box_w // 2)
        right = min(img_w, box_cx + new_box_w // 2)
        top = max(0, box_cy - new_box_h // 2)
        bottom = min(img_h, box_cy + new_box_h // 2)

        cropped = image.crop((left, top, right, bottom))
        return cropped.resize(target_size, Image.LANCZOS)


def add_black_glow_around_logo(base_img, logo_img, x_px, y_px, blur_radius=8, glow_opacity=100):
        base = base_img.convert("RGBA")
        logo = logo_img.convert("RGBA")
        w, h = logo.size
        alpha = logo.split()[-1]
        blurred = alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        shadow = Image.new('RGBA', (w, h), (0,0,0,0))
        shadow.putalpha(blurred.point(lambda p: p * glow_opacity // 100))
        shadow_layer = Image.new('RGBA', (w, h), (0,0,0,255))
        shadow_layer.putalpha(shadow.split()[-1])
        region = base.crop((x_px, y_px, x_px + w, y_px + h))
        region_np = np.array(region).astype(np.float32)
        sh_np = np.array(shadow_layer).astype(np.float32) / 255
        region_np[..., :3] = region_np[..., :3] * (1 - sh_np[..., 3:]) + region_np[..., :3] * sh_np[..., 3:] * 0.5
        base.paste(Image.fromarray(region_np.clip(0,255).astype(np.uint8)), (x_px, y_px))
        base.paste(logo, (x_px, y_px), logo)
        return base.convert("RGB")


def add_blur_background_under_logo(base_img, logo_img, x_px, y_px, blur_radius=10, mask_margin=5):
        base = base_img.convert("RGBA")
        logo = logo_img.convert("RGBA")
        w, h = logo.size
        alpha = logo.split()[-1]
        mask = alpha.point(lambda p: 255 if p > 0 else 0)
        mask = mask.filter(ImageFilter.MaxFilter(mask_margin*2 + 1))
        region = base.crop((x_px, y_px, x_px + w, y_px + h))
        blurred = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blended = Image.composite(blurred, region, mask)
        base.paste(blended, (x_px, y_px))
        return base.convert("RGB")


def merge_logos_horizontally(logo1, logo2, padding=20, separator_text="×", separator_font_size=40):
        try:
            font = ImageFont.truetype("arial.ttf", separator_font_size)
        except:
            font = ImageFont.load_default()

        bbox = font.getbbox(separator_text)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

        h = max(logo1.height, logo2.height, text_height)
        total_width = logo1.width + logo2.width + text_width + 2 * padding
        merged = Image.new("RGBA", (total_width, h), (0, 0, 0, 0))

        y1 = (h - logo1.height) // 2
        y2 = (h - logo2.height) // 2
        text_y = (h - text_height) // 2

        merged.paste(logo1, (0, y1), logo1)

        draw = ImageDraw.Draw(merged)
        text_x = logo1.width + padding
        draw.text((text_x, text_y), separator_text, font=font, fill=(255, 255, 255, 255))

        merged.paste(logo2, (text_x + text_width + padding, y2), logo2)

        return merged


    # =================== Multiprocessing Function ===================
def process_single_image(args):
        """
        Process a single image with all the branding effects.
        This function is designed to work with multiprocessing.
        """
        try:
            # Unpack arguments
            file_data, params = args
            filename, img_bytes = file_data
            
            # Load image from bytes
            img = Image.open(io.BytesIO(img_bytes))
            img = preprocess_uploaded_image(img)
            
            # Resize if too large
            if max(img.size) > 3000:
                img = img.resize((img.width//2, img.height//2), Image.LANCZOS)
            
            # Enhanced subject detection
            img_array = np.array(img)
            bb = enhanced_subject_detection_for_mp(img_array)
            if bb is None:
                bb = (img.width//4, img.height//4, 3*img.width//4, 3*img.height//4)
            
            # Smart resize and crop
            base = smart_resize_preserve_background(
                img, bb, (params['tw'], params['th']), 
                params['ts'], params['bs']
            ).convert("RGBA")
            
            # Add logo if provided
            if params['logo_img_bytes'] is not None:
                logo_img = pickle.loads(params['logo_img_bytes'])
                
                lw = int(params['scale']/100 * base.width)
                lh = int(lw / logo_img.width * logo_img.height)
                logo_res = logo_img.resize((lw, lh), Image.LANCZOS)
                x_px = int((params['x_off']/100) * (base.width - lw))
                y_px = int((params['y_off']/100) * (base.height - lh))
                
                if params['bgblur']:
                    base = add_blur_background_under_logo(
                        base, logo_res, x_px, y_px, params['br'], params['mm']
                    ).convert("RGBA")
                
                if params['shadow']:
                    base = add_black_glow_around_logo(
                        base, logo_res, x_px, y_px, params['sr'], params['so']
                    ).convert("RGBA")
                else:
                    base.paste(logo_res, (x_px, y_px), logo_res)
            
            # Add text overlay if provided
            if params['overlay_text']:
                draw = ImageDraw.Draw(base)
                font_folder = "fonts"
                font_paths = {
                    "Arial": os.path.join(font_folder, "arial.ttf"),
                    "Helvetica": os.path.join(font_folder, "helvetica.ttf"),
                    "Times New Roman": os.path.join(font_folder, "times.ttf"),
                    "Chronicle Display": os.path.join(font_folder, "Chronicle Display Black.ttf"),
                    "Facundo": os.path.join(font_folder, "Facundo.ttf"),
                    "Felidae": os.path.join(font_folder, "Felidae.ttf"),
                    "Edwardian Script ITC": os.path.join(font_folder, "Edwardian.ttf")
                }
                font_file = font_paths.get(params['font_family'], "arial.ttf")
                try:
                    font = ImageFont.truetype(font_file, params['text_size'])
                except:
                    font = ImageFont.load_default()
                
                bbox = draw.textbbox((0, 0), params['overlay_text'], font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                x_text = int((params['text_x_pct']/100) * (base.width - w))
                y_text = int((params['text_y_pct']/100) * (base.height - h))
                draw.text((x_text, y_text), params['overlay_text'], font=font, fill=params['text_color'])
            
            # Convert to RGB and optimize
            final = base.convert("RGB")
            buf = optimize_image(final, params['max_kb'])
            
            return (filename, final, buf)
            
        except Exception as e:
            return (filename, None, str(e))

def main():

    # Enhanced CSS with modern design system
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        /* Global Variables */
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --success-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --warning-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            --dark-bg: #0f1419;
            --card-bg: #1a202c;
            --text-primary: #ffffff;
            --text-secondary: #a0aec0;
            --border-color: #2d3748;
            --accent-blue: #4299e1;
            --accent-purple: #9f7aea;
            --shadow-soft: 0 4px 16px rgba(0, 0, 0, 0.12);
            --shadow-medium: 0 8px 32px rgba(0, 0, 0, 0.18);
            --border-radius: 16px;
            --border-radius-small: 8px;
        }
        
        /* Base Styling */
        .main {
            background: var(--dark-bg);
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background: var(--dark-bg);
        }
        
        /* Main Header */
        .hero-header {
            background: var(--primary-gradient);
            padding: 1rem 0.5rem 2rem 0.5rem;
            border-radius: var(--border-radius);
            margin-bottom: 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .hero-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="150" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="0.5" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            opacity: 0.3;
        }
        
        .hero-header h1 {
            color: var(--text-primary);
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            text-shadow: 0 4px 12px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
        }
        
        .hero-header p {
            color: rgba(255,255,255,0.9);
            font-size: 1.25rem;
            font-weight: 400;
            margin: 0;
            position: relative;
            z-index: 1;
        }
        
        /* Feature Cards */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .feature-card {
            background: var(--card-bg);
            padding: 2rem;
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--primary-gradient);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-medium);
            border-color: var(--accent-blue);
        }
        
        .feature-card:hover::before {
            transform: scaleX(1);
        }
        
        .feature-card h4 {
            color: var(--text-primary);
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }
        
        .feature-card p {
            color: var(--text-secondary);
            font-size: 0.95rem;
            line-height: 1.6;
            margin: 0;
        }
        
        /* Sidebar Styling */
        .css-1d391kg {
            background: var(--card-bg);
            border-right: 1px solid var(--border-color);
        }
        
        .sidebar-section {
            background: var(--primary-gradient);
            padding: 1.5rem;
            border-radius: var(--border-radius);
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        
        .sidebar-section::before {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            width: 100px;
            height: 100px;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            transform: translate(30px, -30px);
        }
        
        .sidebar-section h3 {
            color: var(--text-primary);
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 1;
        }
        
        .sidebar-section p {
            color: rgba(255,255,255,0.8);
            font-size: 0.9rem;
            margin: 0;
            position: relative;
            z-index: 1;
        }
        
        /* Upload Area */
        .upload-container {
            background: var(--card-bg);
            border: 2px dashed var(--border-color);
            border-radius: var(--border-radius);
            padding: 3rem 2rem;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .upload-container:hover {
            border-color: var(--accent-blue);
            background: rgba(66, 153, 225, 0.05);
        }
        
        .upload-container::before {
            content: '📸';
            font-size: 3rem;
            display: block;
            margin-bottom: 1rem;
            opacity: 0.6;
        }
        
        /* File Upload Styling */
        .stFileUploader > div > div {
            background: transparent !important;
            border: none !important;
        }
        
        .stFileUploader label {
            color: var(--text-primary) !important;
            font-weight: 500 !important;
        }
        
        /* Image Grid */
        .image-preview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }
        
        .image-preview-card {
            background: var(--card-bg);
            border-radius: var(--border-radius-small);
            padding: 0.75rem;
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }
        
        .image-preview-card:hover {
            transform: scale(1.02);
            box-shadow: var(--shadow-soft);
        }
        
        /* Process Button */
        .process-button-container {
            display: flex;
            justify-content: center;
            margin: 3rem 0;
        }
        
        .process-button {
            background: var(--success-gradient) !important;
            border: none !important;
            border-radius: 50px !important;
            padding: 1rem 3rem !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: white !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: var(--shadow-soft) !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .process-button:hover {
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-medium) !important;
        }
        
        .process-button::before {
            content: '⚡';
            margin-right: 0.5rem;
        }
        
        /* Results Section */
        .results-header {
            background: var(--card-bg);
            padding: 2rem;
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
            margin: 2rem 0;
            text-align: center;
        }
        
        .results-header h2 {
            color: var(--text-primary);
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        
        .result-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }
        
        .result-card:hover {
            transform: translateY(-8px);
            box-shadow: var(--shadow-medium);
            border-color: var(--accent-purple);
        }
        
        .result-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
        }
        
        .result-card-content {
            padding: 1.5rem;
        }
        
        .result-card-title {
            color: var(--text-primary);
            font-size: 1rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            word-break: break-all;
        }
        
        .result-card-meta {
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }
        
        /* Download Buttons */
        .download-button {
            background: var(--secondary-gradient) !important;
            border: none !important;
            border-radius: var(--border-radius-small) !important;
            padding: 0.75rem 1.5rem !important;
            color: white !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
        }
        
        .download-button:hover {
            transform: translateY(-1px) !important;
            box-shadow: var(--shadow-soft) !important;
        }
        
        .batch-download {
            background: var(--warning-gradient) !important;
            border-radius: 50px !important;
            padding: 1rem 2rem !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            margin: 2rem auto !important;
            display: block !important;
        }
        
        /* Progress Indicators */
        .stProgress > div > div > div > div {
            background: var(--success-gradient) !important;
            border-radius: 10px !important;
        }
        
        /* Status Cards */
        .status-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            padding: 2rem;
            margin: 1.5rem 0;
            position: relative;
            overflow: hidden;
        }
        
        .status-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--success-gradient);
        }
        
        .status-card h3 {
            color: var(--text-primary);
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .status-card p {
            color: var(--text-secondary);
            margin: 0;
            line-height: 1.6;
        }
        
        /* Success/Error Messages */
        .stSuccess {
            background: var(--success-gradient) !important;
            color: white !important;
            border-radius: var(--border-radius-small) !important;
            border: none !important;
        }
        
        .stError {
            background: var(--secondary-gradient) !important;
            color: white !important;
            border-radius: var(--border-radius-small) !important;
            border: none !important;
        }
        
        .stWarning {
            background: var(--warning-gradient) !important;
            color: var(--dark-bg) !important;
            border-radius: var(--border-radius-small) !important;
            border: none !important;
        }
        
        /* Form Controls */
        .stSelectbox > div > div {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--border-radius-small) !important;
            color: var(--text-primary) !important;
        }
        
        .stNumberInput > div > div > input {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--border-radius-small) !important;
            color: var(--text-primary) !important;
        }
        
        .stSlider > div > div > div > div {
            background: var(--primary-gradient) !important;
        }
        
        .stTextInput > div > div > input {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--border-radius-small) !important;
            color: var(--text-primary) !important;
        }
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--border-radius-small) !important;
            color: var(--text-primary) !important;
        }
        
        .streamlit-expanderContent {
            background: var(--card-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-top: none !important;
            border-radius: 0 0 var(--border-radius-small) var(--border-radius-small) !important;
        }
        
        /* Footer */
        .footer {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            padding: 2rem;
            text-align: center;
        }
        
        .footer h3 {
            color: var(--text-primary);
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .footer p {
            color: var(--text-secondary);
            margin: 0.25rem 0;
        }
        
        /* Hide Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .hero-header h1 {
                font-size: 2.5rem;
            }
            
            .hero-header p {
                font-size: 1rem;
            }
            
            .features-grid {
                grid-template-columns: 1fr;
            }
            
            .results-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """, unsafe_allow_html=True)


    @st.cache_resource
    def load_yolo_model():
        return YOLO("yolov8n-seg.pt")


    model = load_yolo_model()


    # =================== Utility Functions ===================
    


    # =================== UI State ===================
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0
    if "stored_files" not in st.session_state:
        st.session_state.stored_files = []
    if "results" not in st.session_state:
        st.session_state.results = []


    # =================== Main UI ===================

    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1>Snipster</h1>
        <p>Your Smart Image Resizing Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature Cards
    st.markdown("""
    <div class="features-grid">
        <div class="feature-card">
            <h4>Professional Branding</h4>
            <p>Add logos, text overlays, and stunning visual effects with pixel-perfect positioning and advanced blending modes.</p>
        </div>
        <div class="feature-card">
            <h4>Lightning-Fast Batch Processing</h4>
            <p>Process hundreds of images simultaneously using intelligent parallel computing for maximum efficiency.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3>Tools & Settings    </h3>
            <p>Configure your image processing pipeline with precision</p>
        </div>
        """, unsafe_allow_html=True)
        
        mode = st.selectbox(
            "Processing Mode:",
            ["Smart Cropper + Branding"],
            help="Select your desired processing workflow"
        )
        
        if st.button("🗑️ Clear All Files", help="Reset all uploaded files and results", type="secondary"):
            st.session_state.upload_key += 1
            st.session_state.stored_files = []
            st.session_state.results = []
            st.rerun()

        
    files = st.file_uploader(
            "Choose image files",
            type=["jpg","jpeg","png"],
            accept_multiple_files=True,
            key=f"up_{st.session_state.upload_key}",
            help="Select one or more images to process (JPG, JPEG, PNG supported)",
            label_visibility="collapsed"
        )
        
    if files:
        st.session_state.stored_files = files
        st.success(f"Successfully uploaded {len(files)} file(s)!")
            
            # Enhanced image preview
        with st.expander("Preview Uploaded Images", expanded=True):
            st.markdown('<div class="image-preview-grid">', unsafe_allow_html=True)
                
            # Create columns for image grid
            cols = st.columns(min(4, len(files)))
            for i, file in enumerate(files[:8]):  # Show first 8 images
                col_idx = i % len(cols)
                with cols[col_idx]:
                    img = Image.open(file)
                    st.image(img, caption=f"{file.name}", use_container_width=True)
                        
            if len(files) > 8:
                st.info(f"... and {len(files) - 8} more files ready for processing")
            
            st.markdown('</div>', unsafe_allow_html=True)


    # Enhanced Sidebar Configuration
    if mode == "Smart Cropper + Branding":
        with st.sidebar:
            st.markdown("---")
            
            # Output Configuration
            st.markdown("""
            <div class="sidebar-section">
                <h3>Quick Controls</h3>
                <p>Configure dimensions and quality</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Dimensions & Quality", expanded=True):
                col_w, col_h = st.columns(2)
                with col_w:
                    tw = st.number_input("Width (px)", 512, 4096, 1200, 100, help="Output image width")
                with col_h:
                    th = st.number_input("Height (px)", 512, 4096, 1800, 100, help="Output image height")
                
                max_kb = st.number_input("Max File Size (KB)", 100, 5000, 800, 50, help="Optimize file size")
                
                # Enhanced aspect ratio display
                aspect_ratio = round(tw/th, 2)
                if aspect_ratio == 0.67:
                    ratio_name = "Portrait (2:3)"
                elif aspect_ratio == 1.0:
                    ratio_name = "Square (1:1)"
                elif aspect_ratio == 1.33:
                    ratio_name = "Standard (4:3)"
                elif aspect_ratio == 1.78:
                    ratio_name = "Widescreen (16:9)"
                else:
                    ratio_name = f"Custom ({aspect_ratio}:1)"
                
                st.success(f"Aspect Ratio: **{ratio_name}**")

            with st.expander("Performance Settings", expanded=False):
                max_workers = st.slider("Parallel Processes", 1, 8, 4, 1, help="More processes = faster processing")
                st.info(f"Using {max_workers} parallel processes")

            with st.expander("Cropping", expanded=False):
                use_space = st.checkbox("Add Head/Foot Space", help="Add extra breathing room around detected subjects")
                if use_space:
                    col_ts, col_bs = st.columns(2)
                    with col_ts:
                        ts = st.number_input("Top Space (px)", 0, 1000, 10)
                    with col_bs:
                        bs = st.number_input("Bottom Space (px)", 0, 1000, 10)
                else:
                    ts = bs = 0

            # Enhanced Branding Configuration
            st.markdown("---")
            st.markdown("""
            <div class="sidebar-section">
                <h3>Branding & Design</h3>
                <p>Add professional branding elements</p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Logo Settings", expanded=True):
                collab_mode = st.checkbox("Collaboration Mode", help="Merge two logos for collaboration posts")
                
                if collab_mode:
                    st.markdown("**Upload Collaboration Logos:**")
                    col_l1, col_l2 = st.columns(2)
                    with col_l1:
                        logo_file1 = st.file_uploader("First Logo", type=["png","jpg","jpeg"], key="logo1_up")
                    with col_l2:
                        logo_file2 = st.file_uploader("Second Logo", type=["png","jpg","jpeg"], key="logo2_up")
                    
                    if logo_file1 and logo_file2:
                        col_prev1, col_prev2 = st.columns(2)
                        with col_prev1:
                            st.image(logo_file1, caption="Logo 1", width=80)
                        with col_prev2:
                            st.image(logo_file2, caption="Logo 2", width=80)
                    
                    merge_padding = st.slider("Logo Spacing", 0, 100, 20, help="Space between logos")
                    separator_text = st.text_input("Separator Symbol", value="×", help="Symbol between logos")
                    separator_font_size = st.slider("Symbol Size", 10, 200, 40)
                else:
                    logo_file = st.file_uploader("Upload Logo", type=["png","jpg","jpeg"], key="logo_up", help="PNG files work best for transparency")
                    if logo_file:
                        st.image(logo_file, caption="Logo Preview", width=120)

                if (collab_mode and logo_file1 and logo_file2) or (not collab_mode and logo_file):
                    st.markdown("**Logo Positioning:**")
                    scale = st.slider("Size (% of width)", 5, 50, 30, help="Logo size relative to image width")
                    
                    col_x, col_y = st.columns(2)
                    with col_x:
                        x_off = st.slider("Horizontal", 0, 100, 50, help="0=Left, 50=Center, 100=Right")
                    with col_y:
                        y_off = st.slider("Vertical", 0, 100, 90, help="0=Top, 50=Middle, 100=Bottom")

            with st.expander("Visual Effects", expanded=False):
                shadow = st.checkbox("Enable Logo Shadow", value=True, help="Add depth with shadow effects")
                if shadow:
                    col_sr, col_so = st.columns(2)
                    with col_sr:
                        sr = st.slider("Shadow Blur", 2, 50, 25, help="Shadow softness")
                    with col_so:
                        so = st.slider("Shadow Opacity", 0, 100, 30, help="Shadow intensity")
                else:
                    sr = so = 0
                    
                bgblur = st.checkbox("Background Blur Under Logo", help="Blur background behind logo for better visibility")
                if bgblur:
                    col_br, col_mm = st.columns(2)
                    with col_br:
                        br = st.slider("Blur Radius", 1, 50, 10, help="Blur intensity")
                    with col_mm:
                        mm = st.slider("Mask Margin", 1, 50, 5, help="Blur area size")
                else:
                    br = mm = 0

            with st.expander("Text Overlay", expanded=False):
                overlay_text = st.text_input("Overlay Text", placeholder="Enter text to overlay on images...")

                if overlay_text:
                    st.markdown("**Text Styling:**")
                    col_size, col_color = st.columns(2)
                    with col_size:
                        text_size = st.slider("Font Size", 10, 200, 40)
                    with col_color:
                        text_color = st.color_picker("Text Color", "#FFFFFF", help="Choose text color")
                    
                    font_family = st.selectbox(
                        "Font Family",
                        [
                            "Arial",
                            "Helvetica", 
                            "Times New Roman",
                            "Chronicle Display",
                            "Facundo",
                            "Felidae",
                            "Edwardian Script ITC"
                        ],
                        help="Select font style"
                    )
                    
                    st.markdown("**Text Position:**")
                    col_tx, col_ty = st.columns(2)
                    with col_tx:
                        text_x_pct = st.slider("Horizontal", 0, 100, 50, key="text_x")
                    with col_ty:
                        text_y_pct = st.slider("Vertical", 0, 100, 95, key="text_y")

        # Enhanced Processing Section
        st.markdown("---")
        
        # Processing button with better styling
        if st.session_state.stored_files:
            if st.button("Start Processing", help="Begin processing all uploaded images", type="primary", use_container_width=True):
                # Processing logic starts here...
                
                # Prepare logo image
                logo_img_bytes = None
                if collab_mode and logo_file1 and logo_file2:
                    tmp1 = Image.open(logo_file1).convert("RGBA")
                    tmp2 = Image.open(logo_file2).convert("RGBA")
                    
                    # Remove white backgrounds
                    datas1 = tmp1.getdata()
                    newData1 = [(r,g,b,0) if r>240 and g>240 and b>240 else (r,g,b,a) for r,g,b,a in datas1]
                    tmp1.putdata(newData1)
                    datas2 = tmp2.getdata()
                    newData2 = [(r,g,b,0) if r>240 and g>240 and b>240 else (r,g,b,a) for r,g,b,a in datas2]
                    tmp2.putdata(newData2)
                    
                    logo_img = merge_logos_horizontally(
                        tmp1, tmp2,
                        padding=merge_padding,
                        separator_text=separator_text,
                        separator_font_size=separator_font_size
                    )
                    logo_img_bytes = pickle.dumps(logo_img)
                    
                elif not collab_mode and logo_file:
                    tmp = Image.open(logo_file).convert("RGBA")
                    datas = tmp.getdata()
                    newData = [(r,g,b,0) if r>240 and g>240 and b>240 else (r,g,b,a) for r,g,b,a in datas]
                    tmp.putdata(newData)
                    logo_img_bytes = pickle.dumps(tmp)

                # Prepare parameters for multiprocessing
                params = {
                    'tw': tw,
                    'th': th,
                    'ts': ts,
                    'bs': bs,
                    'max_kb': max_kb,
                    'logo_img_bytes': logo_img_bytes,
                    'scale': scale if ((collab_mode and logo_file1 and logo_file2) or (not collab_mode and logo_file)) else 30,
                    'x_off': x_off if ((collab_mode and logo_file1 and logo_file2) or (not collab_mode and logo_file)) else 50,
                    'y_off': y_off if ((collab_mode and logo_file1 and logo_file2) or (not collab_mode and logo_file)) else 90,
                    'shadow': shadow,
                    'sr': sr,
                    'so': so,
                    'bgblur': bgblur,
                    'br': br,
                    'mm': mm,
                    'overlay_text': overlay_text,
                    'text_size': text_size if overlay_text else 40,
                    'text_color': text_color if overlay_text else "#FFFFFF",
                    'font_family': font_family if overlay_text else "Arial",
                    'text_x_pct': text_x_pct if overlay_text else 50,
                    'text_y_pct': text_y_pct if overlay_text else 95
                }

                # Prepare file data for multiprocessing
                file_data_list = []
                for f in st.session_state.stored_files:
                    f.seek(0)  # Reset file pointer
                    file_bytes = f.read()
                    file_data_list.append((f.name, file_bytes))

                # Create arguments list for multiprocessing
                args_list = [(file_data, params) for file_data in file_data_list]

                # Enhanced processing status display
                st.markdown("""
                <div class="status-card">
                    <h3>Processing In Progress</h3>
                    <p>Processing <strong>{}</strong> images using <strong>{}</strong> parallel processes...</p>
                </div>
                """.format(len(file_data_list), max_workers), unsafe_allow_html=True)

                # Process images with enhanced progress tracking
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    time_text = st.empty()

                    import time
                    start_time = time.time()

                    try:
                        if __name__ == "__main__" or True:  # Added for Streamlit compatibility
                            with Pool(processes=max_workers) as pool:
                                results = []
                                completed = 0
                                total = len(args_list)
                                
                                # Use imap_unordered to get results as they complete
                                for result in pool.imap_unordered(process_single_image, args_list):
                                    results.append(result)
                                    completed += 1
                                    progress = completed / total
                                    progress_bar.progress(progress)
                                    
                                    elapsed = time.time() - start_time
                                    if completed > 0:
                                        eta = (elapsed / completed) * (total - completed)
                                        status_text.text(f"Processed {completed}/{total} images...")
                                        time_text.text(f"Time: {elapsed:.1f}s | ETA: {eta:.1f}s")
                            
                            # Filter results
                            successful_results = []
                            failed_results = []
                            
                            for result in results:
                                filename, img, buf_or_error = result
                                if img is not None:
                                    successful_results.append(result)
                                else:
                                    failed_results.append((filename, buf_or_error))
                            
                            progress_bar.progress(1.0)
                            total_time = time.time() - start_time
                            status_text.text("Processing Complete!")
                            time_text.text(f"Completed in {total_time:.1f} seconds")
                            
                            if failed_results:
                                st.warning(f"Failed to process {len(failed_results)} images:")
                                for filename, error in failed_results:
                                    st.error(f"{filename}: {error}")
                            
                            if successful_results:
                                st.success(f"Successfully processed {len(successful_results)} images!")
                                st.session_state.results = successful_results
                            else:
                                st.error("No images were processed successfully.")
                                st.session_state.results = []
                                
                    except Exception as e:
                        st.error(f"Multiprocessing error: {str(e)}")
                        st.info("Falling back to sequential processing...")
                        
                        # Fallback processing
                        results = []
                        for i, args in enumerate(args_list):
                            result = process_single_image(args)
                            results.append(result)
                            progress = (i + 1) / len(args_list)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {i + 1}/{len(args_list)} images...")

                        progress_bar.progress(1.0)
                        status_text.text("Sequential processing complete!")
                        st.session_state.results = [r for r in results if r[1] is not None]
            
        else:
            st.markdown("""
            <div class="status-card">
                <h3>No Files Uploaded</h3>
                <p>Please upload some images to begin processing.</p>
            </div>
            """, unsafe_allow_html=True)

    # Enhanced Results Section
    if st.session_state.results:
        st.markdown("---")
        
        # Results header
        st.markdown("""
        <div class="results-header">
            <h2>Processing Results</h2>
            <p>Your processed images are ready for download</p>
        </div>
        """, unsafe_allow_html=True)
        
            # Create ZIP file for batch download
        z = io.BytesIO()
        with zipfile.ZipFile(z, "w") as zf:
            for name, _, buf in st.session_state.results:
                zf.writestr(f"branded_{name}", buf.getvalue())
        z.seek(0)
            
        st.download_button(
            "Download All Images (ZIP)",
            data=z.getvalue(),
            file_name="snipster_processed_images.zip",
            mime="application/zip",
            help="Download all processed images as a ZIP file",
            type="primary",
            use_container_width=True
        )
        
        # Individual results with enhanced grid
        st.markdown("### Individual Results")
        st.markdown('<div class="results-grid">', unsafe_allow_html=True)
        
        # Create responsive grid
    cols_per_row = 3
    num_results = len(st.session_state.results)
    for i in range(0, num_results, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (name, img, buf) in enumerate(st.session_state.results[i:i+cols_per_row]):
            if j < len(cols):
                with cols[j]:
                        # Result card
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)                    
                        # Display image
                    st.image(img, use_container_width=True)
                        
                        # Card content
                    st.markdown('<div class="result-card-content">', unsafe_allow_html=True)
                        
                        # File info
                    file_size_kb = len(buf.getvalue()) // 1024
                    st.markdown(f"""
                    <div class="result-card-title">{name}</div>
                    <div class="result-card-meta">
                        {img.width}×{img.height} pixels<br>
                        {file_size_kb} KB
                    </div>
                    """, unsafe_allow_html=True)

                        # Download button
                    st.download_button(
                        "Download",
                            data=buf.getvalue(),
                            file_name=f"snipster_{name}",
                            mime="image/jpeg",
                            key=f"dl_{i}_{j}",
                            use_container_width=True,
                            help=f"Download {name}"
                        )

    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3>Snipster - AI Image Assistant</h3>
        <p><strong>Professional image processing with intelligent automation</strong></p>
        <p>Smart Cropping • Professional Branding • Lightning Fast Processing</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
