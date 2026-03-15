import streamlit as st
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Uzhavar Ariviyal AI", layout="wide", page_icon="🪴")

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>
/* ------------------------------------------------
GLOBAL PAGE BACKGROUND
------------------------------------------------ */

.stApp{
background: linear-gradient(rgba(232,245,233,0.9), rgba(255,255,255,0.9)),
url('https://www.transparenttextures.com/patterns/leaf.png');
background-color:#f0f4f0;
}


/* ------------------------------------------------
NAVBAR
------------------------------------------------ */

.navbar{
background:#1b5e20;
padding:15px 40px;
border-radius:12px;
display:flex;
justify-content:space-between;
align-items:center;
color:white;
margin-bottom:30px;
}

.nav-title{
font-size:22px;
font-weight:600;
}

.nav-menu{
font-size:16px;
opacity:0.9;
}


/* ------------------------------------------------
HERO SECTION
------------------------------------------------ */

.hero-box{
background:linear-gradient(135deg,#2e7d32,#1b5e20);
padding:60px;
border-radius:20px;
color:white;
text-align:center;
margin-bottom:35px;
box-shadow:0 10px 25px rgba(0,0,0,0.15);
}

.hero-box h1{
font-size:42px;
margin-bottom:10px;
}

.hero-box p{
font-size:18px;
opacity:0.9;
}


/* ------------------------------------------------
FEATURE CARDS (HOME PAGE)
------------------------------------------------ */

.feature-card{
background:white;
padding:30px;
border-radius:18px;
text-align:center;
box-shadow:0 8px 18px rgba(0,0,0,0.1);
transition:0.3s;
}

.feature-card:hover{
transform:translateY(-6px);
box-shadow:0 12px 25px rgba(0,0,0,0.15);
}

.feature-icon{
font-size:45px;
margin-bottom:10px;
}


/* ------------------------------------------------
HOME PAGE ICON CARDS
------------------------------------------------ */

.icon-card{
background:#f5deb3;
padding:35px;
border-radius:18px;
border-left:6px solid #4caf50;
text-align:center;
transition:0.3s;
box-shadow:0 6px 12px rgba(0,0,0,0.08);
}

.icon-card:hover{
transform:translateY(-8px);
box-shadow:0 10px 18px rgba(0,0,0,0.15);
}

.icon-card h3{
font-size:60px;
margin-bottom:15px;
}

.icon-card b{
font-size:24px;
display:block;
margin-bottom:8px;
}

.icon-card p{
font-size:17px;
color:black;
}


/* ------------------------------------------------
UPLOAD SECTION
------------------------------------------------ */

.upload-card{
background:#ffffff;
padding:35px;
border-radius:18px;
box-shadow:0 8px 20px rgba(0,0,0,0.1);
text-align:center;
margin:20px auto;
max-width:900px;
}

.stFileUploader{
border:none;
background:transparent;
}


/* ------------------------------------------------
AI RESULT / IMAGE CARDS
------------------------------------------------ */

.ai-card{
background:#ffffff;
padding:25px;
border-radius:18px;
box-shadow:0 8px 20px rgba(0,0,0,0.1);
}

.card-title{
font-size:22px;
font-weight:600;
color:#2e7d32;
margin-bottom:15px;
}


/* ------------------------------------------------
CONFIDENCE RESULT BOX
------------------------------------------------ */

.confidence-box{
background:#e8f5e9;
padding:10px;
border-radius:10px;
font-size:18px;
margin-top:10px;
}


/* ------------------------------------------------
SCAN ANIMATION
------------------------------------------------ */

.scan-animation{
font-size:18px;
color:#2e7d32;
animation:blink 1s infinite;
}

@keyframes blink{
0%{opacity:1;}
50%{opacity:0.3;}
100%{opacity:1;}
}


/* ------------------------------------------------
BUTTON STYLE
------------------------------------------------ */

.stButton>button{
background:#2e7d32 !important;
color:#f5deb3 !important;
border-radius:10px;
width:100%;
font-size:16px;
height:45px;
}


/* ------------------------------------------------
PAGE TITLES
------------------------------------------------ */

.page-title{
text-align:center;
font-size:36px;
font-weight:bold;
color:#2e7d32;
margin-bottom:20px;
}


/* ------------------------------------------------
FOOTER
------------------------------------------------ */

.footer{
text-align:center;
color:black;
font-size:14px;
padding:10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "Leaf_Disease_128x128.h5")

@st.cache_resource
def load_my_model(path):
    if not os.path.exists(path):
        st.error("Model file not found at path.")
        return None
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model(model_path)

# ---------------------------------------------------
# LABELS
# ---------------------------------------------------
RAW_LABELS = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

TREATMENT_DB = {
    # ---------------- APPLE ----------------
    "Apple - Apple scab": "Apply captan or myclobutanil fungicide. Remove fallen leaves and prune infected branches.",
    "Apple - Black rot": "Remove infected fruits and twigs. Apply copper-based fungicide during growing season.",
    "Apple - Cedar apple rust": "Apply sulfur or myclobutanil fungicide. Remove nearby cedar hosts if possible.",
    "Apple - healthy": "No disease detected. Maintain regular watering and balanced fertilization.",
    
    # ---------------- BLUEBERRY ----------------
    "Blueberry - healthy": "Plant is healthy. Ensure acidic soil and proper irrigation.",
    
    # ---------------- CHERRY ----------------
    "Cherry (including sour) - Powdery mildew": "Apply sulfur fungicide. Improve air circulation and avoid overhead watering.",
    "Cherry (including sour) - healthy": "Plant is healthy. Maintain pruning and balanced nutrients.",
    
    # ---------------- CORN ----------------
    "Corn (maize) - Cercospora leaf spot Gray leaf spot": "Apply strobilurin or triazole fungicide. Use resistant hybrids.",
    "Corn (maize) - Common rust": "Use resistant varieties. Apply fungicide if infection is severe.",
    "Corn (maize) - Northern Leaf Blight": "Use resistant hybrids. Apply fungicide at early infection stage.",
    "Corn (maize) - healthy": "Crop is healthy. Continue proper nitrogen management.",
    
    # ---------------- GRAPE ----------------
    "Grape - Black rot": "Apply mancozeb or myclobutanil. Remove infected berries.",
    "Grape - Esca (Black Measles)": "Prune infected wood. Avoid large pruning wounds.",
    "Grape - Leaf blight (Isariopsis Leaf Spot)": "Apply copper fungicide. Improve vineyard airflow.",
    "Grape - healthy": "Vine is healthy. Maintain proper canopy management.",
    
    # ---------------- ORANGE ----------------
    "Orange - Haunglongbing (Citrus greening)": "Remove infected trees. Control psyllid insects using approved insecticides.",
    
    # ---------------- PEACH ----------------
    "Peach - Bacterial spot": "Apply copper sprays. Use resistant cultivars.",
    "Peach - healthy": "Tree is healthy. Maintain pruning and irrigation schedule.",
    
    # ---------------- PEPPER ----------------
    "Pepper, bell - Bacterial spot": "Use copper bactericides. Avoid overhead irrigation.",
    "Pepper, bell - healthy": "Plant is healthy. Maintain good drainage.",
    
    # ---------------- POTATO ----------------
    "Potato - Early blight": "Apply chlorothalonil fungicide. Practice crop rotation.",
    "Potato - Late blight": "Apply metalaxyl or copper fungicide. Avoid high humidity conditions.",
    "Potato - healthy": "Crop is healthy. Maintain proper spacing and watering.",
    
    # ---------------- RASPBERRY ----------------
    "Raspberry - healthy": "Plant is healthy. Maintain pruning and pest monitoring.",
    
    # ---------------- SOYBEAN ----------------
    "Soybean - healthy": "Crop is healthy. Maintain balanced fertilization.",
    
    # ---------------- SQUASH ----------------
    "Squash - Powdery mildew": "Apply sulfur or potassium bicarbonate spray. Improve air circulation.",
    
    # ---------------- STRAWBERRY ----------------
    "Strawberry - Leaf scorch": "Remove infected leaves. Apply protective fungicide if needed.",
    "Strawberry - healthy": "Plant is healthy. Ensure good soil drainage.",
    
    # ---------------- TOMATO ----------------
    "Tomato - Bacterial spot": "Apply copper-based bactericide. Avoid leaf wetness.",
    "Tomato - Early blight": "Apply fungicide such as chlorothalonil. Remove lower infected leaves.",
    "Tomato - Late blight": "Apply copper fungicide. Destroy infected plants immediately.",
    "Tomato - Leaf Mold": "Improve ventilation. Apply sulfur-based fungicide.",
    "Tomato - Septoria leaf spot": "Remove infected leaves. Apply fungicide early.",
    "Tomato - Spider mites Two-spotted spider mite": "Spray neem oil or insecticidal soap. Maintain humidity.",
    "Tomato - Target Spot": "Apply fungicide. Avoid overhead watering.",
    "Tomato - Tomato Yellow Leaf Curl Virus": "Control whiteflies using insecticide. Remove infected plants.",
    "Tomato - Tomato mosaic virus": "Remove infected plants. Disinfect tools properly.",
    "Tomato - healthy": "Plant is healthy. Continue balanced fertilization and monitoring."
}

def format_label(label):
    return label.replace("___", " - ").replace("_", " ")

# ---------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# ---------------------------------------------------
# NAVIGATION FUNCTIONS
# ---------------------------------------------------
def go_to_home():
    st.session_state.page = 'home'

def go_to_detector():
    st.session_state.page = 'detector'

def go_to_supported():
    st.session_state.page = 'supported'

def go_to_treatment():
    st.session_state.page = 'treatment'

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------
if st.session_state.page == 'home':
    # ---------------- NAVBAR ----------------
    st.markdown("""
    <div class="navbar">
    <div class="nav-title">Smart Crop Disease Detection</div>
    <div class="nav-menu">Upload a leaf image and let AI detect plant diseases instantly.</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- HERO SECTION ----------------
    st.markdown("""
    <div class="hero-box">
    <h1>🌿 Uzhavar Ariviyal AI</h1>
    <p>உழுவார் உலகத்தார்க்கு ஆணிஅஃ தாற்றாது எழுவாரை எல்லாம் பொறுத்து.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Choose a Feature")

    col1, col2, col3 = st.columns(3)

    # ---------- Diagnose ----------
    with col1:
        st.markdown("""
        <div class="feature-card">
        <div class="feature-icon">📸</div>
        <h3>AI Leaf Diagnosis</h3>
        <p>Scan crop leaves and identify diseases instantly using AI.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Start Diagnosis"):
            go_to_detector()

    # ---------- Crop Library ----------
    with col2:
        st.markdown("""
        <div class="feature-card">
        <div class="feature-icon">🌾</div>
        <h3>Crop Library</h3>
        <p>Explore supported crops and disease types detected by the model.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("View Crops"):
            go_to_supported()

    # ---------- Treatment ----------
    with col3:
        st.markdown("""
        <div class="feature-card">
        <div class="feature-icon">🌱</div>
        <h3>Treatment Guide</h3>
        <p>Get recommended treatment methods for detected diseases.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("View Treatment"):
            go_to_treatment()

# ---------------------------------------------------
# DETECTOR PAGE
# ---------------------------------------------------
elif st.session_state.page == 'detector':
    st.button("⬅ Back to Home", on_click=go_to_home)

    st.markdown(
        "<h1 style='text-align:center;color:#2e7d32;'>🌿 Crop Disease Diagnosis</h1>",
        unsafe_allow_html=True
    )

    # ---------------- UPLOAD SECTION ----------------
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag and drop a crop leaf image to start diagnosis",
        type=["jpg","jpeg","png"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        display_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        img_resized = cv.resize(display_img, (128,128)) / 255.0

        with st.spinner("AI is scanning the leaf image..."):
            prediction = model.predict(np.expand_dims(img_resized, axis=0))

        idx = np.argmax(prediction)
        confidence = float(prediction[0][idx]) * 100
        predicted_label = format_label(RAW_LABELS[idx])

        col1, col2 = st.columns(2)

        # ---------------- IMAGE CARD ----------------
        with col1:
            st.markdown('<div class="ai-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📷 Uploaded Image</div>', unsafe_allow_html=True)
            st.image(display_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- RESULT CARD ----------------
        with col2:
            st.markdown('<div class="ai-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🌿 Diagnosis Result</div>', unsafe_allow_html=True)

            if confidence >= 70:
                st.success(predicted_label)
                st.markdown(
                    f'<div class="confidence-box">Confidence Level: {confidence:.2f}%</div>',
                    unsafe_allow_html=True
                )

                treatment = TREATMENT_DB.get(
                    predicted_label,
                    "General Advice: Remove infected leaves, ensure sunlight and avoid overwatering."
                )

                st.markdown("### 🌱 Recommended Treatment")
                st.write(treatment)
            else:
                st.warning("Low confidence. Please upload a clearer leaf image.")

            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# SUPPORTED PAGE
# ---------------------------------------------------
elif st.session_state.page == 'supported':
    st.button("⬅ Back to Home", on_click=go_to_home)
    st.title("🌿 Supported Crops & Diseases")

    st.markdown("### Our AI model recognizes the following conditions:")

    for label in RAW_LABELS:
        st.markdown(f"- {format_label(label)}")

# ---------------------------------------------------
# TREATMENT PAGE
# ---------------------------------------------------
elif st.session_state.page == 'treatment':
    st.button("⬅ Back to Home", on_click=go_to_home)
    st.title("🌿 Treatment Suggestions")

    st.markdown("Select a disease to view recommended treatment:")

    formatted_labels = [format_label(label) for label in RAW_LABELS]
    selected_disease = st.selectbox("Choose Disease", formatted_labels)

    if selected_disease:
        treatment = TREATMENT_DB.get(
            selected_disease,
            "General Advice: Remove infected leaves, ensure proper sunlight, avoid overwatering, and consult local agricultural officer."
        )

        st.success(f"Recommended Treatment for {selected_disease}")
        st.write(treatment)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
© 2026 உழவர் அறிவியல் AI | Uzhavar Ariviyal - AI Based Crop Disease Detection System
</div>
""", unsafe_allow_html=True)
