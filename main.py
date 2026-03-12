import streamlit as st
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import os

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Uzhavar Ariviyal AI", layout="wide", page_icon="🌿")

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.8), rgba(255, 255, 255, 0.8)), 
                    url('https://www.transparenttextures.com/patterns/leaf.png');
        background-color: #f0f4f0;
    }

    .hero-section {
        background-color: #2e7d32;
        padding: 50px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .icon-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4caf50;
        text-align: center;
        transition: transform 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    .icon-card:hover {
        transform: translateY(-5px);
    }

    .stButton>button {
        background-color: #2e7d32 !important;
        color: white !important;
        border-radius: 10px;
        width: 100%;
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
"Apple - Apple scab":
"Apply captan or myclobutanil fungicide. Remove fallen leaves and prune infected branches.",

"Apple - Black rot":
"Remove infected fruits and twigs. Apply copper-based fungicide during growing season.",

"Apple - Cedar apple rust":
"Apply sulfur or myclobutanil fungicide. Remove nearby cedar hosts if possible.",

"Apple - healthy":
"No disease detected. Maintain regular watering and balanced fertilization.",

# ---------------- BLUEBERRY ----------------
"Blueberry - healthy":
"Plant is healthy. Ensure acidic soil and proper irrigation.",

# ---------------- CHERRY ----------------
"Cherry (including sour) - Powdery mildew":
"Apply sulfur fungicide. Improve air circulation and avoid overhead watering.",

"Cherry (including sour) - healthy":
"Plant is healthy. Maintain pruning and balanced nutrients.",

# ---------------- CORN ----------------
"Corn (maize) - Cercospora leaf spot Gray leaf spot":
"Apply strobilurin or triazole fungicide. Use resistant hybrids.",

"Corn (maize) - Common rust":
"Use resistant varieties. Apply fungicide if infection is severe.",

"Corn (maize) - Northern Leaf Blight":
"Use resistant hybrids. Apply fungicide at early infection stage.",

"Corn (maize) - healthy":
"Crop is healthy. Continue proper nitrogen management.",

# ---------------- GRAPE ----------------
"Grape - Black rot":
"Apply mancozeb or myclobutanil. Remove infected berries.",

"Grape - Esca (Black Measles)":
"Prune infected wood. Avoid large pruning wounds.",

"Grape - Leaf blight (Isariopsis Leaf Spot)":
"Apply copper fungicide. Improve vineyard airflow.",

"Grape - healthy":
"Vine is healthy. Maintain proper canopy management.",

# ---------------- ORANGE ----------------
"Orange - Haunglongbing (Citrus greening)":
"Remove infected trees. Control psyllid insects using approved insecticides.",

# ---------------- PEACH ----------------
"Peach - Bacterial spot":
"Apply copper sprays. Use resistant cultivars.",

"Peach - healthy":
"Tree is healthy. Maintain pruning and irrigation schedule.",

# ---------------- PEPPER ----------------
"Pepper, bell - Bacterial spot":
"Use copper bactericides. Avoid overhead irrigation.",

"Pepper, bell - healthy":
"Plant is healthy. Maintain good drainage.",

# ---------------- POTATO ----------------
"Potato - Early blight":
"Apply chlorothalonil fungicide. Practice crop rotation.",

"Potato - Late blight":
"Apply metalaxyl or copper fungicide. Avoid high humidity conditions.",

"Potato - healthy":
"Crop is healthy. Maintain proper spacing and watering.",

# ---------------- RASPBERRY ----------------
"Raspberry - healthy":
"Plant is healthy. Maintain pruning and pest monitoring.",

# ---------------- SOYBEAN ----------------
"Soybean - healthy":
"Crop is healthy. Maintain balanced fertilization.",

# ---------------- SQUASH ----------------
"Squash - Powdery mildew":
"Apply sulfur or potassium bicarbonate spray. Improve air circulation.",

# ---------------- STRAWBERRY ----------------
"Strawberry - Leaf scorch":
"Remove infected leaves. Apply protective fungicide if needed.",

"Strawberry - healthy":
"Plant is healthy. Ensure good soil drainage.",

# ---------------- TOMATO ----------------
"Tomato - Bacterial spot":
"Apply copper-based bactericide. Avoid leaf wetness.",

"Tomato - Early blight":
"Apply fungicide such as chlorothalonil. Remove lower infected leaves.",

"Tomato - Late blight":
"Apply copper fungicide. Destroy infected plants immediately.",

"Tomato - Leaf Mold":
"Improve ventilation. Apply sulfur-based fungicide.",

"Tomato - Septoria leaf spot":
"Remove infected leaves. Apply fungicide early.",

"Tomato - Spider mites Two-spotted spider mite":
"Spray neem oil or insecticidal soap. Maintain humidity.",

"Tomato - Target Spot":
"Apply fungicide. Avoid overhead watering.",

"Tomato - Tomato Yellow Leaf Curl Virus":
"Control whiteflies using insecticide. Remove infected plants.",

"Tomato - Tomato mosaic virus":
"Remove infected plants. Disinfect tools properly.",

"Tomato - healthy":
"Plant is healthy. Continue balanced fertilization and monitoring."
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

    st.markdown("""
        <div class="hero-section">
            <h1>🌿 Uzhavar Ariviyal AI</h1>
            <p> உழுவார் உலகத்தார்க்கு ஆணிஅஃ தாற்றாது எழுவாரை எல்லாம் பொறுத்து.</p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Select an option to begin")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="icon-card">
                <h3>📸</h3>
                <p><b>Leaf Scan</b><br>Upload a single leaf for instant diagnosis.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Start Diagnosis"):
            go_to_detector()

    with col2:
        st.markdown("""
            <div class="icon-card">
                <h3>🍃</h3>
                <p><b>Crop Library</b><br>Supported Plants.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("View Supported Diseases"):
            go_to_supported()

    with col3:
        st.markdown("""
            <div class="icon-card">
                <h3>🏥</h3>
                <p><b>Treatment</b><br>Get cure suggestions.</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Get Cures"):
            go_to_treatment()

# ---------------------------------------------------
# DETECTOR PAGE
# ---------------------------------------------------
elif st.session_state.page == 'detector':

    st.button("⬅ Back to Home", on_click=go_to_home)
    st.title("🔍 Disease Diagnostic Tool")

    if model is None:
        st.error("Model file missing.")
    else:
        uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:

            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
            display_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # Resize for model
            img_resized = cv.resize(display_img, (128, 128)) / 255.0
            prediction = model.predict(np.expand_dims(img_resized, axis=0))
            idx = np.argmax(prediction)
            confidence = float(prediction[0][idx]) * 100

            predicted_label = format_label(RAW_LABELS[idx])

            # Layout
            col1, col2 = st.columns([1, 1])

            # ---------------- IMAGE ----------------
            with col1:
                st.markdown("### 📷 Uploaded Image")
                st.image(display_img, use_container_width=True)

            # ---------------- RESULT + TREATMENT ----------------
            with col2:
                st.markdown("### 🧪 Diagnosis Result")

                if confidence >= 70:
                    st.success(f"**Prediction:** {predicted_label}")
                    st.metric("Confidence Level", f"{confidence:.2f}%")

                    # Get treatment
                    treatment = TREATMENT_DB.get(
                        predicted_label,
                        "General Advice: Remove infected leaves, ensure proper sunlight, avoid overwatering, and consult local agricultural officer."
                    )

                    st.markdown("### 🌿 Recommended Treatment")
                    st.info(treatment)

                else:
                    st.warning("Low confidence. Please upload a clearer leaf image.")

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
st.caption("© 2026 உழவர் அறிவியல் AI | Uzhavar Ariviyal - AI Based Crop Disease Detection System")