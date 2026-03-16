import streamlit as st
import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import os
import plotly.graph_objects as go
from translations import TRANSLATIONS

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Uzhavar Ariviyal AI", layout="wide", page_icon="🪴")
# Language selector (top right)
col1, col2 = st.columns([10,1])

with col2:
    language = st.selectbox(
        "🌐",
        ["en","ta"],
        format_func=lambda x: " ENG" if x=="en" else "தமிழ்"
    )
def t(key):
    return TRANSLATIONS[language].get(key, key)
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
background-color:#4caf50;
}

            
/* ------------------------------------------------
NAVBAR
------------------------------------------------ */

.navbar{
background:#4caf50;
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
font-size:22px;
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



DISEASE_INFO = {

# ---------------- APPLE ----------------
"Apple - Apple scab": {
"description": "Apple scab is a fungal disease that causes dark, scabby spots on leaves and fruits.",
"cause": "Caused by the fungus Venturia inaequalis, which spreads in cool and wet spring weather."
},

"Apple - Black rot": {
"description": "Black rot causes dark circular lesions on leaves and rotting fruits.",
"cause": "Caused by the fungus Botryosphaeria obtusa which survives on dead wood and infected fruits."
},

"Apple - Cedar apple rust": {
"description": "Cedar apple rust produces yellow-orange spots on apple leaves.",
"cause": "Caused by the fungus Gymnosporangium juniperi-virginianae which requires both cedar and apple trees."
},

"Apple - healthy": {
"description": "The apple plant shows no visible signs of disease.",
"cause": "Healthy plant with proper environmental conditions and care."
},

# ---------------- BLUEBERRY ----------------
"Blueberry - healthy": {
"description": "The blueberry plant appears healthy with no disease symptoms.",
"cause": "Proper soil acidity, watering, and healthy plant management."
},

# ---------------- CHERRY ----------------
"Cherry (including sour) - Powdery mildew": {
"description": "Powdery mildew causes white powder-like fungal growth on cherry leaves.",
"cause": "Caused by Podosphaera fungi, common in warm and dry climates with poor airflow."
},

"Cherry (including sour) - healthy": {
"description": "The cherry plant is healthy and free from disease symptoms.",
"cause": "Good plant health maintained through proper pruning and care."
},

# ---------------- CORN ----------------
"Corn (maize) - Cercospora leaf spot Gray leaf spot": {
"description": "Gray leaf spot causes rectangular gray lesions on corn leaves.",
"cause": "Caused by the fungus Cercospora zeae-maydis, thriving in warm humid environments."
},

"Corn (maize) - Common rust": {
"description": "Common rust forms reddish-brown pustules on corn leaves.",
"cause": "Caused by the fungus Puccinia sorghi and spreads through windborne spores."
},

"Corn (maize) - Northern Leaf Blight": {
"description": "Northern leaf blight produces long gray-green lesions on leaves.",
"cause": "Caused by the fungus Exserohilum turcicum during humid weather."
},

"Corn (maize) - healthy": {
"description": "Corn plant is healthy with no signs of infection.",
"cause": "Good crop management and disease-resistant varieties."
},

# ---------------- GRAPE ----------------
"Grape - Black rot": {
"description": "Black rot causes brown circular spots on grape leaves and shriveled fruits.",
"cause": "Caused by the fungus Guignardia bidwellii, which thrives in warm humid weather."
},

"Grape - Esca (Black Measles)": {
"description": "Esca disease leads to leaf discoloration and black spots on grapes.",
"cause": "Caused by multiple fungal pathogens infecting the woody tissue."
},

"Grape - Leaf blight (Isariopsis Leaf Spot)": {
"description": "Leaf blight causes dark angular spots on grape leaves.",
"cause": "Caused by the fungus Isariopsis clavispora under humid conditions."
},

"Grape - healthy": {
"description": "Grape vine is healthy without visible disease symptoms.",
"cause": "Proper vineyard management and disease prevention."
},

# ---------------- ORANGE ----------------
"Orange - Haunglongbing (Citrus greening)": {
"description": "Citrus greening causes yellowing leaves and bitter, misshapen fruits.",
"cause": "Caused by Candidatus Liberibacter bacteria transmitted by citrus psyllid insects."
},

# ---------------- PEACH ----------------
"Peach - Bacterial spot": {
"description": "Bacterial spot causes small dark lesions on peach leaves and fruits.",
"cause": "Caused by the bacterium Xanthomonas campestris spread through rain splash."
},

"Peach - healthy": {
"description": "Peach tree appears healthy with no disease symptoms.",
"cause": "Healthy orchard management and proper irrigation."
},

# ---------------- PEPPER ----------------
"Pepper, bell - Bacterial spot": {
"description": "Bacterial spot causes dark water-soaked spots on pepper leaves.",
"cause": "Caused by Xanthomonas bacteria and spreads through water droplets."
},

"Pepper, bell - healthy": {
"description": "Pepper plant is healthy without signs of infection.",
"cause": "Good drainage and crop care."
},

# ---------------- POTATO ----------------
"Potato - Early blight": {
"description": "Early blight causes brown spots with concentric rings on potato leaves.",
"cause": "Caused by Alternaria solani fungus in warm humid environments."
},

"Potato - Late blight": {
"description": "Late blight causes large dark lesions and rapid plant decay.",
"cause": "Caused by Phytophthora infestans under cool and moist conditions."
},

"Potato - healthy": {
"description": "Potato crop is healthy without disease symptoms.",
"cause": "Proper crop rotation and disease management."
},

# ---------------- RASPBERRY ----------------
"Raspberry - healthy": {
"description": "Raspberry plant shows no visible disease symptoms.",
"cause": "Healthy growth conditions."
},

# ---------------- SOYBEAN ----------------
"Soybean - healthy": {
"description": "Soybean crop is healthy with normal leaf development.",
"cause": "Balanced fertilization and proper crop management."
},

# ---------------- SQUASH ----------------
"Squash - Powdery mildew": {
"description": "Powdery mildew forms white powder-like spots on squash leaves.",
"cause": "Caused by fungal pathogens in warm dry climates."
},

# ---------------- STRAWBERRY ----------------
"Strawberry - Leaf scorch": {
"description": "Leaf scorch causes reddish-purple spots on strawberry leaves.",
"cause": "Caused by the fungus Diplocarpon earlianum."
},

"Strawberry - healthy": {
"description": "Strawberry plant is healthy with no disease symptoms.",
"cause": "Proper soil drainage and plant care."
},

# ---------------- TOMATO ----------------
"Tomato - Bacterial spot": {
"description": "Bacterial spot causes dark lesions on tomato leaves and fruits.",
"cause": "Caused by Xanthomonas bacteria spread by water and wind."
},

"Tomato - Early blight": {
"description": "Early blight produces concentric brown spots on leaves.",
"cause": "Caused by the fungus Alternaria solani."
},

"Tomato - Late blight": {
"description": "Late blight causes large dark lesions and rapid plant collapse.",
"cause": "Caused by Phytophthora infestans during cool humid weather."
},

"Tomato - Leaf Mold": {
"description": "Leaf mold causes yellow spots on the upper leaf surface.",
"cause": "Caused by the fungus Passalora fulva under high humidity."
},

"Tomato - Septoria leaf spot": {
"description": "Septoria leaf spot causes small circular gray spots on leaves.",
"cause": "Caused by the fungus Septoria lycopersici."
},

"Tomato - Spider mites Two-spotted spider mite": {
"description": "Spider mites cause yellow speckled leaves and webbing.",
"cause": "Caused by tiny mites feeding on leaf tissue."
},

"Tomato - Target Spot": {
"description": "Target spot produces circular lesions with ring patterns.",
"cause": "Caused by Corynespora cassiicola fungus."
},

"Tomato - Tomato Yellow Leaf Curl Virus": {
"description": "This virus causes yellowing and curling of tomato leaves.",
"cause": "Transmitted by whitefly insects."
},

"Tomato - Tomato mosaic virus": {
"description": "Tomato mosaic virus causes mottled light and dark green leaf patterns.",
"cause": "Spread through infected seeds, tools, and plant contact."
},

"Tomato - healthy": {
"description": "Tomato plant is healthy with no disease symptoms.",
"cause": "Balanced nutrients and proper crop management."
}

}

CROP_EMOJI = {
    "Apple": "🍎",
    "Blueberry": "🫐",
    "Cherry (including sour)": "🍒",
    "Corn (maize)": "🌽",
    "Grape": "🍇",
    "Orange": "🍊",
    "Peach": "🍑",
    "Pepper, bell": "🫑",
    "Potato": "🥔",
    "Raspberry": "🍓",
    "Soybean": "🌱",
    "Squash": "🎃",
    "Strawberry": "🍓",
    "Tomato": "🍅"
}

def format_label(label):
    return label.replace("___", " - ").replace("_", " ")


def confidence_meter(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "AI Confidence Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2e7d32"},
            'steps': [
                {'range': [0, 50], 'color': "#ffcdd2"},
                {'range': [50, 75], 'color': "#fff9c4"},
                {'range': [75, 100], 'color': "#c8e6c9"}
            ],
        }
    ))

    fig.update_layout(height=250)
    return fig

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
    st.markdown(f"""
    <div class="navbar">
    <div class="nav-title">{t("navbar_title")}</div>
    <div class="nav-menu">{t("navbar_desc")}</div></div>
    """, unsafe_allow_html=True)

    # ---------------- HERO SECTION ----------------
    st.markdown(f"""
    <div class="hero-box">
    <h1>🌿 {t("app_title")}</h1>
    <p>உழுவார் உலகத்தார்க்கு ஆணிஅஃ தாற்றாது எழுவாரை எல்லாம் பொறுத்து.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Choose a Feature")

    col1, col2, col3 = st.columns(3)

    # ---------- Diagnose ----------
    with col1:
        st.markdown(f"""
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
        st.markdown(f"""
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
        st.markdown(f"""
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
        f"""<h1 style='text-align:center;color:#2e7d32;'>🌿 {t('diagnosis_title')}</h1>""",
        unsafe_allow_html=True
    )

    # ---------------- UPLOAD SECTION ----------------
    st.markdown(f"""
<div class="result-card">

<h3 style="text-align:center;color:#2e7d32;">AI Leaf Analysis</h3>

<p style="font-size:18px;">
{t("upload_text")}
The system will analyze the leaf using a trained deep learning model
and identify possible plant diseases from the dataset.
</p>

<p style="font-size:20px;">
After detection, the system will display:
<br>✔ Disease Name
<br>✔ Recommended Treatment
</p>

</div>
""", unsafe_allow_html=True)
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
            st.markdown(f'<div class="card-title">🌿 {t("disease_result")}</div>', unsafe_allow_html=True)
 
            
            if confidence >= 70:
                st.success(predicted_label)
                # AI Confidence Meter
                st.plotly_chart(
                confidence_meter(confidence),
                use_container_width=True
                )
                

                treatment = TREATMENT_DB.get(
                    predicted_label,
                    "General Advice: Remove infected leaves, ensure sunlight and avoid overwatering."
                )

                st.markdown("### 🌱 " + t("recommended_treatment"))
                st.markdown(
    f"""
    <div style="
        font-size:20px;
        background:#e8f5e9;
        padding:15px;
        border-radius:10px;
        color:#1b5e20;
        font-weight:500;
    ">
    {treatment}
    </div>
    """,
    unsafe_allow_html=True
)
            else:
                st.warning("Low confidence. Please upload a clearer leaf image.")

            st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# SUPPORTED PAGE
# ---------------------------------------------------
elif st.session_state.page == 'supported':
    st.button("⬅ Back to Home", on_click=go_to_home)

    st.markdown(
f"<h1 style='text-align:center;color:#2e7d32;'>🌿 {t('supported_title')}</h1>",
unsafe_allow_html=True
)
    st.markdown("### 📊 Crop Disease Distribution")

    crop_counts = {}
    for label in RAW_LABELS:
        formatted = format_label(label)
        crop = formatted.split(" - ")[0]

        if crop not in crop_counts:
            crop_counts[crop] = 0

        crop_counts[crop] += 1

    # Show bar chart once after loop
    st.bar_chart(crop_counts)

    # ---------------- SEARCH FILTER ----------------
    search = st.text_input("### 🔎 Search crop or disease")

    if search:
        filtered_labels = [
            label for label in RAW_LABELS
            if search.lower() in format_label(label).lower()
        ]
    else:
        filtered_labels = RAW_LABELS

    # Group diseases by crop
    crop_dict = {}
    for label in filtered_labels:
        formatted = format_label(label)
        crop = formatted.split(" - ")[0]
        disease = formatted.split(" - ")[1]

        if crop not in crop_dict:
            crop_dict[crop] = []

        crop_dict[crop].append(disease)

    cols = st.columns(3)

    i = 0
    for crop, diseases in crop_dict.items():
        with cols[i % 3]:
            emoji = CROP_EMOJI.get(crop, "🌿")

            st.markdown(f"""
            <div class="ai-card">
            <div class="card-title">{emoji} {crop}</div>
            """, unsafe_allow_html=True)

            for d in diseases:
                st.markdown(f"• {d}")

            st.markdown("</div>", unsafe_allow_html=True)

        i += 1

# ---------------------------------------------------
# TREATMENT PAGE
# ---------------------------------------------------

elif st.session_state.page == 'treatment':

    st.button("⬅ Back to Home", on_click=go_to_home)

    st.markdown(
        f"<h1 style='text-align:center;color:#2e7d32;'>🌿 {t('treatment_title')}</h1>",
        unsafe_allow_html=True
    )

    st.markdown("### 🔎 Search or select a disease")

    # ---------- SEARCH BAR ----------
    search = st.text_input(t("search_disease"))

    formatted_labels = [format_label(label) for label in RAW_LABELS]

    filtered = [
        label for label in formatted_labels
        if search.lower() in label.lower()
    ]

    disease_list = filtered if search else formatted_labels

    # ---------- DROPDOWN ----------
    selected_disease = st.selectbox(t("choose_disease"), disease_list)

    if selected_disease:
        treatment = TREATMENT_DB.get(
            selected_disease,
            "General Advice: Remove infected leaves, ensure sunlight, avoid overwatering and consult an agriculture officer."
        )

        info = DISEASE_INFO.get(selected_disease)

        if info:
            st.markdown("### 🧬 Disease Description")
            st.markdown(
                f"""
                <div style="
                background:#fff3e0;
                padding:18px;
                border-radius:10px;
                font-size:17px;
                border-left:6px solid #fb8c00;
                ">
                {info['description']}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 🦠 Cause of Disease")
            st.markdown(
                f"""
                <div style="
                background:#ffebee;
                padding:18px;
                border-radius:10px;
                font-size:17px;
                border-left:6px solid #e53935;
                ">
                {info['cause']}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ---------- HEALTH STATUS ----------
            if "healthy" in selected_disease.lower():
                st.success(t("healthy_status"))
            else:
                st.warning(t("disease_detected"))

            st.markdown(f"### {selected_disease}")

            # ---------- TREATMENT CARD ----------
            st.markdown(
                f"""
                <div style="
                    background:#e8f5e9;
                    padding:20px;
                    border-radius:12px;
                    font-size:18px;
                    color:#1b5e20;
                    border-left:6px solid #2e7d32;
                ">
                🌱 {treatment}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ---------- PREVENTION TIPS ----------
            st.markdown(f"### 🛡️ {t('prevention_tips')}")
            st.markdown("""
            • Maintain proper crop spacing  
            • Avoid excessive leaf wetness  
            • Remove infected leaves early  
            • Use disease-resistant crop varieties  
            """)


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
© 2026 உழவர் அறிவியல் AI | Uzhavar Ariviyal - AI Based Crop Disease Detection System
</div>
""", unsafe_allow_html=True)
