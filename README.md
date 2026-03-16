# 🌿 Uzhavar Ariviyal AI

### AI-Powered Crop Disease Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red)
![License](https://img.shields.io/badge/License-MIT-green)

Uzhavar Ariviyal AI is a deep learning based web application that detects plant diseases from leaf images and provides recommended treatments.
The system helps farmers, agriculture students, and researchers quickly identify crop diseases using Artificial Intelligence.

The application uses a trained convolutional neural network model to analyze leaf images and predict plant diseases across multiple crop types.

---

# 🚀 Features

🌱 AI-powered crop disease detection
📷 Upload leaf images for instant diagnosis
🧠 Deep learning model trained using TensorFlow
💊 Treatment suggestions for detected diseases
🌿 Supports multiple crops and plant diseases
💻 Simple and interactive web interface
⚡ Fast predictions using optimized image preprocessing
---
## 🌐 Language

This application supports multilingual user interface.

Supported languages:
- 🇬🇧 English
- 🇮🇳 Tamil
  
Language can be switched from the dropdown available in the application UI.

---

# 🧠 Supported Crops

The AI model can detect diseases in crops such as:

* Apple
* Blueberry
* Cherry
* Corn (Maize)
* Grape
* Orange
* Peach
* Pepper
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato

Each prediction includes recommended treatment suggestions.

---

# 🏗 Project Architecture

```
Uzhavar-Ariviyal-AI
│
├── model
│   └── Leaf_Disease_128x128.h5
|   |___train_model.py
|
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Workflow

```
Leaf Image Upload
        │
        ▼
Image Preprocessing (OpenCV)
        │
        ▼
Deep Learning Model (TensorFlow)
        │
        ▼
Disease Prediction
        │
        ▼
Treatment Recommendation
```

---

# 🛠 Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* OpenCV
* NumPy

---

# ⚙ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/uzhavar-ariviyal-ai.git
cd uzhavar-ariviyal-ai
```

### 2. Create Virtual Environment

```bash
python -m venv uzhavar-ariviyal-ai
```

Activate environment

Windows

```bash
uzhavar-ariviyal-ai\Scripts\activate
```

Linux / Mac

```bash
source uzhavar-ariviyal-ai/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the Application

```bash
streamlit run main.py
```

The application will open in your browser.

---

# 📊 Model Information

Model Type: Convolutional Neural Network (CNN)
Framework: TensorFlow / Keras
Input Size: 128 × 128 RGB Image
Dataset: Plant leaf disease dataset with multiple crop classes

---

# 🎯 Project Objective

The goal of this project is to demonstrate how Artificial Intelligence can assist agriculture by enabling early disease detection and providing helpful treatment recommendations for farmers.

---

# 🌍 Future Improvements

* Mobile friendly interface
* Real-time camera detection
* More crop disease classes
* Multilingual support
* Grad-CAM visualization for AI explainability

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

Developed by **Nithin**
AI Based Crop Disease Detection Project

---

⭐ If you find this project useful, consider giving the repository a star.
