# 🏦 E-Signature Fraud Detection System

> AI-powered signature authentication for banking security — built with Python, Streamlit & Machine Learning

---

## 📋 Abstract

This project presents an **E-Signature Fraud Detection System** using machine learning techniques to verify the authenticity of digital signatures. The system detects forged signatures and enhances security in electronic transactions by providing a simple, efficient, and reliable verification solution for banking environments.

---

## 📁 Project Structure

```
signguard/
│
├── app.py                        ← Flask REST API (alternative backend)
├── train.py                      ← Train all 5 models at once (CLI)
├── train_cnn.py                  ← Train CNN model only
├── train_svm.py                  ← Train SVM model only
├── train_random_forest.py        ← Train Random Forest only
├── train_knn.py                  ← Train KNN model only
├── train_logistic_regression.py  ← Train Logistic Regression only
├── preprocess_dataset.py         ← Sort & preprocess raw dataset images
├── streamlit_app.py              ← Main web application (Streamlit UI)
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
│
├── datasets/
│   ├── genuine/                  ← Preprocessed genuine signature images
│   └── forged/                   ← Preprocessed forged signature images
│
├── raw_dataset/
│   └── extract/
│       ├── 001/                  ← Genuine signatures (person 1)
│       ├── 001_forg/             ← Forged signatures (person 1)
│       ├── 002/                  ← Genuine signatures (person 2)
│       ├── 002_forg/             ← Forged signatures (person 2)
│       └── ... (686 persons)
│
├── models/
│   ├── classifiers.py            ← All 5 model definitions
│   ├── svm.pkl                   ← Saved SVM model
│   ├── random_forest.pkl         ← Saved Random Forest model
│   ├── knn.pkl                   ← Saved KNN model
│   ├── logistic_regression.pkl   ← Saved Logistic Regression model
│   └── cnn_model.h5              ← Saved CNN model (TensorFlow)
│
├── utils/
│   └── feature_extractor.py      ← 20-D feature extraction pipeline
│
├── reports/
│   └── training_results.json     ← Model performance metrics
│
├── static/
│   ├── css/style.css             ← Flask UI stylesheet
│   └── js/
│       ├── canvas.js             ← Signature drawing canvas
│       └── app.js                ← Flask UI logic
│
└── templates/
    ├── base.html                 ← Base layout
    ├── dashboard.html            ← Dashboard page
    ├── train.html                ← Training page
    ├── verify.html               ← Verify page
    └── compare.html              ← Compare page
```

---

## 🧠 Machine Learning Models (5)

| # | Model | Type | Accuracy |
|---|-------|------|----------|
| 1 | **CNN** | Deep Learning | **67.43%** 🏆 |
| 2 | **SVM** | Classical ML | 61.76% |
| 3 | **Random Forest** | Ensemble | 62.44% |
| 4 | **KNN** | Instance-based | 59.36% |
| 5 | **Logistic Regression** | Linear | 56.94% |

---

## 🔬 Feature Extraction (20-Dimensional Vector)

Each signature image is processed into a **20-D feature vector**:

| Feature | Description |
|---------|-------------|
| Pixel Density | Ink ratio to total image area |
| Total Ink | Normalised ink pixel count |
| Aspect Ratio | Bounding box width / height |
| BBox H & W | Normalised bounding box dimensions |
| H-Proj Mean & Std | Horizontal ink histogram statistics |
| V-Proj Mean & Std | Vertical ink histogram statistics |
| Transitions | Ink ↔ background stroke crossings |
| Edge Density | Gradient magnitude (pen pressure proxy) |
| CoM X & Y | Centre of mass coordinates |
| Zone Map (7) | 4×2 spatial grid ink density |

---

## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| Total Images | **14,626** |
| Genuine Signatures | 7,313 |
| Forged Signatures | 7,313 |
| Number of Persons | 686 |
| Raw Format | JPG/JPEG |
| Preprocessed Format | PNG (128×64) |

---

## 🖥 Web Application Pages

| Page | Description |
|------|-------------|
| 🔐 **Login** | Secure officer authentication |
| ✍ **Verify Signature** | Draw or upload a signature → instant verdict |
| 🔍 **Compare Signatures** | Reference vs Test → similarity score |
| ⚙ **System & Training** | Dashboard, model stats, retrain models |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Step 1 — Clone / Extract Project
```bash
cd "C:\Users\Vedant\OneDrive\Desktop\project intern\signguard"
```

### Step 2 — Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Install TensorFlow (for CNN)
```bash
pip install tensorflow
```

### Step 5 — Preprocess Dataset
```bash
python preprocess_dataset.py
```

### Step 6 — Train Models
```bash
# Train all 5 models at once
python train.py

# OR train individually
python train_svm.py
python train_random_forest.py
python train_knn.py
python train_logistic_regression.py
python train_cnn.py
```

### Step 7 — Run the App
```bash
streamlit run streamlit_app.py
```

### Step 8 — Open Browser
```
http://localhost:8501
```

**Login credentials:**
- Username: `admin`
- Password: `admin123`

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit, Plotly, streamlit-drawable-canvas |
| Backend | Python, Flask |
| Deep Learning | TensorFlow / Keras |
| Classical ML | scikit-learn |
| Image Processing | Pillow (PIL), NumPy |
| Data | pandas |

---

## 📈 System Pipeline

```
Raw Dataset (14,626 images)
        ↓
Preprocessing (grayscale → crop → resize 128×64)
        ↓
Feature Extraction (20-D vector per image)
        ↓
Model Training (CNN / SVM / RF / KNN / LR)
        ↓
Web Application (Streamlit UI)
        ↓
Real-time Fraud Detection
```

---

## 📄 Login Credentials

| Role | Username | Password |
|------|----------|----------|
| Officer / Admin | `admin` | `admin123` |

---

## 👨‍💻 Internship Details

- **Organization:** Excerpt Technologies Pvt Ltd, Bengaluru
- **Project:** E-Signature Fraud Detection System
- **Domain:** Machine Learning / Computer Vision
- **Year:** 2025–2026

---

## 📝 License

This project is developed for academic internship purposes at Excerpt Technologies Pvt Ltd.
