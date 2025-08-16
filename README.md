# Parkinson’s Detection ML Clinic App

🎯 **Live Demo**: [Click to Launch App](https://parkinsons-detection-ml-clinic-farzana-lrualqrejpwdikb9ktyuoa.streamlit.app)

This Streamlit app predicts Parkinson’s disease from voice recordings using machine learning. It extracts acoustic features from `.wav` files and classifies them using a trained model. Built for clinical research visibility and recruiter review.

---

## 🧠 Features

- Upload `.wav` voice samples  
- Extract 22+ acoustic features using `librosa`  
- Predict Parkinson’s status using a trained ML model  
- Real-time audio playback of uploaded `.wav` files  
- Model confidence score display (if supported by classifier)  
- Download prediction as CSV  
- Clean, responsive UI built with Streamlit

---

## 🩺 Clinical Relevance

Voice changes are early indicators of Parkinson’s. This app demonstrates how ML can assist in non-invasive screening and remote diagnostics.

---

## 🚀 Technologies Used

- Python, Streamlit  
- Librosa, NumPy, Pandas  
- Scikit-learn, Pickle

---

## 📁 Files

- `app.py`: Streamlit interface and prediction logic  
- `model.pkl`: Trained classifier  
- `requirements.txt`: Dependencies

---

# 🧠 Parkinson's Detection ML Clinic – Farzana

A voice-based machine learning app that predicts Parkinson’s disease using acoustic biomarkers extracted from `.wav` recordings. Built with Streamlit and scikit-learn.

---

## 🧬 Clinical Relevance

Parkinson’s disease often presents with subtle vocal impairments before motor symptoms become pronounced. This tool leverages acoustic biomarkers—MFCCs, spectral features, and RMS energy—to support early detection. It aligns with modern trends in telemedicine and AI-driven diagnostics, and can be adapted for use in clinical trials, outpatient screening, or remote patient monitoring.

---

## 🚀 Features

- Upload `.wav` voice recordings  
- Extract 22 vocal features using `librosa`  
  ***Note: These 22 features are extracted from a single `.wav` voice recording using `librosa`—not 22 separate recordings. This includes MFCCs, spectral features, and RMS energy.***  
- Predict Parkinson’s status using RandomForestClassifier  
- Streamlit-based UI for easy interaction  
- Audio playback and prediction display

---

## ✅ Demo Status

Tested with real voice input recorded by the developer.  
**Prediction: Parkinson’s Negative**  
App successfully extracts features and returns prediction via Streamlit interface.

---

## ⚠️ Limitations & Future Work

- Model trained on limited dataset; may not generalize across populations  
- Future improvements:  
  - Expand dataset with multilingual and age-diverse samples  
  - Integrate deep learning models  
  - Validate predictions with clinical trial data  
  - Add real-time voice recording functionality

---

## 🛠️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py'''

Project Walkthrough
This section provides a step-by-step guide to how the project was built, making it easy for learners and reviewers to follow.

1. Dataset Source
Used the UCI Parkinson’s Dataset

Contains voice recordings and biomedical voice measurements from patients with and without Parkinson’s

2. Feature Extraction
Used librosa to extract 22 acoustic features from .wav files:

MFCCs (Mel-frequency cepstral coefficients)

Spectral centroid, bandwidth, rolloff

RMS energy and zero-crossing rate

3. Model Training
Trained a RandomForestClassifier using scikit-learn

Input: 22 extracted features

Output: Binary prediction (Parkinson’s Positive or Negative)

Saved model as model.pkl using pickle

4. Streamlit App Setup
Built a web interface using Streamlit

Allows users to upload .wav files

Displays prediction result and audio playback

Shows model confidence score (if supported)

5. Testing with Real Voice
Recorded a .wav file using mobile voice recorder

Uploaded to the app and received prediction: Parkinson’s Negative

6. Deployment
Deployed to Streamlit Cloud

Includes app.py, model.pkl, and requirements.txt
