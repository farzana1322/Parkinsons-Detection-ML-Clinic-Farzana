# 🧠 Parkinson's Detection ML Clinic – Farzana

A voice-based machine learning app that predicts Parkinson’s disease using acoustic biomarkers extracted from `.wav` recordings. Built with Streamlit and scikit-learn.

---

## 🧬 Clinical Relevance

Parkinson’s disease often presents with subtle vocal impairments before motor symptoms become pronounced. This tool leverages acoustic biomarkers—MFCCs, spectral features, and RMS energy—to support early detection. It aligns with modern trends in telemedicine and AI-driven diagnostics, and can be adapted for use in clinical trials, outpatient screening, or remote patient monitoring.

---

## 🚀 Features

- Upload `.wav` voice recordings
- Extract 22 vocal features using `librosa`
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
streamlit run app.py










