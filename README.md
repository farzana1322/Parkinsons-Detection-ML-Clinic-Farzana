# 🧠 Parkinson’s Detection ML Clinic – Farzana

This project presents a voice-based machine learning solution for the early detection of Parkinson’s disease, developed with clinical research applications in mind. By analyzing subtle vocal impairments—often among the earliest signs of Parkinson’s—this tool aims to support timely diagnosis and intervention.

It combines:
- 🎙️ **Librosa** for extracting acoustic features  
- 🧠 **scikit-learn** for predictive modeling  
- 🌐 **Streamlit** for building an intuitive, clinician-friendly interface

Whether used in research settings or as a prototype for real-world screening tools, this app reflects a commitment to **accessible, data-driven healthcare innovation**.

---

## 🔬 Features
- Extracts vocal biomarkers from `.wav` files
- Predicts Parkinson’s likelihood using trained ML models
- Interactive web app for clinicians and researchers
- Clean UI with real-time feedback

---

## 🛠️ Tech Stack

| Tool           | Purpose                     |
|----------------|-----------------------------|
| Python         | Core programming language   |
| Librosa        | Audio feature extraction    |
| scikit-learn   | Machine learning models     |
| Streamlit      | Web app interface           |
| Pandas & NumPy | Data handling & preprocessing |

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/farzana1322/Parkinsons-Detection-ML-Clinic-Farzana.git
cd Parkinsons-Detection-ML-Clinic-Farzana
pip install -r requirements.txt
streamlit run app.py
