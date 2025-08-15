import streamlit as st
import librosa
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Triggering rebuild to reload model.pkl

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Sidebar UI
st.sidebar.title("🧠 Parkinson's Detection App")
st.sidebar.markdown("""
Upload `.wav` voice samples to check for Parkinson's indicators using machine learning.

**Steps:**
1. Record voice samples (preferably sustained vowel sounds)
2. Save as `.wav` format
3. Upload one or more files below
4. Click **Predict** to view results
5. Download all predictions as CSV

📎 [Download Sample .wav](https://github.com/farzana1322/Parkinsons-Detection-ML-Clinic-Farzana/raw/main/sample_voice.wav)

⚠️ **Disclaimer:** This app is for educational and research purposes only. It is not a diagnostic tool.
""")

# Main UI
st.title("🎙️ Parkinson's Detection from Voice")
uploaded_files = st.file_uploader("Upload voice files (.wav)", type=["wav"], accept_multiple_files=True)

# Initialize session state
if "result" not in st.session_state:
    st.session_state.result = None

results = []

if uploaded_files and st.button("Predict"):
    for file in uploaded_files:
        y, sr = librosa.load(file, sr=None)

        # Feature extraction
        features = []
        features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
        features.append(np.mean(librosa.feature.rms(y=y)[0]))
        features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=17)
        for mfcc in mfccs:
            features.append(np.mean(mfcc))

        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        result = "Parkinson's Positive" if prediction[0] == 1 else "Parkinson's Negative"
        results.append((file.name, result))

    st.session_state.result = results

# Display results and download
if st.session_state.result:
    for filename, result in st.session_state.result:
        st.success(f"{filename}: {result}")

    output_df = pd.DataFrame(st.session_state.result, columns=["Filename", "Prediction"])
    csv = output_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download All Predictions as CSV",
        data=csv,
        file_name='batch_parkinsons_predictions.csv',
        mime='text/csv'
    )

# Evaluation Metrics
st.subheader("📊 Model Evaluation Metrics")
if os.path.exists("X_test_final.csv") and os.path.exists("y_test_final.csv"):
    try:
        X_test = pd.read_csv("X_test_final.csv")
        y_test = pd.read_csv("y_test_final.csv")["label"]

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        st.write(f"**Accuracy:** {accuracy:.2f}")
        st.write(f"**Precision:** {precision:.2f}")
        st.write(f"**Recall:** {recall:.2f}")
        st.write("**Confusion Matrix:**")
        st.write(conf_matrix)
    except Exception as e:
        st.warning("Error loading evaluation metrics. Please check file format.")
else:
    st.warning("Evaluation metrics not available. Please upload X_test_final.csv and y_test_final.csv.")
