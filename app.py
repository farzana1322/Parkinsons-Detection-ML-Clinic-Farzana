import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Sidebar UI
st.sidebar.title("🧠 Parkinson's Detection App")
st.sidebar.markdown("""
Upload a `.wav` voice sample to check for Parkinson's indicators using machine learning.

**Steps:**
1. Record a voice sample (preferably sustained vowel sounds)
2. Save as `.wav` format
3. Upload using the main panel
4. Click **Predict** to view result
5. Download prediction as CSV

📎 [Download Sample .wav](https://github.com/farzana1322/Parkinsons-Detection-ML-Clinic-Farzana/raw/main/sample_voice.wav)

⚠️ **Disclaimer:** This app is for educational and research purposes only. It is not a diagnostic tool.
""")

# Main UI
st.title("🎙️ Parkinson's Detection from Voice")
uploaded_file = st.file_uploader("Upload your voice (.wav)", type=["wav"])

# Initialize session state
if "result" not in st.session_state:
    st.session_state.result = None

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=None)

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

    if st.button("Predict"):
        prediction = model.predict(features)
        st.session_state.result = "Parkinson's Positive" if prediction[0] == 1 else "Parkinson's Negative"

if st.session_state.result:
    st.success(f"Prediction: {st.session_state.result}")

    # Download button
    output_df = pd.DataFrame({'Prediction': [st.session_state.result]})
    csv = output_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Prediction as CSV",
        data=csv,
        file_name='parkinsons_prediction.csv',
        mime='text/csv'
    )
