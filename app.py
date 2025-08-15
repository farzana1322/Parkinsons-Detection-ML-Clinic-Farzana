import streamlit as st
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Parkinson's Detection from Voice")
uploaded_file = st.file_uploader("Upload your voice (.wav)", type=["wav"])

if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=None)
    
    # Feature extraction
    features = []
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    features.append(np.mean(librosa.feature.rms(y)))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=17)
    for mfcc in mfccs:
        features.append(np.mean(mfcc))

    features = np.array(features).reshape(1, -1)

    if st.button("Predict"):
        prediction = model.predict(features)
        result = "Parkinson's Positive" if prediction[0] == 1 else "Parkinson's Negative"
        st.success(f"Prediction: {result}")

        # Download button
        output_df = pd.DataFrame({'Prediction': [result]})
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Prediction as CSV",
            data=csv,
            file_name='parkinsons_prediction.csv',
            mime='text/csv'
        )
