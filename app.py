import streamlit as st
import librosa
import numpy as np
import joblib

# Load model
model = joblib.load("emotion_model.pkl")
# scaler = joblib.load("scaler.pkl")  # Uncomment if you used scaling

# Reverse label map
inv_map = {0: "Angry 😡", 1: "Happy 😀", 2: "Sad 😢"}

# Streamlit App
st.title("🎙️ Human Emotion Detection from Voice")
st.write("Upload your voice (.wav) to see what emotion it contains.")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

# Feature extractor
def extract_features(file):
    y, sr = librosa.load(file, res_type='kaiser_fast')
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    features = np.hstack([mfcc, chroma, mel])
    return features.reshape(1, -1)

# Predict and show
if uploaded_file is not None:
    st.audio(uploaded_file)
    try:
        features = extract_features(uploaded_file)
        # features = scaler.transform(features)  # If using scaling
        prediction = model.predict(features)[0]
        st.success(f"**Predicted Emotion:** {inv_map[prediction]}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")