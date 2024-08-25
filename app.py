import streamlit as st
import os
import time
from preprocessing import predict_emotion
from modelutils import load_model
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(page_title="Audio Emotion Detector", layout="wide")

# Load model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Title and description
st.title("🎭 Audio Emotion Detector")
st.write("Upload an audio file to detect the emotion in the speech.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.audio(uploaded_file, format='audio/wav')

    with col2:
        # Display a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(100):
            # Update progress bar
            progress_bar.progress(i + 1)
            status_text.text(f"Processing: {i+1}%")
            time.sleep(0.01)

        # Save uploaded file temporarily
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Predict emotion
        predicted_emotion = predict_emotion(os.path.join("temp_audio.wav"), model)

        # Display result
        st.success(f"Predicted Emotion: {predicted_emotion}")

        

    # Display waveform and spectrogram
    st.write("### Audio Visualization")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    y, sr = librosa.load(os.path.join("temp_audio.wav"))
    
    # Waveform
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title('Waveform')
    
    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
    ax2.set_title('Spectrogram')
    
    plt.tight_layout()
    st.pyplot(fig)
    # Remove temporary file
    os.remove("temp_audio.wav")

