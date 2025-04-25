import streamlit as st
import os
import time
from preprocessing import predict_emotion
from modelutils import load_model
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import soundfile as sf  # Add explicit soundfile dependency

# Set page configuration
st.set_page_config(page_title="Audio Emotion Detector", layout="wide")

# Load model
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Helper function for robust audio loading
def load_audio(file_path):
    """Load audio with better error handling and multiple fallback methods"""
    try:
        # First try soundfile (it's faster)
        import soundfile as sf
        audio, sr = sf.read(file_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert stereo to mono
        return audio, sr
    except Exception as sf_error:
        st.warning(f"SoundFile loading failed, trying librosa: {sf_error}")
        try:
            # Fall back to librosa
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            return audio, sr
        except Exception as e:
            raise Exception(f"Could not load audio: {e}")

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

        # Create a temporary file with a proper extension
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            # Save uploaded file temporarily
            temp_file.write(uploaded_file.getbuffer())
        
        try:
            # Predict emotion
            predicted_emotion = predict_emotion(temp_path, model)
            
            # Display result
            st.success(f"Predicted Emotion: {predicted_emotion}")
            
            # Display waveform and spectrogram
            st.write("### Audio Visualization")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Load audio data
            try:
                # Use our robust loading function
                y, sr = load_audio(temp_path)
                
                # Waveform
                librosa.display.waveshow(y, sr=sr, ax=ax1)
                ax1.set_title('Waveform')
                
                # Spectrogram
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
                ax2.set_title('Spectrogram')
                
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error visualizing audio: {str(e)}")
                
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
        finally:
            # Remove temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)