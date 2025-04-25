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
        # Play the audio
        st.audio(uploaded_file, format='audio/wav')

    with col2:
        with st.spinner('Processing audio...'):
            try:
                # Get file extension
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                if not file_extension:
                    file_extension = '.wav'  # Default extension
                
                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(uploaded_file.getbuffer())
                    # Ensure all data is written
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                
                # Process after ensuring file is fully written
                predicted_emotion = predict_emotion(temp_path, model)
                
                # Display result with emoji
                emoji_map = {
                    "happy": "😊", 
                    "sad": "😢", 
                    "angry": "😠", 
                    "neutral": "😐",
                    "fearful": "😨",
                    "disgust": "🤢",
                    "surprised": "😲"
                }
                emoji = emoji_map.get(predicted_emotion, "")
                st.success(f"Predicted Emotion: {predicted_emotion} {emoji}")
                
                # Display waveform and spectrogram
                st.write("### Audio Visualization")
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Load audio data
                try:
                    y, sr = librosa.load(temp_path, sr=None)
                    
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
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)