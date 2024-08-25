import streamlit as st
from preprocessing import predict_emotion
from modelutils import  load_model
import os


model = load_model()
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
if uploaded_file is not None:
    st.write("Processing audio...")
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    predicted_emotion = predict_emotion(os.path.join("temp_audio.wav"), model)
    st.write(f"Predicted Emotion: {predicted_emotion}")
    os.remove("temp_audio.wav")
