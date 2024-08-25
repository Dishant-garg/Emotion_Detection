import os
import numpy as np
import librosa
from scipy.signal import wiener

emotion_labels = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fearful",
    5: "disgust",
    6: "surprised"
}

def reduce_noise(audio, frame_len=2048):
    return wiener(audio, mysize=frame_len, noise=None)

def remove_silence(audio, sr, frame_length=2048):
    intervals = librosa.effects.split(audio, top_db=20, frame_length=frame_length)
    return np.concatenate([audio[start:end] for start, end in intervals])

def preprocess_audio(file_path, num_mfcc=40, n_mels=128):
    
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')
    audio = librosa.util.normalize(audio)
    
    
    audio = reduce_noise(audio)
    audio = remove_silence(audio, sr)
    
    
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc).T, axis=0)
    mfccs_deltas = librosa.feature.delta(mfccs)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
    mel_spec = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels), axis=1)
    

    features = np.concatenate([mfccs, mfccs_deltas, [spectral_centroid], chroma, mel_spec])
    
    return features


def predict_emotion(file_path, model):
    features = preprocess_audio(file_path)
    
    
    features = np.expand_dims(features, axis=0)  
    features = np.expand_dims(features, axis=-1)  
    
    prediction = model.predict(features)
    predicted_emotion_index = np.argmax(prediction)
    predicted_label = emotion_labels[predicted_emotion_index]
    
    return predicted_label


