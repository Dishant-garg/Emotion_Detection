import os
import numpy as np
import librosa
from scipy.signal import wiener
import warnings

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
    if len(intervals) == 0:  # No non-silent intervals found
        return audio
    return np.concatenate([audio[start:end] for start, end in intervals])

def preprocess_audio(file_path, num_mfcc=40, n_mels=128):
    # Suppress warnings during loading
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Try loading with different parameters for better compatibility
            audio, sr = librosa.load(file_path, mono=True, sr=None)
        except Exception as e1:
            try:
                # Fallback to default sample rate
                audio, sr = librosa.load(file_path, mono=True, sr=22050)
            except Exception as e2:
                raise Exception(f"Failed to load audio file: {str(e1)}. Second attempt: {str(e2)}")
    
    # Normalize audio
    audio = librosa.util.normalize(audio)
    
    # Apply preprocessing
    try:
        audio = reduce_noise(audio)
        audio = remove_silence(audio, sr)
        
        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc).T, axis=0)
        mfccs_deltas = librosa.feature.delta(mfccs)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)
        mel_spec = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels), axis=1)
        
        features = np.concatenate([mfccs, mfccs_deltas, [spectral_centroid], chroma, mel_spec])
        
        return features
    except Exception as e:
        raise Exception(f"Error extracting features: {str(e)}")

def predict_emotion(file_path, model):
    try:
        features = preprocess_audio(file_path)
        
        features = np.expand_dims(features, axis=0)  
        features = np.expand_dims(features, axis=-1)  
        
        # Verbose prediction
        prediction = model.predict(features, verbose=0)  # Suppress prediction output
        predicted_emotion_index = np.argmax(prediction)
        predicted_label = emotion_labels[predicted_emotion_index]
        
        return predicted_label
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")