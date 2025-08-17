import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def extract_features(y, sr, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract audio features from audio data - simplified version
    
    Args:
        y (np.array): Audio time series
        sr (int): Sample rate
        n_mfcc (int): Number of MFCC coefficients to extract
        n_fft (int): Length of the FFT window
        hop_length (int): Number of samples between successive frames
        
    Returns:
        features (np.array): Extracted features ready for model input
    """
    # Make sure the audio is long enough for analysis
    if len(y) < sr * 0.5:  # If shorter than 0.5 seconds
        y = np.pad(y, (0, int(sr * 0.5) - len(y)))
    
    # Extract only the most important features
    
    # 1. Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # 2. Calculate statistics over the MFCCs to get fixed-length features
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    # 3. Extract some basic acoustic features as single values
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)
    zcr_mean = np.mean(zcr)
    
    # Root Mean Square value (energy)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    rms_mean = np.mean(rms)
    
    # Create a simple feature vector (all flat arrays)
    features = np.concatenate([
        mfccs_mean,  # Shape: (n_mfcc,)
        mfccs_std,   # Shape: (n_mfcc,)
        [zcr_mean],  # Shape: (1,)
        [rms_mean]   # Shape: (1,)
    ])
    
    # Reshape for model input (batch size of 1)
    features = features.reshape(1, -1)
    
    return features

def extract_mfccs_for_training(audio_file, n_mfcc=13):
    """
    Extract MFCC features for model training
    
    Args:
        audio_file (str): Path to audio file
        n_mfcc (int): Number of MFCC coefficients to extract
        
    Returns:
        mfccs (np.array): MFCC features
    """
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Normalize MFCC
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    
    return mfccs

def segment_audio(y, sr, segment_length=3.0, overlap=0.5):
    """
    Segment audio into fixed-length segments with overlap
    
    Args:
        y (np.array): Audio time series
        sr (int): Sample rate
        segment_length (float): Length of each segment in seconds
        overlap (float): Overlap between segments in seconds
        
    Returns:
        segments (list): List of audio segments
    """
    # Calculate segment length and hop length in samples
    segment_samples = int(segment_length * sr)
    hop_samples = int((segment_length - overlap) * sr)
    
    # Create segments
    segments = []
    for i in range(0, len(y) - segment_samples + 1, hop_samples):
        segment = y[i:i + segment_samples]
        segments.append(segment)
    
    # If no segments were created (audio shorter than segment_length)
    if len(segments) == 0:
        # Pad the audio if it's shorter than segment_length
        if len(y) < segment_samples:
            padding = segment_samples - len(y)
            segment = np.pad(y, (0, padding), 'constant')
            segments.append(segment)
        else:
            segments.append(y[:segment_samples])
    
    return segments
