import numpy as np
import librosa
import os
import glob
from sklearn.model_selection import train_test_split
from feature_extraction import extract_features, segment_audio

def preprocess_audio_file(file_path, target_sr=16000):
    """
    Preprocess an audio file for feature extraction
    
    Args:
        file_path (str): Path to the audio file
        target_sr (int): Target sample rate
        
    Returns:
        y (np.array): Audio time series
        sr (int): Sample rate
    """
    # Load the audio file with resampling
    y, sr = librosa.load(file_path, sr=target_sr)
    
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # Normalize
    y = librosa.util.normalize(y)
    
    return y, sr

def prepare_telugu_dataset(dataset_path, test_size=0.2, val_size=0.1):
    """
    Prepare a Telugu speech emotion dataset for training
    
    Args:
        dataset_path (str): Path to the dataset
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation
        
    Returns:
        X_train, X_val, X_test: Features for training, validation, and testing
        y_train, y_val, y_test: Labels for training, validation, and testing
    """
    features = []
    labels = []
    
    # This function would typically process real data
    # For this implementation, we'll assume a specific directory structure
    # e.g., dataset_path/emotion_label/audio_files
    
    # List of emotions (folders)
    emotions = ["angry", "happy", "sad", "neutral", "fear", "surprise"]
    
    for i, emotion in enumerate(emotions):
        emotion_dir = os.path.join(dataset_path, emotion)
        
        # Skip if directory doesn't exist
        if not os.path.exists(emotion_dir):
            continue
        
        # Process each audio file
        for audio_file in glob.glob(os.path.join(emotion_dir, "*.wav")):
            try:
                # Preprocess audio
                y, sr = preprocess_audio_file(audio_file)
                
                # Segment audio into fixed-length segments
                segments = segment_audio(y, sr)
                
                # Extract features from each segment
                for segment in segments:
                    feature_vector = extract_features(segment, sr)
                    features.append(feature_vector.flatten())
                    labels.append(i)  # Use index as the label
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    # Convert labels to one-hot encoding manually
    def to_categorical(y, num_classes):
        """Manual implementation of one-hot encoding"""
        y_cat = np.zeros((len(y), num_classes))
        for i in range(len(y)):
            y_cat[i, y[i]] = 1
        return y_cat
        
    y_categorical = to_categorical(y, num_classes=len(emotions))
    
    # Split data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=test_size + val_size, random_state=42
    )
    
    # Further split temporary set into validation and test sets
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1-val_ratio, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def augment_audio(y, sr, augmentations=None):
    """
    Apply data augmentation to audio samples
    
    Args:
        y (np.array): Audio time series
        sr (int): Sample rate
        augmentations (list): List of augmentation types to apply
        
    Returns:
        augmented_audio (list): List of augmented audio samples
    """
    augmented_audio = []
    
    if augmentations is None:
        augmentations = ['pitch', 'speed', 'noise']
    
    # Original audio
    augmented_audio.append((y, 'original'))
    
    # Pitch shift
    if 'pitch' in augmentations:
        for n_steps in [-2, 2]:
            y_pitch_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            augmented_audio.append((y_pitch_shifted, f'pitch_{n_steps}'))
    
    # Speed change
    if 'speed' in augmentations:
        for speed_factor in [0.9, 1.1]:
            y_speed = librosa.effects.time_stretch(y, rate=speed_factor)
            augmented_audio.append((y_speed, f'speed_{speed_factor}'))
    
    # Add noise
    if 'noise' in augmentations:
        noise_factor = 0.005
        noise = np.random.randn(len(y))
        y_noise = y + noise_factor * noise
        augmented_audio.append((y_noise, 'noise'))
    
    return augmented_audio
