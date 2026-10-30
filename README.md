# Telugu Speech Emotion Analysis

## Overview

Telugu Speech Emotion Analysis is an interactive audio-analysis application developed using Python and Streamlit.

The application allows users to upload Telugu speech recordings, extract acoustic features, visualize audio characteristics, and explore illustrative emotion-analysis outputs.

It uses Librosa for audio processing and provides interactive visualizations to help understand speech-related acoustic features.

## Key Features

### 1. Audio Upload
- Supports WAV, MP3, and OGG audio formats.
- Allows users to upload speech recordings for analysis.

### 2. Audio Feature Extraction
Extracts acoustic features including:
- Mel-Frequency Cepstral Coefficients (MFCC)
- Zero Crossing Rate (ZCR)
- Root Mean Square (RMS) Energy

### 3. Audio Visualization
- Displays audio waveform visualizations.
- Provides MFCC feature visualizations.
- Helps explore acoustic patterns in speech recordings.

### 4. Emotion Analysis Interface
- Displays illustrative emotion-score outputs.
- Provides interactive charts through Plotly.
- Presents audio analysis results through a Streamlit interface.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Application Framework | Streamlit |
| Audio Processing | Librosa |
| Numerical Processing | NumPy |
| Visualization | Plotly, Matplotlib |
| Machine Learning Library | Scikit-learn |

## Project Structure

```text
TeluguEmotionRecognizer/
├── app.py
├── feature_extraction.py
├── model.py
├── preprocess.py
└── utils.py
```

## How It Works

1. The user uploads a speech recording.
2. The application loads and processes the audio using Librosa.
3. Acoustic features such as MFCC, ZCR, and RMS Energy are extracted.
4. Audio waveform and feature visualizations are generated.
5. The application displays illustrative emotion-score outputs.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Shreeya309/Telugu-Speech-Emotion-Detection.git
```

### 2. Navigate to the Project Directory

```bash
cd Telugu-Speech-Emotion-Detection/TeluguEmotionRecognizer
```

### 3. Install Dependencies

```bash
pip install streamlit librosa numpy plotly matplotlib scikit-learn
```

### 4. Run the Application

```bash
streamlit run app.py
```

## Current Limitations

The current implementation is an audio feature extraction and visualization prototype.

The existing model initialization uses randomly generated data rather than a trained model using labelled Telugu speech recordings. The displayed emotion scores are illustrative and have not been validated using a real test dataset.

The performance metrics shown in the interface are fixed example values and should not be interpreted as actual model evaluation results.

## Future Improvements

- Integrate a properly labelled Telugu speech-emotion dataset.
- Implement a reproducible Random Forest training pipeline.
- Train and evaluate the model using real audio samples.
- Add confusion matrix and actual classification metrics.
- Explore deep learning approaches such as CNN and LSTM.
- Improve audio preprocessing and model generalization.
