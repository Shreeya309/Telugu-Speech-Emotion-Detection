# Telugu-Speech-Emotion-Detection
## Overview

Telugu Speech Emotion Detection is a machine learning-based web application that analyzes Telugu speech recordings and predicts the speaker's emotion. The application provides an interactive interface for uploading audio files, extracting speech features, visualizing audio characteristics, and displaying emotion prediction results.

Built using Python and Streamlit, the project combines audio signal processing with machine learning to demonstrate the complete workflow of speech emotion recognition.

## Features

* Upload Telugu speech recordings in WAV, MP3, or OGG format.
* Extract audio features such as MFCC, Zero Crossing Rate, and RMS Energy.
* Predict emotions using a Random Forest machine learning model.
* Visualize audio waveforms and MFCC feature maps.
* Display prediction confidence through interactive charts.
* User-friendly Streamlit interface for real-time emotion analysis.

## Technologies Used

* Python
* Streamlit
* Scikit-learn
* Librosa
* NumPy
* Pandas
* Plotly
* Matplotlib

## How It Works

1. Upload a Telugu speech audio file.
2. The application preprocesses the audio and extracts relevant speech features.
3. A trained Random Forest classifier analyzes the extracted features.
4. The predicted emotion and confidence scores are displayed along with interactive visualizations of the audio signal and extracted features.

## Learning Outcomes

This project helped strengthen my understanding of:

* Audio signal processing
* Feature extraction using Librosa
* Machine learning with Scikit-learn
* Data visualization with Plotly and Matplotlib
* Building interactive web applications using Streamlit

## Future Improvements

* Train the model on a larger Telugu speech emotion dataset.
* Implement deep learning models such as LSTM or CNN for improved accuracy.
* Support multilingual speech emotion recognition.
* Deploy the application on a cloud platform for public access.
* Add model evaluation metrics and performance comparison.
