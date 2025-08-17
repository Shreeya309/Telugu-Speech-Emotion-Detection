import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import librosa
import librosa.display
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import streamlit as st

def plot_waveform(y, sr):
    """
    Create a plotly visualization of the audio waveform
    
    Args:
        y (np.array): Audio time series
        sr (int): Sample rate
        
    Returns:
        fig: Plotly figure object
    """
    time = np.linspace(0, len(y) / sr, len(y))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=y,
        mode='lines',
        line=dict(color='#1f77b4', width=1),
        name='Waveform'
    ))
    
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        hovermode="x",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def plot_mfcc(y, sr, n_mfcc=13):
    """
    Create a plotly visualization of MFCC features
    
    Args:
        y (np.array): Audio time series
        sr (int): Sample rate
        n_mfcc (int): Number of MFCC coefficients to extract
        
    Returns:
        fig: Plotly figure object
    """
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Create time axis in seconds
    times = librosa.times_like(mfccs, sr=sr)
    
    # Create heatmap
    fig = px.imshow(
        mfccs, 
        x=times,
        y=np.arange(n_mfcc),
        aspect="auto",
        color_continuous_scale='viridis',
        origin='lower'
    )
    
    fig.update_layout(
        title="MFCC Features",
        xaxis_title="Time (s)",
        yaxis_title="MFCC Coefficients",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig

def display_confidence(emotion_probs, emotion_labels):
    """
    Create a bar chart to display emotion confidence scores
    
    Args:
        emotion_probs (np.array): Probabilities for each emotion
        emotion_labels (list): List of emotion labels
        
    Returns:
        fig: Plotly figure object
    """
    # Convert probabilities to percentages
    confidence_values = emotion_probs * 100
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=emotion_labels,
        x=confidence_values,
        orientation='h',
        marker=dict(
            color=confidence_values,
            colorscale='Viridis',
            colorbar=dict(title="Confidence (%)"),
        ),
        text=[f"{val:.2f}%" for val in confidence_values],
        textposition='auto',
        hoverinfo='text+x',
        hoverlabel=dict(font=dict(size=14))
    ))
    
    # Update layout
    fig.update_layout(
        title="Emotion Confidence Scores",
        xaxis_title="Confidence (%)",
        yaxis_title="Emotion",
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(range=[0, 100])
    )
    
    return fig
