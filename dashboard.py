import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go

st.set_page_config(page_title="Ocean Dashboard", layout="wide")

st.title("🌊 Ocean Data Dashboard")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    df['moving_avg'] = df['wl'].rolling(3).mean()

    def lowpass(data):
        b, a = butter(2, 0.1)
        return filtfilt(b, a, data)

    df['lowpass'] = lowpass(df['wl'].fillna(method='bfill'))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['wl'], name='Water Level'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['moving_avg'], name='Moving Avg'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['lowpass'], name='Lowpass'))

    st.plotly_chart(fig, use_container_width=True)
