import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import plotly.graph_objects as go

# Config
st.set_page_config(page_title="Ocean Dashboard", layout="wide")
st.title("Ocean Data Dashboard")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Merapikan Kolom
    df.columns = df.columns.str.lower().str.strip()

    # Mendeteksi Kolom
    time_col = [col for col in df.columns if "time" in col][0]
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col])
    df = df.sort_values(time_col)

    # Kolom Numerik
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    param = st.selectbox("Pilih Parameter", numeric_cols)

    df[param] = pd.to_numeric(df[param], errors='coerce')

    # Memotong Outlier
    q1 = df[param].quantile(0.01)
    q99 = df[param].quantile(0.99)
    df[param] = df[param].clip(q1, q99)

    # Memilih Data
    st.subheader("Pilih Data yang Ditampilkan")

    show_raw = st.checkbox("Raw Data", True)
    show_avg = st.checkbox("Average (per jam)")
    show_moving = st.checkbox("Moving Average")
    show_filter = st.checkbox("Signal Filter")

    # Parameter
    if show_avg:
        jam_avg = st.slider("Average (jam)", 1, 25, 3)
        df_resample = (
            df.set_index(time_col)[[param]]
            .resample(f'{jam_avg}h')
            .mean()
            .reset_index()
        )

    if show_moving:
        window = st.slider("Window Moving Average", 1, 50, 5)
        df['moving_avg'] = df[param].rolling(window=window, min_periods=1).mean()

    if show_filter:
        filter_type = st.selectbox("Tipe Filter", ["lowpass", "highpass", "bandpass"])
        order = st.slider("Filter Order", 1, 5, 2)

        if filter_type in ["lowpass", "highpass"]:
            cutoff = st.slider("Cutoff (0.01 - 0.5)", 0.01, 0.5, 0.05)

        if filter_type == "bandpass":
            low = st.slider("Low Cut", 0.01, 0.5, 0.05)
            high = st.slider("High Cut", 0.05, 0.9, 0.2)

        def apply_filter(data):
            data = data.fillna(method='bfill').fillna(method='ffill')
            if len(data) < 10:
                return data
            try:
                if filter_type == "lowpass":
                    b, a = butter(order, cutoff, btype='low')
                elif filter_type == "highpass":
                    b, a = butter(order, cutoff, btype='high')
                elif filter_type == "bandpass":
                    b, a = butter(order, [low, high], btype='band')

                return filtfilt(b, a, data)
            except:
                return data

        df['filtered'] = apply_filter(df[param])

   
    # Statistik Data
    st.subheader("Statistik Data")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Max", round(df[param].max(), 2))
    col2.metric("Min", round(df[param].min(), 2))
    col3.metric("Mean", round(df[param].mean(), 2))
    col4.metric("Std Dev", round(df[param].std(), 2))
    col5.metric("Count", int(df[param].count()))

    # Plot Grafik
    fig = go.Figure()

    if show_raw:
        fig.add_trace(go.Scatter(
            x=df[time_col], y=df[param],
            name='Raw Data'
        ))

    if show_moving:
        fig.add_trace(go.Scatter(
            x=df[time_col], y=df['moving_avg'],
            name='Moving Avg'
        ))

    if show_filter:
        fig.add_trace(go.Scatter(
            x=df[time_col], y=df['filtered'],
            name=f'{filter_type.capitalize()} Filter'
        ))

    if show_avg:
        fig.add_trace(go.Scatter(
            x=df_resample[time_col], y=df_resample[param],
            name=f'Average {jam_avg} Jam',
            line=dict(dash='dash')
        ))

    fig.update_layout(template="plotly_dark")

    st.plotly_chart(fig, use_container_width=True)
