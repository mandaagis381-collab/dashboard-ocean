import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import plotly.express as px
from scipy.signal import butter, filtfilt
import utide

# --- 1. CONFIG & UI STYLE ---
st.set_page_config(page_title="OceanData Pro Analytics", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f2027, #2c5364);
        color: #e0e0e0;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1c92d2, #185a9d);
    }

    h1, h2, h3 {
        color: #00e5ff;
    }

</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("🌊 OceanData Pro")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

# --- 3. DATA ---
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    cols = df.select_dtypes(include=np.number).columns
    target = st.selectbox("Pilih Variabel", cols)

    df_clean = df[['timestamp', target]].copy()
    df_clean[target] = df_clean[target].interpolate()
    df_clean = df_clean.dropna()
    df_clean.columns = ['time', 'raw']

    # --- STATISTIK DASAR ---
    st.subheader("📊 Statistik Data")
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Mean", round(df_clean['raw'].mean(), 3))
    col2.metric("Max", round(df_clean['raw'].max(), 3))
    col3.metric("Min", round(df_clean['raw'].min(), 3))
    col4.metric("Std Dev", round(df_clean['raw'].std(), 3))
    col5.metric("Count", df_clean['raw'].count())

    # --- FILTER ---
    st.sidebar.subheader("⚙️ Filter")

    pilihan_jam = st.sidebar.selectbox("Window:", [
        "1 Jam", "3 Jam", "12 Jam", "24 Jam", "25 Jam", "Custom"
    ])

    if pilihan_jam == "1 Jam": window = 60
    elif pilihan_jam == "3 Jam": window = 180
    elif pilihan_jam == "12 Jam": window = 720
    elif pilihan_jam == "24 Jam": window = 1440
    elif pilihan_jam == "25 Jam": window = 1500
    else:
        window = st.sidebar.number_input("Custom", 10, 5000, 60)

    filter_type = st.sidebar.selectbox("Tipe Filter", [
        "Raw", "Averaging", "Moving Average", "Lowpass"
    ])

    # --- FILTER LOGIC ---
    df_plot = df_clean.copy()

    if filter_type == "Averaging":
        df_plot['filtered'] = df_plot['raw'].rolling(window=window).mean()

    elif filter_type == "Moving Average":
        df_plot['filtered'] = df_plot['raw'].rolling(window=window, center=True).mean()

    elif filter_type == "Lowpass":
        try:
            b, a = butter(4, 1/window, btype='low')
            df_plot['filtered'] = filtfilt(b, a, df_plot['raw'])
        except:
            st.warning("Window terlalu kecil")

    # --- GRAFIK ---
    st.subheader("📈 Grafik Time Series")

    if filter_type == "Raw":
        chart_data = df_plot
        y_col = 'raw'
    else:
        chart_data = df_plot.dropna()
        chart_data = chart_data.melt('time', ['raw', 'filtered'])

    chart = alt.Chart(chart_data).mark_line().encode(
        x='time:T',
        y=alt.Y('value:Q' if filter_type != "Raw" else 'raw:Q', scale=alt.Scale(zero=False)),
        color='variable:N' if filter_type != "Raw" else alt.value("#00e5ff")
    ).properties(height=450).interactive()

    st.altair_chart(chart, use_container_width=True)

    # --- PASUT ---
    st.subheader("🌊 Analisis Pasut (UTide)")

    try:
        time = df_clean['time'].values

        coef = utide.solve(
            time,
            df_clean['raw'].values,
            lat=-6.0,
            method='ols',
            trend=False
        )

        predict = utide.reconstruct(time, coef)

        df_pasut = df_clean.copy()
        df_pasut['Prediksi'] = predict.h

        chart2 = alt.Chart(df_pasut.melt('time', ['raw', 'Prediksi'])).mark_line().encode(
            x='time:T',
            y=alt.Y('value:Q', scale=alt.Scale(zero=False)),
            color='variable:N'
        ).properties(height=400).interactive()

        st.altair_chart(chart2, use_container_width=True)

    except Exception as e:
        st.error(f"Error pasut: {e}")

else:
    st.info("Upload data dulu ya 👋")
