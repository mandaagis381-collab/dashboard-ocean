import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import plotly.express as px
from scipy.signal import butter, filtfilt
import utide

# --- CONFIG ---
st.set_page_config(page_title="OceanData Pro Analytics", layout="wide")

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #11151c; }
    h1, h2, h3 { color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🌊 OceanData")
    pilihan = st.radio("Menu", ["Dashboard","Data Cleaning","Visualisasi","Analisis Pasut","Windrose"])
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# --- DATA ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    cols = df.select_dtypes(include=np.number).columns
    target = st.sidebar.selectbox("Pilih Data", cols)

    df_clean = df[['timestamp', target]].copy()
    df_clean[target] = df_clean[target].interpolate()
    df_clean.columns = ['time','raw']

# ===================== PASUT =====================
    if pilihan == "Analisis Pasut":
        st.header("🌊 Analisis Pasang Surut")

        data = df_clean['raw'].copy()

        # --- FIX WATER LEVEL 300-400 ---
        if data.mean() > 100:
            data = data / 100   # cm → meter

        data = data - data.mean()  # hilangkan offset

        time = df_clean['time'].values

        coef = utide.solve(time, data.values, lat=-6, method='ols')
        pred = utide.reconstruct(time, coef)

        df_pasut = pd.DataFrame({
            'time': time,
            'observasi': data,
            'prediksi': pred.h
        })

        # --- RESIDU ---
        df_pasut['residu'] = df_pasut['observasi'] - df_pasut['prediksi']

        chart = alt.Chart(df_pasut).transform_fold(
            ['observasi','prediksi','residu']
        ).mark_line().encode(
            x='time:T',
            y=alt.Y('value:Q', scale=alt.Scale(zero=False)),
            color=alt.Color('key:N',
                scale=alt.Scale(
                    domain=['observasi','prediksi','residu'],
                    range=['#00d4ff','#ffaa00','#ff4da6']  # pink residu
                )
            )
        ).properties(height=400).interactive()

        st.altair_chart(chart, use_container_width=True)

# ===================== WINDROSE =====================
    elif pilihan == "Windrose":
        st.header(f"🍃 Windrose ({target})")

        df_rose = df.copy()
        df_rose['dir_bin'] = (np.round(df_rose[target]/22.5)*22.5)%360
        counts = df_rose.groupby('dir_bin').size().reset_index(name='count')

        fig = px.bar_polar(counts, r='count', theta='dir_bin')

        # --- TAMBAH ARAH ANGIN ---
        fig.update_layout(
            polar=dict(
                angularaxis=dict(
                    tickmode='array',
                    tickvals=[0,45,90,135,180,225,270,315],
                    ticktext=['N','NE','E','SE','S','SW','W','NW'],
                    rotation=90,
                    direction='clockwise'
                )
            )
        )

        st.plotly_chart(fig, use_container_width=True)

# ===================== VISUAL =====================
    elif pilihan == "Visualisasi":
        st.line_chart(df_clean.set_index('time')['raw'])

# ===================== CLEANING =====================
    elif pilihan == "Data Cleaning":
        st.line_chart(df_clean.set_index('time')['raw'])

# ===================== DASHBOARD =====================
    else:
        st.dataframe(df_clean)

else:
    st.info("Upload data dulu")
