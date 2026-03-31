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
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #11151c; border-right: 1px solid #1f2937; }
    .menu-header { 
        color: #58a6ff; font-weight: 800; margin-top: 35px; margin-bottom: 15px; 
        font-size: 13px; letter-spacing: 2px; text-transform: uppercase; 
        padding-left: 10px; border-left: 3px solid #00d4ff; 
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; border-radius: 4px 4px 0 0; background-color: #161b22; color: #888; }
    .stTabs [aria-selected="true"] { background-color: #007bff !important; color: white !important; }
    h1, h2, h3 { color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#00d4ff; margin-bottom:0;'>🌊 OceanData</h2>", unsafe_allow_html=True)
    st.caption("Platform Analisis Data Kelautan")
    
    all_options = ["🏠 Dashboard", "📊 Statistika Data", "📂 Data Cleaning", "📈 Visualisasi", "🔍 Analisis Scatter", "🌊 Analisis Pasut", "🍃 Windrose"]
    
    st.markdown("<div class='menu-header'>MAIN MENU</div>", unsafe_allow_html=True)
    pilihan = st.radio("Navigasi", all_options, label_visibility="collapsed")
    
    uploaded_file = st.file_uploader("Upload File CSV/Excel", type=["csv", "xlsx"])

# --- 3. LOGIKA DATA ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=None, engine='python')

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    target = st.sidebar.selectbox("Pilih Variabel Analisis:", cols_num)

    df_clean = df[['timestamp', target]].copy()
    df_clean[target] = df_clean[target].interpolate(method='linear', limit_direction='both')
    df_clean = df_clean.dropna()
    df_clean.columns = ['time', 'raw']

    # =====================
    # DASHBOARD
    # =====================
    if pilihan == "🏠 Dashboard":
        st.header(f"🏠 Dashboard: {target}")
        st.dataframe(df_clean.head(100), use_container_width=True)

    # =====================
    # STATISTIKA
    # =====================
    elif pilihan == "📊 Statistika Data":
        st.header(f"📊 Statistika Data: {target}")
        data = df_clean['raw']

        col1, col2, col3 = st.columns(3)
        col1.metric("Minimum", round(data.min(), 3))
        col2.metric("Maximum", round(data.max(), 3))
        col3.metric("Mean", round(data.mean(), 3))

        col4, col5 = st.columns(2)
        col4.metric("Standard Deviasi", round(data.std(), 3))
        col5.metric("Jumlah Data", int(data.count()))

    # =====================
    # CLEANING
    # =====================
    elif pilihan == "📂 Data Cleaning":
        st.header("📂 Preprocessing: Despiking")
        thresh = st.slider("Threshold (Std Dev):", 1.0, 5.0, 3.0)
        mean, std = df_clean['raw'].mean(), df_clean['raw'].std()
        df_cleaned = df_clean[((df_clean['raw'] - mean).abs() / std) <= thresh].copy()
        st.line_chart(df_cleaned.set_index('time')['raw'])

    # =====================
    # VISUALISASI
    # =====================
    elif pilihan == "📈 Visualisasi":
        st.header("📈 Analisis Deret Waktu")
        st.altair_chart(
            alt.Chart(df_clean).mark_line(color='#00d4ff').encode(
                x='time:T',
                y=alt.Y('raw:Q', scale=alt.Scale(zero=False))
            ).properties(height=450).interactive(),
            use_container_width=True
        )

    # =====================
    # SCATTER
    # =====================
    elif pilihan == "🔍 Analisis Scatter":
        st.header("🔍 Analisis Scatter")

        cols = df.select_dtypes(include=[np.number]).columns.tolist()
        x_var = st.selectbox("X", cols)
        y_var = st.selectbox("Y", cols, index=1 if len(cols)>1 else 0)

        df_scatter = df[[x_var, y_var]].dropna()

        fig = px.scatter(df_scatter, x=x_var, y=y_var, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Korelasi", round(df_scatter[x_var].corr(df_scatter[y_var]),3))

    # PASUT (FINAL)
    
       elif pilihan == "🌊 Analisis Pasut":
    st.header("🌊 Analisis Pasang Surut")

    if any(x in target.lower() for x in ['level', 'height', 'elevasi']):

        time = df_clean['time'].values

        with st.spinner('Menghitung Harmonik...'):
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

        # ===== GRAFIK =====
        st.subheader("Grafik Observasi vs Prediksi")

        chart = alt.Chart(df_pasut.melt('time', ['raw', 'Prediksi'])).mark_line().encode(
            x='time:T',
            y=alt.Y('value:Q', title='Elevasi (cm)', scale=alt.Scale(zero=False)),
            color=alt.Color('variable:N',
                            title='Keterangan',
                            scale=alt.Scale(range=['#00d4ff', '#ff4b4b']))
        ).properties(height=400).interactive()

        st.altair_chart(chart, use_container_width=True)

        # ===== TABEL HARMONIK =====
        st.subheader("Komponen Harmonik Utama")

        df_harmonik = pd.DataFrame({
            "Komponen": coef.name,
            "Amplitudo (m)": np.round(coef.A, 3),
            "Fase (°)": np.round(coef.g, 2)
        })

        komponen_penting = ["M2", "S2", "K1", "O1", "K2", "N2"]
        df_harmonik = df_harmonik[df_harmonik["Komponen"].isin(komponen_penting)]

        st.dataframe(df_harmonik, use_container_width=True)

        # ===== FORMZAHL =====
        st.subheader("Analisis Tipe Pasut (Formzahl)")
        durasi_hari = (df_clean['time'].max() - df_clean['time'].min()).days

        if durasi_hari >= 29:
            try:
                A = dict(zip(coef.name, coef.A))
                F = (A.get('K1',0) + A.get('O1',0)) / (A.get('M2',1e-6) + A.get('S2',1e-6))

                st.metric("Nilai Formzahl (F)", round(F, 3))

                if F < 0.25:
                    tipe = "Semi-diurnal"
                elif F < 1.5:
                    tipe = "Mixed condong semi-diurnal"
                elif F < 3:
                    tipe = "Mixed condong diurnal"
                else:
                    tipe = "Diurnal"

                st.success(f"Tipe Pasut: {tipe}")

            # --- TABEL ---
            st.subheader("Komponen Harmonik Utama")
            df_harmonik = pd.DataFrame({
                "Komponen": coef.name,
                "Amplitudo (m)": np.round(coef.A, 3),
                "Fase (°)": np.round(coef.g, 2)
            })

            komponen_penting = ["M2", "S2", "K1", "O1", "K2", "N2"]
            df_harmonik = df_harmonik[df_harmonik["Komponen"].isin(komponen_penting)]

            st.dataframe(df_harmonik, use_container_width=True)

            # --- FORMZAHL ---
            st.subheader("Analisis Tipe Pasut (Formzahl)")
            durasi_hari = (df_clean['time'].max() - df_clean['time'].min()).days

            if durasi_hari >= 29:
                try:
                    A = dict(zip(coef.name, coef.A))
                    F = (A.get('K1',0) + A.get('O1',0)) / (A.get('M2',1e-6) + A.get('S2',1e-6))

                    st.metric("Nilai Formzahl (F)", round(F, 3))

                    if F < 0.25:
                        tipe = "Semi-diurnal"
                    elif F < 1.5:
                        tipe = "Mixed condong semi-diurnal"
                    elif F < 3:
                        tipe = "Mixed condong diurnal"
                    else:
                        tipe = "Diurnal"

                    st.success(f"Tipe Pasut: {tipe}")

                except:
                    st.warning("Perhitungan gagal")

            else:
                st.info("⚠️ Formzahl tidak dapat dihitung karena panjang data terlalu pendek.")

    # =====================
    # WINDROSE
    # =====================
    elif pilihan == "🍃 Windrose":
        st.header(f"🍃 Windrose ({target})")

        if "wind" in target.lower():
            df_rose = df[[target]].dropna()
            df_rose['dir_bin'] = (np.round(df_rose[target] / 22.5) * 22.5) % 360
            counts = df_rose.groupby(['dir_bin']).size().reset_index(name='count')

            fig = px.bar_polar(counts, r="count", theta="dir_bin", template="plotly_dark")

            fig.update_layout(
                polar=dict(
                    angularaxis=dict(
                        rotation=90,
                        direction='clockwise',
                        tickvals=[0,90,180,270],
                        ticktext=['N','E','S','W']
                    )
                )
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Pilih variabel wind")

else:
    st.info("👋 Silahkan upload data dulu")
