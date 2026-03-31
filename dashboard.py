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
    st.markdown("<br>", unsafe_allow_html=True)
    
    all_options = ["🏠 Dashboard", "📂 Data Cleaning", "📈 Visualisasi", "🔍 Analisis Scatter", "🌊 Analisis Pasut", "🍃 Windrose"]
    
    st.markdown("<div class='menu-header'>MAIN MENU</div>", unsafe_allow_html=True)
    pilihan = st.radio("Navigasi", all_options, label_visibility="collapsed")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload File CSV/Excel", type=["csv", "xlsx"])

# --- 3. DATA ---
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

    # --- DASHBOARD ---
    if pilihan == "🏠 Dashboard":
        st.header(f"🏠 Dashboard: {target}")
        st.dataframe(df_clean.head(100), use_container_width=True)

    # --- CLEANING ---
    elif pilihan == "📂 Data Cleaning":
        st.header("📂 Preprocessing: Despiking")
        thresh = st.slider("Threshold (Std Dev):", 1.0, 5.0, 3.0)
        mean, std = df_clean['raw'].mean(), df_clean['raw'].std()
        df_cleaned = df_clean[((df_clean['raw'] - mean).abs() / std) <= thresh].copy()
        st.line_chart(df_cleaned.set_index('time')['raw'])

    # --- VISUALISASI ---
    elif pilihan == "📈 Visualisasi":
        st.header("📈 Analisis Deret Waktu (Time Series)")
        st.sidebar.markdown("### Setting Filter")
        pilihan_jam = st.sidebar.selectbox("Pilih Jendela Waktu:", ["1 Jam", "3 Jam", "12 Jam", "24 Jam", "25 Jam (Eliminasi Pasut)", "Custom"])
        
        if pilihan_jam == "1 Jam": window_size = 60
        elif pilihan_jam == "3 Jam": window_size = 180
        elif pilihan_jam == "12 Jam": window_size = 720
        elif pilihan_jam == "24 Jam": window_size = 1440
        elif pilihan_jam == "25 Jam (Eliminasi Pasut)": window_size = 1500
        else: window_size = st.sidebar.number_input("Masukkan Jumlah Poin:", 5, 5000, 60)

        t_raw, t_avg, t_ma, t_lp = st.tabs(["📄 Data Raw", "📊 Averaging", "📈 Moving Average", "📉 Low Pass"])
        
        with t_raw:
            st.altair_chart(alt.Chart(df_clean).mark_line(color='#00d4ff').encode(
                x='time:T',
                y=alt.Y('raw:Q', scale=alt.Scale(zero=False))
            ).properties(height=450).interactive(), use_container_width=True)
        
        with t_avg:
            df_avg = df_clean.copy(); df_avg['filtered'] = df_avg['raw'].rolling(window=window_size).mean()
            st.altair_chart(alt.Chart(df_avg.melt('time', ['raw', 'filtered'])).mark_line().encode(
                x='time:T', y='value:Q', color='variable:N'
            ).properties(height=450).interactive(), use_container_width=True)

        with t_ma:
            df_ma = df_clean.copy(); df_ma['filtered'] = df_ma['raw'].rolling(window=window_size, center=True).mean()
            st.altair_chart(alt.Chart(df_ma.melt('time', ['raw', 'filtered'])).mark_line().encode(
                x='time:T', y='value:Q', color='variable:N'
            ).properties(height=450).interactive(), use_container_width=True)

        with t_lp:
            try:
                b, a = butter(4, 1/window_size, btype='low')
                df_lp = df_clean.copy(); df_lp['filtered'] = filtfilt(b, a, df_lp['raw'])
                st.altair_chart(alt.Chart(df_lp.melt('time', ['raw', 'filtered'])).mark_line().encode(
                    x='time:T', y='value:Q', color='variable:N'
                ).properties(height=450).interactive(), use_container_width=True)
            except:
                st.error("Window terlalu kecil.")

    # --- SCATTER 
    elif pilihan == "🔍 Analisis Scatter":
        st.header("🔍 Analisis Scatter Plot")
        
        st.info("💡 Petunjuk: Pilih variabel X dan variabel Y yang berbeda. **Tidak boleh memilih variabel yang sama.**")

        col_x = st.selectbox("Pilih Variabel X", cols_num)
        col_y = st.selectbox("Pilih Variabel Y", cols_num)

        if col_x == col_y:
            st.error("⚠️ Variabel X dan Y tidak boleh sama! Silakan pilih variabel lain.")
        else:
            if col_x and col_y:
                df_scatter = df[[col_x, col_y]].dropna()

                chart = alt.Chart(df_scatter).mark_circle(size=60).encode(
                    x=col_x,
                    y=col_y,
                    tooltip=[col_x, col_y]
                ).interactive().properties(height=500)

                st.altair_chart(chart, use_container_width=True)

    # --- PASUT ---
    elif pilihan == "🌊 Analisis Pasut":
        st.header("🌊 Analisis Pasang Surut")

        if any(x in target.lower() for x in ['level', 'height', 'elevasi']):

            time = df_clean['time'].values

            # NORMALISASI 0–1
            data = df_clean['raw']
            data = (data - data.min()) / (data.max() - data.min())

            with st.spinner('Menghitung Harmonik...'):
                coef = utide.solve(time, data.values, lat=-6.0, method='ols', trend=False)
                predict = utide.reconstruct(time, coef)

            df_pasut = pd.DataFrame({
                'time': time,
                'Observasi': data,
                'Prediksi': predict.h
            })

            st.subheader("Grafik Observasi vs Prediksi")
            chart = alt.Chart(df_pasut.melt('time')).mark_line().encode(
                x='time:T',
                y=alt.Y('value:Q', scale=alt.Scale(zero=False)),
                color=alt.Color('variable:N',
                                scale=alt.Scale(domain=['Observasi','Prediksi'],
                                                range=['#00d4ff','#ff4b4b']))
            ).properties(height=400).interactive()

            st.altair_chart(chart, use_container_width=True)

            st.subheader("Konstanta Harmonik Utama")

            df_coef = pd.DataFrame({
                "Komponen": coef.name,
                "Amplitudo": coef.A,
                "Fase": coef.g
            })

            utama = ['M2', 'S2', 'K1', 'O1']
            df_utama = df_coef[df_coef['Komponen'].isin(utama)].reset_index(drop=True)

            c1, c2 = st.columns(2)
            c1.table(df_utama)

            try:
                amps = dict(zip(df_utama['Komponen'], df_utama['Amplitudo']))
                F = (amps['K1'] + amps['O1']) / (amps['M2'] + amps['S2'])

                c2.metric("Bilangan Formzahl (F)", round(F, 3))

                if F <= 0.25:
                    tipe = "Harian Ganda (Semidiurnal)"
                elif F <= 1.5:
                    tipe = "Campuran Dominan Ganda"
                elif F <= 3.0:
                    tipe = "Campuran Dominan Tunggal"
                else:
                    tipe = "Harian Tunggal (Diurnal)"

                c2.success(f"Tipe Pasut: {tipe}")

            except:
                c2.info("Data kurang panjang untuk hitung Formzahl")

        else:
            st.warning("⚠️ Pilih data Water Level / Elevasi dulu")

    # --- WINDROSE ---

    elif pilihan == "🍃 Windrose":

        if "wind" in target.lower():

            df_rose = df[[target]].copy()

            df_rose['dir_bin'] = (np.round(df_rose[target]/22.5)*22.5)%360

            counts = df_rose.groupby('dir_bin').size().reset_index(name='count')



            fig = px.bar_polar(counts, r="count", theta="dir_bin", template="plotly_dark")



            fig.update_layout(

                polar=dict(

                    angularaxis=dict(

                        tickvals=[0,45,90,135,180,225,270,315],

                        ticktext=['N','NE','E','SE','S','SW','W','NW'],

                        tickfont=dict(size=14, color='black', family='Arial Black'),

                        rotation=90,

                        direction='clockwise'

                    )

                )

            )



            st.plotly_chart(fig, use_container_width=True)

        else:

            st.error("Pilih variabel wind")


else:
    st.info("👋 Upload file dulu ya")
