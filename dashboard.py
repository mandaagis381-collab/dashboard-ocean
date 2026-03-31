import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import plotly.express as px
from scipy.signal import butter, filtfilt
import utide

# --- 1. CONFIG & UI STYLE ---
st.set_page_config(page_title="OceanData Pro Analytics", layout="wide", page_icon="🌊")

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

# --- 3. DATA PROCESSING ---
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            df = pd.read_excel(uploaded_file)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
        target = st.sidebar.selectbox("Pilih Variabel Utama Analisis:", cols_num)

        # Dataset dasar
        df_clean = df.copy()
        for col in cols_num:
            df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        df_clean = df_clean.dropna()

        # --- LOGIKA NAVIGASI ---
        
        if pilihan == "🏠 Dashboard":
            st.header(f"🏠 Dashboard: {target}")
            st.dataframe(df_clean.head(100), use_container_width=True)
            st.metric("Total Baris Data", len(df_clean))

        elif pilihan == "📂 Data Cleaning":
            st.header("📂 Preprocessing: Despiking")
            thresh = st.slider("Threshold (Std Dev):", 1.0, 5.0, 3.0)
            mean, std = df_clean[target].mean(), df_clean[target].std()
            df_cleaned = df_clean[((df_clean[target] - mean).abs() / std) <= thresh].copy()
            st.line_chart(df_cleaned.set_index('timestamp')[target] if 'timestamp' in df_cleaned.columns else df_cleaned[target])

        elif pilihan == "📈 Visualisasi":
            st.header("📈 Analisis Deret Waktu")
            window_size = st.sidebar.number_input("Window Size (Poin):", 5, 5000, 60)
            st.line_chart(df_clean.set_index('timestamp')[target] if 'timestamp' in df_clean.columns else df_clean[target])

        elif pilihan == "🔍 Analisis Scatter":
            st.header("🔍 Analisis Scatter Plot")
            # PESAN PERINGATAN REVISI
            st.info("💡 Petunjuk: Pilih dua variabel yang berbeda untuk melihat hubungan korelasi. **Tidak boleh memilih variabel X dan Y yang sama.**")

            c1, c2 = st.columns(2)
            with c1:
                col_x = st.selectbox("Pilih Variabel X", cols_num, key="x_scat")
            with c2:
                col_y = st.selectbox("Pilih Variabel Y", cols_num, key="y_scat")

            if col_x == col_y:
                st.error("⚠️ Kesalahan: Variabel X dan Y tidak boleh sama. Silakan pilih variabel yang berbeda.")
            else:
                chart = alt.Chart(df_clean).mark_circle(size=60, color='#00d4ff').encode(
                    x=alt.X(col_x, scale=alt.Scale(zero=False)),
                    y=alt.Y(col_y, scale=alt.Scale(zero=False)),
                    tooltip=[col_x, col_y]
                ).interactive().properties(height=500)
                st.altair_chart(chart, use_container_width=True)

        elif pilihan == "🌊 Analisis Pasut":
            st.header("🌊 Modul: Analisis Pasang Surut")
            if any(x in target.lower() for x in ['level', 'height', 'elevasi']):
                time = pd.to_numeric(df_clean['timestamp']) / 1e9 / 3600 if 'timestamp' in df_clean.columns else np.arange(len(df_clean))
                coef = utide.solve(time.values, df_clean[target].values, lat=-6.0, method='ols', conf_int='none')
                st.success("Analisis Harmonik Selesai")
                st.write("Konstanta ditemukan:", coef.name)
            else:
                st.warning("Pilih variabel water level.")

        elif pilihan == "🍃 Windrose":
            st.header("🍃 Windrose")
            if "wind" in target.lower() or "dir" in target.lower():
                fig = px.bar_polar(df_clean, r=target, theta=target, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Pilih variabel arah angin.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👋 Silakan upload file CSV/Excel di sidebar untuk memulai.")
