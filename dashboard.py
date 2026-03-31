import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt
import utide

# ================================
# CONFIG
# ================================
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

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.markdown("<h2 style='color:#00d4ff;'>🌊 OceanData</h2>", unsafe_allow_html=True)
    st.caption("Platform Analisis Data Kelautan")

    menu = [
        "🏠 Dashboard",
        "📂 Data Cleaning",
        "📈 Visualisasi",
        "🔍 Analisis Scatter",
        "🌊 Analisis Pasut",
        "🍃 Windrose"
    ]

    st.markdown("<div class='menu-header'>MAIN MENU</div>", unsafe_allow_html=True)
    pilihan = st.radio("Navigasi", menu, label_visibility="collapsed")

    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

# ================================
# DATA
# ================================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, sep=None, engine="python")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    cols_num = df.select_dtypes(include=np.number).columns.tolist()

    target = st.sidebar.selectbox("Pilih Variabel Analisis", cols_num)

    df_clean = df[["timestamp", target]].dropna().copy()
    df_clean[target] = df_clean[target].interpolate(limit_direction="both")
    df_clean.columns = ["time", "raw"]

    # ================================
    # DASHBOARD
    # ================================
    if pilihan == "🏠 Dashboard":

        st.header(f"Dashboard {target}")
        st.dataframe(df_clean.head(100), use_container_width=True)

    # ================================
    # DATA CLEANING
    # ================================
    elif pilihan == "📂 Data Cleaning":

        st.header("Preprocessing Despiking")

        thresh = st.slider("Threshold Std Dev", 1.0, 5.0, 3.0)

        mean = df_clean["raw"].mean()
        std = df_clean["raw"].std()

        df_despike = df_clean[
            ((df_clean["raw"] - mean).abs() / std) <= thresh
        ]

        st.line_chart(df_despike.set_index("time")["raw"])

    # ================================
    # VISUALISASI
    # ================================
    elif pilihan == "📈 Visualisasi":

        st.header("Analisis Deret Waktu")

        # CUT hanya jika water level
        if "water" in target.lower() or "level" in target.lower() or "elevasi" in target.lower():
            df_clean = df_clean[
                (df_clean["raw"] >= 300) &
                (df_clean["raw"] <= 400)
            ]

        pilihan_jam = st.sidebar.selectbox(
            "Pilih Jendela Waktu",
            ["1 Jam", "3 Jam", "12 Jam", "24 Jam", "25 Jam (Eliminasi Pasut)", "Custom"]
        )

        if pilihan_jam == "1 Jam":
            window_size = 60
        elif pilihan_jam == "3 Jam":
            window_size = 180
        elif pilihan_jam == "12 Jam":
            window_size = 720
        elif pilihan_jam == "24 Jam":
            window_size = 1440
        elif pilihan_jam == "25 Jam (Eliminasi Pasut)":
            window_size = 1500
        else:
            window_size = st.sidebar.number_input("Masukkan poin", 5, 5000, 60)

        tab1, tab2, tab3, tab4 = st.tabs(
            ["Data Raw", "Averaging", "Moving Average", "Low Pass"]
        )

        # RAW
        with tab1:
            st.altair_chart(
                alt.Chart(df_clean)
                .mark_line(color="#00d4ff")
                .encode(
                    x="time:T",
                    y=alt.Y("raw:Q", scale=alt.Scale(zero=False))
                )
                .properties(height=450)
                .interactive(),
                use_container_width=True
            )

        # AVERAGE
        with tab2:

            df_avg = df_clean.copy()
            df_avg["filtered"] = df_avg["raw"].rolling(window_size).mean()

            st.altair_chart(
                alt.Chart(df_avg.melt("time"))
                .mark_line()
                .encode(
                    x="time:T",
                    y="value:Q",
                    color="variable:N"
                )
                .properties(height=450)
                .interactive(),
                use_container_width=True
            )

        # MOVING
        with tab3:

            df_ma = df_clean.copy()
            df_ma["filtered"] = df_ma["raw"].rolling(window_size, center=True).mean()

            st.altair_chart(
                alt.Chart(df_ma.melt("time"))
                .mark_line()
                .encode(
                    x="time:T",
                    y="value:Q",
                    color="variable:N"
                )
                .properties(height=450)
                .interactive(),
                use_container_width=True
            )

        # LOWPASS
        with tab4:

            try:
                b, a = butter(4, 1/window_size, btype="low")
                df_lp = df_clean.copy()
                df_lp["filtered"] = filtfilt(b, a, df_lp["raw"])

                st.altair_chart(
                    alt.Chart(df_lp.melt("time"))
                    .mark_line()
                    .encode(
                        x="time:T",
                        y="value:Q",
                        color="variable:N"
                    )
                    .properties(height=450)
                    .interactive(),
                    use_container_width=True
                )

            except:
                st.error("Window terlalu kecil")

    # ================================
    # SCATTER
    # ================================
    elif pilihan == "🔍 Analisis Scatter":

        st.header("Scatter Plot")

        x = st.selectbox("Variabel X", cols_num)
        y = st.selectbox("Variabel Y", cols_num)

        if x != y:

            df_scatter = df[[x, y]].dropna()

            chart = alt.Chart(df_scatter).mark_circle(size=60).encode(
                x=x,
                y=y,
                tooltip=[x, y]
            ).interactive().properties(height=500)

            st.altair_chart(chart, use_container_width=True)

        else:
            st.error("Variabel tidak boleh sama")

    # ================================
    # ANALISIS PASUT
    # ================================
    elif pilihan == "🌊 Analisis Pasut":

        st.header("Analisis Pasang Surut")

        if "water_level" in df.columns:

            df_pasut = df[["timestamp", "water_level"]].dropna()

            df_pasut = df_pasut[
                (df_pasut["water_level"] >= 300) &
                (df_pasut["water_level"] <= 400)
            ]

            time = pd.to_datetime(df_pasut["timestamp"]).to_pydatetime()
            elev = df_pasut["water_level"].values

            st.subheader("Grafik Elevasi")

            st.line_chart(
                df_pasut.set_index("timestamp")["water_level"]
            )

            st.subheader("UTide Harmonik")

            coef = utide.solve(time, elev, lat=-6, method="ols", trend=False)
            pred = utide.reconstruct(time, coef)

            df_utide = pd.DataFrame({
                "time": df_pasut["timestamp"],
                "Observasi": elev,
                "Prediksi": pred.h
            })

            st.line_chart(df_utide.set_index("time"))

            st.subheader("Komponen Harmonik Utama")

            df_coef = pd.DataFrame({
                "Komponen": coef.name,
                "Amplitudo": coef.A,
                "Fase": coef.g
            })

            utama = ["M2", "S2", "K1", "O1"]
            df_utama = df_coef[df_coef["Komponen"].isin(utama)]

            st.table(df_utama)

            st.write("Jumlah komponen utama:", len(df_utama))

            st.info("""
Keterangan:

Amplitudo = Besarnya pengaruh komponen pasut terhadap elevasi air laut  
Fase = Waktu terjadinya puncak gelombang pasut  

Formzahl tidak dapat dihitung karena panjang data tidak mencukupi
""")

        else:
            st.error("Kolom water_level tidak ditemukan")

    # ================================
    # WINDROSE
    # ================================
    elif pilihan == "🍃 Windrose":

        st.header("Windrose Angin")

        if "wind_direction_avg" in df.columns and "wind_speed_avg" in df.columns:

            df_wind = df[
                ["wind_direction_avg", "wind_speed_avg"]
            ].dropna()

            speed_bins = [0,1,2,3,4,5,10]
            df_wind["speed_class"] = pd.cut(
                df_wind["wind_speed_avg"],
                speed_bins
            )

            dir_bins = np.arange(0,361,45)

            df_wind["dir_bin"] = pd.cut(
                df_wind["wind_direction_avg"],
                dir_bins
            )

            windrose = df_wind.groupby(
                ["dir_bin", "speed_class"]
            ).size().reset_index(name="freq")

            theta = windrose["dir_bin"].astype(str)

            fig = go.Figure()

            for sp in windrose["speed_class"].unique():

                subset = windrose[
                    windrose["speed_class"] == sp
                ]

                fig.add_trace(
                    go.Barpolar(
                        r=subset["freq"],
                        theta=subset["dir_bin"].astype(str),
                        name=str(sp)
                    )
                )

            fig.update_layout(
                template="plotly_dark",
                polar=dict(
                    angularaxis=dict(
                        direction="clockwise",
                        rotation=90
                    )
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Kolom wind tidak ditemukan")

else:
    st.info("Upload file dulu")
