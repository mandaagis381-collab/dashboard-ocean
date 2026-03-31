import streamlit as st 
import pandas as pd
import altair as alt
import numpy as np
import plotly.graph_objects as go
import utide

# --- CONFIG ---
st.set_page_config(page_title="OceanData Pro Analytics", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #11151c; border-right: 1px solid #1f2937; }
    h1, h2, h3 { color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 style='color:#00d4ff;'>🌊 OceanData</h2>", unsafe_allow_html=True)

    menu = [
        "🏠 Dashboard",
        "📂 Data Cleaning",   # ✅ TAMBAHAN
        "📈 Visualisasi",
        "🔍 Scatter",
        "🌊 Analisis Pasut",
        "🍃 Windrose"
    ]

    pilihan = st.radio("Menu", menu)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv","xlsx"])

# --- DATA ---
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    cols_num = df.select_dtypes(include=np.number).columns.tolist()

# =========================
# DASHBOARD
# =========================
    if pilihan == "🏠 Dashboard":

        st.header("Dashboard Data")
        st.dataframe(df.head(100), use_container_width=True)

# =========================
# DATA CLEANING (BARU)
# =========================
    elif pilihan == "📂 Data Cleaning":

        st.header("Data Cleaning")

        target = st.selectbox("Pilih Variabel", cols_num)

        df_clean = df[['timestamp', target]].copy()

        # INTERPOLATE
        df_clean[target] = df_clean[target].interpolate(method='linear', limit_direction='both')

        # DESPIKING
        thresh = st.slider("Threshold (Std Dev)", 1.0, 5.0, 3.0)

        mean = df_clean[target].mean()
        std = df_clean[target].std()

        df_cleaned = df_clean[
            ((df_clean[target] - mean).abs() / std) <= thresh
        ]

        st.subheader("Hasil Cleaning")

        chart = alt.Chart(df_cleaned).mark_line(color='#00d4ff').encode(
            x='timestamp:T',
            y=alt.Y(f'{target}:Q', scale=alt.Scale(zero=False))
        ).properties(height=450).interactive()

        st.altair_chart(chart, use_container_width=True)

        st.caption("Metode: Interpolasi + Despiking (Std Dev)")

# =========================
# VISUALISASI
# =========================
    elif pilihan == "📈 Visualisasi":

        target = st.selectbox("Pilih Variabel", cols_num)

        df_plot = df[['timestamp', target]].dropna()

        Q1 = df_plot[target].quantile(0.25)
        Q3 = df_plot[target].quantile(0.75)
        IQR = Q3 - Q1

        df_plot = df_plot[
            (df_plot[target] >= Q1 - 1.5*IQR) &
            (df_plot[target] <= Q3 + 1.5*IQR)
        ]

        st.subheader("Filter Waktu")

        metode = st.selectbox(
            "Metode",
            ["Raw Data", "Average", "Moving Average", "Lowpass"]
        )

        pilihan_jam = st.selectbox(
            "Window (jam)",
            ["1 jam", "3 jam", "12 jam", "25 jam", "Custom"]
        )

        if pilihan_jam == "Custom":
            jam = st.number_input("Masukkan jam", min_value=1, value=6)
        else:
            jam = int(pilihan_jam.split()[0])

        df_filter = df_plot.copy()

        if metode == "Raw Data":
            df_filter['filtered'] = df_filter[target]

        elif metode == "Average":
            df_filter = df_filter.set_index("timestamp")
            df_filter['filtered'] = df_filter[target].resample(f"{jam}H").mean()
            df_filter = df_filter.reset_index()

        elif metode == "Moving Average":
            df_filter['filtered'] = df_filter[target].rolling(window=jam, center=True).mean()

        elif metode == "Lowpass":
            df_filter['filtered'] = df_filter[target].rolling(window=jam, center=True).mean()

        df_filter = df_filter.dropna()

        chart = alt.Chart(df_filter).mark_line(color='#00d4ff').encode(
            x='timestamp:T',
            y=alt.Y('filtered:Q', scale=alt.Scale(zero=False))
        ).properties(height=450).interactive()

        st.altair_chart(chart, use_container_width=True)

# =========================
# SCATTER
# =========================
    elif pilihan == "🔍 Scatter":

        col_x = st.selectbox("Variabel X", cols_num)
        col_y = st.selectbox("Variabel Y", cols_num)

        if col_x != col_y:

            df_scatter = df[[col_x, col_y]].dropna()

            for col in [col_x, col_y]:
                Q1 = df_scatter[col].quantile(0.25)
                Q3 = df_scatter[col].quantile(0.75)
                IQR = Q3 - Q1
                df_scatter = df_scatter[
                    (df_scatter[col] >= Q1 - 1.5*IQR) &
                    (df_scatter[col] <= Q3 + 1.5*IQR)
                ]

            chart = alt.Chart(df_scatter).mark_circle(size=60).encode(
                x=col_x,
                y=col_y
            ).interactive().properties(height=500)

            st.altair_chart(chart, use_container_width=True)

        else:
            st.error("Variabel tidak boleh sama")

# =========================
# ANALISIS PASUT
# =========================
    elif pilihan == "🌊 Analisis Pasut":

        st.header("Analisis Pasang Surut")

        if "water_level" in df.columns:

            df_pasut = df[['timestamp','water_level']].dropna()

            df_pasut = df_pasut.sort_values("timestamp")

            time = df_pasut['timestamp'].values
            elev = df_pasut['water_level'].values

            coef = utide.solve(time, elev, lat=-6, method='ols', trend=False)
            pred = utide.reconstruct(time, coef)

            df_utide = pd.DataFrame({
                "time": time,
                "Observasi": elev,
                "Prediksi": pred.h
            })

            chart = alt.Chart(
                df_utide.melt('time')
            ).mark_line().encode(
                x='time:T',
                y='value:Q',
                color='variable:N'
            ).properties(height=400).interactive()

            st.altair_chart(chart, use_container_width=True)

        else:
            st.error("Tidak ada water_level")

# =========================
# WINDROSE
# =========================
    elif pilihan == "🍃 Windrose":

        st.header("Windrose")

        wind_cols = [col for col in df.columns if "wind" in col.lower()]

        if len(wind_cols) > 0:

            st.info("Pilih yang wind ya")

            col_dir = st.selectbox("Pilih arah angin", wind_cols)
            col_spd = st.selectbox("Pilih kecepatan angin", wind_cols)

            if col_dir != col_spd:

                df_wind = df[[col_dir, col_spd]].dropna()

                dir = df_wind[col_dir]
                speed = df_wind[col_spd]

                bins_dir = np.arange(0,361,30)
                bins_speed = [0,2,4,6,8,10,15]

                df_wind['dir_bin'] = pd.cut(dir, bins_dir)
                df_wind['spd_bin'] = pd.cut(speed, bins_speed)

                table = pd.crosstab(df_wind['dir_bin'], df_wind['spd_bin'])

                theta = bins_dir[:-1]

                fig = go.Figure()

                for col in table.columns:
                    fig.add_trace(go.Barpolar(
                        r=table[col],
                        theta=theta,
                        name=str(col)
                    ))

                fig.update_layout(
                    template="plotly_dark",
                    polar=dict(
                        angularaxis=dict(rotation=90, direction='clockwise')
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Direction dan speed harus beda")

        else:
            st.error("Tidak ada kolom wind")

else:
    st.info("Upload file dulu ya")
