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
# VISUALISASI (SUDAH DI CUT)
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

        chart = alt.Chart(df_plot).mark_line(color='#00d4ff').encode(
            x='timestamp:T',
            y=alt.Y(f'{target}:Q', scale=alt.Scale(zero=False))
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

            df_pasut = df_pasut[
                (df_pasut['water_level'] >= 300) &
                (df_pasut['water_level'] <= 400)
            ]

            time = df_pasut['timestamp'].values
            elev = df_pasut['water_level'].values

            st.subheader("Time Series Elevasi")

            chart = alt.Chart(df_pasut).mark_line(color='#00d4ff').encode(
                x='timestamp:T',
                y=alt.Y('water_level:Q', scale=alt.Scale(domain=[300,400]))
            ).properties(height=400).interactive()

            st.altair_chart(chart, use_container_width=True)

            st.subheader("Harmonik UTide")

            coef = utide.solve(
                time,
                elev,
                lat=-6,
                method='ols',
                trend=False
            )

            pred = utide.reconstruct(time, coef)

            df_utide = pd.DataFrame({
                "time": time,
                "Observasi": elev,
                "Prediksi": pred.h
            })

            chart2 = alt.Chart(
                df_utide.melt('time')
            ).mark_line().encode(
                x='time:T',
                y=alt.Y('value:Q', scale=alt.Scale(domain=[300,400])),
                color='variable:N'
            ).properties(height=400).interactive()

            st.altair_chart(chart2, use_container_width=True)

            st.subheader("Konstanta Harmonik")

            df_coef = pd.DataFrame({
                "Komponen": coef.name,
                "Amplitudo": coef.A,
                "Fase": coef.g
            })

            utama = ['M2','S2','K1','O1']
            df_utama = df_coef[df_coef['Komponen'].isin(utama)]

            st.table(df_utama)

            try:
                M2 = df_utama[df_utama['Komponen']=='M2']['Amplitudo'].values[0]
                S2 = df_utama[df_utama['Komponen']=='S2']['Amplitudo'].values[0]
                K1 = df_utama[df_utama['Komponen']=='K1']['Amplitudo'].values[0]
                O1 = df_utama[df_utama['Komponen']=='O1']['Amplitudo'].values[0]

                F = (K1 + O1) / (M2 + S2)

                st.subheader("Formzahl")
                st.write(f"F = {F:.3f}")

                if F < 0.25:
                    tipe = "Semi-diurnal"
                elif F < 1.5:
                    tipe = "Campuran condong semi-diurnal"
                elif F < 3:
                    tipe = "Campuran condong diurnal"
                else:
                    tipe = "Diurnal"

                st.success(f"Tipe: {tipe}")

            except:
                st.warning("Komponen tidak lengkap")

            st.info("Formzahl belum optimal karena panjang data pendek")

        else:
            st.error("Tidak ada water_level")

# =========================
# WINDROSE (YANG SUDAH DIUBAH)
# =========================
    elif pilihan == "🍃 Windrose":

        st.header("Windrose")

        wind_cols = [col for col in df.columns if "wind" in col.lower()]

        if len(wind_cols) > 0:

            st.info("Pilih yang wind ya")

            col_dir = st.selectbox("Pilih arah angin (direction)", wind_cols)
            col_spd = st.selectbox("Pilih kecepatan angin (speed)", wind_cols)

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
                        angularaxis=dict(
                            rotation=90,
                            direction='clockwise'
                        )
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Direction dan speed harus beda")

        else:
            st.error("Tidak ada kolom wind")

else:
    st.info("Upload file dulu ya")
