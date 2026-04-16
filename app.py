import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import google.generativeai as genai

# CONFIGURACIÓN
st.set_page_config(page_title="App Estadística con IA", layout="wide")

st.title("📊 App de Análisis Estadístico con IA")

# TABS
tabs = st.tabs([
    "📂 Datos",
    "📊 Visualización",
    "📈 Prueba Z",
    "🤖 IA"
])

# =========================
# TAB 0 - DATOS
# =========================
with tabs[0]:
    st.header("Carga de datos")

    metodo = st.radio("Selecciona método", ["CSV", "Datos sintéticos"])

    if metodo == "CSV":
        archivo = st.file_uploader("Sube tu CSV", type=["csv"])
        if archivo:
            df = pd.read_csv(archivo)
            st.write(df.head())
            st.session_state["data"] = df

    elif metodo == "Datos sintéticos":
        n = st.slider("Número de datos", 30, 500, 100)
        media = st.number_input("Media", value=50)
        std = st.number_input("Desviación estándar", value=10)

        datos = np.random.normal(media, std, n)
        df = pd.DataFrame({"valores": datos})

        st.write(df.head())
        st.session_state["data"] = df


# =========================
# TAB 1 - VISUALIZACIÓN
# =========================
with tabs[1]:
    st.header("Visualización de distribuciones")

    if "data" in st.session_state:
        df = st.session_state["data"]
        columna = st.selectbox("Selecciona variable", df.columns)

        data = df[columna]

        col1, col2 = st.columns(2)

        # Histograma + KDE
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(data, kde=True, ax=ax)
            ax.set_title("Histograma + KDE")
            st.pyplot(fig)

        # Boxplot
        with col2:
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=data, ax=ax2)
            ax2.set_title("Boxplot")
            st.pyplot(fig2)

        # Estadísticos
        media = data.mean()
        mediana = data.median()
        std = data.std()

        st.subheader("Resumen estadístico")
        st.write(f"Media: {media:.2f}")
        st.write(f"Mediana: {mediana:.2f}")
        st.write(f"Desviación estándar: {std:.2f}")

        # Interpretación automática
        st.subheader("Interpretación")

        if abs(media - mediana) < std * 0.1:
            st.write("La distribución parece aproximadamente simétrica.")
        else:
            st.write("La distribución presenta sesgo.")

        # Outliers
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        outliers = data[(data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)]

        if len(outliers) > 0:
            st.write(f"Se detectaron {len(outliers)} outliers.")
        else:
            st.write("No se detectaron outliers significativos.")

    else:
        st.warning("Primero carga datos en la pestaña anterior")