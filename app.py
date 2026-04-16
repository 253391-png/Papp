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