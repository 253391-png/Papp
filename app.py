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
        columna = st.selectbox("Selecciona variable", df.columns, key="vis_col")

        data = df[columna]

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            sns.histplot(data, kde=True, ax=ax)
            ax.set_title("Histograma + KDE")
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=data, ax=ax2)
            ax2.set_title("Boxplot")
            st.pyplot(fig2)

        media = data.mean()
        mediana = data.median()
        std = data.std()

        st.subheader("Resumen estadístico")
        st.write(f"Media: {media:.2f}")
        st.write(f"Mediana: {mediana:.2f}")
        st.write(f"Desviación estándar: {std:.2f}")

        st.subheader("Interpretación")

        if abs(media - mediana) < std * 0.1:
            st.write("La distribución parece aproximadamente simétrica.")
        else:
            st.write("La distribución presenta sesgo.")

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


# =========================
# TAB 2 - PRUEBA Z
# =========================
with tabs[2]:
    st.header("Prueba de hipótesis (Z)")

    if "data" in st.session_state:
        df = st.session_state["data"]
        columna = st.selectbox("Selecciona variable", df.columns, key="z_col")

        data = df[columna]

        st.subheader("Parámetros de la prueba")

        mu = st.number_input("Media hipotética (H0)", value=50.0)
        sigma = st.number_input("Desviación poblacional (σ)", value=10.0)
        alpha = st.slider("Nivel de significancia (α)", 0.01, 0.1, 0.05)

        tipo = st.selectbox("Tipo de prueba", ["bilateral", "izquierda", "derecha"], key="tipo_test")

        n = len(data)
        x_bar = np.mean(data)

        z = (x_bar - mu) / (sigma / np.sqrt(n))

        st.subheader("Resultados")

        st.write(f"Media muestral: {x_bar:.4f}")
        st.write(f"Tamaño de muestra: {n}")
        st.write(f"Estadístico Z: {z:.4f}")

        if tipo == "bilateral":
            p = 2 * (1 - norm.cdf(abs(z)))
        elif tipo == "derecha":
            p = 1 - norm.cdf(z)
        else:
            p = norm.cdf(z)

        st.write(f"p-value: {p:.6f}")

        if p < alpha:
            decision = "rechazar"
            st.error("Se rechaza la hipótesis nula (H0)")
        else:
            decision = "no rechazar"
            st.success("No se rechaza la hipótesis nula (H0)")

        st.subheader("Interpretación")

        if decision == "rechazar":
            st.write(
                f"Con un nivel de significancia de {alpha}, existe evidencia estadísticamente significativa "
                f"para rechazar la hipótesis nula."
            )
        else:
            st.write(
                f"No existe evidencia suficiente para rechazar la hipótesis nula con un nivel de significancia de {alpha}."
            )

        st.session_state["z_result"] = {
            "media_muestral": x_bar,
            "media_hipotetica": mu,
            "n": n,
            "sigma": sigma,
            "alpha": alpha,
            "tipo": tipo,
            "z": z,
            "p_value": p,
            "decision": decision
        }

    else:
        st.warning("Primero carga datos en la pestaña de Datos")