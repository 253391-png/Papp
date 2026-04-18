# =============================================================================
# STATLAB — App de Análisis Estadístico con IA
# =============================================================================
# Módulos implementados:
#   1. Carga de datos (CSV o datos sintéticos)
#   2. Visualización de distribuciones (Histograma, KDE, Boxplot, Q-Q, Violín)
#   3. Prueba de hipótesis Z (bilateral, cola izquierda, cola derecha)
#   4. Asistente IA con Google Gemini
# =============================================================================

# ── Librerías estándar y de terceros ─────────────────────────────────────────
import os
import streamlit as st          # Framework para la interfaz web interactiva
import pandas as pd             # Manejo y lectura de datos tabulares
import numpy as np              # Operaciones numéricas y generación de datos
import matplotlib.pyplot as plt # Graficación de distribuciones y región crítica
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns           # Graficación estadística de alto nivel
from scipy.stats import norm, shapiro, skew, kurtosis
# norm     → distribución normal (CDF, PDF, PPF) para calcular p-value y Z crítico
# shapiro  → prueba de Shapiro-Wilk para verificar normalidad
# skew     → coeficiente de asimetría (sesgo) de la distribución
# kurtosis → medida del apuntalamiento / colas de la distribución
import google.generativeai as genai  # SDK de Google Gemini para el módulo de IA
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIGURACIÓN DE LA PÁGINA ──────────────────────────────────────────────
st.set_page_config(
    page_title="StatLab · Análisis Estadístico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── TEMA GLOBAL DE MATPLOTLIB ───────────────────────────────────────────────
# Se aplica a todas las gráficas para mantener el estilo oscuro consistente
plt.rcParams.update({
    "figure.facecolor":  "#0f1623",
    "axes.facecolor":    "#0f1623",
    "axes.edgecolor":    "#1e2d45",
    "axes.labelcolor":   "#94a3b8",
    "axes.titlecolor":   "#e2e8f0",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "grid.color":        "#1e2d45",
    "grid.linewidth":    0.6,
    "legend.facecolor":  "#0f1623",
    "legend.edgecolor":  "#1e2d45",
    "legend.labelcolor": "#94a3b8",
    "legend.fontsize":   9,
    "text.color":        "#e2e8f0",
    "font.family":       "monospace",
})

# Paleta de colores principal usada en gráficas
ACCENT   = "#00e5ff"   # Cian — líneas principales, KDE, curva normal
ACCENT2  = "#7c3aed"   # Violeta — histograma, boxplot
ACCENT3  = "#f59e0b"   # Ámbar — estadístico Z calculado, mediana
DANGER   = "#ef4444"   # Rojo — zona de rechazo de H0
SUCCESS  = "#10b981"   # Verde — zona de no rechazo / mensajes positivos

# ─── CSS PERSONALIZADO ───────────────────────────────────────────────────────
# Estilos visuales inyectados como HTML para personalizar la UI de Streamlit
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:      #0a0e17;
    --bg2:     #0f1623;
    --bg3:     #151d2e;
    --accent:  #00e5ff;
    --accent2: #7c3aed;
    --accent3: #f59e0b;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --border:  rgba(0,229,255,0.13);
    --card:    rgba(15,22,35,0.95);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: radial-gradient(ellipse at 8% 5%,  rgba(124,58,237,0.10) 0%, transparent 50%),
                radial-gradient(ellipse at 92% 95%, rgba(0,229,255,0.07) 0%, transparent 50%),
                var(--bg) !important;
}

section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stSelectbox > div,
section[data-testid="stSidebar"] .stSlider > div { margin-top: 4px; }

.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    padding: 10px 20px !important;
    text-transform: uppercase !important;
    transition: color 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: rgba(0,229,255,0.04) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

input, textarea, .stTextInput input, .stNumberInput input {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}
input:focus, .stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.15) !important;
}
.stSelectbox > div > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}
.stSlider [data-baseweb="slider"] { margin-top: 4px; }
.stSlider [data-baseweb="thumb"]  { background: var(--accent) !important; }
.stSlider [data-baseweb="track-fill"] { background: var(--accent) !important; }

.stButton > button {
    background: linear-gradient(135deg, rgba(0,229,255,0.12), rgba(124,58,237,0.12)) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.06em !important;
    border-radius: 6px !important;
    padding: 8px 20px !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(0,229,255,0.2) !important;
    box-shadow: 0 0 16px rgba(0,229,255,0.2) !important;
}

.stSuccess {
    background: rgba(16,185,129,0.08) !important;
    border: 1px solid rgba(16,185,129,0.3) !important;
    border-radius: 8px !important;
}
.stError, .stWarning {
    background: rgba(239,68,68,0.08) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 8px !important;
}
.stInfo {
    background: rgba(0,229,255,0.06) !important;
    border: 1px solid rgba(0,229,255,0.2) !important;
    border-radius: 8px !important;
}

.stFileUploader > div {
    background: var(--bg3) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}

.stRadio label { font-size: 13px !important; }
.stRadio [data-testid="stMarkdownContainer"] { color: var(--muted) !important; }

.stat-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: rgba(0,229,255,0.35); }
.stat-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 6px;
}
.stat-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
}
.stat-card .unit {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    margin-top: 3px;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.02em;
    margin-bottom: 4px;
    border-left: 3px solid var(--accent);
    padding-left: 12px;
}
.section-sub {
    font-size: 12px;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
    margin-bottom: 1.2rem;
    padding-left: 15px;
}

.decision-reject {
    display: inline-block;
    background: rgba(239,68,68,0.12);
    border: 1px solid rgba(239,68,68,0.4);
    color: #fca5a5;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.06em;
    padding: 6px 16px;
    border-radius: 20px;
    text-transform: uppercase;
}
.decision-accept {
    display: inline-block;
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.4);
    color: #6ee7b7;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.06em;
    padding: 6px 16px;
    border-radius: 20px;
    text-transform: uppercase;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    background: linear-gradient(135deg, #00e5ff 0%, #7c3aed 60%, #f59e0b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 4px;
}
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }

.ai-box {
    background: rgba(124,58,237,0.06);
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    font-size: 13px;
    line-height: 1.7;
    color: var(--text);
    white-space: pre-wrap;
}

.hyp-box {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 18px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: var(--text);
    margin-bottom: 8px;
}
.hyp-box span { color: var(--accent); font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─── FUNCIONES AUXILIARES ────────────────────────────────────────────────────

def stat_card(label: str, value: str, unit: str = ""):
    """Genera el HTML de una tarjeta de métrica estilizada."""
    return f"""
    <div class="stat-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="unit">{unit}</div>
    </div>"""

def section_header(title: str, sub: str = ""):
    """Renderiza un encabezado de sección con línea lateral de acento."""
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)

def fig_to_st(fig):
    """Muestra una figura de Matplotlib en Streamlit y libera memoria."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
        <div class="hero-title" style="font-size:24px">StatLab</div>
        <div class="hero-sub">Estadística · IA · Python</div>
    </div>
    <hr class="divider">
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;margin-bottom:8px">Navegación</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:12px;color:var(--muted);line-height:2">
    01 · <span style="color:#e2e8f0">Carga de datos</span><br>
    02 · <span style="color:#e2e8f0">Visualización</span><br>
    03 · <span style="color:#e2e8f0">Prueba Z</span><br>
    04 · <span style="color:#e2e8f0">Asistente IA</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;margin-bottom:8px">Estado del sistema</div>', unsafe_allow_html=True)

    # Indicadores de estado — se verifican en session_state de Streamlit
    has_data = "data" in st.session_state       # True si ya se cargaron datos
    has_test = "z_result" in st.session_state   # True si ya se ejecutó la prueba Z

    st.markdown(f"""
    <div style="font-family:Space Mono,monospace;font-size:11px;line-height:2.2">
    <span style="color:{'#10b981' if has_data else '#ef4444'}">{'●' if has_data else '○'}</span>
    <span style="color:var(--muted);margin-left:6px">Datos cargados</span><br>
    <span style="color:{'#10b981' if has_test else '#ef4444'}">{'●' if has_test else '○'}</span>
    <span style="color:var(--muted);margin-left:6px">Prueba ejecutada</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;color:var(--muted);font-family:Space Mono,monospace">v2.0 · Prueba Z · Gemini API</div>', unsafe_allow_html=True)


# ─── TÍTULO PRINCIPAL ────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:2rem">
    <div class="hero-title">Análisis Estadístico<br>con Inteligencia Artificial</div>
    <div class="hero-sub" style="margin-top:10px">Distribuciones · Prueba Z · Asistente Gemini</div>
</div>
""", unsafe_allow_html=True)


# ─── PESTAÑAS PRINCIPALES ────────────────────────────────────────────────────
tabs = st.tabs(["01 · Datos", "02 · Visualización", "03 · Prueba Z", "04 · IA Gemini"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section_header("Carga de datos", "Fuente · CSV o generación sintética")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        metodo = st.radio(
            "Método de entrada",
            ["📂 Cargar CSV", "⚙️ Datos sintéticos"],
            label_visibility="collapsed",
        )

    with col_right:
        # Muestra resumen del dataset si ya hay datos cargados en memoria
        if "data" in st.session_state:
            n_rows = len(st.session_state["data"])
            n_cols = len(st.session_state["data"].columns)
            st.markdown(
                f'<div style="display:flex;gap:12px">'
                + stat_card("Filas", str(n_rows), "registros")
                + stat_card("Columnas", str(n_cols), "variables")
                + "</div>",
                unsafe_allow_html=True,
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Opción A: Carga de archivo CSV ────────────────────────────────────
    if metodo == "📂 Cargar CSV":
        archivo = st.file_uploader(
            "Arrastra tu archivo CSV aquí",
            type=["csv"],
            help="Formatos soportados: UTF-8, separador coma",
        )
        if archivo:
            # Leer CSV con pandas y guardar en session_state para uso entre pestañas
            df = pd.read_csv(archivo)
            st.session_state["data"] = df
            st.success(f"✓ Archivo cargado · {len(df):,} filas · {len(df.columns)} columnas")
            st.dataframe(df.head(10), use_container_width=True, hide_index=False)

    # ── Opción B: Generación de datos sintéticos ──────────────────────────
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            n = st.slider("Muestras (n)", 30, 1000, 150, step=10)
        with c2:
            media_s = st.number_input("Media (μ)", value=50.0, step=1.0)
        with c3:
            std_s = st.number_input("Desv. estándar (σ)", value=10.0, min_value=0.1, step=0.5)

        # Generar n datos de una distribución Normal con parámetros definidos por el usuario
        # np.random.seed(42) asegura reproducibilidad (misma semilla = mismos datos)
        np.random.seed(42)
        datos = np.random.normal(media_s, std_s, n)
        df = pd.DataFrame({"valores": datos})

        if st.button("Generar datos"):
            st.session_state["data"] = df
            st.success(f"✓ Datos generados · n={n} · μ={media_s} · σ={std_s}")

        st.dataframe(df.head(8), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VISUALIZACIÓN DE DISTRIBUCIONES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section_header("Visualización de distribuciones", "Histograma · KDE · Boxplot · Q-Q Plot")

    if "data" not in st.session_state:
        st.info("⟵ Carga datos en la pestaña anterior para continuar.")
    else:
        df = st.session_state["data"]
        columna = st.selectbox("Variable a analizar", df.columns, key="vis_col")
        data = df[columna].dropna()   # Eliminar valores nulos antes de calcular

        # ── ESTADÍSTICOS DESCRIPTIVOS ─────────────────────────────────────
        # Describen la forma, centro y dispersión de la distribución

        mu   = float(data.mean())     # Media aritmética — centro de los datos
        med  = float(data.median())   # Mediana — valor central, robusto a outliers
        sd   = float(data.std())      # Desviación estándar — dispersión promedio

        # Asimetría (skewness): mide si la distribución tiene cola más larga a un lado
        #   sk ≈ 0  → distribución simétrica
        #   sk > 0  → cola a la derecha (la mayoría de datos están a la izquierda)
        #   sk < 0  → cola a la izquierda (la mayoría de datos están a la derecha)
        sk = float(skew(data))

        # Kurtosis: mide qué tan "pesadas" son las colas respecto a una normal
        #   kurt ≈ 0  → mesocúrtica (similar a la normal)
        #   kurt > 0  → leptocúrtica (pico alto, colas pesadas)
        #   kurt < 0  → platocúrtica (más plana, colas ligeras)
        kurt = float(kurtosis(data))

        # Cuartiles para el Rango Intercuartílico (IQR)
        q1  = float(data.quantile(0.25))   # Q1: 25% de datos están por debajo
        q3  = float(data.quantile(0.75))   # Q3: 75% de datos están por debajo
        iqr = q3 - q1                       # IQR = Q3 - Q1

        # Regla de Tukey para detectar outliers:
        # Un dato es outlier si cae fuera de [Q1 - 1.5*IQR , Q3 + 1.5*IQR]
        n_out = int(((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).sum())

        # Prueba de Shapiro-Wilk: contrasta si los datos provienen de una normal
        #   H0 (Shapiro): los datos son normales
        #   Si p > 0.05 → no se rechaza H0 → se asume normalidad
        #   Limitada a 5000 obs. por eficiencia computacional
        stat_sw, p_sw = shapiro(data[:5000])

        # Tarjetas de métricas principales
        cards_html = (
            '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1.5rem">'
            + stat_card("Media", f"{mu:.2f}")
            + stat_card("Mediana", f"{med:.2f}")
            + stat_card("Desv. std", f"{sd:.2f}")
            + stat_card("Asimetría", f"{sk:.3f}")
            + "</div>"
        )
        st.markdown(cards_html, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="medium")

        # ── GRÁFICA 1: Histograma + KDE ───────────────────────────────────
        # Histograma: muestra la frecuencia empírica de los datos por intervalos
        # KDE (Kernel Density Estimation): versión suavizada y continua del histograma
        # Normal teórica: curva N(μ, σ²) para comparar visualmente con los datos
        with col1:
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            # density=True normaliza el histograma para que el área total = 1
            # (comparable con la función de densidad de probabilidad)
            ax.hist(data, bins="auto", color=ACCENT2, alpha=0.55, density=True, label="Histograma")

            # KDE con Gaussian Kernel: estimación no paramétrica de la distribución
            xg = np.linspace(data.min(), data.max(), 400)
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            ax.plot(xg, kde(xg), color=ACCENT, lw=2, label="KDE")

            # Curva normal teórica con la misma μ y σ que los datos muestrales
            ax.plot(xg, norm.pdf(xg, mu, sd), color=ACCENT3, lw=1.4, ls="--", alpha=0.8, label="Normal teórica")

            # Líneas verticales en media y mediana
            # Si están muy separadas → distribución sesgada
            ax.axvline(mu,  color=ACCENT,  lw=1, ls=":", alpha=0.7)  # Línea de la media
            ax.axvline(med, color=ACCENT3, lw=1, ls=":", alpha=0.7)  # Línea de la mediana

            ax.set_title("Histograma + KDE")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig_to_st(fig)

        # ── GRÁFICA 2: Boxplot ────────────────────────────────────────────
        # Resume la distribución en 5 estadísticos: mín, Q1, mediana, Q3, máx
        # Los puntos fuera de los bigotes son outliers (regla de Tukey)
        # El ancho de la caja representa el IQR (donde está el 50% central de datos)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
            bp = ax2.boxplot(
                data,
                vert=False,         # Orientación horizontal
                patch_artist=True,  # Permite colorear la caja
                notch=False,
                medianprops=dict(color=ACCENT, lw=2),                       # Línea mediana
                boxprops=dict(facecolor=ACCENT2+"44", edgecolor=ACCENT2),   # Caja
                whiskerprops=dict(color="#94a3b8"),                         # Bigotes
                capprops=dict(color="#94a3b8"),                             # Extremos
                flierprops=dict(marker="o", color=ACCENT3, markersize=4, alpha=0.7),  # Outliers
            )
            ax2.set_title("Boxplot")
            ax2.set_yticks([])
            ax2.grid(True, alpha=0.3, axis="x")
            fig2.tight_layout()
            fig_to_st(fig2)

        col3, col4 = st.columns(2, gap="medium")

        # ── GRÁFICA 3: Q-Q Plot ───────────────────────────────────────────
        # Compara cuantiles empíricos de los datos vs cuantiles teóricos normales
        # Interpretación:
        #   Puntos sobre la línea → datos normales
        #   Curva hacia arriba/abajo en los extremos → colas más pesadas/ligeras
        #   Desviación sistemática → no normalidad
        with col3:
            fig3, ax3 = plt.subplots(figsize=(5.5, 3.6))
            from scipy.stats import probplot
            # probplot retorna cuantiles teóricos (osm) y muestrales (osr)
            (osm, osr), (slope, intercept, r) = probplot(data, dist="norm")
            ax3.scatter(osm, osr, color=ACCENT2, s=12, alpha=0.6, label="Datos")
            # Línea de referencia: si los datos son perfectamente normales, los puntos caen aquí
            ax3.plot(osm, slope*np.array(osm)+intercept, color=ACCENT, lw=1.5, label=f"Ref (R²={r**2:.3f})")
            ax3.set_title("Q-Q Plot (Normal)")
            ax3.set_xlabel("Cuantiles teóricos")
            ax3.set_ylabel("Cuantiles muestrales")
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
            fig3.tight_layout()
            fig_to_st(fig3)

        # ── GRÁFICA 4: Violin Plot ────────────────────────────────────────
        # Combina boxplot con KDE: el ancho del violín en cada punto refleja la densidad
        # Permite ver la distribución completa, no solo el resumen del boxplot
        with col4:
            fig4, ax4 = plt.subplots(figsize=(5.5, 3.6))
            vp = ax4.violinplot(data, vert=False, showmeans=True, showmedians=True)
            for body in vp["bodies"]:
                body.set_facecolor(ACCENT2)
                body.set_alpha(0.45)
            vp["cmeans"].set_color(ACCENT)    # Línea de la media
            vp["cmedians"].set_color(ACCENT3) # Línea de la mediana
            ax4.set_title("Violin Plot")
            ax4.set_yticks([])
            ax4.grid(True, alpha=0.3, axis="x")
            fig4.tight_layout()
            fig_to_st(fig4)

        # ── INTERPRETACIÓN AUTOMÁTICA ─────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        section_header("Interpretación automática", "Análisis basado en estadísticos descriptivos")

        c_i1, c_i2 = st.columns(2, gap="large")

        with c_i1:
            # Normalidad: basada en Shapiro-Wilk
            # p > 0.05 → no hay evidencia para rechazar que los datos son normales
            normal_flag = p_sw > 0.05
            st.markdown(f"""
            <div style="background:{'rgba(16,185,129,0.07)' if normal_flag else 'rgba(239,68,68,0.07)'};
                        border:1px solid {'rgba(16,185,129,0.3)' if normal_flag else 'rgba(239,68,68,0.3)'};
                        border-radius:8px;padding:14px 18px;margin-bottom:12px">
                <div style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:.1em;
                            color:var(--muted);text-transform:uppercase;margin-bottom:6px">
                    ¿Distribución normal?
                </div>
                <div style="font-size:13px;color:{'#6ee7b7' if normal_flag else '#fca5a5'}">
                    {'✓ Indica normalidad' if normal_flag else '✗ No parece normal'}
                </div>
                <div style="font-size:11px;color:var(--muted);margin-top:4px">
                    Shapiro-Wilk: W={stat_sw:.4f} · p={p_sw:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Clasificación del sesgo por rangos convencionales
            if abs(sk) < 0.5:
                sesgo_txt, sesgo_col = "Simétrica (sesgo ≈ 0)", "#6ee7b7"
            elif sk > 0:
                sesgo_txt, sesgo_col = f"Sesgo positivo (derecha) · sk={sk:.2f}", "#fcd34d"
            else:
                sesgo_txt, sesgo_col = f"Sesgo negativo (izquierda) · sk={sk:.2f}", "#fcd34d"

            st.markdown(f"""
            <div style="background:rgba(15,22,35,0.8);border:1px solid var(--border);
                        border-radius:8px;padding:14px 18px;margin-bottom:12px">
                <div style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:.1em;
                            color:var(--muted);text-transform:uppercase;margin-bottom:6px">Sesgo</div>
                <div style="font-size:13px;color:{sesgo_col}">{sesgo_txt}</div>
                <div style="font-size:11px;color:var(--muted);margin-top:4px">Kurtosis: {kurt:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with c_i2:
            # Outliers detectados con regla de Tukey (IQR)
            out_col = "#fca5a5" if n_out > 0 else "#6ee7b7"
            st.markdown(f"""
            <div style="background:{'rgba(239,68,68,0.07)' if n_out>0 else 'rgba(16,185,129,0.07)'};
                        border:1px solid {'rgba(239,68,68,0.3)' if n_out>0 else 'rgba(16,185,129,0.3)'};
                        border-radius:8px;padding:14px 18px;margin-bottom:12px">
                <div style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:.1em;
                            color:var(--muted);text-transform:uppercase;margin-bottom:6px">Outliers (IQR)</div>
                <div style="font-size:13px;color:{out_col}">
                    {'✗ ' + str(n_out) + ' valores atípicos detectados' if n_out > 0 else '✓ Sin outliers significativos'}
                </div>
                <div style="font-size:11px;color:var(--muted);margin-top:4px">
                    Q1={q1:.2f} · Q3={q3:.2f} · IQR={iqr:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background:rgba(15,22,35,0.8);border:1px solid var(--border);
                        border-radius:8px;padding:14px 18px">
                <div style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:.1em;
                            color:var(--muted);text-transform:uppercase;margin-bottom:6px">Rango</div>
                <div style="font-size:13px;color:var(--text)">
                    [{data.min():.2f},  {data.max():.2f}]
                </div>
                <div style="font-size:11px;color:var(--muted);margin-top:4px">
                    Amplitud: {data.max()-data.min():.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PRUEBA DE HIPÓTESIS Z
# ══════════════════════════════════════════════════════════════════════════════
# La prueba Z de una muestra contrasta si la media poblacional μ es igual a
# un valor hipotético μ₀, cuando se conoce la varianza poblacional σ² y n ≥ 30.
#
# Supuestos requeridos:
#   1. Varianza poblacional σ² CONOCIDA
#   2. n ≥ 30 (Teorema del Límite Central garantiza normalidad de x̄)
#   3. Observaciones independientes entre sí
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section_header("Prueba de hipótesis Z", "Varianza conocida · n ≥ 30")

    if "data" not in st.session_state:
        st.info("⟵ Carga datos primero.")
    else:
        df    = st.session_state["data"]
        col_z = st.selectbox("Variable", df.columns, key="z_col")
        data  = df[col_z].dropna()
        n     = len(data)

        # Verificar supuesto de tamaño mínimo de muestra
        if n < 30:
            st.warning(f"⚠ n={n} < 30. La prueba Z requiere muestras grandes.")

        # ── DEFINICIÓN DE HIPÓTESIS ───────────────────────────────────────
        # H₀ (Hipótesis nula): lo que se asume verdadero por defecto
        #   → Afirma que la media poblacional es igual al valor de referencia μ₀
        #   → Se mantiene hasta tener evidencia estadística suficiente en contra
        #
        # H₁ (Hipótesis alternativa): lo que el investigador quiere demostrar
        #   → Depende del tipo de prueba seleccionado:
        #
        #   Bilateral    → H₁: μ ≠ μ₀   (la media es distinta, sin importar dirección)
        #   Cola derecha → H₁: μ > μ₀   (la media es mayor que μ₀)
        #   Cola izquierda → H₁: μ < μ₀ (la media es menor que μ₀)
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        section_header("Parámetros", "Define las hipótesis y condiciones")

        p1, p2, p3, p4 = st.columns(4, gap="medium")
        with p1:
            # μ₀: valor de la media bajo H₀ (el valor que queremos contrastar)
            mu0 = st.number_input("H₀: μ =", value=50.0, step=1.0, key="mu0")
        with p2:
            # σ: desviación estándar POBLACIONAL
            # Debe conocerse de antemano (dato histórico, norma técnica, etc.)
            # Si no se conoce, se usaría prueba t en su lugar
            sigma = st.number_input("σ poblacional", value=10.0, min_value=0.01, step=0.5, key="sigma")
        with p3:
            # α (nivel de significancia): probabilidad máxima aceptable de
            # cometer Error Tipo I (rechazar H₀ cuando en realidad es verdadera)
            # Valores comunes: 0.01 (1%), 0.05 (5%), 0.10 (10%)
            alpha = st.selectbox("Nivel α", [0.01, 0.05, 0.10], index=1, key="alpha")
        with p4:
            tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola derecha", "Cola izquierda"], key="tipo")

        # ── VISUALIZACIÓN DE HIPÓTESIS ────────────────────────────────────
        # Mapea el tipo de prueba al operador de la hipótesis alternativa H₁
        tipo_map = {"Bilateral": "≠", "Cola derecha": ">", "Cola izquierda": "<"}
        op = tipo_map[tipo]
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin:1rem 0">
            <div class="hyp-box">H₀: μ <span>=</span> {mu0}</div>
            <div class="hyp-box">H₁: μ <span>{op}</span> {mu0}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── CÁLCULO DEL ESTADÍSTICO DE PRUEBA Z ──────────────────────────
        #
        # Fórmula:        x̄ - μ₀
        #           Z = ──────────
        #                 σ / √n
        #
        #   x̄  = media muestral (calculada de los datos)
        #   μ₀ = media hipotética bajo H₀ (definida por el usuario)
        #   σ  = desviación estándar poblacional (conocida, ingresada por el usuario)
        #   n  = tamaño de muestra
        #   σ/√n = error estándar de la media (qué tan precisa es x̄ como estimador de μ)
        #
        # Interpretación: Z indica cuántas desviaciones estándar está x̄ de μ₀.
        # Bajo H₀, Z sigue una distribución Normal Estándar N(0,1).
        x_bar = float(data.mean())                      # x̄: media muestral
        z_val = (x_bar - mu0) / (sigma / np.sqrt(n))   # Estadístico Z calculado

        # ── CÁLCULO DE p-value Y VALOR CRÍTICO Z ─────────────────────────
        #
        # p-value: probabilidad de obtener un estadístico tan extremo o más
        #          si H₀ fuera verdadera.
        #          → Si p-value < α: evidencia suficiente para rechazar H₀
        #          → Si p-value ≥ α: no hay evidencia suficiente
        #
        # Valor crítico Z (z_crit): umbral que delimita la región de rechazo
        #   Se obtiene con norm.ppf(p): función cuantil de la normal estándar
        #   PPF = Percent Point Function (inversa de CDF)
        #
        # norm.cdf(z): P(Z ≤ z) — probabilidad acumulada hasta z
        # norm.ppf(p): valor z tal que P(Z ≤ z) = p

        if tipo == "Bilateral":
            # H₁: μ ≠ μ₀ → zona de rechazo en AMBAS colas
            # p-value = 2 × P(Z > |z_calculado|) = 2 × (1 - CDF(|Z|))
            # z_crit positivo: p.ej. 1.96 para α=0.05 (α/2 en cada cola)
            p_val  = 2 * (1 - norm.cdf(abs(z_val)))
            z_crit = norm.ppf(1 - alpha / 2)

        elif tipo == "Cola derecha":
            # H₁: μ > μ₀ → zona de rechazo solo en la cola derecha
            # p-value = P(Z > z_calculado) = 1 - CDF(Z)
            # z_crit: p.ej. 1.645 para α=0.05
            p_val  = 1 - norm.cdf(z_val)
            z_crit = norm.ppf(1 - alpha)

        else:
            # H₁: μ < μ₀ → zona de rechazo solo en la cola izquierda
            # p-value = P(Z < z_calculado) = CDF(Z)
            # z_crit: valor negativo, p.ej. -1.645 para α=0.05
            p_val  = norm.cdf(z_val)
            z_crit = norm.ppf(alpha)

        # ── REGLA DE DECISIÓN ─────────────────────────────────────────────
        # Criterio del p-value (equivalente al criterio del valor crítico):
        #   Si p-value < α  → RECHAZAR H₀ (el resultado es estadísticamente significativo)
        #   Si p-value ≥ α  → NO RECHAZAR H₀ (sin evidencia suficiente)
        #
        # IMPORTANTE: "No rechazar H₀" NO significa que H₀ sea verdadera,
        # solo que no tenemos suficiente evidencia en su contra con esta muestra.
        rechaza = p_val < alpha

        # ── Tarjetas de resultados ────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        section_header("Resultados", "Estadístico calculado vs región crítica")

        r1, r2, r3, r4, r5 = st.columns(5, gap="small")
        html_cards = (
            '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:1.2rem">'
            + stat_card("Media muestral", f"{x_bar:.4f}")
            + stat_card("Estadístico Z", f"{z_val:.4f}")
            + stat_card("Valor crítico", f"±{z_crit:.4f}" if tipo == "Bilateral" else f"{z_crit:.4f}")
            + stat_card("p-value", f"{p_val:.5f}")
            + stat_card("n", str(n), "obs.")
            + "</div>"
        )
        st.markdown(html_cards, unsafe_allow_html=True)

        # Badge visual con la decisión final
        if rechaza:
            st.markdown('<span class="decision-reject">✗ Se rechaza H₀</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="decision-accept">✓ No se rechaza H₀</span>', unsafe_allow_html=True)

        # ── GRÁFICA: DISTRIBUCIÓN N(0,1) Y REGIÓN CRÍTICA ────────────────
        # Visualiza dónde cae el estadístico Z calculado respecto a las zonas de
        # rechazo y no rechazo de H₀
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        section_header("Distribución y región crítica", "Zona de rechazo marcada según α y tipo de prueba")

        # Eje X: de -4.5 a 4.5 σ (cubre >99.99% del área bajo la curva normal)
        x = np.linspace(-4.5, 4.5, 1200)
        y = norm.pdf(x)   # PDF de la N(0,1) — función de densidad de probabilidad

        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(x, y, color=ACCENT, lw=2, label="N(0,1)")
        ax.fill_between(x, y, alpha=0.08, color=ACCENT)  # Área bajo curva (sutil)

        # Colorear la zona de rechazo según el tipo de prueba elegido
        if tipo == "Bilateral":
            # Ambas colas: rechaza si Z > z_crit O Z < -z_crit
            mask_r = x >= z_crit
            mask_l = x <= -z_crit
            ax.fill_between(x, y, where=mask_r, color=DANGER, alpha=0.45, label=f"Rechazo α/2={alpha/2}")
            ax.fill_between(x, y, where=mask_l, color=DANGER, alpha=0.45)
            ax.axvline( z_crit, color=DANGER, lw=1.2, ls="--", alpha=0.8)
            ax.axvline(-z_crit, color=DANGER, lw=1.2, ls="--", alpha=0.8)
        elif tipo == "Cola derecha":
            # Cola derecha: rechaza si Z > z_crit
            ax.fill_between(x, y, where=(x >= z_crit), color=DANGER, alpha=0.45, label=f"Rechazo α={alpha}")
            ax.axvline(z_crit, color=DANGER, lw=1.2, ls="--", alpha=0.8)
        else:
            # Cola izquierda: rechaza si Z < z_crit (z_crit es negativo)
            ax.fill_between(x, y, where=(x <= z_crit), color=DANGER, alpha=0.45, label=f"Rechazo α={alpha}")
            ax.axvline(z_crit, color=DANGER, lw=1.2, ls="--", alpha=0.8)

        # Marcar el estadístico Z calculado con línea vertical y punto
        ax.axvline(z_val, color=ACCENT3, lw=2, label=f"Z calc = {z_val:.3f}")
        ax.scatter([z_val], [norm.pdf(z_val)], color=ACCENT3, s=70, zorder=5)

        # Anotación con flecha apuntando al valor Z calculado
        y_top = max(norm.pdf(z_val), 0.05)
        ax.annotate(
            f"Z = {z_val:.3f}",
            xy=(z_val, norm.pdf(z_val)),
            xytext=(z_val + 0.4, norm.pdf(z_val) + 0.06),
            fontsize=9, color=ACCENT3,
            arrowprops=dict(arrowstyle="->", color=ACCENT3, lw=0.8),
        )

        # Etiqueta en el centro indicando la zona de NO rechazo
        ax.text(0, norm.pdf(0) * 0.45, "No rechazo\nH₀", ha="center", va="center",
                fontsize=9, color="#94a3b8", style="italic")

        ax.set_xlim(-4.5, 4.5)
        ax.set_xlabel("Z")
        ax.set_ylabel("Densidad")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig_to_st(fig)

        # ── INTERPRETACIÓN TEXTUAL DE LA DECISIÓN ────────────────────────
        if rechaza:
            interp = (
                f"Con α = {alpha} y un p-value = {p_val:.5f} < α, existe evidencia estadística "
                f"suficiente para <strong style='color:#fca5a5'>rechazar H₀</strong>. "
                f"El estadístico Z = {z_val:.4f} cae dentro de la región crítica."
            )
        else:
            interp = (
                f"Con α = {alpha} y un p-value = {p_val:.5f} ≥ α, <strong style='color:#6ee7b7'>no se rechaza H₀</strong>. "
                f"El estadístico Z = {z_val:.4f} no alcanza la región crítica. "
                f"No hay evidencia suficiente para concluir que μ {op} {mu0}."
            )

        st.markdown(f"""
        <div style="background:rgba(15,22,35,0.9);border:1px solid var(--border);
                    border-radius:10px;padding:16px 20px;margin-top:1rem;
                    font-size:13px;line-height:1.7;color:var(--text)">
            {interp}
        </div>
        """, unsafe_allow_html=True)

        # ── GUARDAR RESULTADOS EN SESSION STATE ───────────────────────────
        # Persiste todos los valores para que el Tab 3 (IA) pueda accederlos
        st.session_state["z_result"] = {
            "media_muestral":    x_bar,
            "media_hipotetica":  mu0,
            "n":                 n,
            "sigma":             sigma,
            "alpha":             alpha,
            "tipo":              tipo,
            "z":                 z_val,
            "p_value":           p_val,
            "decision":          "rechazar" if rechaza else "no rechazar",
            "z_critico":         z_crit,
            "columna":           col_z,
        }
        st.session_state["interp_auto"] = interp


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ASISTENTE IA CON GOOGLE GEMINI
# ══════════════════════════════════════════════════════════════════════════════
# Envía el RESUMEN de la prueba Z (no datos crudos) a la API de Gemini.
# La IA analiza si la decisión es correcta, verifica supuestos y sugiere mejoras.
# El estudiante puede comparar la respuesta de la IA con su propia decisión.
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section_header("Asistente IA · Gemini", "Análisis estadístico interpretado por inteligencia artificial")

    if "z_result" not in st.session_state:
        st.info("⟵ Realiza primero la Prueba Z para habilitar el asistente.")
    else:
        datos = st.session_state["z_result"]

        # API Key: se obtiene del input del usuario o de variable de entorno
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="AIza...",
            help="Obtén tu clave en https://aistudio.google.com/",
        )
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY", "")  # Fallback a variable de entorno

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        section_header("Resumen enviado a la IA", "Parámetros y resultados de la prueba Z")

        s1, s2 = st.columns(2, gap="large")
        with s1:
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
            {stat_card("Media muestral", f"{datos['media_muestral']:.4f}")}
            {stat_card("H₀: μ", f"{datos['media_hipotetica']}")}
            {stat_card("n", str(datos['n']), "obs")}
            {stat_card("σ", str(datos['sigma']))}
            </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
            {stat_card("Estadístico Z", f"{datos['z']:.4f}")}
            {stat_card("p-value", f"{datos['p_value']:.5f}")}
            {stat_card("α", str(datos['alpha']))}
            {stat_card("Tipo", datos['tipo'][:8])}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── CONSTRUCCIÓN DEL PROMPT ───────────────────────────────────────
        # El prompt incluye el contexto estadístico completo para que Gemini
        # pueda verificar la decisión e interpretar los resultados.
        # Se envía el RESUMEN estadístico, NO los datos crudos.
        # El prompt es editable para que el estudiante pueda experimentar.
        default_prompt = f"""Actúa como un experto en estadística inferencial explicando a un estudiante universitario.

Se realizó una prueba Z con los siguientes resultados:
- Variable analizada: {datos['columna']}
- Media muestral (x̄) = {datos['media_muestral']:.4f}
- Media hipotética H₀ (μ₀) = {datos['media_hipotetica']}
- Tamaño de muestra (n) = {datos['n']}
- Desviación estándar poblacional (σ) = {datos['sigma']}
- Nivel de significancia (α) = {datos['alpha']}
- Tipo de prueba: {datos['tipo']}
- Estadístico Z calculado = {datos['z']:.4f}
- Valor crítico Z = {datos['z_critico']:.4f}
- p-value = {datos['p_value']:.6f}
- Decisión: {datos['decision']} H₀

Responde de forma estructurada y clara:
1. ¿La decisión estadística es correcta? Verifica Z y p-value.
2. ¿Qué significa este resultado en términos prácticos?
3. ¿Se cumplen los supuestos de la prueba Z?
4. Riesgos de error Tipo I y Tipo II en este contexto.
5. ¿Recomendarías alguna prueba alternativa?

Sé directo, pedagógico y evita repetir los datos innecesariamente."""

        prompt = st.text_area(
            "Prompt enviado a Gemini (editable)",
            value=default_prompt,
            height=200,
        )

        if st.button("🤖 Analizar con Gemini"):
            if not api_key:
                st.error("⚠ Ingresa tu API Key de Gemini para continuar.")
            else:
                try:
                    # Configurar cliente de Gemini con la clave API
                    genai.configure(api_key=api_key)
                    # Usar el modelo gemini-1.5-flash (rápido y eficiente)
                    model = genai.GenerativeModel("gemini-1.5-flash")

                    # Llamada a la API — generate_content envía el prompt y espera respuesta
                    with st.spinner("Consultando a Gemini..."):
                        response = model.generate_content(prompt)

                    resp_text = response.text   # Texto generado por Gemini

                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    section_header("Respuesta de Gemini", "Análisis generado por IA")
                    st.markdown(f'<div class="ai-box">{resp_text}</div>', unsafe_allow_html=True)

                    # ── COMPARACIÓN: IA vs DECISIÓN ESTADÍSTICA ───────────
                    # Analiza el texto de Gemini para detectar si concuerda
                    # con la decisión calculada matemáticamente por la app
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    section_header("Comparación · IA vs Decisión estadística")

                    decision_app = datos["decision"]
                    resp_lower   = resp_text.lower()

                    # Buscar palabras clave en la respuesta para determinar concordancia
                    coincide = (
                        ("rechazar" in resp_lower and "no rechazar" not in resp_lower and decision_app == "rechazar")
                        or ("no rechazar" in resp_lower and decision_app == "no rechazar")
                        or ("no se rechaza" in resp_lower and decision_app == "no rechazar")
                        or ("se rechaza" in resp_lower and "no se rechaza" not in resp_lower and decision_app == "rechazar")
                    )

                    ic1, ic2 = st.columns(2, gap="large")
                    with ic1:
                        st.markdown(f"""
                        <div style="background:rgba(15,22,35,0.9);border:1px solid var(--border);
                                    border-radius:10px;padding:16px 20px">
                            <div style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:.1em;
                                        color:var(--muted);text-transform:uppercase;margin-bottom:8px">
                                Decisión app
                            </div>
                            {'<span class="decision-reject">✗ Rechaza H₀</span>' if decision_app=="rechazar" else '<span class="decision-accept">✓ No rechaza H₀</span>'}
                        </div>
                        """, unsafe_allow_html=True)

                    with ic2:
                        st.markdown(f"""
                        <div style="background:{'rgba(16,185,129,0.07)' if coincide else 'rgba(245,158,11,0.07)'};
                                    border:1px solid {'rgba(16,185,129,0.3)' if coincide else 'rgba(245,158,11,0.3)'};
                                    border-radius:10px;padding:16px 20px">
                            <div style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:.1em;
                                        color:var(--muted);text-transform:uppercase;margin-bottom:8px">
                                Concordancia IA
                            </div>
                            <div style="font-size:13px;color:{'#6ee7b7' if coincide else '#fcd34d'}">
                                {'✓ La IA coincide con la decisión estadística' if coincide else '⚠ Posible discrepancia — analiza el contexto'}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Reflexión crítica: invita al estudiante a comparar ambos análisis
                    # Esto cumple con el requisito del profe de "comparar con decisión del estudiante"
                    st.markdown(f"""
                    <div style="background:rgba(124,58,237,0.06);border:1px solid rgba(124,58,237,0.2);
                                border-radius:10px;padding:16px 20px;margin-top:1rem;font-size:13px;
                                line-height:1.7;color:var(--text)">
                        <strong style="color:var(--accent2)">Reflexión crítica:</strong><br>
                        {'La IA respalda la decisión, lo que indica consistencia. Aun así, evalúa si los supuestos del modelo (normalidad, σ conocida) se cumplen en tu contexto real.' if coincide else 'Existe discrepancia. Puede deberse a la interpretación contextual de la IA, supuestos distintos o ambigüedad en el prompt. Compara ambos razonamientos y justifica tu decisión.'}
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error al conectar con Gemini: {str(e)}")
                    st.info("Verifica que tu API Key sea válida y que tengas acceso a la API de Gemini.")