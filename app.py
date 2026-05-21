import streamlit as st
import os
import tempfile
import matplotlib.pyplot as plt
import sys

# ── Path setup ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from logic.audio_processor import AudioProcessor
from logic.model_handler import ModelHandler
from utils.config_manager import ConfigManager

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Géneros Musicales",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* General */
    .main .block-container { padding-top: 2rem; }

    /* Title */
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Result cards */
    .result-card {
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .result-success { background: #1a3a1a; border-left: 4px solid #4caf50; }
    .result-warning { background: #3a2e00; border-left: 4px solid #ff9800; }
    .result-error   { background: #3a1a1a; border-left: 4px solid #f44336; }
    .result-card h2 { margin: 0 0 0.25rem; font-size: 1.5rem; }
    .result-card p  { margin: 0; color: #bbb; font-size: 0.9rem; }

    /* Genre badge */
    .genre-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #64b5f6;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #ccc;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    /* Confidence bar label */
    .conf-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        color: #aaa;
        margin-bottom: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ─────────────────────────────────────────────────────────────────
GENRE_TRANSLATIONS = {
    "blues": "Blues", "classical": "Clásica", "country": "Country",
    "disco": "Disco", "hiphop": "Hip Hop", "jazz": "Jazz",
    "lofi": "Lofi", "metal": "Metal", "pop": "Pop",
    "reggae": "Reggae", "reggaeton": "Reggaeton", "rock": "Rock",
    "trap": "Trap",
}

GENRES = sorted(GENRE_TRANSLATIONS.keys())


def translate(genre: str) -> str:
    return GENRE_TRANSLATIONS.get(genre.lower(), genre)


# ── Session state init ───────────────────────────────────────────────────────
def init_state():
    defaults = {
        "config": ConfigManager(),
        "model_handler": None,
        "result": None,          # dict with genre, confidence, all_scores
        "spectrogram_buf": None, # BytesIO PNG for display
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
cfg: ConfigManager = st.session_state.config


def get_model() -> ModelHandler:
    """Lazy-load model; reload if path changed."""
    model_path = cfg.get("model_path")
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.getcwd(), model_path)

    handler = st.session_state.model_handler
    if handler is None or not handler.is_loaded():
        handler = ModelHandler(model_path)
        st.session_state.model_handler = handler
    return handler


def reset_result():
    st.session_state.result = None
    st.session_state.spectrogram_buf = None


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Ajustes")

    threshold = st.slider(
        "Umbral de confianza",
        min_value=0, max_value=100,
        value=int(cfg.get("confidence_threshold") * 100),
        format="%d%%"
    )

    if st.button("💾 Guardar ajustes", use_container_width=True):
        cfg.set("confidence_threshold", threshold / 100.0)
        st.success("Ajustes guardados ✓")


# ── Main content ─────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">🎵 Clasificador de Géneros Musicales</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Sube un archivo de audio y analiza su género.</div>',
    unsafe_allow_html=True
)

# ── File upload ──────────────────────────────────
uploaded_file = st.file_uploader(
    "Selecciona un archivo de audio",
    type=["mp3", "wav", "ogg", "flac"],
    on_change=reset_result,
    label_visibility="collapsed",
)

if uploaded_file:
    col_info, col_btn = st.columns([4, 1])
    with col_info:
        st.markdown(f"**Archivo:** `{uploaded_file.name}`  &nbsp; `{uploaded_file.size / 1024:.1f} KB`")
    with col_btn:
        classify_clicked = st.button("🔍 Clasificar", type="primary", use_container_width=True)

    if classify_clicked:
        reset_result()
        # Write to a temp file so librosa can read it
        suffix = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        features = None
        spec_buf = None
        try:
            with st.spinner("Extrayendo espectrograma y consultando el modelo…"):
                features, spec_buf = AudioProcessor.extract_features(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        if features is None:
            st.error("❌ No se pudieron extraer características. Comprueba que el archivo sea válido.")
        else:
            model = get_model()
            if not model.is_loaded():
                st.error(f"❌ No se pudo cargar el modelo desde `{cfg.get('model_path')}`. Revisa la ruta en los ajustes.")
            else:
                with st.spinner("Prediciendo género…"):
                    genre, confidence, all_scores = model.predict(features)

                st.session_state.result = {
                    "genre": genre,
                    "confidence": confidence,
                    "all_scores": all_scores,
                }
                st.session_state.spectrogram_buf = spec_buf

# ── Result display ────────────────────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result
    genre = result["genre"]
    confidence = result["confidence"]
    all_scores = result["all_scores"]
    threshold_val = cfg.get("confidence_threshold")

    st.divider()

    col_result, col_spec = st.columns([1, 1], gap="large")

    with col_result:
        st.markdown('<div class="section-header">Resultado</div>', unsafe_allow_html=True)

        if confidence >= threshold_val:
            card_class = "result-success"
            icon = "✅"
            label = "Género predicho"
        else:
            card_class = "result-warning"
            icon = "⚠️"
            label = "Género sugerido (confianza baja)"

        genre_es = translate(genre)
        st.markdown(f"""
        <div class="result-card {card_class}">
            <h2>{icon} {genre_es}</h2>
            <p>{label} · Confianza: <strong>{confidence*100:.1f}%</strong> · Umbral: {threshold_val*100:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Top scores bar chart
        st.markdown('<div class="section-header" style="margin-top:1.2rem">Distribución de probabilidades</div>', unsafe_allow_html=True)
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        labels = [translate(g) for g, _ in sorted_scores]
        values = [v * 100 for _, v in sorted_scores]

        colors = ["#4caf50" if g == genre else "#1e3a5f" for g, _ in sorted_scores]

        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.6)
        ax.set_xlabel("Probabilidad (%)", color="#aaa")
        ax.tick_params(colors="#ccc", labelsize=9)
        ax.spines[:].set_color("#333")
        ax.xaxis.label.set_color("#aaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_spec:
        st.markdown('<div class="section-header">Espectrograma Mel</div>', unsafe_allow_html=True)
        if st.session_state.spectrogram_buf:
            st.image(st.session_state.spectrogram_buf, use_container_width=True)



elif not uploaded_file:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #555;">
        <div style="font-size: 3rem;">🎼</div>
        <p style="margin-top: 0.5rem;">Sube un archivo de audio para comenzar</p>
        <p style="font-size: 0.85rem;">Formatos soportados: MP3, WAV, OGG, FLAC</p>
    </div>
    """, unsafe_allow_html=True)
