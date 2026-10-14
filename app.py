
import streamlit as st

st.set_page_config(
    page_title="News app",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
    .stApp {
        background: white;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    .main-title {
        font-family: "Times New Roman", Times, serif;
        font-size: 76px;
        font-weight: bold;
        color: #000;
        text-align: center;
        line-height: 1.15;
        margin-bottom: 12px;
    }

    .subtitle {
        font-family: "Times New Roman", Times, serif;
        font-size: 26px;
        color: rgba(71, 85, 105, 0.65);
        text-align: center;
        margin-bottom: 25px;
    }

    div[data-testid="stTextInput"] input {
        height: 52px;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        font-family: "Times New Roman", Times, serif;
        font-size: 18px;
        padding-left: 16px;
    }

    div[data-testid="stFormSubmitButton"] button {
        height: 52px;
        width: 60px;
        background: #0866f5;
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 24px;
        padding: 0;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #0052cc;
        color: white;
    }

    @media (max-width: 600px) {
        .main-title {
            font-size: 52px;
        }

        .subtitle {
            font-size: 21px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Interfaz ----------------

# Espacio superior ajustable para centrar verticalmente
st.write("")
st.write("")
st.write("")
st.write("")

# Título y subtítulo
st.markdown('<div class="main-title">News app</div>',
            unsafe_allow_html=True)

st.markdown('<div class="subtitle">Search for news!</div>',
            unsafe_allow_html=True)

# Barra centrada horizontalmente
_, center, _ = st.columns([1, 5, 1])


with st.form("news_search_form"):
    input_col, button_col = st.columns(
        [8, 1],
        vertical_alignment="center"
    )

    with input_col:
        topic = st.text_input(
            "Search topic",
            placeholder="Enter a topic, keyword or event...",
            label_visibility="collapsed"
        )

    with button_col:
        submitted = st.form_submit_button("⌕")
