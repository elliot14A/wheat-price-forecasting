import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
WEB_DIR = FILE_PATH.parents[1]
PROFILE_HTML = WEB_DIR / "profile.html"

st.set_page_config(page_title="wheat-price-prediction", layout="wide")


@st.cache_data
def get_data_profiling():
    HtmlFile = open(PROFILE_HTML, "r", encoding="utf-8")
    source_code = HtmlFile.read()
    return source_code


with st.container():
    source_code = get_data_profiling()
    components.html(source_code, height=950, scrolling=True)
