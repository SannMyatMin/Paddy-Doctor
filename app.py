import streamlit as st
from streamlit_option_menu import option_menu
import src.home as home
import src.CropCare as CropCare
import src.marketplace as marketplace
import src.news as news


st.set_page_config(page_title="Crop Care AI", layout="wide", page_icon="🌾")

st.markdown("""
<style>
section[data-testid="stSidebar"] { width: 220px !important; }
.sidebar-title { font-size: 22px !important; font-weight: bold; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-title">🌾 Crop Care AI</div>', unsafe_allow_html=True)
    
    main_menu = option_menu(
        menu_title=None,
        options=["Home", "CropCare", "Marketplace", "News"],
        icons=["house", "leaf", "shop", "newspaper"],
        menu_icon="cast",
        default_index=0,
        styles={
            "nav-link-selected": {"background-color": "#2e7d32"},
        }
    )

if main_menu == "Home":
    home.app()
elif main_menu == "CropCare":
    CropCare.app()
elif main_menu == "Marketplace":
    marketplace.app()
elif main_menu == "News":
    news.app()