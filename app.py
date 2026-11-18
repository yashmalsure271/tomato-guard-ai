from __future__ import annotations

import streamlit as st

from utils.config import FAVICON_PATH, LOGO_PATH, PROJECT_NAME
from utils.ui import inject_css

st.set_page_config(
    page_title=f"{PROJECT_NAME} | Tomato leaf diagnostics",
    page_icon=str(FAVICON_PATH),
    layout="wide",
    initial_sidebar_state="auto",
)

inject_css()
st.logo(str(LOGO_PATH), size="large")

pages = {
    "Workspace": [
        st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
        st.Page("views/prediction.py", title="Prediction", icon=":material/biotech:"),
        st.Page("views/dashboard.py", title="Dashboard", icon=":material/insights:"),
    ],
    "Project": [
        st.Page("views/about.py", title="About", icon=":material/info:"),
    ],
}

navigation = st.navigation(pages, position="sidebar")
navigation.run()
