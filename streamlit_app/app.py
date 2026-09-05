import streamlit as st

from views.home import show_home
from views.detection import show_detection
from views.dashboard import show_dashboard
from views.about import show_about

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="Cross-Platform Cyberbullying Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Sidebar Navigation
# ==========================================
st.sidebar.title("🛡️ Navigation")

page = st.sidebar.radio(
    "Menu",
    [
        "🏠 Home",
        "🔍 Detection",
        "📊 Dashboard",
        "ℹ️ About"
    ],
    label_visibility="collapsed"
)

# ==========================================
# Page Routing
# ==========================================
if page == "🏠 Home":
    show_home()

elif page == "🔍 Detection":
    show_detection()

elif page == "📊 Dashboard":
    show_dashboard()

elif page == "ℹ️ About":
    show_about()