import streamlit as st


def show_home():

    st.title("🛡️ Cross-Platform Cyberbullying Detection System")

    st.markdown(
        """
        ### AI-Powered Detection of Cyberbullying Across Social Media Platforms

        Detect cyberbullying using Machine Learning, Deep Learning and Transformer-based models.
        This application provides real-time text analysis, live Twitter detection and an
        interactive dashboard for cross-platform cyberbullying monitoring.
        """
    )

    st.divider()

    # ---------------------------------------------------------
    # Banner
    # ---------------------------------------------------------
    #st.image(
    #    "assets/banner.png",
    #    use_container_width=True
    #)

    st.divider()

    # ---------------------------------------------------------
    # Overview
    # ---------------------------------------------------------
    st.subheader("📖 Project Overview")

    st.info(
        """
        Cyberbullying has become one of the most serious online safety issues across
        modern social media platforms. This project applies Artificial Intelligence
        techniques to automatically identify harmful content from Twitter, Facebook,
        and Instagram.

        The system supports multiple machine learning and transformer models,
        enabling comparative evaluation as well as real-time prediction.
        """
    )

    st.divider()

    # ---------------------------------------------------------
    # Highlights
    # ---------------------------------------------------------
    st.subheader("📊 Project Highlights")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Models",
            value="4"
        )

    with col2:
        st.metric(
            label="Platforms",
            value="3"
        )

    with col3:
        st.metric(
            label="Detection",
            value="Real-Time"
        )

    with col4:
        st.metric(
            label="Dashboard",
            value="Interactive"
        )

    # ---------------------------------------------------------
    # Features
    # ---------------------------------------------------------
    st.divider()

    st.subheader("✨ Features")

    left, right = st.columns(2)

    with left:

        st.success("📝 Manual Text Detection")
        st.write(
            "Analyze any social media post entered by the user."
        )

        st.success("🤖 Multiple AI Models")
        st.write(
            "Choose between Logistic Regression, LSTM, CNN-BiLSTM, BERT and DistilBERT."
        )

    with right:

        st.success("🐦 Live Twitter Detection")
        st.write(
            "Fetch live tweets and perform cyberbullying detection."
        )

        st.success("📊 Interactive Dashboard")
        st.write(
            "Visualize detection statistics, logs and cross-platform analysis."
        )

    # ---------------------------------------------------------
    # Workflow
    # ---------------------------------------------------------
    st.divider()

    st.subheader("⚙️ Detection Workflow")

    st.code(
    """
    User Input / Twitter API
            │
            ▼
    Text Preprocessing
            │
            ▼
    Selected AI Model
            │
            ▼
    Cyberbullying Detection
            │
            ▼
    Confidence Score
            │
            ▼
    Detection Log & Dashboard
    """
    )
    # ---------------------------------------------------------
    # Supported Models
    # ---------------------------------------------------------
    st.divider()

    st.subheader("🧠 Supported Models")

    c1, c2, c3, c4 = st.columns(4)

    c1.info("LSTM")
    c2.info("CNN-\nBiLSTM")
    c3.info("BERT")
    c4.info("DistilBERT")

    # ---------------------------------------------------------
    # Get Started
    # ---------------------------------------------------------
    st.divider()

    st.subheader("🚀 Get Started")

    st.markdown("""
    Use the **Navigation Panel** to:

    - 🔍 Detect cyberbullying manually
    - 🐦 Analyze live Twitter posts
    - 📊 Explore the Admin Dashboard
    - ℹ️ Learn more about the project
    """)