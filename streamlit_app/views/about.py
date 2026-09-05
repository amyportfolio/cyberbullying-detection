import streamlit as st


def show_about():

    st.title("ℹ️ About")

    st.markdown(
        """
        This application demonstrates an intelligent **Cross-Platform Cyberbullying Detection System**
        capable of detecting harmful content across multiple social media platforms using
        Machine Learning and Deep Learning techniques.
        """
    )

    st.divider()

    # ---------------------------------------------------
    # Project Overview
    # ---------------------------------------------------

    st.subheader("🎯 Project Objective")

    st.info(
        """
        The objective of this project is to automatically identify cyberbullying
        content on social media using Artificial Intelligence models and evaluate
        their performance across multiple platforms.
        """
    )

    st.divider()

    # ---------------------------------------------------
    # Datasets
    # ---------------------------------------------------

    st.subheader("📂 Datasets")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("🐦 Twitter")

    with col2:
        st.success("📘 Facebook")

    with col3:
        st.success("📷 Instagram")

    st.divider()

    # ---------------------------------------------------
    # Models
    # ---------------------------------------------------

    st.subheader("🤖 Implemented Models")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        ✅ Logistic Regression

        ✅ LSTM

        ✅ CNN-BiLSTM
        """)

    with col2:

        st.markdown("""
        ✅ BERT

        ✅ DistilBERT
        """)

    st.divider()

    # ---------------------------------------------------
    # Features
    # ---------------------------------------------------

    st.subheader("🚀 Features")

    st.markdown("""
    - Manual Text Detection

    - Twitter API Detection

    - Multiple Model Comparison

    - Cross-Platform Evaluation

    - Interactive Dashboard

    - Real-time Prediction
    """)

    st.divider()

    # ---------------------------------------------------
    # Technologies
    # ---------------------------------------------------

    st.subheader("💻 Technologies Used")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        st.markdown("""
        • Python

        • Streamlit

        • Pandas
        """)

    with tech2:

        st.markdown("""
        • PyTorch

        • Transformers

        • Scikit-learn
        """)

    with tech3:

        st.markdown("""
        • Twitter API

        • Matplotlib

        • NumPy
        """)

    st.divider()

    # ---------------------------------------------------
    # Developer
    # ---------------------------------------------------

    st.subheader("👨‍🎓 Developer")

    st.markdown(
        """
        **Name:** Amina Mulla

        **Programme:** M.Tech (Information Technology)

        **Project:** Cross-Platform Cyberbullying Detection System
        """
    )

    st.divider()

    # ---------------------------------------------------
    # Footer
    # ---------------------------------------------------

    st.caption(
        "© 2026 Cross-Platform Cyberbullying Detection System | M.Tech Dissertation Project"
    )