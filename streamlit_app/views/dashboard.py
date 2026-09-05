import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt


def show_dashboard():

    st.title("📊 Admin Dashboard")

    # ============================
    # Admin Login
    # ============================

    if "admin_logged_in" not in st.session_state:
        st.session_state.admin_logged_in = False


    if not st.session_state.admin_logged_in:

        st.subheader("🔐 Admin Login")

        username = st.text_input("Username")
        password = st.text_input(
            "Password",
            type="password"
        )

        if st.button("Login"):

            if username == "admin" and password == "admin123":

                st.session_state.admin_logged_in = True
                st.success("Login Successful")
                st.rerun()

            else:
                st.error("Invalid credentials")

        return


    # ============================
    # Logout
    # ============================

    col1, col2 = st.columns([6,1])

    with col2:

        if st.button("Logout"):

            st.session_state.admin_logged_in = False
            st.rerun()


    st.divider()


    # ============================
    # Load Detection Logs
    # ============================


    log_path = os.path.join(
        "data",
        "detection_log.csv"
    )


    if not os.path.exists(log_path):

        st.warning(
            "No detection records available yet."
        )

        return


    df = pd.read_csv(log_path)


    if df.empty:

        st.warning(
            "Detection log is empty."
        )

        return



    # ============================
    # Overview Cards
    # ============================

    st.subheader("📌 Detection Overview")


    total = len(df)

    cyber = (
        df["prediction"]
        .eq("Cyberbullying")
        .sum()
    )

    normal = (
        df["prediction"]
        .eq("Not Cyberbullying")
        .sum()
    )


    col1,col2,col3 = st.columns(3)


    col1.metric(
        "Total Detections",
        total
    )

    col2.metric(
        "Cyberbullying Detected",
        cyber
    )

    col3.metric(
        "Safe Content",
        normal
    )



    st.divider()



    # ============================
    # Charts
    # ============================

    st.subheader("📈 Detection Analytics")


    col1,col2 = st.columns(2)


    with col1:

        st.write(
            "Prediction Distribution"
        )

        prediction_count = (
            df["prediction"]
            .value_counts()
        )


        st.bar_chart(
            prediction_count
        )


    with col2:

        st.write(
            "Detection Source"
        )


        source_count = (
            df["source"]
            .value_counts()
        )


        st.bar_chart(
            source_count
        )



    st.divider()



    # ============================
    # Model Usage
    # ============================


    if "model" in df.columns:

        st.subheader(
            "🤖 Model Usage"
        )

        st.bar_chart(
            df["model"]
            .value_counts()
        )



    # ============================
    # Confidence Analysis
    # ============================


    if "confidence" in df.columns:


        st.subheader(
            "🎯 Confidence Distribution"
        )


        fig,ax = plt.subplots(
            figsize=(8,3)
        )


        ax.hist(
            df["confidence"],
            bins=10
        )


        ax.set_xlabel(
            "Confidence"
        )

        ax.set_ylabel(
            "Count"
        )


        st.pyplot(fig)



    st.divider()



    # ============================
    # Detection Logs
    # ============================


    st.subheader(
        "📝 Detection History"
    )


    st.dataframe(
        df.sort_values(
            "timestamp",
            ascending=False
        ),
        use_container_width=True
    )



    # ============================
    # Cross Platform Analysis
    # ============================


    st.divider()


    st.subheader(
        "🌍 Cross Platform Dataset Analysis"
    )


    cross_path = os.path.join(
        "data",
        "processed",
        "cross_platform_dataset_processed.csv"
    )


    if os.path.exists(cross_path):


        cross_df = pd.read_csv(
            cross_path
        )


        col1,col2 = st.columns(2)



        with col1:

            st.write(
                "Platform Distribution"
            )


            st.bar_chart(
                cross_df["Platform"]
                .value_counts()
            )



        with col2:


            st.write(
                "Cyberbullying Distribution"
            )


            platform_labels = pd.crosstab(
                cross_df["Platform"],
                cross_df["label_encoded"]
            )


            st.bar_chart(
                platform_labels
            )


    else:

        st.info(
            "Cross-platform dataset not found."
        )