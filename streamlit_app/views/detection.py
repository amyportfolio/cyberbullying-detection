import streamlit as st
from models.model_loader import predict, prediction_label
from utils.logger import log_prediction
from utils.twitter_api import fetch_mixed_tweets
from models.model_loader import predict, prediction_label
from utils.logger import log_prediction

# ----------------------------------------------------
# Detection Page
# ----------------------------------------------------

def show_detection():

    st.title("🔍 Cyberbullying Detection")

    st.write(
        "Select a trained model to view its performance and perform real-time cyberbullying detection."
    )

    st.divider()

    # ----------------------------------------------------
    # Accuracy Database
    # ----------------------------------------------------

    MODEL_RESULTS = {

        "Logistic Regression": {
            "twitter": 84.97,
            "cross": 69.00,
            "deploy": False
        },

        "LSTM": {
            "twitter": 85.77,
            "cross": 68.47,
            "deploy": False
        },

        "CNN-BiLSTM": {
            "twitter": 88.47,
            "cross": 83.79,
            "deploy": True
        },

        "BERT": {
            "twitter": 90.42,
            "cross": 74.00,
            "deploy": False
        },

        "DistilBERT": {
            "twitter": 90.38,
            "cross": 75.00,
            "deploy": True
        }

    }

# ----------------------------------------------------
# Model Selection
# ----------------------------------------------------

    selected_model = st.selectbox(

        "Select Model",

        ["Select Model"] + list(MODEL_RESULTS.keys())

    )

    # Session state
    if "active_model" not in st.session_state:
        st.session_state.active_model = None

    # ----------------------------------------------------
    # Show Accuracy
    # ----------------------------------------------------

    if selected_model != "Select Model":

        st.divider()

        st.subheader("📊 Model Performance")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(

                "Twitter Accuracy",

                f"{MODEL_RESULTS[selected_model]['twitter']}%"

            )

        with col2:

            st.metric(

                "Cross Platform Accuracy",

                f"{MODEL_RESULTS[selected_model]['cross']}%"

            )

        st.divider()

        # ----------------------------------------------------
        # Deployment Availability
        # ----------------------------------------------------

        if MODEL_RESULTS[selected_model]["deploy"]:

            st.success(
                f"✅ {selected_model} is available for real-time detection."
            )

            if st.button(

                f"Use {selected_model}",

                use_container_width=True

            ):

                st.session_state.active_model = selected_model

                st.success(
                    f"{selected_model} selected successfully."
                )

        else:

            st.info(
                "ℹ️ This model is available only for performance comparison."
            )

    else:

        st.info(
            "Please select a model to view its performance."
        )

    # ----------------------------------------------------
    # Active Model
    # ----------------------------------------------------

    if st.session_state.active_model is not None:

        st.divider()

        st.success(
            f"🤖 Current Detection Model : {st.session_state.active_model}"
        )

    else:

        st.warning(
            "No detection model has been selected yet."
        )

    st.divider()

    # ----------------------------------------------------
    # Detection Tabs
    # ----------------------------------------------------

    manual_tab, twitter_tab = st.tabs(

        [

            "📝 Manual Detection",

            "🐦 Twitter API Detection"

        ]

    )
    # ====================================================
    # Manual Detection
    # ====================================================

    with manual_tab:

        st.subheader("Manual Text Detection")

        text = st.text_area(

            "Enter Text",

            placeholder="Type your tweet or social media comment here...",

            height=180

        )

        col1, col2 = st.columns(2)

        detect = col1.button(
            "Detect",
            use_container_width=True
        )

        clear = col2.button(
            "Clear",
            use_container_width=True
        )

        if clear:
            st.rerun()

        if detect:

            if not MODEL_RESULTS[selected_model]["deploy"]:

                st.warning(
                    "Please select CNN-BiLSTM or DistilBERT for real-time prediction."
                )

            elif text.strip() == "":

                st.warning("Please enter some text.")

            else:

                try:

                    prediction, confidence = predict(
                        selected_model,
                        text
                    )

                    label = prediction_label(prediction)

                    log_prediction(
                        model=selected_model,
                        source="Manual",
                        text=text,
                        prediction=label,
                        confidence=confidence
                    )

                    if prediction == 1:

                        st.error("🚨 Cyberbullying Detected")

                    else:

                        st.success("✅ No Cyberbullying Detected")

                    st.metric(
                        label="Prediction",
                        value=label
                    )

                    st.progress(float(confidence))

                    st.caption(
                        f"Confidence Score : {confidence * 100:.2f}%"
                    )

                except Exception as e:

                    st.error("Prediction Failed")

                    st.exception(e)

    # ====================================================
    # Twitter Detection
    # ====================================================

    with twitter_tab:

        st.subheader("Twitter API Detection")

        topic = st.text_input(
            "Search Topic (Optional)",
            placeholder="Leave blank to fetch tweets from a random popular topic"
        )

        st.caption(
            "Leave the topic blank to automatically fetch tweets from a random popular topic."
        )

        tweet_count = st.slider(
            "Number of Tweets",
            10,
            20,
            10
        )

        fetch = st.button(
            "Fetch Tweets",
            use_container_width=True
        )

        if fetch:

            if not MODEL_RESULTS[selected_model]["deploy"]:

                st.warning(
                    "Please select CNN-BiLSTM or DistilBERT for real-time prediction."
                )

            else:

                tweets = fetch_mixed_tweets(
                    topic=topic,
                    max_results=tweet_count
                )

                if tweets.empty:

                    st.error("No tweets were fetched.")

                else:

                    # ---------------------------------------
                    # Source Information
                    # ---------------------------------------

                    if tweets.iloc[0]["topic"] == "Demo Dataset":

                        st.info(
                            "⚠️ Twitter API is currently unavailable. Showing tweets from the local demo dataset."
                        )

                    else:

                        st.success(
                            f"Showing live tweets for '{tweets.iloc[0]['topic']}'."
                        )

                    # ---------------------------------------
                    # Prediction
                    # ---------------------------------------

                    cyber_count = 0
                    safe_count = 0

                    for _, row in tweets.iterrows():

                        if "text" in row.index:
                            tweet = row["text"]
                        elif "clean_text" in row.index:
                            tweet = row["clean_text"]
                        else:
                            tweet = ""

                        prediction, confidence = predict(
                            selected_model,
                            tweet
                        )

                        label = prediction_label(prediction)

                        log_prediction(
                            model=selected_model,
                            source="Twitter API",
                            text=tweet,
                            prediction=label,
                            confidence=confidence
                        )

                        if prediction == 1:
                            cyber_count += 1
                        else:
                            safe_count += 1

                    # ---------------------------------------
                    # Summary
                    # ---------------------------------------

                    st.divider()

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Tweets Fetched",
                            len(tweets)
                        )

                    with col2:
                        st.metric(
                            "Cyberbullying",
                            cyber_count
                        )

                    with col3:
                        st.metric(
                            "Safe",
                            safe_count
                        )

                    st.divider()

                    # ---------------------------------------
                    # Tweet List
                    # ---------------------------------------

                    for index, row in tweets.iterrows():

                        if "text" in row.index:
                            tweet = row["text"]
                        elif "clean_text" in row.index:
                            tweet = row["clean_text"]
                        else:
                            tweet = ""

                        prediction, confidence = predict(
                            selected_model,
                            tweet
                        )

                        label = prediction_label(prediction)

                        left, right = st.columns([9, 2])

                        with left:

                            st.write(
                                f"**Tweet {index + 1}**"
                            )

                            st.write(tweet)

                        with right:

                            if prediction == 1:

                                st.error("🚨\nCyberbullying")

                            else:

                                st.success("✅\nSafe")

                        st.divider()