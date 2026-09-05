# Deep Learning-Based Cyberbullying Detection

A machine learning and deep learning-based system for detecting cyberbullying in social media text, with evaluation across datasets from different social media platforms and an interactive Streamlit application for detection.

## Project Overview

Cyberbullying is a growing concern on social media platforms, where harmful or abusive content can spread rapidly. This project investigates automated cyberbullying detection using Natural Language Processing (NLP), machine learning, and deep learning techniques.

The project focuses not only on model performance on the training platform but also on evaluating how well the models generalize to data from other social media platforms.

## Objectives

* Develop an automated cyberbullying detection system for social media text.
* Apply NLP-based preprocessing to noisy social media content.
* Implement and compare traditional machine learning and deep learning models.
* Evaluate model performance using multiple classification metrics.
* Analyze model generalization on an independent cross-platform dataset.
* Develop a Streamlit-based interface for cyberbullying detection.
* Provide an interactive dashboard for analyzing detection results.

## Models Implemented

The project includes the following models:

* **Logistic Regression** with TF-IDF features
* **LSTM**
* **CNN-BiLSTM with Attention**
* **BERT**
* **DistilBERT**

Logistic Regression is used as a traditional machine learning baseline, while the deep learning and transformer-based models are evaluated for their ability to capture more complex linguistic patterns.

## Data Preprocessing

The preprocessing pipeline includes:

1. Dataset standardization
2. English language filtering
3. Lowercase conversion
4. URL and mention removal
5. Contraction expansion
6. Repeated-character normalization
7. Slang replacement
8. Emoji conversion
9. Removal of unnecessary characters
10. Stopword handling
11. Lemmatization
12. Removal of empty records
13. Duplicate removal
14. Label encoding

The datasets are standardized to provide a consistent input format across different social media sources.

## Dataset and Evaluation

The project uses social media datasets for model training and independent evaluation.

The primary training dataset is divided into training, validation, and test subsets. An independent multi-platform dataset is used for cross-platform evaluation to examine model generalization beyond the training data.

For privacy, licensing, and repository-size considerations, the datasets are **not included in this repository**.

## Evaluation Metrics

The implemented models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

Performance is analyzed both on the primary test dataset and on independent cross-platform data.

## Project Structure

```text
cyberbullying-detection/
│
├── api/
│   └── Live tweet data collection modules
│
├── models/
│   └── Model-related implementation
│
├── notebooks/
│   ├── Data understanding
│   ├── Exploratory data analysis
│   ├── Preprocessing
│   ├── Model experimentation
│   └── Evaluation and visualization
│
├── results/
│   ├── Evaluation figures
│   └── Analysis outputs
│
├── src/
│   ├── data_fetch.py
│   ├── preprocess.py
│   └── utils.py
│
├── streamlit_app/
│   └── Streamlit application
│
├── .gitignore
├── config.yaml
├── requirements.txt
└── README.md
```

## Streamlit Application

The project includes an interactive Streamlit application that provides a user interface for cyberbullying detection.

The application includes functionality for:

* Text-based cyberbullying detection
* Simulated social media posts
* Live tweet data collection
* Prediction and confidence display
* Detection logging
* Administrative analysis and dashboard views

API credentials are stored locally in environment variables and are **not included in this repository**.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/cyberbullying-detection.git
cd cyberbullying-detection
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root for any required API credentials.

Example:

```text
TWITTER_BEARER_TOKEN=your_token_here
```

Do not commit the `.env` file to GitHub.

## Running the Streamlit Application

From the project root, run:

```bash
streamlit run streamlit_app/app.py
```

The application will open in your browser.

## Research Focus

The project emphasizes **testing, evaluation, and model generalization**. In particular, the independent cross-platform evaluation is used to analyze the limitations of models when applied to social media text originating from platforms different from the training data.

The results demonstrate that strong performance on a platform-specific test set does not necessarily guarantee equivalent performance on data from another platform.

## Limitations

* Social media language changes rapidly and contains slang, abbreviations, emojis, and contextual expressions.
* Cross-platform datasets may differ in vocabulary, writing style, and class distribution.
* API availability and rate limits can affect live data collection.
* Model performance may vary depending on dataset characteristics and preprocessing.
* The current system focuses primarily on textual information.

## Future Scope

Future improvements may include:

* Multilingual cyberbullying detection
* Multimodal analysis using text, images, and video
* Improved domain adaptation for cross-platform deployment
* Real-time monitoring at larger scale
* Explainable AI techniques for prediction interpretation
* Continuous model updating using newly collected data
* Improved handling of sarcasm, context, and implicit cyberbullying

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* TensorFlow / Keras
* PyTorch
* Hugging Face Transformers
* Streamlit
* Jupyter Notebook

## Academic Project

This repository contains the implementation and experimental work associated with an **M.Tech dissertation project on cyberbullying detection in social media text**.

The repository is intended for academic and research purposes.
