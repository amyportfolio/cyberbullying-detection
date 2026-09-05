from models.distilbert import predict as distilbert_predict
from models.cnn_bilstm import predict as cnn_bilstm_predict


# ==========================================================
# Supported Models
# ==========================================================

AVAILABLE_MODELS = [
    "CNN-BiLSTM",
    "DistilBERT"
]


# ==========================================================
# Prediction Router
# ==========================================================

def predict(model_name, text):

    """
    Parameters
    ----------
    model_name : str
        CNN-BiLSTM or DistilBERT

    text : str
        Input text

    Returns
    -------
    prediction : int
        0 = Not Cyberbullying
        1 = Cyberbullying

    confidence : float
        Confidence score
    """

    if model_name == "CNN-BiLSTM":

        return cnn_bilstm_predict(text)

    elif model_name == "DistilBERT":

        return distilbert_predict(text)

    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )


# ==========================================================
# Prediction Label
# ==========================================================

def prediction_label(prediction):

    return (
        "Cyberbullying"
        if prediction == 1
        else "Not Cyberbullying"
    )