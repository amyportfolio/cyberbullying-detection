import os
import torch

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

# ---------------------------------------------------
# Device
# ---------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------
# Model Path
# ---------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)
print(BASE_DIR)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "distilbert"
)
print(MODEL_PATH)

# ---------------------------------------------------
# Global Variables
# ---------------------------------------------------

tokenizer = None
model = None

# ---------------------------------------------------
# Load Model (Only Once)
# ---------------------------------------------------

def load_model():

    global tokenizer
    global model

    if tokenizer is None:

        tokenizer = DistilBertTokenizerFast.from_pretrained(
            MODEL_PATH
        )

    if model is None:

        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_PATH
        )

        model.to(device)
        model.eval()

# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------

def predict(text):

    load_model()

    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    encoding = {
        k: v.to(device)
        for k, v in encoding.items()
    }

    with torch.no_grad():

        outputs = model(**encoding)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    return (
        int(prediction.item()),
        float(confidence.item())
    )