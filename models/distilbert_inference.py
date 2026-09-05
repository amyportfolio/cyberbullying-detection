import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "distilbert_model")

@torch.no_grad()
def load_distilbert():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict_text(text, tokenizer, model):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)

    label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][label].item()

    return label, confidence
