import torch
import os
from .cnn_architecture import CNN_BiLSTM_Attention


def load_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(
        os.path.dirname(__file__),
        "cnn_bilstm_model.pt"
    )

    checkpoint = torch.load(model_path, map_location=device)

    vocab = checkpoint["vocab"]
    max_len = checkpoint["max_len"]

    model = CNN_BiLSTM_Attention(len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return {
        "model": model,
        "vocab": vocab,
        "max_len": max_len,
        "device": device
    }


def tokenize(text):
    return text.lower().split()


def predict_text(text, model_data):
    model = model_data["model"]
    vocab = model_data["vocab"]
    max_len = model_data["max_len"]
    device = model_data["device"]

    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    ids = ids[:max_len]

    tensor = torch.tensor(ids).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.sigmoid(output).item()

    label = 1 if prob > 0.5 else 0

    return label, prob
