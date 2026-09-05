import os
import re
import torch
import torch.nn as nn

# ==========================================================
# Device
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# Model Path
# ==========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "cnn_bilstm",
    "cnn_bilstm_attention.pt"
)

# ==========================================================
# CNN-BiLSTM + Attention Model
# ==========================================================

class CNN_BiLSTM_Attention(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 96, kernel_size=3, padding="same"),
            nn.Conv1d(embed_dim, 96, kernel_size=4, padding="same"),
            nn.Conv1d(embed_dim, 96, kernel_size=5, padding="same")
        ])

        self.lstm = nn.LSTM(
            input_size=288,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)

        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):

        x = self.embedding(x)

        x = x.permute(0, 2, 1)

        conv_outputs = [
            torch.relu(conv(x))
            for conv in self.convs
        ]

        x = torch.cat(conv_outputs, dim=1)

        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        attention_weights = torch.softmax(
            self.attention(lstm_out),
            dim=1
        )

        context = torch.sum(
            attention_weights * lstm_out,
            dim=1
        )

        context = self.layer_norm(context)

        context = self.dropout(context)

        return self.fc(context)

# ==========================================================
# Globals
# ==========================================================

model = None
vocab = None
MAX_LEN = None

# ==========================================================
# Load Model
# ==========================================================

def load_model():

    global model
    global vocab
    global MAX_LEN

    if model is not None:
        return

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    print(MODEL_PATH)
    print(checkpoint.keys())

    vocab = checkpoint["vocab"]

    MAX_LEN = checkpoint["max_len"]

    model = CNN_BiLSTM_Attention(

        vocab_size=checkpoint["vocab_size"],

        embed_dim=checkpoint["embed_dim"],

        hidden_dim=checkpoint["hidden_dim"]

    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

# ==========================================================
# Text Cleaning
# ==========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# ==========================================================
# Encode Sentence
# ==========================================================

def encode_sentence(text):

    text = clean_text(text)

    tokens = text.split()

    unk_id = vocab.get("<UNK>", 1)

    ids = [
        vocab.get(token, unk_id)
        for token in tokens
    ]

    ids = ids[:MAX_LEN]

    if len(ids) < MAX_LEN:
        ids.extend([0] * (MAX_LEN - len(ids)))

    return torch.tensor(ids).unsqueeze(0)

# ==========================================================
# Prediction
# ==========================================================

def predict(text):

    load_model()

    inputs = encode_sentence(text).to(device)

    with torch.no_grad():

        logits = model(inputs)

        probability = torch.sigmoid(logits)

        confidence = probability.item()

        prediction = 1 if confidence >= 0.5 else 0

    return prediction, confidence