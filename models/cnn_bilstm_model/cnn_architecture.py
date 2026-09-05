import torch
import torch.nn as nn

class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 128, kernel_size=3, padding='same'),
            nn.Conv1d(embed_dim, 128, kernel_size=4, padding='same'),
            nn.Conv1d(embed_dim, 128, kernel_size=5, padding='same')
        ])

        self.lstm = nn.LSTM(
            input_size=128 * 3,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        conv_outs = [torch.relu(conv(x)) for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        context = self.layer_norm(context)
        context = self.dropout(context)

        return self.fc(context)
