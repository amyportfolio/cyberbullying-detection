from models.model_loader import predict, prediction_label

text = "You are so stupid. Nobody likes you."

print("\n===== DistilBERT =====")

prediction, confidence = predict(
    "DistilBERT",
    text
)

print(
    prediction_label(prediction),
    confidence
)

print("\n===== CNN-BiLSTM =====")

prediction, confidence = predict(
    "CNN-BiLSTM",
    text
)

print(
    prediction_label(prediction),
    confidence
)