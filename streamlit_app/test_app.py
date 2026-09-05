import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.distilbert_inference import load_distilbert, predict_text

tokenizer, model = load_distilbert()
print("✅ Model loaded successfully")

text = "You are nicest persion i met"
label, confidence = predict_text(text, tokenizer, model)

print(text, label, confidence)

# 1 -> cyberbullying
# 0 -> not cyberbullying
