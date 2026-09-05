import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os
from sklearn.preprocessing import LabelEncoder
import emoji

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# File paths
DATA_PATH = os.path.join("data", "raw", "kaggle_dataset.csv")
SLANG_PATH = os.path.join("data", "resources", "slang_dict.csv")
OUTPUT_PATH = os.path.join("data", "processed", "preprocessed_data.csv")

# 🔹 Load slang dictionary from CSV file
def load_slang_dict(path=SLANG_PATH):
    if os.path.exists(path):
        slang_df = pd.read_csv(path)
        slang_dict = dict(zip(slang_df['keyword'].str.lower(), slang_df['meaning'].str.lower()))
        print(f"✅ Loaded {len(slang_dict)} slang replacements from '{path}'.")
        return slang_dict
    else:
        print("⚠️ slang_dict.csv not found, using small fallback dictionary.")
        return {
            "u": "you",
            "ur": "your",
            "r": "are",
            "lol": "laughing out loud",
            "wtf": "what the hell",
            "idk": "i do not know",
            "brb": "be right back"
        }

# Convert emojis to text (😊 → smiling face)
def convert_emojis(text):
    return emoji.demojize(text, delimiters=(" ", " "))

# Replace slang words
def replace_slang(text, slang_dict):
    words = text.split()
    return ' '.join([slang_dict.get(word.lower(), word) for word in words])

# Clean text function
def clean_text(text, slang_dict):
    text = str(text).lower()                             # Lowercase
    text = convert_emojis(text)                          # Convert emojis
    text = replace_slang(text, slang_dict)               # Replace slang
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'@\w+|#', '', text)                   # Remove mentions/hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)              # Keep only letters
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()             # Remove extra spaces
    return text

# Preprocess dataset
def preprocess_data(df, slang_dict):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Clean and lemmatize
    df['clean_text'] = df['tweet_text'].apply(lambda x: clean_text(x, slang_dict))
    df['clean_text'] = df['clean_text'].apply(
        lambda x: ' '.join([
            lemmatizer.lemmatize(word)
            for word in x.split()
            if word not in stop_words
        ])
    )

    print("✅ Text cleaning, slang replacement, emoji conversion, and lemmatization done!")

    # Encode labels (if column exists)
    if 'cyberbullying_type' in df.columns:
        label_encoder = LabelEncoder()
        df['label_encoded'] = label_encoder.fit_transform(df['cyberbullying_type'])
        print("\n✅ Labels encoded:")
        print(df[['cyberbullying_type', 'label_encoded']].drop_duplicates())

    # Rearrange columns
    desired_order = ['tweet_text', 'clean_text', 'cyberbullying_type', 'label_encoded']
    df = df[[col for col in desired_order if col in df.columns]]

    # Show sample
    print("\n🧹 Sample of preprocessed data:")
    print(df.head(7))

    return df

# Save preprocessed data
def save_preprocessed_data(df, output_path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Preprocessed data saved successfully at: {output_path}")

# Main
if __name__ == "__main__":
    print("🚀 Starting preprocessing...\n")
    slang_dict = load_slang_dict()
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset loaded! Shape: {df.shape}\n")
    df = preprocess_data(df, slang_dict)
    save_preprocessed_data(df)
