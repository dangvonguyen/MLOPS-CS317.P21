import joblib
import argparse
import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download("stopwords")

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    words = [w for w in text if w not in stops]
    return " ".join(words)

def predict(model_path: str, input_texts: list):
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Cannot find model at: {model_path}")
        return
    
    model = joblib.load(model_path)

    # Clean inputs
    cleaned = [clean_text(text) for text in input_texts]
    predictions = model.predict(cleaned)

    for i, pred in enumerate(predictions):
        label = "Positive" if pred == 1 else "Negative"
        print(f"🗣 \"{input_texts[i]}\" → 💬 Sentiment: {label}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, nargs="+", required=True, help="Text(s) to classify")
    parser.add_argument("--model", type=str, default="model/sentiment_model.joblib", help="Path to model file")
    args = parser.parse_args()

    predict(args.model, args.text)
