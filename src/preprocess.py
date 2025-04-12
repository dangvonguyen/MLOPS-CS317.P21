import pandas as pd
import os
import re
import argparse
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download("stopwords")

def tweet_to_words(tweet: str) -> str:
    # Bỏ link, @user, RT, ký tự đặc biệt và hạ về chữ thường
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r"@\w+", "", tweet)
    tweet = re.sub(r"[^a-zA-Z]", " ", tweet)  # chỉ giữ lại chữ cái
    tweet = tweet.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in tweet if w not in stops]
    return " ".join(meaningful_words)

def preprocess(input_path, output_dir):
    df = pd.read_csv(input_path)

    # Giữ lại cột text và sentiment
    df = df[["text", "airline_sentiment"]]
    df = df[df["airline_sentiment"].isin(["positive", "negative"])]
    df = df.dropna()

    # Làm sạch dữ liệu văn bản
    df["text"] = df["text"].apply(tweet_to_words)
    df = df[df["text"].str.strip() != ""]
    # Gán nhãn nhị phân
    df["label"] = df["airline_sentiment"].map({"positive": 1, "negative": 0})

    # Chia train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/Tweets.csv")
    parser.add_argument("--output", type=str, default="data/processed")
    args = parser.parse_args()

    preprocess(args.input, args.output)
