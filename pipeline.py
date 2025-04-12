import os

if __name__ == "__main__":
    print("🔄 Step 1: Preprocessing...")
    os.system("python src/preprocess.py --input data/Tweets.csv --output data/processed")

    print("\n🧠 Step 2: Training...")
    os.system("python src/train.py --train data/processed/train.csv --model model/sentiment_model.joblib --exp Sentiment_Analysis_SVM")

    print("\n📊 Step 3: Evaluation...")
    os.system("python src/evaluate.py --model model/sentiment_model.joblib --test data/processed/test.csv")

    print("\n✅ Pipeline completed.")
