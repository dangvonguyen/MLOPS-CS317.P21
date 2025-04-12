from prefect import task, flow
import os

@task
def preprocess():
    print("🔄 Step 1: Preprocessing...")
    os.system("python src/preprocess.py --input data/Tweets.csv --output data/processed")

@task
def train():
    print("🧠 Step 2: Training...")
    os.system("python src/train.py --train data/processed/train.csv --model model/sentiment_model.joblib --exp Sentiment_Analysis_SVM")

@task
def evaluate():
    print("📊 Step 3: Evaluation...")
    os.system("python src/evaluate.py --model model/sentiment_model.joblib --test data/processed/test.csv")

@flow(name="Sentiment Analysis MLOps Pipeline")
def sentiment_pipeline():
    preprocess()
    train()
    evaluate()
    print("✅ Pipeline (Prefect) completed.")

if __name__ == "__main__":
    sentiment_pipeline()
