import pandas as pd
import joblib
import argparse
import mlflow
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate(model_path: str, test_path: str):
    model = joblib.load(model_path)
    df = pd.read_csv(test_path)
    X_test, y_test = df["text"], df["label"]

    preds = model.predict(X_test)

    report = classification_report(y_test, preds, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_test, preds))

    with open("mlflow_run_id.txt") as f:
        run_id = f.read()

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "test_accuracy": report["accuracy"],
            "test_precision": report["weighted avg"]["precision"],
            "test_recall": report["weighted avg"]["recall"],
            "test_f1": report["weighted avg"]["f1-score"]
        })

        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/confusion_matrix.png")
        mlflow.log_artifact("plots/confusion_matrix.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model/sentiment_model.joblib")
    parser.add_argument("--test", type=str, default="data/processed/test.csv")
    args = parser.parse_args()

    evaluate(args.model, args.test)