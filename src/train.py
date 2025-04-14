import pandas as pd
import joblib
import os
import mlflow
import argparse
import optuna
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def objective(trial, X, y):
    max_features = trial.suggest_int("tfidf_max_features", 500, 2000)
    C = trial.suggest_float("svm_C", 0.1, 10.0, log=True)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features)),
        ("svm", SVC(kernel="linear", C=C, probability=True))
    ])
    score = cross_val_score(pipeline, X, y, scoring="accuracy", cv=3).mean()
    return score

def train_with_optuna(train_path, model_path, experiment_name="Sentiment_Analysis_SVM"):
    df = pd.read_csv(train_path)
    X, y = df["text"], df["label"]

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_param("dataset_source", "https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment")
        mlflow.log_param("dataset_version", "v1.0")
        mlflow.log_param("dataset_path", train_path)

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X, y), n_trials=10)

        best_params = study.best_params
        mlflow.log_params(best_params)

        final_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=best_params["tfidf_max_features"])),
            ("svm", SVC(kernel="linear", C=best_params["svm_C"], probability=True))
        ])
        final_pipeline.fit(X, y)
        acc = final_pipeline.score(X, y)
        mlflow.log_metric("train_accuracy", acc)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(final_pipeline, model_path)
        mlflow.sklearn.log_model(final_pipeline, artifact_path="sentiment_model")

        with open("mlflow_run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"Best params: {best_params}")
        print(f"Final model accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/processed/train.csv")
    parser.add_argument("--model", type=str, default="model/sentiment_model.joblib")
    parser.add_argument("--exp", type=str, default="Sentiment_Analysis_SVM")
    args = parser.parse_args()

    train_with_optuna(args.train, args.model, args.exp)