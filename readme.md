# ✨ Sentiment Analysis MLOps Pipeline

This project demonstrates a complete **MLOps pipeline** for sentiment classification on airline tweets using traditional ML (SVM), with features including:
- Data preprocessing
- Hyperparameter tuning (Optuna)
- Experiment tracking (MLflow)
- Task orchestration (Prefect)
- Deployment-ready model structure

---

## 📁 Project Structure

```
.
├── data/                      # Raw and processed data
├── model/                    # Saved trained model (.joblib)
├── plots/                    # Confusion matrix and other visualizations
├── src/                      # All core pipeline scripts
│   ├── preprocess.py         # Data cleaning and split
│   ├── train.py              # Model training + tuning + MLflow logging
│   ├── evaluate.py           # Evaluation and confusion matrix
├── pipeline.py               # Simple sequential pipeline (no orchestration)
├── pipeline_prefect.py       # Prefect-based orchestrated pipeline ✅
├── predict.py                # Predict sentiment from custom input
├── requirements.txt          # Full library dependencies (with versions)
├── README.md
```

---

## ✅ Features

| Feature                              | Description |
|--------------------------------------|-------------|
| **Preprocessing**                    | Clean text, remove stopwords, split data |
| **Model**                            | SVM (with TF-IDF vectorizer) |
| **Hyperparameter Tuning**            | ✅ Via [Optuna](https://optuna.org/) |
| **Experiment Tracking**              | ✅ Via [MLflow](https://mlflow.org/) — log metrics, params, model |
| **Checkpoints**                      | ✅ Model saved as `.joblib`, also logged to MLflow |
| **Task Orchestration**               | ✅ Using [Prefect](https://prefect.io) |
| **Custom Prediction**                | Use `predict.py` with any custom input |
| **Confusion Matrix**                 | Plot and log matrix to MLflow |

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run full pipeline

- ✅ Simple mode:
```bash
python pipeline.py
```

- ✅ Orchestrated mode (Prefect):
```bash
python pipeline_prefect.py
```

### 3. Run MLflow UI (optional)

```bash
mlflow ui
```

Visit `http://localhost:5000` to view experiment tracking.

---

## 🤖 Predict from custom input

```bash
python predict.py --text "The flight was awesome!" "Worst airline experience"
```

---

## 📌 Notes

- MLflow logs model with full reproducibility: `requirements.txt`, `conda.yaml`, `MLmodel`
- You can register models, deploy via MLflow or convert to API using FastAPI/Flask (optional)
- Prefect flow can be scaled or visualized with `prefect orion start`

---

## 👨‍💻 Team Members

| Name               | Student ID  |
|--------------------|-------------|
| **Lê Huỳnh Giang** | 22520356    |
| **Võ Nguyên Đăng** | 22520197    |
| **Văn Quốc Khánh** | 22520658    |
