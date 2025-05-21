# ✨ Sentiment Analysis MLOps Pipeline

This project demonstrates a complete **MLOps pipeline** for sentiment classification on airline tweets using traditional ML (SVM), with features including:

- Data preprocessing
- Hyperparameter tuning (Optuna)
- Experiment tracking (MLflow)
- Task orchestration (Prefect)
- Deployment-ready model structure

---

## 🔧 Prerequisites

- **Python 3.12** or higher
- **UV** package manager (`pip install uv`)
- **Docker and Docker Compose** for containerization

## 📁 Project Structure

```
.
├── data/                      # Raw and processed data
├── model/                     # Saved trained model (.joblib)
├── src/                       # All core pipeline scripts
│   ├── preprocess.py          # Data cleaning and split
│   ├── train.py               # Model training + tuning + MLflow logging
│   ├── evaluate.py            # Evaluation and confusion matrix
├── api.py                     # FastAPI service for model deployment
├── pipeline.py                # Simple sequential pipeline (no orchestration)
├── pipeline_prefect.py        # Prefect-based orchestrated pipeline ✅
├── predict.py                 # Predict sentiment from custom input
```

---

## ✅ Features

| Feature                   | Description                                                       |
| ------------------------- | ----------------------------------------------------------------- |
| **Preprocessing**         | Clean text, remove stopwords, split data                          |
| **Model**                 | SVM (with TF-IDF vectorizer)                                      |
| **Hyperparameter Tuning** | ✅ Via [Optuna](https://optuna.org/)                              |
| **Experiment Tracking**   | ✅ Via [MLflow](https://mlflow.org/) — log metrics, params, model |
| **Checkpoints**           | ✅ Model saved as `.joblib`, also logged to MLflow                |
| **Task Orchestration**    | ✅ Using [Prefect](https://prefect.io)                            |
| **REST API**              | ✅ Using [FastAPI](https://fastapi.tiangolo.com/)                 |
| **Containerization**      | ✅ Using Docker with multi-stage builds                           |
| **Custom Prediction**     | Use `predict.py` with any custom input                            |
| **Confusion Matrix**      | Plot and log matrix to MLflow                                     |

---

## 🚀 Setup

### Option 1: Local setup with UV

```bash
$ uv sync
```

### Option 2: Docker setup

```bash
$ docker-compose up --build
```

## ▶️ How to Run

### 1. Run full pipeline

- ✅ Simple mode:

```bash
$ python pipeline.py
```

- ✅ Orchestrated mode (Prefect):

```bash
$ python pipeline_prefect.py
```

### 2. Run MLflow UI (optional)

```bash
$ mlflow ui
```

Visit `http://localhost:5000` to view experiment tracking.

---

## 🤖 Predict from custom input

```bash
$ python predict.py --text "The flight was awesome!" "Worst airline experience"
```

## 🌐 API Deployment

### Option 1: Run API service directly

```bash
$ python api.py
```

### Option 2: Using Docker

```bash
$ docker-compose up --build -d
```

Visit `http://localhost:8000/docs` for Swagger UI documentation.

### API Usage Example

```bash
$ curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{"texts": ["The flight was awesome!", "Worst airline experience"]}'
```

Response:

```json
[
  { "text": "The flight was awesome!", "sentiment": "positive", "label": 1 },
  { "text": "Worst airline experience", "sentiment": "negative", "label": 0 }
]
```

## 📹 Demo

![Demo](./assets/demo_call_api.gif)

---

## 📌 Notes

- MLflow logs model with full reproducibility
- The Docker setup uses multi-stage builds for smaller image size
- Security best practices implemented: non-root user, optimized dependency management with UV
- Healthcheck endpoint at `/health` for container orchestration systems
- Prefect flow can be scaled or visualized with `prefect orion start`

---

## 👨‍💻 Team Members

| Name               | Student ID |
| ------------------ | ---------- |
| **Lê Huỳnh Giang** | 22520356   |
| **Võ Nguyên Đăng** | 22520197   |
| **Văn Quốc Khánh** | 22520658   |
