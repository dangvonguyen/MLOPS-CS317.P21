import json
import logging
import logging.config
import os
import re
import time
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException, Request
from nltk.corpus import stopwords
from prometheus_client import CollectorRegistry, Histogram, Summary
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from pydantic import BaseModel


def setup_logging():
    with open("logging_config.json", "r") as f:
        config = json.load(f)

    logging.config.dictConfig(config)


# Initialize logger
setup_logging()
logger = logging.getLogger("api")


# Load model
MODEL_PATH = "model/sentiment_model.joblib"

# Create a custom registry to avoid duplicate registrations
registry = CollectorRegistry()

# Define custom metrics
MODEL_INFERENCE_TIME = Summary(
    "model_inference_time_seconds",
    "Time spent on model inference in seconds",
    registry=registry,
)
MODEL_CONFIDENCE = Histogram(
    "model_confidence_score",
    "Model confidence scores",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    registry=registry,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Starting up sentiment analysis API")
    if not os.path.exists(MODEL_PATH):
        error_msg = f"Model file not found at {MODEL_PATH}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    logger.info("Model loaded successfully")
    yield

    logger.info("Shutting down API")
    model = None


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

# Setup Prometheus instrumentation
instrumentator = Instrumentator(registry=registry)
instrumentator.add(metrics.latency())
instrumentator.add(metrics.requests())
instrumentator.add(metrics.request_size())
instrumentator.add(metrics.response_size())
instrumentator.instrument(app, metric_namespace="sentiment_api").expose(
    app, include_in_schema=False, should_gzip=True
)
logger.info("Prometheus instrumentation configured")


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method

    logger.info(f"Request started: {method} {path}")

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        logger.info(
            f"Request completed: {method} {path} - {response.status_code} in {process_time:.4f}s"
        )
        return response

    except Exception as exc:
        process_time = time.time() - start_time
        logger.error(
            f"Request failed: {method} {path} after {process_time:.4f}s", exc_info=True
        )
        raise exc


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    words = [w for w in text if w not in stops]
    return " ".join(words)


class TextInput(BaseModel):
    texts: list[str]


class PredictionResult(BaseModel):
    text: str
    sentiment: str
    label: int
    confidence: float


@app.post("/predict", response_model=list[PredictionResult])
async def predict(input_data: TextInput):
    try:
        logger.info(f"Received prediction request with {len(input_data.texts)} text(s)")
        # Measure inference time
        start_time = time.time()

        # Clean inputs
        cleaned_texts = [clean_text(text) for text in input_data.texts]
        preds = model.predict(cleaned_texts)

        # Get confidence scores
        confidence_scores = None
        if hasattr(model, "predict_proba"):
            confidence_scores = model.predict_proba(cleaned_texts)

        inference_time = time.time() - start_time
        MODEL_INFERENCE_TIME.observe(inference_time)
        logger.debug(f"Model inference completed in {inference_time:.4f} seconds")

        results = []
        for i, pred in enumerate(preds):
            sentiment = "positive" if pred == 1 else "negative"

            # Calculate confidence score
            confidence = 0.0
            if confidence_scores is not None:
                confidence = confidence_scores[i][pred]
                MODEL_CONFIDENCE.observe(confidence)

            results.append(
                PredictionResult(
                    text=input_data.texts[i],
                    sentiment=sentiment,
                    label=pred,
                    confidence=float(confidence),
                )
            )
        logger.info(
            f"Prediction completed successfully for {len(input_data.texts)} text(s)"
        )
        return results

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server with uvicorn")
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
