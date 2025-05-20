import os
import re

from contextlib import asynccontextmanager
import joblib
from fastapi import FastAPI, HTTPException
from nltk.corpus import stopwords
from pydantic import BaseModel

# Load model
MODEL_PATH = "model/sentiment_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    yield
    model = None


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


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


@app.post("/predict", response_model=list[PredictionResult])
async def predict(input_data: TextInput):
    try:
        # Clean inputs
        cleaned_texts = [clean_text(text) for text in input_data.texts]
        preds = model.predict(cleaned_texts)

        results = []
        for i, pred in enumerate(preds):
            sentiment = "positive" if pred == 1 else "negative"
            results.append(
                PredictionResult(
                    text=input_data.texts[i],
                    sentiment=sentiment,
                    label=pred,
                )
            )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
