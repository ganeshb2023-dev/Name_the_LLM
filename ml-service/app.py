"""FastAPI inference API for the trained Name That LLM classifier."""
from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from scipy.special import softmax

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(ROOT / "model" / "classifier.joblib")))
API_KEY = os.environ.get("CLASSIFIER_API_KEY", "").strip()
MAX_CHARS = 20_000
MIN_CHARS = 20

app = FastAPI(
    title="Name That LLM Classifier API",
    version="1.0.0",
    description="Text classifier trained on the provided labelled Name That LLM competition training data.",
)

model: Any = None
load_error: str | None = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:  # Keep health endpoint available for deployment diagnostics.
    load_error = f"{type(exc).__name__}: model could not be loaded"

class PredictRequest(BaseModel):
    text: str = Field(min_length=MIN_CHARS, max_length=MAX_CHARS)

class Score(BaseModel):
    label: str
    score: float

class PredictResponse(BaseModel):
    label: str
    scores: list[Score]
    score_type: str = "normalized_relative_score_not_calibrated_probability"

def verify_key(authorization: str | None) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

@app.get("/")
def root():
    return {"service": "Name That LLM Classifier API", "health": "/health", "predict": "POST /predict"}

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail=load_error or "Model unavailable")
    return {"status": "ok", "model_loaded": True, "classes": len(model.named_steps["model"].classes_)}

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, authorization: str | None = Header(default=None)):
    verify_key(authorization)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    text = payload.text.strip()
    if len(text) < MIN_CHARS:
        raise HTTPException(status_code=422, detail=f"Text must contain at least {MIN_CHARS} non-whitespace characters.")
    if len(text) > MAX_CHARS:
        raise HTTPException(status_code=422, detail=f"Text must be at most {MAX_CHARS} characters.")
    classifier = model.named_steps["model"]
    scores = np.asarray(model.decision_function([text])[0], dtype=np.float64)
    probabilities = softmax(scores)
    labels = classifier.classes_
    order = np.argsort(probabilities)[::-1][:5]
    results = [{"label": str(labels[i]), "score": float(probabilities[i])} for i in order]
    return {"label": results[0]["label"], "scores": results, "score_type": "normalized_relative_score_not_calibrated_probability"}
