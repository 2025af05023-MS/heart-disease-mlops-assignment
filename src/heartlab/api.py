"""FastAPI serving with aggregate Prometheus metrics and privacy-aware logs."""

import json
import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from heartlab.data import FEATURES

logger = logging.getLogger("heartlab.api")
logging.basicConfig(level=logging.INFO, format="%(message)s")
REQUESTS = Counter("heartlab_requests_total", "Prediction requests", ["status"])
LATENCY = Histogram("heartlab_prediction_seconds", "Prediction latency")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=1, le=4)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=800)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=1, le=3)
    ca: int | None = Field(default=None, ge=0, le=3)
    thal: int | None = Field(default=None)


app = FastAPI(title="Cleveland Heart Disease Classifier", version="1.0.0")


def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact is unavailable")
    if not hasattr(app.state, "model"):
        app.state.model = joblib.load(MODEL_PATH)
    return app.state.model


@app.get("/health")
def health():
    try:
        get_model()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="Model artifact cannot be loaded") from None
    return {"status": "ready"}


@app.post("/predict")
def predict(patient: PatientFeatures):
    start = time.monotonic()
    try:
        model = get_model()
        frame = pd.DataFrame([patient.model_dump()])[FEATURES]
        probability = float(model.predict_proba(frame)[0, 1])
        result = {"prediction": int(probability >= 0.5),
                  "confidence": round(max(probability, 1 - probability), 4),
                  "probability_positive": round(probability, 4),
                  "model_version": "1.0.0"}
        REQUESTS.labels(status="success").inc()
        logger.info(json.dumps({"event": "prediction", "status": "success",
                                "duration_ms": round((time.monotonic() - start) * 1000, 2)}))
        return result
    except HTTPException:
        REQUESTS.labels(status="unavailable").inc()
        raise
    except Exception:
        REQUESTS.labels(status="error").inc()
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from None
    finally:
        LATENCY.observe(time.monotonic() - start)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
