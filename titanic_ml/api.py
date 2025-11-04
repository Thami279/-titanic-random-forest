"""FastAPI application for serving Titanic survival predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from titanic_ml import config
from titanic_ml.models.training import train_models
from titanic_ml.utils.preprocessing import engineer_features

app = FastAPI(title="Titanic Survival API", version="1.0.0")

_PIPELINE = None
_TRAINING_REPORT = {}


class PassengerRecord(BaseModel):
    """Schema for incoming prediction requests."""

    Name: str = Field(..., example="Smith, Mr. John")
    Pclass: int = Field(..., ge=1, le=3, example=1)
    Sex: str = Field(..., regex="^(male|female)$", example="female")
    Age: Optional[float] = Field(None, ge=0, le=100, example=29.0)
    SibSp: int = Field(..., ge=0, le=10, example=0)
    Parch: int = Field(..., ge=0, le=10, example=0)
    Fare: float = Field(..., ge=0, le=600, example=72.0)
    Embarked: Optional[str] = Field(None, regex="^(S|C|Q)$", example="C")
    Ticket: str = Field(..., example="PC 17599")
    Cabin: Optional[str] = Field(None, example="C85")

    @validator("Embarked")
    def uppercase_embarked(cls, value: Optional[str]) -> Optional[str]:
        return value.upper() if isinstance(value, str) else value


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    survived: int
    probability: float
    model: str
    training_data_hash: str


@app.on_event("startup")
def load_model() -> None:
    """Train or load the best-performing model into memory."""
    global _PIPELINE, _TRAINING_REPORT

    result = train_models(Path(config.DATA_PATH), use_mlflow=False, fast_mode=True)
    _PIPELINE = result.best_model.pipeline
    _TRAINING_REPORT = result.to_report()


@app.get("/health")
def healthcheck() -> dict:
    """Health endpoint for liveness probes."""
    if _PIPELINE is None:
        return {"status": "starting"}
    return {"status": "ok", "model": _TRAINING_REPORT.get("best_model", {}).get("model")}


@app.post("/predict", response_model=PredictionResponse)
def predict(record: PassengerRecord) -> PredictionResponse:
    """Predict survival probability for a passenger record."""
    if _PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not initialised yet")

    df = pd.DataFrame([record.dict()])
    engineered = engineer_features(df)
    missing_features = set(config.BASELINE_FEATURES) - set(engineered.columns)
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Engineered features missing: {', '.join(sorted(missing_features))}",
        )

    X = engineered[config.BASELINE_FEATURES]

    try:
        probabilities = _PIPELINE.predict_proba(X)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
    except AttributeError:
        predictions = _PIPELINE.predict(X)
        probabilities = predictions.astype(float)

    return PredictionResponse(
        survived=int(predictions[0]),
        probability=float(probabilities[0]),
        model=_TRAINING_REPORT.get("best_model", {}).get("model", "unknown"),
        training_data_hash=_TRAINING_REPORT.get("data_hash", "unknown"),
    )


@app.get("/report")
def report() -> dict:
    """Return the latest training summary."""
    if not _TRAINING_REPORT:
        raise HTTPException(status_code=503, detail="Training report unavailable")
    return _TRAINING_REPORT
