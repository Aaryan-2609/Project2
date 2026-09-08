"""
FastAPI inference service for the DASS-42 severity models.

Loads the 3 best-model artifacts (Depression/Anxiety/Stress) saved by
scripts/run_pipeline.py and exposes:

  POST /predict            -> severity + probabilities for all 3 targets
  GET  /health              -> liveness check
  GET  /models/info          -> which model/feature-mode is loaded per target

Run:  uvicorn api.main:app --reload --port 8000
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import config

app = FastAPI(
    title="DASS-42 Mental State Classifier API",
    description="Predicts Depression / Anxiety / Stress severity from personality, "
                "demographic and (optionally) DASS-item features. Not a diagnostic tool.",
    version="1.0.0",
)

_MODELS: Dict[str, dict] = {}


@app.on_event("startup")
def load_models():
    for target in config.TARGETS:
        path = config.MODELS_DIR / f"best_model_{target}.joblib"
        if path.exists():
            _MODELS[target] = joblib.load(path)
        else:
            _MODELS[target] = None  # allows the API to start even before training


class TIPIInput(BaseModel):
    tipi1: int = Field(..., ge=1, le=7)
    tipi2: int = Field(..., ge=1, le=7)
    tipi3: int = Field(..., ge=1, le=7)
    tipi4: int = Field(..., ge=1, le=7)
    tipi5: int = Field(..., ge=1, le=7)
    tipi6: int = Field(..., ge=1, le=7)
    tipi7: int = Field(..., ge=1, le=7)
    tipi8: int = Field(..., ge=1, le=7)
    tipi9: int = Field(..., ge=1, le=7)
    tipi10: int = Field(..., ge=1, le=7)


class PredictionRequest(BaseModel):
    tipi: TIPIInput
    vocab_score: int = Field(..., ge=0, le=14, description="Count of real words correctly recognized (VCL, excluding the 2 fake-word checks)")
    education: int = Field(..., ge=1, le=4)
    urban: int = Field(..., ge=1, le=3)
    gender: int = Field(..., ge=1, le=3)
    engnat: int = Field(..., ge=1, le=2)
    age: int = Field(..., ge=13, le=100)
    hand: int = Field(..., ge=1, le=3)
    religion: int = Field(..., ge=1, le=12)
    orientation: int = Field(..., ge=1, le=5)
    race: int = Field(..., ge=1, le=6)
    voted: int = Field(..., ge=1, le=2)
    married: int = Field(..., ge=1, le=3)
    familysize: int = Field(..., ge=0, le=20)
    introelapse: float = Field(30, ge=0)
    testelapse: float = Field(300, ge=0)
    surveyelapse: float = Field(60, ge=0)


class TargetPrediction(BaseModel):
    severity: str
    probabilities: Dict[str, float]
    model_used: str


class PredictionResponse(BaseModel):
    depression: TargetPrediction
    anxiety: TargetPrediction
    stress: TargetPrediction
    disclaimer: str = (
        "This is a research/educational tool, NOT a medical diagnosis. "
        "If you are struggling, please speak with a qualified mental health "
        "professional or a crisis line in your country."
    )


def _row_from_request(req: PredictionRequest) -> pd.DataFrame:
    """Rebuild the same engineered features the model was trained on, from
    the raw request fields (mirrors src/feature_engineering.py logic for the
    'metadata_only' mode)."""
    def r(x):
        return 8 - x

    row = {
        "trait_extraversion": (req.tipi.tipi1 + r(req.tipi.tipi6)) / 2,
        "trait_agreeableness": (r(req.tipi.tipi2) + req.tipi.tipi7) / 2,
        "trait_conscientiousness": (req.tipi.tipi3 + r(req.tipi.tipi8)) / 2,
        "trait_emotional_stability": (r(req.tipi.tipi4) + req.tipi.tipi9) / 2,
        "trait_openness": (req.tipi.tipi5 + r(req.tipi.tipi10)) / 2,
        "vocab_score": req.vocab_score,
        "age": req.age,
        "familysize": req.familysize,
        "introelapse": req.introelapse,
        "testelapse": req.testelapse,
        "surveyelapse": req.surveyelapse,
        f"education_{req.education}": 1,
        f"urban_{req.urban}": 1,
        f"gender_{req.gender}": 1,
        f"engnat_{req.engnat}": 1,
        f"hand_{req.hand}": 1,
        f"religion_{req.religion}": 1,
        f"orientation_{req.orientation}": 1,
        f"race_{req.race}": 1,
        f"voted_{req.voted}": 1,
        f"married_{req.married}": 1,
    }
    return pd.DataFrame([row])


def _predict_target(target: str, req: PredictionRequest) -> TargetPrediction:
    bundle = _MODELS.get(target)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=f"No trained model available for {target}. Run scripts/run_pipeline.py first.",
        )
    model, feature_names = bundle["model"], bundle["feature_names"]

    row = _row_from_request(req)
    row = row.reindex(columns=feature_names, fill_value=0)  # align one-hot columns exactly

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["clf"].classes_

    return TargetPrediction(
        severity=str(pred),
        probabilities={str(c): float(p) for c, p in zip(classes, proba)},
        model_used=bundle["model_name"],
    )


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": {k: v is not None for k, v in _MODELS.items()}}


@app.get("/models/info")
def models_info():
    return {
        target: {
            "model_name": bundle["model_name"] if bundle else None,
            "feature_mode": bundle["feature_mode"] if bundle else None,
            "n_features": len(bundle["feature_names"]) if bundle else None,
        }
        for target, bundle in _MODELS.items()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    return PredictionResponse(
        depression=_predict_target("Depression", req),
        anxiety=_predict_target("Anxiety", req),
        stress=_predict_target("Stress", req),
    )
