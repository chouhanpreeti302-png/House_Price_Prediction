from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional, Dict, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# -------------------------
# Paths (relative to repo)
# -------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH   = BASE_DIR / "models" / "catboost_model.pkl"
ENC_PATH     = BASE_DIR / "models" / "label_encoders.pkl"
IMPUTER_PATH = BASE_DIR / "models" / "imputer.pkl"
SCALER_PATH  = BASE_DIR / "models" / "scaler.pkl"

SCHEMA_PATH  = BASE_DIR / "schema" / "feature_schema.json"


# -------------------------
# Feature schema (must match training)
# -------------------------
FEATURE_ORDER = [
    "Location", "Size", "Bedrooms", "Bathrooms", "Year Built", "Type",
    "Sold_Year", "Sold_Month", "Sold_Quarter", "Property_Age", "Condition_Ordinal"
]

CAT_COLS = ["Location", "Type"]
NUM_COLS = [
    "Size", "Bedrooms", "Bathrooms", "Year Built",
    "Sold_Year", "Sold_Month", "Sold_Quarter",
    "Property_Age", "Condition_Ordinal"
]

COND_MAP = {"Poor": 0, "Fair": 1, "Good": 2, "New": 3}


# -------------------------
# Load artifacts once at startup
# -------------------------
def load_artifacts():
    for p in [MODEL_PATH, ENC_PATH, IMPUTER_PATH, SCALER_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    model = joblib.load(MODEL_PATH)
    label_encoders = joblib.load(ENC_PATH)    # dict: {col -> LabelEncoder}
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, label_encoders, imputer, scaler


try:
    model, label_encoders, imputer, scaler = load_artifacts()
except Exception as e:
    # If this fails on Render, /health will reveal it in logs.
    model = label_encoders = imputer = scaler = None
    LOAD_ERROR = str(e)
else:
    LOAD_ERROR = None


# -------------------------
# API setup
# -------------------------
app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    description="Predict house prices using a CatBoost model trained on log1p(price). Returns expm1(pred)."
)


class PredictRequest(BaseModel):
    Location: str = Field(..., example="CityA")
    Size: float = Field(..., example=1800)
    Bedrooms: float = Field(..., example=3)
    Bathrooms: float = Field(..., example=2)
    Year_Built: float = Field(..., alias="Year Built", example=2005)
    Condition: Literal["Poor", "Fair", "Good", "New"] = Field(..., example="Good")
    Type: str = Field(..., example="Single Family")
    Date_Sold: str = Field(..., alias="Date Sold", example="2024-06-01")

    class Config:
        populate_by_name = True  # allow both "Year Built" and "Year_Built"


def feature_engineer_row(payload: PredictRequest) -> pd.DataFrame:
    dt = pd.to_datetime(payload.Date_Sold, errors="coerce")
    if pd.isna(dt):
        raise ValueError("Invalid 'Date Sold'. Use format YYYY-MM-DD.")

    sold_year = int(dt.year)
    sold_month = int(dt.month)
    sold_quarter = int((sold_month - 1) // 3 + 1)

    cond_ord = float(COND_MAP[payload.Condition])

    row = {
        "Location": str(payload.Location),
        "Size": float(payload.Size),
        "Bedrooms": float(payload.Bedrooms),
        "Bathrooms": float(payload.Bathrooms),
        "Year Built": float(payload.Year_Built),
        "Type": str(payload.Type),
        "Sold_Year": sold_year,
        "Sold_Month": sold_month,
        "Sold_Quarter": sold_quarter,
        "Property_Age": sold_year - float(payload.Year_Built),
        "Condition_Ordinal": cond_ord
    }

    return pd.DataFrame([row], columns=FEATURE_ORDER)


def preprocess_one(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()

    # label encode categoricals, reject unseen categories
    for col in CAT_COLS:
        le = label_encoders.get(col)
        if le is None:
            raise ValueError(f"Missing label encoder for column: {col}")

        val = str(Xc.loc[0, col])
        if val not in le.classes_:
            raise ValueError(
                f"Unknown {col}='{val}'. Allowed values: {list(le.classes_)}"
            )
        Xc[col] = le.transform([val])[0]

    # impute
    X_imp = pd.DataFrame(imputer.transform(Xc), columns=Xc.columns)

    # scale numeric features
    X_imp[NUM_COLS] = scaler.transform(X_imp[NUM_COLS])

    return X_imp


def predict_price(payload: PredictRequest) -> Dict[str, Any]:
    X = feature_engineer_row(payload)
    X_scaled = preprocess_one(X)

    # model predicts log1p(price)
    pred_log = float(model.predict(X_scaled.values)[0])
    pred = float(np.expm1(pred_log))

    return {
        "predicted_price": pred,
        "predicted_log1p_price": pred_log
    }


@app.get("/")
def root():
    return {
        "message": "House Price Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health")
def health():
    if LOAD_ERROR is not None:
        return {"status": "error", "detail": LOAD_ERROR}
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if LOAD_ERROR is not None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {LOAD_ERROR}")

    try:
        return predict_price(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/schema")
def schema():
    # Optional: return your schema file contents if you have it
    if SCHEMA_PATH.exists():
        return {"schema_file": str(SCHEMA_PATH)}
    return {"schema_file": None, "note": "schema/feature_schema.json not found (optional)."}
