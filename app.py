from pathlib import Path
import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_heart_disease_model.joblib"
FEATURES = ["max_heart_rate_achieved", "exercise_induced_angina", "st_depression", "age", "ldl", "hdl", "resting_bp_systolic", "exercise_minutes_per_week", "hba1c", "bmi"]
app = FastAPI(title="CardioCheck — Educational Assessment")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
model = None
model_error = None
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Add your newly trained 10-feature best_heart_disease_model.joblib to the project root")
    model = joblib.load(MODEL_PATH)  # Only load trusted artifacts
    names = list(getattr(model, "feature_names_in_", []))
    if names != FEATURES:
        raise ValueError(f"Incompatible model features. Expected {FEATURES}; got {names}")
except Exception as exc:
    model_error = str(exc)
    logging.warning("Model unavailable: %s", model_error)
    model = None

class PatientData(BaseModel):
    max_heart_rate_achieved: float = Field(ge=40, le=250)
    exercise_induced_angina: int = Field(ge=0, le=1)
    st_depression: float = Field(ge=0, le=10)
    age: float = Field(ge=18, le=110)
    ldl: float = Field(ge=0, le=400)
    hdl: float = Field(ge=1, le=150)
    resting_bp_systolic: float = Field(ge=60, le=260)
    exercise_minutes_per_week: float = Field(ge=0, le=2000)
    hba1c: float = Field(ge=3, le=20)
    bmi: float = Field(ge=10, le=80)

@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")

@app.get("/api/status")
def status():
    return {"model_ready": model is not None, "message": "Ready" if model else model_error}

@app.post("/api/predict")
def predict(patient: PatientData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable: " + str(model_error))
    data = patient.model_dump()
    frame = pd.DataFrame([[data[k] for k in FEATURES]], columns=FEATURES)
    try:
        predicted = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0][1])
    except Exception:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Unable to generate prediction. Check model compatibility and input encodings.")
    return {"prediction": predicted, "risk_label": "Positive classification" if predicted else "Negative classification", "model_probability": round(probability, 4), "note": "Educational model output, not a diagnosis or validated personal risk estimate."}
