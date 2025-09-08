
import os, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

MODEL_PATH = os.environ.get("MODEL_PATH","/models/model.pkl")
app = FastAPI(title="Penguins Inference API")

model = None
meta = {}

class PenguinFeatures(BaseModel):
    island: Optional[str] = None
    bill_length_mm: Optional[float] = None
    bill_depth_mm: Optional[float] = None
    flipper_length_mm: Optional[float] = None
    body_mass_g: Optional[float] = None
    sex: Optional[str] = None

def try_load_model():
    global model, meta
    if model is None and os.path.exists(MODEL_PATH):
        payload = joblib.load(MODEL_PATH)
        model = payload["pipeline"]
        meta = {k:v for k,v in payload.items() if k!="pipeline"}

@app.on_event("startup")
def startup_event():
    try_load_model()

@app.get("/")
def root():
    exists = os.path.exists(MODEL_PATH)
    return {"status":"ok", "model_loaded": model is not None, "model_file_exists": exists, "meta": meta if model else None}

@app.post("/predict")
def predict(f: PenguinFeatures):
    try_load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Train the model first (run the Airflow DAG).")
    x = [{
        "island": f.island,
        "bill_length_mm": f.bill_length_mm,
        "bill_depth_mm": f.bill_depth_mm,
        "flipper_length_mm": f.flipper_length_mm,
        "body_mass_g": f.body_mass_g,
        "sex": f.sex,
    }]
    try:
        X = pd.DataFrame(x)
        y = model.predict(X)[0]
        proba = None
        try:
            probs = model.predict_proba(X)[0]
            labels = getattr(model, "classes_", None)
            if labels is not None:
                proba = {str(lbl): float(p) for lbl, p in zip(labels, probs)}
        except Exception:
            pass
        return {"prediction": str(y), "proba": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
