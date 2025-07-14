import joblib
import os
import glob

import pandas as pd 

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from api.schemas import InsuranceInput, PredictionOut

BEST_MODEL_DIR = 'artifacts/best_model'

MODEL_PATH = glob.glob(os.path.join(BEST_MODEL_DIR, "*.joblib"))[0] 

pipeline = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Medical Charges API",
    description="Predict annual medical charge",
    version="v1.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    return open("static/index.html").read()

@app.post("/predict", response_model=PredictionOut)
def predict(payload: InsuranceInput):
    df = pd.DataFrame([payload.model_dump()])
    pred = round(pipeline.predict(df)[0], 2)
    return PredictionOut(charges=float(pred))
