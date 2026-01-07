# src/predict.py

import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

MODEL_PATH = "models/full_dropout_pipeline.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Student Dropout Prediction API")

class StudentInput(BaseModel):
    data: Dict[str, Any]

@app.post("/predict")
def predict(student: StudentInput):
    prediction = model.predict(student.data)[0]

    return {
        "dropout_prediction": int(prediction)
    }
