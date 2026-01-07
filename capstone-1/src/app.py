from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

from src.inference import InferencePipeline

app = FastAPI()
pipeline = InferencePipeline()


class EncodedInput(BaseModel):
    features: dict[str, float]


@app.post("/predict")
def predict(data: EncodedInput):
    prob = pipeline.predict_proba(data.features)
    return {
        "dropout_probability": prob,
        "dropout": prob >= 0.5
    }
