from fastapi import FastAPI
from src.inference import InferencePipeline

app = FastAPI()
pipeline = InferencePipeline()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    prob = pipeline.predict_proba(data)
    return {
        "dropout_probability": prob,
        "dropout": prob >= 0.5
    }
