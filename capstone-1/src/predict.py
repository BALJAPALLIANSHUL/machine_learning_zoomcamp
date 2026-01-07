# src/predict.py

# This is a simple script to test the inference pipeline independently.
from src.inference import InferencePipeline

pipeline = InferencePipeline()

sample = {
    "Age at enrollment": 22,
    "Gender": 1,
    "Debtor": 0,
    "Scholarship holder": 1,
    "Tuition fees up to date": 1
}

prob = pipeline.predict_proba(sample)

print("Dropout probability:", prob)
print("Dropout:", prob >= 0.5)
