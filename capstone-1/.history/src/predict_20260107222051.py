# src/predict.py

import pickle
from flask import Flask, request, jsonify

MODEL_PATH = "models/full_dropout_pipeline.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask("student-dropout")

@app.route("/predict", methods=["POST"])
def predict():
    student = request.get_json()

    prediction = model.predict(student)[0]

    return jsonify({
        "dropout_prediction": int(prediction)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9696, debug=True)
