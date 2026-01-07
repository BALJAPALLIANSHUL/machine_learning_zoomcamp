
import pickle
from flask import Flask, request, jsonify

MODEL_PATH = "models/model.bin"

with open(MODEL_PATH, "rb") as f:
    dv, model = pickle.load(f)

app = Flask("capstone")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    X = dv.transform([data])
    y_pred = model.predict(X)[0]
    return jsonify({"prediction": int(y_pred)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
