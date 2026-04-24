from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("pulse_model.pkl")

# Add this
@app.route("/", methods=["GET"])
def home():
    return "Pulse Diagnosis API is running"

# ML API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = [
        data["mean"],
        data["std"],
        data["variance"],
        data["max"],
        data["min"],
        data["range"],
        data["energy"]
    ]

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]

    if prediction == 0:
        status = "Normal"
        risk = "Low"
    else:
        status = "Abnormal"
        risk = "High"

    return jsonify({
        "status": status,
        "risk_level": risk
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    #app.run(host="127.0.0.1", port=5000)