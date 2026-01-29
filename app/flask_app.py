from flask import Flask, request, jsonify
from app.inference import predict
import sys
import os


app = Flask(__name__)

# health check for ALB
@app.route("/", methods=["GET"])
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["GET", "POST"])
def predict_churn():
    data = request.get_json()
    result = predict(data)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
