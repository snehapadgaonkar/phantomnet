"""
api.py
-------
Flask REST API for real-time attack detection using PhantomNet.
"""

from flask import Flask, request, jsonify
import pandas as pd
import torch
from infer import load_model, infer_from_dataframe
from data_loader import load_config

app = Flask(__name__)

CFG = load_config("config.yaml")
MODEL, SCALER, DEVICE = load_model("artifacts/models/phantomnet_best.pt", CFG)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "PhantomNet API is running"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            df = pd.read_csv(request.files["file"])
        else:
            data = request.json
            df = pd.DataFrame(data)

        results = infer_from_dataframe(df, MODEL, SCALER, DEVICE)
        return jsonify(results.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
