from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)

model = joblib.load("stroke_model.pkl")
THRESHOLD = 0.029698

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    X = np.array([
        float(data["gender"]),
        float(data["age"]),
        float(data["bmi"]),
        float(data["sbp"]),
        float(data["dbp"]),
        float(data["glucose"]),
        float(data["smoking"]),
        float(data["drinking"])
    ]).reshape(1,-1)

    prob = float(model.predict_proba(X)[0][1]) * 100
    prob = round(prob,2)

    # 위험도 등급
    if prob >= 20:
        risk_class = "result-high"
        risk_text = "고위험"
    elif prob >= 10:
        risk_class = "result-medium"
        risk_text = "중위험"
    else:
        risk_class = "result-low"
        risk_text = "저위험"

    return jsonify({
        "prob": prob,
        "risk_text": risk_text,
        "risk_class": risk_class,
        "advice": "AI 건강 조언: 물 충분히 마시고 절주/저염 식단 유지하세요."
    })
