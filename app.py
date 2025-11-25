from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)

model = joblib.load("stroke_model.pkl")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob):
    if not GROQ_API_KEY:
        return "AI 조언을 가져올 수 없습니다."

    prompt = f"""
    사용자의 뇌졸중 발병 확률은 {prob}% 입니다.
    식습관, 운동, 위험 신호, 생활습관 관리 조언을 한국어로 5줄 정도로 알려주세요.
    """

    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
    )
    return r.json()["choices"][0]["message"]["content"].strip()

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
    ]).reshape(1, -1)

    prob = float(model.predict_proba(X)[0][1]) * 100
    prob = round(prob, 2)

    advice = generate_advice(prob)

    return jsonify({
        "prob": prob,
        "advice": advice
    })
