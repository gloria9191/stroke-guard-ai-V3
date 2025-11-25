from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)

# 모델 로드
model = joblib.load("stroke_model.pkl")

THRESHOLD = 0.029698

# GROQ LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob):
    if not GROQ_API_KEY:
        return "생활습관 관리와 정기검진을 통해 꾸준히 건강을 지켜보세요."

    prompt = f"""
    사용자의 뇌졸중 발병 확률이 {prob}%로 계산되었습니다.
    의료 지식을 기반으로 생활습관, 식단, 운동, 주의해야 할 증상 등을 포함하여
    5줄 정도로 구체적인 조언을 해주세요.
    """

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            },
            timeout=10
        )

        resp = r.json()
        return resp["choices"][0]["message"]["content"]

    except Exception as e:
        # 절대 JSON을 깨지 않음
        return "건강을 위해 규칙적인 운동, 절주, 금연, 충분한 수면을 유지해보세요."


    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model":"llama-3.1-70b-versatile",
                "messages":[{"role":"user","content":prompt}],
                "temperature":0.7
            }
        )
        return r.json()["choices"][0]["message"]["content"]

    except Exception:
        return "건강관리: 규칙적인 운동과 식습관 관리를 통해 위험도를 낮출 수 있습니다."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
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
    except:
        return jsonify({"error":"입력값 파싱 오류"}), 400

    prob = float(model.predict_proba(X)[0][1]) * 100
    prob = round(prob, 2)

    # 위험군 판정
    if prob >= 20:
        risk_class = "result-high"
        risk_text = "고위험"
    elif prob >= 10:
        risk_class = "result-medium"
        risk_text = "중위험"
    else:
        risk_class = "result-low"
        risk_text = "저위험"

    # AI 조언 생성
    advice = generate_advice(prob)

    return jsonify({
        "prob": prob,
        "risk_text": risk_text,
        "risk_class": risk_class,
        "advice": advice
    })

