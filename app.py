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
        return "생활습관 관리로 건강을 유지하세요."

    prompt = f"""사용자의 뇌졸중 발병 확률은 {prob}% 입니다.
고혈압·고혈당·흡연·음주·스트레스·운동 등을 고려해
5줄 이내의 구체적이고 실천 가능한 건강 조언을 한국어로 작성하세요.
불필요한 특수문자(*, •, -, 숫자 자동 생성 등) 절대 넣지 마세요.
"""

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

