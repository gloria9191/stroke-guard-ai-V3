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
import requests
import json

def generate_advice(prob):
    # 기본 fallback
    fallback = (
        "1. 규칙적인 유산소 운동을 유지하세요.<br>"
        "2. 염분과 당분을 줄이고 채소·과일을 충분히 섭취하세요.<br>"
        "3. 흡연은 즉시 중단하고 음주는 절주하세요.<br>"
        "4. 혈압·혈당을 주기적으로 점검하세요.<br>"
        "5. 체중 관리와 스트레스 조절에 신경 쓰세요."
    )
    
    if not GROQ_API_KEY:
        return fallback

    try:
        prompt = f"""
        사용자의 뇌졸중 발병 확률이 {prob:.2f}%입니다.
        위험도에 맞춰 생활습관·식단·운동 5가지 조언을 한국어로 간결히 써주세요.
        각 항목은 번호를 붙이고 <br>로 줄바꿈해주세요.
        영어, 한자, 이상한 문자 없이 자연스러운 한국어로만 출력하세요.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )

        if response.status_code != 200:
            print("LLM status error:", response.text)
            return fallback

        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()

        if not content:
            return fallback

        # HTML safe 처리
        content = content.replace("\n", "<br>")

        return content

    except Exception as e:
        print("LLM ERROR:", e)
        return fallback

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 이 데이터를 predict로 넘기지 않고 index에서 처리하니까
        # 바로 result.html 렌더링해줘야 함
        data = request.form.to_dict()

        # BMI가 정상적으로 들어왔는지 확인
        # 문자열 -> float 변환
        for k in data:
            try:
                data[k] = float(data[k])
            except:
                pass

        # 모델 예측 (LLM 부분은 아래 predict 로직 그대로 복사하면 됨)
        prob, risk_text, risk_class = model_predict(data)

        # LLM 조언 생성
        advice = generate_advice(prob)

        return render_template("result.html",
                               prob=prob,
                               risk_text=risk_text,
                               risk_class=risk_class,
                               advice=advice)

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

