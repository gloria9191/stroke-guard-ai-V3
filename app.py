from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import requests
import os
import threading

app = Flask(__name__)

model = joblib.load("stroke_model.pkl")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def fetch_llm(prob, result_dict):
    """LLM을 별도 스레드에서 요청해서 timeout 방지"""
    try:
        prompt = f"""
        사용자의 뇌졸중 발병 확률은 {prob}% 입니다.
        생활습관, 식단, 운동, 주의해야 할 조기 경고신호를 5줄로 조언하세요.
        """

        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            },
            timeout=5
        )
        result_dict["advice"] = r.json()["choices"][0]["message"]["content"].strip()

    except:
        result_dict["advice"] = "LLM 조언을 불러오지 못했습니다. 입력값을 기반으로 건강관리를 꾸준히 해주세요."


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.get_json()

        # LLM 조언 생성 (네 기존 함수 사용)
        prob = 0   # 모델 제거했으니까 필요 없다면 0 또는 계산된 값 넣기
        advice = generate_advice(prob)

        # risk_text / risk_class 계산 (원래 쓰던 로직 그대로)
        risk_text = "저위험"
        risk_class = "#1dd1a1"

        return render_template(
            "index.html",
            prob=prob,
            risk_text=risk_text,
            risk_class=risk_class,
            advice=advice
        )

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    X = np.array([
        data["gender"],
        data["age"],
        data["bmi"],
        data["sbp"],
        data["dbp"],
        data["glucose"],
        data["smoking"],
        data["drinking"]
    ]).reshape(1, -1).astype(float)

    prob = round(float(model.predict_proba(X)[0][1]) * 100, 2)

    result = {"advice": ""}
    t = threading.Thread(target=fetch_llm, args=(prob, result))
    t.start()
    t.join(timeout=5)

    return jsonify({
        "prob": prob,
        "advice": result["advice"]
    })
