import os
import pickle
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
import requests

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 10000))

print("🔄 Loading stroke_model.pkl ...")
with open("stroke_model.pkl", "rb") as f:
    model = pickle.load(f)
print("✅ 모델 로드 완료")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def get_llm_advice(data, prob):
    if not GROQ_API_KEY:
        return "규칙적 운동, 금연·절주, 혈압·혈당 관리가 중요합니다."

    prompt = f"""
당신은 서울대병원 신경과 전문의입니다.

환자 정보:
- 나이: {data['age']}세
- BMI: {data['bmi']}
- 혈압: {data['sbp']}/{data['dbp']}
- 혈당: {data['glucose']}
- 흡연: {data['smoking']}
- 음주: {data['drinking']}

뇌졸중 확률은 {prob:.1f}% 입니다.
5문장 정도로 생활습관 조언을 해주세요.
"""

    try:
        r = requests.post(GROQ_URL, json={
            "model": "llama-3.1-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "max_tokens": 350
        }, headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, timeout=20)

        return r.json()["choices"][0]["message"]["content"].replace("\n", "<br>")

    except Exception as e:
        print("LLM ERROR:", e)
        return "AI 조언 생성 오류 발생 – 기본 건강 조언을 참고해주세요."


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    d = {
        "gender": float(request.form.get("gender")),
        "age": float(request.form.get("age")),
        "bmi": float(request.form.get("bmi")),
        "sbp": float(request.form.get("sbp")),
        "dbp": float(request.form.get("dbp")),
        "glucose": float(request.form.get("glucose")),
        "smoking": float(request.form.get("smoking")),
        "drinking": float(request.form.get("drinking"))
    }

    X = np.array([[d[k] for k in d]])
    prob = float(model.predict_proba(X)[0][1] * 100)

    if prob > 70:
        risk_class = "high"
        risk_text = "고위험"
    elif prob > 30:
        risk_class = "medium"
        risk_text = "주의 필요"
    else:
        risk_class = "low"
        risk_text = "안전"

    advice = get_llm_advice(d, prob)

    return render_template(
        "result.html",
        prob=f"{prob:.1f}",
        risk_class=risk_class,
        risk_text=risk_text,
        advice=advice
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
