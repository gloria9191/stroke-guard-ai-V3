from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import requests

app = Flask(__name__)

model = joblib.load("stroke_model.pkl")
THRESHOLD = 0.029698

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob):
    if not GROQ_API_KEY:
        return "⚠️ AI 코멘트를 불러올 수 없습니다."
    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": f"뇌졸중 발병 확률 {prob}%일 때 조언 5줄"}],
                "max_tokens": 150
            }
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return "⚠️ AI 조언 생성 실패"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    print("📌 FORM DATA:", dict(request.form))

    def safe_float(v):
        try:
            return float(v)
        except:
            return 0.0

    gender   = safe_float(request.form.get("gender"))
    age      = safe_float(request.form.get("age"))
    bmi      = safe_float(request.form.get("bmi"))
    sbp      = safe_float(request.form.get("sbp"))
    dbp      = safe_float(request.form.get("dbp"))
    glucose  = safe_float(request.form.get("glucose"))
    smoking  = safe_float(request.form.get("smoking"))
    drinking = safe_float(request.form.get("drinking"))

    try:
        X = np.array([[gender, age, bmi, sbp, dbp, glucose, smoking, drinking]])
        prob = model.predict_proba(X)[0][1]
    except Exception as e:
        return f"<h1>⚠️ 예측 오류: {e}</h1>"

    prob_percent = round(prob * 100, 2)
    risk_text = "고위험" if prob >= THRESHOLD else "저위험"
    risk_class = "high" if prob >= THRESHOLD else "low"
    advice = generate_advice(prob_percent)

    return render_template(
        "result.html",
        prob=prob_percent,
        risk_text=risk_text,
        risk_class=risk_class,
        advice=advice
    )
