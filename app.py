from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)

model = joblib.load("stroke_model.pkl")
THRESHOLD = 0.029698
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob):
    if not GROQ_API_KEY:
        return "AI 코멘트를 불러올 수 없습니다."

    prompt = f"""
    사용자의 뇌졸중 발병 확률이 {prob}%로 계산되었습니다.
    식습관, 운동, 위험요인 관리 등 구체적 조언을 4~5줄로 작성해주세요.
    """

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model":"llama3-8b-8192", "messages":[{"role":"user","content":prompt}]}
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return "생활습관 개선이 중요합니다. 혈압/혈당/흡연 여부를 관리하시기 바랍니다."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender = float(request.form.get("gender", 1))
        age = float(request.form.get("age", 60))
        bmi = float(request.form.get("bmi", 23))
        sbp = float(request.form.get("sbp", 120))
        dbp = float(request.form.get("dbp", 80))
        glucose = float(request.form.get("glucose", 90))
        smoking = float(request.form.get("smoking", 0))
        drinking = float(request.form.get("drinking", 0))

        x = np.array([[gender, age, bmi, sbp, dbp, glucose, smoking, drinking]])
        prob = model.predict_proba(x)[0][1]

        prob_percent = round(prob * 100, 2)

        risk_class = "high" if prob > THRESHOLD else "low"
        risk_text = "고위험" if prob > THRESHOLD else "저위험"

        advice = generate_advice(prob_percent)

        return jsonify({
            "prob": prob_percent,
            "risk_text": risk_text,
            "risk_class": risk_class,
            "advice": advice
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
