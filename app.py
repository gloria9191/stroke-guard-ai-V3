from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import requests
import os

app = Flask(__name__)

# 모델 로드
model = joblib.load("stroke_model.pkl")

THRESHOLD = 0.03

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob):
    if not GROQ_API_KEY:
        return "AI 조언 기능을 사용할 수 없습니다."

    prompt = f"""
    사용자의 뇌졸중 발병 확률은 {prob}% 입니다.
    생활습관, 운동, 식단 개선 등을 포함한 구체적인 조언을 5줄 작성해주세요.
    """

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        data = r.json()
        return data["choices"][0]["message"]["content"]

    except:
        return "AI 조언 생성 중 오류가 발생했습니다."


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        gender   = float(request.form.get("gender"))
        age      = float(request.form.get("age"))
        bmi      = float(request.form.get("bmi"))
        sbp      = float(request.form.get("sbp"))
        dbp      = float(request.form.get("dbp"))
        glucose  = float(request.form.get("glucose"))
        smoking  = float(request.form.get("smoking"))
        drinking = float(request.form.get("drinking"))

        X = np.array([[gender, age, bmi, sbp, dbp, glucose, smoking, drinking]])
        prob = float(model.predict_proba(X)[0][1])

        prob_percent = round(prob * 100, 2)

        risk_class = "high" if prob > THRESHOLD else "low"
        risk_text  = "고위험" if prob > THRESHOLD else "저위험"

        advice = generate_advice(prob_percent)

        return render_template(
            "result.html",
            prob=prob_percent,
            risk_class=risk_class,
            risk_text=risk_text,
            advice=advice
        )

    except Exception as e:
        return f"오류 발생: {str(e)}"


if __name__ == "__main__":
    app.run()
