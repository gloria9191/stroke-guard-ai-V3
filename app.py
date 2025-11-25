from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import requests

app = Flask(__name__)

# MODEL LOAD
model = joblib.load("stroke_model.pkl")
THRESHOLD = 0.029698

# ADVICE
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
                "messages": [{"role": "user", "content": f"뇌졸중 발병 확률 {prob}%일 때 생활습관, 운동, 식습관 조언 5줄"}],
                "max_tokens": 200
            }
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return "⚠️ AI 조언 생성 중 오류가 발생했습니다."


# ROUTES
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
    except Exception as e:
        return f"입력 오류 발생: {e}"

    # 모델 입력 (정확히 8개)
    X = np.array([[gender, age, bmi, sbp, dbp, glucose, smoking, drinking]])

    prob = model.predict_proba(X)[0][1]
    prob_percent = round(prob * 100, 2)

    risk_class = "high" if prob >= THRESHOLD else "low"
    risk_text  = "고위험" if prob >= THRESHOLD else "저위험"

    advice = generate_advice(prob_percent)

    return render_template(
        "result.html",
        prob=prob_percent,
        risk_class=risk_class,
        risk_text=risk_text,
        advice=advice
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
