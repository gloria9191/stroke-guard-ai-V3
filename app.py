from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)

# ----------------------------
# 1) 모델 로드
# ----------------------------
model = joblib.load("stroke_model.pkl")
THRESHOLD = 0.029698   # recall 0.915 달성 threshold

# ----------------------------
# 2) GROQ LLM
# ----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob):
    if not GROQ_API_KEY:
        return "⚠️ AI 코멘트를 불러올 수 없습니다."

    prompt = f"""
    사용자의 뇌졸중 발병 확률이 {prob}%로 계산되었습니다.
    의료 지식 기반으로 생활습관, 식단, 운동, 주의해야 할 증상 등을 포함하여
    5줄 정도로 구체적인 조언을 해주세요.
    """

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200
            }
        )
        res = r.json()
        return res["choices"][0]["message"]["content"]
    except:
        return "⚠️ AI 조언 생성 중 오류가 발생했습니다."


# ----------------------------
# 3) ROUTES
# ----------------------------
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

    X = np.array([[gender, age, bmi, sbp, dbp, glucose, smoking, drinking]])

    prob = model.predict_proba(X)[0][1]
    prob_percent = round(prob * 100, 2)

    if prob >= THRESHOLD:
        risk_class = "high"
        risk_text = "고위험"
    else:
        risk_class = "low"
        risk_text = "저위험"

    advice = generate_advice(prob_percent)

    return render_template(
        "result.html",
        prob=prob_percent,
        risk_class=risk_class,
        risk_text=risk_text,
        advice=advice
    )


# ----------------------------
# 4) MAIN
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
