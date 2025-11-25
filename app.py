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
THRESHOLD = 0.029698   # recall 0.915 때 쓰던 threshold

# ----------------------------
# 2) LLM 조언 생성 (Groq)
# ----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob):
    if not GROQ_API_KEY:
        return "AI 코멘트를 불러올 수 없습니다."

    prompt = f"""
    사용자의 뇌졸중 발병 확률이 {prob}%입니다.
    한국어로 자연스럽게, 의료적 조언을 5문장 이내로 작성하세요.
    불필요한 기호(*, #)를 절대 넣지 마세요.
    """

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "mixtral-8x7b-32768",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return "AI 조언 생성 중 오류가 발생했습니다."


# ----------------------------
# 3) 라우트
# ----------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def predict():
    try:
        gender = int(request.form.get("gender"))
        age = float(request.form.get("age"))
        bmi = float(request.form.get("bmi"))
        sbp = float(request.form.get("sbp"))
        dbp = float(request.form.get("dbp"))
        glucose = float(request.form.get("glucose"))
        smoking = int(request.form.get("smoking"))
        drinking = int(request.form.get("drinking"))
    except:
        return "입력 오류"

    X = np.array([[gender, age, bmi, sbp, dbp, glucose, smoking, drinking]])
    prob = float(model.predict_proba(X)[0][1]) * 100

    # 위험군 분류
    if prob >= THRESHOLD * 100:
        risk_text = "고위험"
        risk_class = "linear-gradient(135deg,#ff6b6b,#feca57)"
    elif prob >= 3:
        risk_text = "중위험"
        risk_class = "linear-gradient(135deg,#feca57,#ff9ff3)"
    else:
        risk_text = "저위험"
        risk_class = "linear-gradient(135deg,#1dd1a1,#54a0ff)"

    advice = generate_advice(round(prob, 2))

    return render_template(
        "index.html",
        prob=round(prob, 2),
        risk_text=risk_text,
        risk_class=risk_class,
        advice=advice,
        show_result=True
    )


if __name__ == "__main__":
    app.run()
