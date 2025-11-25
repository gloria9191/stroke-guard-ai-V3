import os
import pickle
import numpy as np
from flask import Flask, request, render_template
import requests

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 10000))

print("🔄 Loading stroke_model.pkl ...")
with open("stroke_model.pkl", "rb") as f:
    model = pickle.load(f)
print("✅ 모델 로드 완료")

# Groq LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_llm_advice(data, prob):
    if not GROQ_API_KEY:
        return "규칙적인 운동, 혈압/혈당 관리, 절주·금연을 실천해보세요."

    prompt = f"""
당신은 서울대병원 신경과 전문의입니다.

환자 정보:
- 나이: {data['age']}세 | 성별: {'남성' if data['gender']==1 else '여성'}
- BMI: {data['bmi']} | 혈압: {data['sbp']}/{data['dbp']}
- 공복혈당: {data['glucose']}
- 흡연: {'예' if data['smoking']==1 else '아니오'}
- 음주: {'예' if data['drinking']==1 else '아니오'}
예측 확률: {prob:.1f}%

이 환자에게 따뜻하고 현실적인 조언을 5문장으로 주세요.
"""

    try:
        r = requests.post(
            GROQ_URL,
            json={
                "model": "llama-3.1-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 350
            },
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=15,
        )
        return r.json()["choices"][0]["message"]["content"].replace("\n", "<br>")

    except Exception:
        return "일시적 오류입니다. 생활습관 관리가 중요합니다."


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = {k: float(request.form[k]) for k in request.form}

        X = np.array([[d["gender"], d["age"], d["bmi"], d["sbp"], d["dbp"],
                       d["glucose"], d["smoking"], d["drinking"]]])

        prob = model.predict_proba(X)[0][1] * 100

        # 색상 구간
        if prob > 70:
            rc = "linear-gradient(135deg,#ff6b6b,#feca57)"
            rt = "고위험"
        elif prob > 30:
            rc = "linear-gradient(135deg,#feca57,#ff9ff3)"
            rt = "주의 필요"
        else:
            rc = "linear-gradient(135deg,#1dd1a1,#54a0ff)"
            rt = "안전"

        advice = get_llm_advice(d, prob)

        return render_template(
            "index.html",
            prob=f"{prob:.1f}",
            risk_class=rc,
            risk_text=rt,
            advice=advice,
            show_result=True
        )

    except Exception as e:
        return f"<h1>Error</h1><p>{e}</p>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
