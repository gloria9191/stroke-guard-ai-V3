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


def get_val(form, key):
    try:
        return float(form.get(key, 0))
    except:
        return 0.0


def get_llm_advice(data, prob):
    if not GROQ_API_KEY:
        return "혈압·혈당 관리, 운동, 절주·금연이 중요합니다."

    prompt = f"""
당신은 서울대병원 신경과 전문의입니다.

환자 정보:
- 나이: {data['age']}세 | 성별: {'남성' if data['gender']==1 else '여성'}
- BMI: {data['bmi']:.1f} | 혈압: {data['sbp']}/{data['dbp']}
- 혈당: {data['glucose']:.1f}
- 흡연: {data['smoking']} | 음주: {data['drinking']}

뇌졸중 발생 확률은 {prob:.1f}%입니다.
생활습관 조언을 5문장으로 작성해주세요.
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
            timeout=15
        )
        return r.json()["choices"][0]["message"]["content"].replace("\n", "<br>")
    except:
        return "일시적 지연입니다. 기본적인 혈압·혈당 관리가 필요합니다."


@app.route("/", methods=["GET"])
def survey_page():
    return render_template("survey.html")


@app.route("/predict", methods=["POST"])
def predict():
    d = {
        "gender": get_val(request.form, "gender"),
        "age": get_val(request.form, "age"),
        "bmi": get_val(request.form, "bmi"),
        "sbp": get_val(request.form, "sbp"),
        "dbp": get_val(request.form, "dbp"),
        "glucose": get_val(request.form, "glucose"),
        "smoking": get_val(request.form, "smoking"),
        "drinking": get_val(request.form, "drinking"),
    }

    X = np.array([[d[k] for k in d]])
    prob = float(model.predict_proba(X)[0][1] * 100)

    if prob > 70:
        risk_class = "result-high"
        risk_text = "고위험"
    elif prob > 30:
        risk_class = "result-medium"
        risk_text = "주의 필요"
    else:
        risk_class = "result-low"
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
