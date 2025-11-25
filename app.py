import os
import pickle
import numpy as np
from flask import Flask, request, render_template
import requests

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 10000))

# ================================
# 모델 로드
# ================================
print("🔄 Loading stroke_model.pkl ...")
with open("stroke_model.pkl", "rb") as f:
    model = pickle.load(f)
print("✅ 모델 로드 완료")

# ================================
# Groq LLM
# ================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def safe_float(v):
    try:
        return float(v)
    except:
        return 0.0

def get_llm_advice(data, prob):
    if not GROQ_API_KEY:
        return "혈압·혈당 관리, 금연, 절주, 규칙적인 운동을 실천하면 뇌졸중 위험을 크게 줄일 수 있습니다.<br>꾸준한 관리가 가장 중요합니다."

    prompt = f"""
당신은 서울대병원 신경과 전문의이자 환자에게 가장 따뜻하게 설명하는 의사입니다.

환자 정보
- 나이: {data['age']}
- 성별: {"남성" if data['gender']==1 else "여성"}
- BMI: {data['bmi']}
- 혈압: {data['sbp']}/{data['dbp']}
- 공복혈당: {data['glucose']}
- 흡연: {data['smoking']}
- 음주: {data['drinking']}
- 뇌졸중 예측 확률: {prob:.1f}%

위 환자를 위해 5문장 정도의 따뜻하고 현실적인 생활습관 조언을 의료진의 말투로 작성해 주세요.
"""

    try:
        r = requests.post(
            GROQ_URL,
            json={
                "model": "llama-3.1-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 300
            },
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=12,
        )
        return r.json()["choices"][0]["message"]["content"].replace("\n", "<br>")
    except:
        return "현재 서버가 혼잡합니다.<br>규칙적인 운동과 생활습관 관리가 가장 중요합니다."


# ================================
# GET → 설문 UI만
# POST → 결과 화면만
# ================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("survey.html")

    # POST (predict)
    d = {
        "gender": safe_float(request.form.get("gender")),
        "age": safe_float(request.form.get("age")),
        "bmi": safe_float(request.form.get("bmi")),
        "sbp": safe_float(request.form.get("sbp")),
        "dbp": safe_float(request.form.get("dbp")),
        "glucose": safe_float(request.form.get("glucose")),
        "smoking": safe_float(request.form.get("smoking")),
        "drinking": safe_float(request.form.get("drinking")),
    }

    X = np.array([[d[k] for k in d]])
    prob = float(model.predict_proba(X)[0][1] * 100)

    # 위험도 색상
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
        "result.html",
        prob=f"{prob:.1f}",
        risk_class=rc,
        risk_text=rt,
        advice=advice
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
