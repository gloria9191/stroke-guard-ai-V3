import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string
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

# 안전하게 form value 가져오기
def get_val(form, key):
    try:
        return float(form.get(key, 0))
    except:
        return 0.0


# LLM 조언 생성 함수
def get_llm_advice(data, prob):
    if not GROQ_API_KEY:
        return "혈압과 혈당 관리, 규칙적인 운동, 금연·절주를 실천해보세요.<br>작은 변화가 큰 차이를 만듭니다."

    prompt = f"""
당신은 서울대병원 신경과 전문의이자 가장 따뜻한 의사입니다.

환자 정보:
- 나이: {data['age']}세 | 성별: {'남성' if data['gender']==1 else '여성'}
- BMI: {data['bmi']:.1f} | 혈압: {data['sbp']}/{data['dbp']} mmHg | 공복혈당: {data['glucose']:.1f} mg/dL
- 흡연: {'합니다' if data['smoking']==1 else '하지 않습니다'} | 음주: {'합니다' if data['drinking']==1 else '하지 않습니다'}

뇌졸중 예측 확률은 {prob:.1f}%입니다.
현실적이고 따뜻한 생활습관 조언을 5문장 정도로 부탁드립니다.
"""

    try:
        r = requests.post(GROQ_URL, json={
            "model": "llama-3.1-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "max_tokens": 350
        }, headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, timeout=15)

        return r.json()["choices"][0]["message"]["content"].replace("\n", "<br>")

    except Exception as e:
        print("LLM ERROR:", e)
        return "현재 서버가 혼잡합니다.<br>규칙적 운동, 금연·절주, 혈압·혈당 관리가 중요합니다."


# ===============================
# HTML
# ===============================

HTML = """  
(여기엔 네가 쓴 긴 HTML 그대로 들어감 — 생략 X, 위 코드 그대로 복사)
"""

# ===============================
# 라우트
# ===============================

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            d = {
                "gender": get_val(request.form, "gender"),
                "age": get_val(request.form, "age"),
                "bmi": get_val(request.form, "bmi"),
                "sbp": get_val(request.form, "sbp"),
                "dbp": get_val(request.form, "dbp"),
                "glucose": get_val(request.form, "glucose"),
                "smoking": get_val(request.form, "smoking"),
                "drinking": get_val(request.form, "drinking")
            }

            X = np.array([[d[k] for k in d]])
            prob = float(model.predict_proba(X)[0][1] * 100)

            # 등급
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

            return render_template_string(
                HTML,
                prob=f"{prob:.1f}",
                risk_class=rc,
                risk_text=rt,
                advice=advice
            )

        except Exception as e:
            return f"<h1>오류 발생</h1><p>{e}</p>"

    return HTML


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
