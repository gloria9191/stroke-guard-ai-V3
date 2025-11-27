from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)

# ------------------------------------------------
# 1) 모델 로드
# ------------------------------------------------
print("🔄 Loading stroke_model.pkl ...")
model = joblib.load("stroke_model.pkl")
print("✅ 모델 로드 완료")

THRESHOLD = 0.029698   # recall 0.915 기준 threshold

# ------------------------------------------------
# 2) GROQ API 설정
# ------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob, user_data):
    if not GROQ_API_KEY:
        return "AI 조언을 불러올 수 없습니다."

    prompt = f"""
사용자의 건강 정보를 기반으로 생활습관 개선을 위한 의료 조언을 작성하세요.
외국어 금지. 한자 금지. 기본 문장 부호(.,!? ) 외의 기호 사용 금지.
사용자 정보를 직접 반영하여 의사가 설명하는 말투로 작성하세요.
불필요하게 어려운 표현을 사용하지 말고 5줄 이내로 간결하게 정리하세요.

사용자 정보:
성별 {user_data['gender']}
나이 {user_data['age']}
BMI {user_data['bmi']}
수축기 혈압 {user_data['sbp']}
이완기 혈압 {user_data['dbp']}
공복 혈당 {user_data['glucose']}
흡연 여부 {user_data['smoking']}
음주 빈도 {user_data['drinking']}
"""

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4,
            }
        )
        response = r.json()
        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("LLM Error:", e)
        return "AI 조언을 불러오지 못했습니다."


# ------------------------------------------------
# 3) 라우팅
# ------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        gender    = float(data["gender"])
        age       = float(data["age"])
        bmi       = float(data["bmi"])
        sbp       = float(data["sbp"])
        dbp       = float(data["dbp"])
        glucose   = float(data["glucose"])
        smoking   = float(data["smoking"])
        drinking  = float(data["drinking"])

        X = np.array([[gender, age, bmi, sbp, dbp, glucose, smoking, drinking]])
        proba = model.predict_proba(X)[0][1]
        prob_percent = round(proba * 100, 1)

        risk_class = "result-low"
        risk_text  = "저위험"

        if proba >= THRESHOLD:
            risk_class = "result-high"
            risk_text  = "고위험"

        advice = generate_advice(prob_percent)

        return jsonify({
            "prob": prob_percent,
            "risk_text": risk_text,
            "risk_class": risk_class,
            "advice": advice
        })

    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"})


# ------------------------------------------------
# Render: run() 절대 실행 X
# ------------------------------------------------
if __name__ == "__main__":
    pass
