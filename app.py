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

def generate_advice(prob):
    if not GROQ_API_KEY:
        return "AI 조언 생성이 활성화되지 않았습니다."

    prompt = f"""
    사용자의 건강 정보를 기반으로 의료 전문가의 관점에서 생활습관 개선 조언을 작성하세요.
    외국어 금지.
    한자 금지.
    기본 문장 부호(.,!? ) 외 특수기호 금지.
    지나치게 어려운 표현 금지.
    성별, 만 나이, BMI, 혈압, 혈당, 흡연, 음주 정보를 반영할 것.
    의사가 직접 설명해주는 말투로 5줄 이내로 작성하세요.
    
    사용자 정보:
    성별: {gender}
    나이: {age}
    BMI: {bmi}
    수축기 혈압: {sbp}
    이완기 혈압: {dbp}
    공복 혈당: {glucose}
    흡연 여부: {smoking}
    음주 빈도: {drinking}
    """


    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6
            },
            timeout=15
        )
        ans = r.json()
        return ans["choices"][0]["message"]["content"].strip()
    except Exception:
        return "AI 조언 생성 중 오류가 발생했습니다."


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
