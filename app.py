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
    사용자의 뇌졸중 발병 확률은 {prob}% 입니다.

    한국인의 생활습관 기준으로,
    - 식이요법
    - 운동
    - 혈압/혈당 관리
    - 위험 신호 체크
    - 금연/절주 조언

    5줄 이내로 따뜻하고 이해하기 쉬운 문장으로 작성해주세요.
    영어, 특수문자 없이 한국어로만 출력하세요.
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
                "temperature": 0.6,
                "max_tokens": 200
            },
            timeout=12
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

        # 모델 입력
        X = np.array([[gender, age, bmi, sbp, dbp, glucose, smoking, drinking]])
        proba = model.predict_proba(X)[0][1]
        prob_percent = round(proba * 100, 1)

        # 위험군 결정
        risk_class = "result-low"
        risk_text  = "저위험"

        if proba >= THRESHOLD:
            risk_class = "result-high"
            risk_text  = "고위험"

        # AI 조언
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
# 4) Render에서는 run() 금지
# ------------------------------------------------
if __name__ == "__main__":
    pass
