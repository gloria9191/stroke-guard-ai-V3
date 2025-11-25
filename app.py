from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)

# ----------------------------
# 1) MODEL LOAD
# ----------------------------
try:
    model = joblib.load("stroke_model.pkl")
    print("MODEL LOADED")
except Exception as e:
    print("===================")
    print("MODEL LOAD ERROR:", e)
    print("===================")

THRESHOLD = 0.029698

# ----------------------------
# 2) GROQ LLM
# ----------------------------
# ----------------------------
# 2) GROQ LLM
# ----------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob):
    if not GROQ_API_KEY:
        return "⚠️ AI 코멘트를 불러올 수 없습니다."

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }

        prompt = f"""
        당신은 전문 의료 상담 AI입니다.

        다음 사용자의 뇌졸중 발병 위험은 {prob}% 입니다.
        출력 형식은 반드시 아래 조건을 지키세요.
        
        [조건]
        1) 한국어로만 작성
        2) 1~5번 항목 형태 유지
        3) 번호 뒤에는 마침표만 사용 (예: 1.)
        4) 절대 영어/중국어/일본어 섞이지 않게
        5) 과학적 근거 기반 조언으로 고급스럽고 자연스럽게 작성
        6) 문장은 모두 존댓말
        7) "이 모델은 의료 목적이 아니다" 같은 문구 금지
        
        [예시 출력 형식]
        1. 나트륨 섭취를 줄이고 식단을 균형 있게 유지하세요.
        2. 규칙적인 유산소 운동을 주 3~5회 실천하세요.
        3. 혈압·혈당을 정기적으로 모니터링하세요.
        4. 금연을 유지하고 음주는 절제하세요.
        5. 스트레스 관리와 충분한 수면을 유지하세요.
        
        위 조건을 지켜서, 사용자의 위험도에 맞는 맞춤형 조언 5가지를 작성하세요.
        """


        data = {
            "model": "llama3-8b-8192",    # 🔥 현재 Groq에서 가장 안정적으로 지원되는 모델
            "messages": [
                {"role": "system", "content": "당신은 전문 의료 상담가입니다."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.6
        }

        res = requests.post(url, headers=headers, json=data)
        
        if res.status_code != 200:
            return "⚠️ AI 조언 생성 중 오류가 발생했습니다."

        out = res.json()
        return out["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return "⚠️ AI 조언 생성 중 오류가 발생했습니다."


# ----------------------------
# 3) HOME
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ----------------------------
# 4) PREDICT (JSON)
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        X = np.array([[ 
            float(data["gender"]),
            float(data["age"]),
            float(data["bmi"]),
            float(data["sbp"]),
            float(data["dbp"]),
            float(data["glucose"]),
            float(data["smoking"]),
            float(data["drinking"]),
        ]])

        prob = float(model.predict_proba(X)[0][1])
        prob_percent = round(prob * 100, 2)

        # 위험군 판단
        if prob >= THRESHOLD:
            risk_class = "result-high"
            risk_text = "고위험"
        else:
            risk_class = "result-low"
            risk_text = "저위험"

        advice = generate_advice(prob_percent)

        return jsonify({
            "prob": prob_percent,
            "risk_class": risk_class,
            "risk_text": risk_text,
            "advice": advice
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


# ----------------------------
# MAIN (절대 app.run() 넣지 마세요)
# ----------------------------
if __name__ == "__main__":
    app.run()
