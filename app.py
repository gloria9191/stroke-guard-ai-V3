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
        사용자의 뇌졸중 발병 확률은 {prob}%입니다.
        한국인 생활습관 기준으로,
        나트륨, 운동, 금연, 혈압·혈당관리 등을 포함해
        5줄 이내의 고급 건강 조언을 한국어로 제공하세요.
        절대로 영어·기호·한자 섞어서 쓰지 말고 자연스럽게 한국어로만 작성하세요.
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
