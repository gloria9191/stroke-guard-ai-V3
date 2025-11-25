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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob):
    if not GROQ_API_KEY:
        return "AI 조언을 불러올 수 없습니다."

    prompt = f"""
    사용자의 뇌졸중 발병 확률은 {prob}% 입니다.
    건강 관리, 생활습관, 운동, 식습관, 주의할 점을
    5줄 이내의 한국어로 자연스럽게 조언해 주세요.
    """

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mixtral-8x7b-32768",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=15
        )
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except:
        return "AI 조언 생성 중 오류가 발생했습니다."


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
