from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import requests
import os

app = Flask(__name__)

model = joblib.load("stroke_model.pkl")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def generate_advice(prob):
    if not GROQ_API_KEY:
        return "AI 조언을 불러올 수 없습니다."

    prompt = f"""
    사용자의 뇌졸중 발생 위험도는 {prob}%입니다.
    의료 지식을 기반으로 식습관, 운동, 생활 습관, 주의해야 할 증상을 포함한 맞춤형 조언을 작성해주세요.
    5줄 이내 한국어로 자연스럽게.
    """

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }
        )
        return r.json()["choices"][0]["message"]["content"]
    except:
        return "AI 조언 생성 중 오류가 발생했습니다."


def preprocess(data):
    X = np.array([
        float(data["gender"]),
        float(data["age"]),
        float(data["bmi"]),
        float(data["sbp"]),
        float(data["dbp"]),
        float(data["glucose"]),
        float(data["smoking"]),
        float(data["drinking"])
    ]).reshape(1, -1)
    return X


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        X = preprocess(data)

        prob = float(model.predict_proba(X)[0][1]) * 100
        prob = round(prob, 2)

        if prob >= 20:
            risk_class = "result-high"
            risk_text = "고위험"
        elif prob >= 10:
            risk_class = "result-medium"
            risk_text = "중위험"
        else:
            risk_class = "result-low"
            risk_text = "저위험"

        advice = generate_advice(prob)

        return jsonify({
            "prob": prob,
            "risk_text": risk_text,
            "risk_class": risk_class,
            "advice": advice
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run()
