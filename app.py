from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# 모델 로드
model = joblib.load("stroke_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        print("📌 /predict called")
        print("📌 request.json:", request.json)

        data = request.json
        if data is None:
            print("❌ request.json is None")
            return jsonify({"error": "No JSON received"}), 400

        X = np.array([
            float(data.get("gender", 0)),
            float(data.get("age", 0)),
            float(data.get("bmi", 0)),
            float(data.get("sbp", 0)),
            float(data.get("dbp", 0)),
            float(data.get("glucose", 0)),
            float(data.get("smoking", 0)),
            float(data.get("drinking", 0))
        ]).reshape(1, -1)

        print("📌 Input X:", X)

        prob = float(model.predict_proba(X)[0][1]) * 100
        prob = round(prob, 2)
        print("📌 prob:", prob)

        if prob >= 20:
            risk_class = "result-high"
            risk_text = "고위험"
        elif prob >= 10:
            risk_class = "result-medium"
            risk_text = "중위험"
        else:
            risk_class = "result-low"
            risk_text = "저위험"

        return jsonify({
            "prob": prob,
            "risk_text": risk_text,
            "risk_class": risk_class,
            "advice": "AI 건강 조언: 물 충분히 마시고 절주/저염 식단 유지하세요."
        })

    except Exception as e:
        print("🔥 ERROR in /predict:", str(e))
        return jsonify({"error": str(e)}), 500
