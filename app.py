from flask import Flask, render_template, request
import pickle
import numpy as np
import os

MODEL_PATH = "stroke_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"MODEL NOT FOUND: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

FEATURES = ["Sex","Age","BMI","SBP","DBP","Glucose","Smoking","Alcohol"]

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    explanation = None
    if request.method == "POST":
        values = []
        for f in FEATURES:
            values.append(float(request.form[f]))

        arr = np.array([values])
        prob = float(model.predict_proba(arr)[0][1])
        result = round(prob*100, 2)

        if prob >= 0.20:
            explanation = "⚠ 높은 위험군: 전문 검사 및 추가적인 진료가 권장됩니다."
        elif prob >= 0.10:
            explanation = "중간 위험군: 생활습관 개선 및 정기 검진을 추천드립니다."
        else:
            explanation = "낮은 위험군: 현재 위험은 낮지만 지속적인 관리가 중요합니다."

    return render_template("index.html", result=result, explanation=explanation)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
