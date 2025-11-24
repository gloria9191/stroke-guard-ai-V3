
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

MODEL_PATH = "stroke_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sex = float(request.form["sex"])
    age = float(request.form["age"])
    bmi = float(request.form["bmi"])
    sbp = float(request.form["sbp"])
    dbp = float(request.form["dbp"])
    glucose = float(request.form["glucose"])
    smoking = float(request.form["smoking"])
    alcohol = float(request.form["alcohol"])

    X = np.array([[sex, age, bmi, sbp, dbp, glucose, smoking, alcohol]])
    prob = model.predict_proba(X)[0][1]
    prob_percent = round(prob * 100, 2)

    comment = []
    if sbp > 130:
        comment.append("혈압이 높습니다.")
    if glucose > 110:
        comment.append("혈당이 높습니다.")
    if bmi > 25:
        comment.append("체중 관리가 필요합니다.")
    if smoking == 1:
        comment.append("흡연은 위험합니다.")
    if alcohol == 1:
        comment.append("음주가 위험도를 높일 수 있습니다.")

    if len(comment) == 0:
        comment = ["전반적으로 건강 지표가 양호합니다."]
    else:
        comment.insert(0, "⚠ 위험 요소 감지:")

    return render_template("result.html", prob=prob_percent, comment=comment)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
