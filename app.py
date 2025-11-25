from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load("stroke_model.pkl")
THRESHOLD = 0.029698

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    age = float(data["age"])
    fbs = float(data["fbs"])

    X = [[age, fbs]]
    prob = float(model.predict_proba(X)[0][1])

    prob_percent = round(prob * 100, 2)

    return jsonify({
        "prob": prob_percent,
        "advice": "생활습관 개선 조언: 물 많이 마시고 적절한 운동!"  # 임시
    })

# ❗ 절대 넣지 말 것
# if __name__ == "__main__":
#     app.run(debug=True)
