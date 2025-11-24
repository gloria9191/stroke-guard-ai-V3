import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string
import requests

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 10000))

# 모델 로드
with open("stroke_model.pkl", "rb") as f:
    model = pickle.load(f)

# Groq LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_llm_advice(data, prob):
    if not GROQ_API_KEY:
        return "혈압·혈당 관리, 금연·절주, 매일 30분 걷기를 권장드립니다.<br>정기 검진 잊지 마세요!"

    prompt = f"""서울대병원 신경과 전문의입니다. 다음 환자분께 따뜻한 조언 부탁드립니다:
- 성별: {'남성' if data['gender']==1 else '여성'}
- 나이: {data['age']}세
- BMI: {data['bmi']:.1f}, 혈압: {data['sbp']}/{data['dbp']} mmHg
- 공복혈당: {data['glucose']:.1f} mg/dL
- 흡연: {'합니다' if data['smoking']==1 else '하지 않습니다'}
- 음주: {'합니다' if data['drinking']==1 else '하지 않습니다'}
- 뇌졸중 위험도: {prob:.1f}%

현실적이고 구체적인 생활습관 조언 4~6문장으로 부탁드립니다."""
    try:
        r = requests.post(GROQ_URL, json={
            "model": "llama-3.1-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, "max_tokens": 300
        }, headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, timeout=20)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].replace("\n", "<br>")
    except:
        pass
    return "현재 서버가 혼잡합니다.<br>혈압·혈당 관리와 규칙적인 운동, 금연·절주를 권장드립니다."

HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>StrokeGuard AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {font-family:'Noto Sans KR',sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);color:white}
        .hero {padding:100px 0;text-align:center;background:rgba(0,0,0,0.6);border-radius:0 0 50px 50px}
        .card {background:rgba(255,255,255,0.97);color:#333;border-radius:25px;box-shadow:0 20px 40px rgba(0,0,0,0.2)}
        .btn-step {background:#5e42a6;color:white;padding:15px 60px;border-radius:50px;font-size:1.3rem}
        .btn-opt {width:48%;padding:20px;font-size:1.3rem;border:3px solid #5e42a6;border-radius:15px;background:white}
        .btn-opt.active {background:#5e42a6;color:white}
        .progress {height:10px;border-radius:10px}
        .result-high {background:#e74c3c}
        .result-medium {background:#f39c12}
        .result-low {background:#27ae60}
    </style>
</head>
<body>
<div class="hero">
    <h1 class="display-3 fw-bold">StrokeGuard AI</h1>
    <p class="lead fs-3">인공지능 뇌졸중 조기 예측 시스템</p>
</div>

<div class="container my-5" id="survey">
    <div class="row justify-content-center">
        <div class="col-lg-8">
            <div class="card p-5">
                <div class="progress mb-5"><div class="progress-bar bg-success" id="prog" style="width:12.5%"></div></div>
                <h2 class="text-center text-primary mb-5 fs-3" id="question">1/8 성별을 선택해주세요</h2>

                <div id="step1" class="text-center mb-4">
                    <button class="btn btn-opt active" data-value="1">남성</button>
                    <button class="btn btn-opt" data-value="2">여성</button>
                </div>
                <div id="step2" class="d-none text-center"><input type="number" class="form-control form-control-lg text-center" id="age" placeholder="나이 (예: 65)"></div>
                <div id="step3" class="d-none text-center"><input type="number" step="0.1" class="form-control form-control-lg text-center" id="bmi" placeholder="BMI (예: 25.4)"></div>
                <div id="step4" class="d-none text-center"><input type="number" class="form-control form-control-lg text-center" id="sbp" placeholder="수축기 혈압 (예: 140)"></div>
                <div id="step5" class="d-none text-center"><input type="number" class="form-control form-control-lg text-center" id="dbp" placeholder="이완기 혈압 (예: 90)"></div>
                <div id="step6" class="d-none text-center"><input type="number" step="0.1" class="form-control form-control-lg" id="glucose" placeholder="공복 혈당 (예: 120.0)"></div>
                <div id="step7" class="d-none text-center mb-4">
                    <p class="fs-4">흡연하시나요?</p>
                    <button class="btn btn-opt active" data-value="0">아니오</button>
                    <button class="btn btn-opt" data-value="1">예</button>
                </div>
                <div id="step8" class="d-none text-center mb-4">
                    <p class="fs-4">음주하시나요?</p>
                    <button class="btn btn-opt active" data-value="0">아니오</button>
                    <button class="btn btn-opt" data-value="1">예</button>
                </div>

                <div class="text-center mt-5">
                    <button id="prev" class="btn btn-secondary btn-lg me-3 d-none">이전</button>
                    <button id="next" class="btn btn-step">다음</button>
                    <button id="submit" class="btn btn-danger btn-lg d-none">예측하기</button>
                </div>
            </div>
        </div>
    </div>
</div>

<div class="container my-5 d-none" id="result">
    <div class="row justify-content-center">
        <div class="col-lg-8">
            <div class="card p-5 text-white text-center {{risk_class}}">
                <h1 class="display-1 fw-bold">{{prob}}%</h1>
                <h2 class="display-5">{{risk_text}}군</h2>
                <canvas id="gauge" width="300" height="150"></canvas>
                <div class="mt-5 fs-4 lh-lg">{{advice|safe}}</div>
                <button class="btn btn-light btn-lg mt-4" onclick="location.reload()">다시 검사하기</button>
            </div>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
const questions = ["성별을 선택해주세요","나이를 입력해주세요","BMI를 입력해주세요","수축기 혈압(SBP)을 입력해주세요","이완기 혈압(DBP)을 입력해주세요","공복 혈당 수치를 입력해주세요","흡연하시나요?","음주하시나요?"];
let step = 1;
const data = {gender:1, smoking:0, drinking:0};

document.getElementById("next").onclick = () => {
    if(step === 1) data.gender = document.querySelector("#step1 .active").dataset.value;
    if(step === 7) data.smoking = document.querySelector("#step7 .active").dataset.value;
    if(step === 8) data.drinking = document.querySelector("#step8 .active").dataset.value;
    if(step >= 2 && step <= 6){
        const keys = ["","age","bmi","sbp","dbp","glucose"];
        data[keys[step-1]] = document.getElementById(keys[step-1]).value;
        if(!data[keys[step-1]]) return alert("값을 입력해주세요!");
    }
    if(step < 8){
        document.getElementById("step"+step).classList.add("d-none");
        step++;
        document.getElementById("step"+step).classList.remove("d-none");
        document.getElementById("question").innerHTML = step+"/8 "+questions[step-1];
        document.getElementById("prog").style.width = (step/8*100)+"%";
        document.getElementById("prev").classList.remove("d-none");
        if(step === 8){ document.getElementById("next").classList.add("d-none"); document.getElementById("submit").classList.remove("d-none"); }
    }
};

document.getElementById("prev").onclick = () => {
    document.getElementById("step"+step).classList.add("d-none");
    step--;
    document.getElementById("step"+step).classList.remove("d-none");
    document.getElementById("question").innerHTML = step+"/8 "+questions[step-1];
    document.getElementById("prog").style.width = (step/8*100)+"%";
    if(step === 1) document.getElementById("prev").classList.add("d-none");
    document.getElementById("submit").classList.add("d-none");
    document.getElementById("next").classList.remove("d-none");
};

document.querySelectorAll(".btn-opt").forEach(b => b.onclick = function(){
    this.parentNode.querySelectorAll(".btn-opt").forEach(x => x.classList.remove("active"));
    this.classList.add("active");
});

document.getElementById("submit").onclick = () => {
    fetch("/", {
        method: "POST",
        headers: {"Content-Type": "application/x-www-form-urlencoded"},
        body: new URLSearchParams(data)
    })
    .then(r => r.text())
    .then(html => document.body.innerHTML = html);
};
</script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            d = {k: float(request.form[k]) for k in ["gender","age","bmi","sbp","dbp","glucose","smoking","drinking"]}
            X = np.array([[d["gender"], d["age"], d["bmi"], d["sbp"], d["dbp"], d["glucose"], d["smoking"], d["drinking"]]])
            prob = model.predict_proba(X)[0][1] * 100
            advice = get_llm_advice(d, prob)
            rc = "result-high" if prob > 70 else "result-medium" if prob > 30 else "result-low"
            rt = "고위험" if prob > 70 else "주의 필요" if prob > 30 else "안전"
            return render_template_string(HTML, prob=f"{prob:.1f}", risk_class=rc, risk_text=rt, advice=advice)
        except Exception as e:
            return f"<h1>오류 발생</h1><p>{e}</p><button onclick='history.back()'>돌아가기</button>"
    return HTML

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
