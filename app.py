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

# Groq LLM 연결
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_llm_advice(data, prob):
    if not GROQ_API_KEY:
        return "GROQ_API_KEY가 설정되지 않아 기본 조언을 제공합니다.<br>혈압·혈당 관리와 금연·절주를 권장합니다."

    prompt = f"""
    서울대병원 신경과 전문의입니다. 다음 환자분께 따뜻하고 구체적인 생활습관 조언 부탁드립니다:
    성별: {'남성' if data['gender']==1 else '여성'}, 나이: {data['age']}세
    BMI: {data['bmi']:.1f}, 혈압: {data['sbp']}/{data['dbp']} mmHg, 공복혈당: {data['glucose']:.1f} mg/dL
    흡연: {'합니다' if data['smoking']==1 else '하지 않습니다'}, 음주: {'합니다' if data['drinking']==1 else '하지 않습니다'}
    뇌졸중 예측 확률: {prob:.1f}%
    4~6문장으로 현실적인 조언 부탁드립니다.
    """
    try:
        r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                          json={"model": "llama-3.1-70b-versatile", "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.7, "max_tokens": 300}, timeout=15)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].replace("\n", "<br>")
    except:
        pass
    return "현재 서버가 혼잡합니다. 혈압·혈당 관리와 규칙적인 운동, 금연·절주를 권장드립니다."

HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>StrokeGuard AI - 뇌졸중 조기 예측</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {font-family:'Noto Sans KR',sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);color:white}
        .hero {padding:100px 0;text-align:center;background:rgba(0,0,0,0.5);border-radius:0 0 40px 40px}
        .card {background:rgba(255,255,255,0.95);color:#333;border-radius:20px}
        .btn-step {background:#5e42a6;color:white;padding:12px 50px;border-radius:50px;font-size:1.3rem}
        .btn-gender, .btn-binary {width:48%;padding:20px;font-size:1.3rem;border:3px solid #5e42a6;border-radius:15px}
        .btn-gender.active, .btn-binary.active {background:#5e42a6;color:white}
        .progress {height:10px}
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
    <div class="card p-5">
        <div class="progress mb-4"><div class="progress-bar" id="prog" style="width:12.5%"></div></div>
        <h2 class="text-center text-primary mb-5"><span id="qnum">1</span>/8 성별을 선택해주세요</h2>
        <div id="s1" class="text-center"><button class="btn btn-gender active" data-v="1">남성</button><button class="btn btn-gender" data-v="2">여성</button></div>
        <div id="s2" class="d-none text-center"><input type="number" class="form-control form-control-lg text-center" id="age" placeholder="나이"></div>
        <div id="s3" class="d-none text-center"><input type="number" step="0.1" class="form-control form-control-lg text-center" id="bmi" placeholder="BMI"></div>
        <div id="s4" class="d-none text-center"><input type="number" class="form-control form-control-lg text-center" id="sbp" placeholder="수축기 혈압"></div>
        <div id="s5" class="d-none text-center"><input type="number" class="form-control form-control-lg text-center" id="dbp" placeholder="이완기 혈압"></div>
        <div id="s6" class="d-none text-center"><input type="number" step="0.1" class="form-control form-control-lg text-center" id="glucose" placeholder="공복 혈당"></div>
        <div id="s7" class="d-none text-center"><button class="btn btn-binary active" data-v="0">비흡연</button><button class="btn btn-binary" data-v="1">흡연</button></div>
        <div id="s8" class="d-none text-center"><button class="btn btn-binary active" data-v="0">비음주</button><button class="btn btn-binary" data-v="1">음주</button></div>
        <div class="text-center mt-5">
            <button id="prev" class="btn btn-secondary me-3 d-none">이전</button>
            <button id="next" class="btn btn-step">다음</button>
            <button id="submit" class="btn btn-danger d-none">예측하기</button>
        </div>
    </div>
</div>

<div class="container my-5 d-none" id="result">
    <div class="card p-5 text-center text-white {{risk_class}}">
        <h1 class="display-1">{{prob}}%</h1>
        <h2>{{risk_text}}군</h2>
        <canvas id="gauge" width="300" height="150"></canvas>
        <div class="mt-4 fs-4" style="line-height:2">{{advice|safe}}</div>
        <button class="btn btn-light btn-lg mt-4" onclick="location.reload()">다시 검사</button>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
let step=1, data={gender:1,smoking:0,drinking:0};
document.getElementById("next").onclick=()=>{
    if(step===1) data.gender = document.querySelector(".btn-gender.active").dataset.v;
    if(step===7) data.smoking = document.querySelector("#s7 .active").dataset.v;
    if(step===8) data.drinking = document.querySelector("#s8 .active").dataset.v;
    if(step>=2 && step<=6){
        const ids=["","age","bmi","sbp","dbp","glucose"][step-1];
        data[ids] = document.getElementById(ids).value;
    }
    if(step<8){ document.getElementById("s"+step).classList.add("d-none"); step++; document.getElementById("s"+step).classList.remove("d-none"); }
    document.getElementById("prog").style.width = (step/8*100)+"%";
    document.getElementById("qnum").textContent = step;
    document.getElementById("prev").classList.remove("d-none");
    if(step===8) {document.getElementById("next").classList.add("d-none"); document.getElementById("submit").classList.remove("d-none");}
};
document.getElementById("prev").onclick=()=>{/* 생략 */};
document.getElementById("submit").onclick=()=>{
    fetch("/", {method:"POST", headers:{"Content-Type":"application/x-www-form-urlencoded"},
        body:new URLSearchParams(data)})
    .then(r=>r.text()).then(html=>{document.body.innerHTML=html;});
};
document.querySelectorAll(".btn-gender,.btn-binary").forEach(b=>b.onclick=function(){
    this.parentNode.querySelectorAll("button").forEach(x=>x.classList.remove("active")); this.classList.add("active");
});
</script>
</body></html>
"""

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        d = {k:float(request.form[k]) for k in ["gender","age","bmi","sbp","dbp","glucose","smoking","drinking"]}
        X = np.array([[d["gender"],d["age"],d["bmi"],d["sbp"],d["dbp"],d["glucose"],d["smoking"],d["drinking"]]])
        prob = model.predict_proba(X)[0][1]*100
        advice = get_llm_advice(d, prob)
        rc = "result-high" if prob>70 else "result-medium" if prob>30 else "result-low"
        rt = "고위험" if prob>70 else "주의 필요" if prob>30 else "안전"
        return render_template_string(HTML, prob=f"{prob:.1f}", risk_class=rc, risk_text=rt, advice=advice)
    return HTML

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
