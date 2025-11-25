import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string
import requests

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 10000))

print("🔄 Loading stroke_model.pkl ...")
with open("stroke_model.pkl", "rb") as f:
    model = pickle.load(f)
print("✅ 모델 로드 완료")

# Groq LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# 안전하게 form value 가져오기
def get_val(form, key):
    try:
        return float(form.get(key, 0))
    except:
        return 0.0


# LLM 조언 생성 함수
def get_llm_advice(data, prob):
    if not GROQ_API_KEY:
        return "혈압과 혈당 관리, 규칙적인 운동, 금연·절주를 실천해보세요.<br>작은 변화가 큰 차이를 만듭니다."

    prompt = f"""
당신은 서울대병원 신경과 전문의이자 가장 따뜻한 의사입니다.

환자 정보:
- 나이: {data['age']}세 | 성별: {'남성' if data['gender']==1 else '여성'}
- BMI: {data['bmi']:.1f} | 혈압: {data['sbp']}/{data['dbp']} mmHg | 공복혈당: {data['glucose']:.1f} mg/dL
- 흡연: {'합니다' if data['smoking']==1 else '하지 않습니다'} | 음주: {'합니다' if data['drinking']==1 else '하지 않습니다'}

뇌졸중 예측 확률은 {prob:.1f}%입니다.
현실적이고 따뜻한 생활습관 조언을 5문장 정도로 부탁드립니다.
"""

    try:
        r = requests.post(GROQ_URL, json={
            "model": "llama-3.1-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "max_tokens": 350
        }, headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, timeout=15)

        return r.json()["choices"][0]["message"]["content"].replace("\n", "<br>")

    except Exception as e:
        print("LLM ERROR:", e)
        return "현재 서버가 혼잡합니다.<br>규칙적 운동, 금연·절주, 혈압·혈당 관리가 중요합니다."


# ===============================
# HTML
# ===============================


HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>StrokeGuard AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Yoon+Gothic+700:wght@700&family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">
    <style>
        body {margin:0;background:#0f0f23;color:white;font-family:'Noto Sans KR',sans-serif;overflow-x:hidden}
        .hero{min-height:100vh;background:linear-gradient(135deg,#0f0f23 0%,#1a1a3a 50%,#2d1b69 100%);display:flex;align-items:center;justify-content:center;position:relative}
        .hero::before{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:url('https://images.unsplash.com/photo-1638207213803-3c093c7cd506?ixlib=rb-4.0.3&auto=format&fit=crop&q=80') center/cover;opacity:0.15}
        .hero-content{position:relative;z-index:2;max-width:900px;text-align:center;padding:0 20px}
        .title{font-size:clamp(4rem,10vw,7rem);font-weight:900;letter-spacing:-3px;margin:0 0 1.5rem;line-height:0.9;font-family:'Yoon Gothic 700',sans-serif}
        .stroke{background:linear-gradient(90deg,#ff6b6b,#feca57);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
        .guard{color:#a29bfe;text-shadow:0 0 30px rgba(162,155,254,0.6)}
        .subtitle{font-size:1.5rem;font-weight:300;margin:0 0 3rem;opacity:0.9;line-height:1.7}
        .badges{display:flex;flex-wrap:wrap;gap:14px;justify-content:center;margin:0 0 4rem}
        .badge{background:rgba(255,255,255,0.12);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.2);padding:10px 24px;border-radius:50px;font-weight:500;transition:all 0.3s}
        .badge:hover{background:rgba(162,155,254,0.3);transform:translateY(-3px)}
        .tagline{font-size:1.6rem;font-weight:400;opacity:0.9}
        .icon{margin-right:12px;background:linear-gradient(135deg,#54a0ff,#a29bfe);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
        .card{background:rgba(255,255,255,0.95);color:#333;border-radius:28px;box-shadow:0 20px 60px rgba(80,40,150,0.2);transition:0.4s}
        .card:hover{transform:translateY(-10px)}
        .btn-opt{width:48%;padding:22px;font-size:1.4rem;border:3px solid #6c5ce7;border-radius:20px;background:white;color:#6c5ce7;font-weight:700;transition:0.3s}
        .btn-opt.active,.btn-opt:hover{background:#6c5ce7;color:white;transform:scale(1.05)}
        .btn-step{background:linear-gradient(135deg,#a29bfe,#6c5ce7);color:white;padding:18px 70px;border-radius:50px;font-size:1.4rem;font-weight:700;box-shadow:0 15px 35px rgba(108,92,231,0.4)}
        .progress{height:12px;border-radius:12px;background:#e0e0e0}
        .progress-bar{background:linear-gradient(90deg,#a29bfe,#6c5ce7)}
        .result-high{background:linear-gradient(135deg,#ff6b6b,#feca57)}
        .result-medium{background:linear-gradient(135deg,#feca57,#ff9ff3)}
        .result-low{background:linear-gradient(135deg,#1dd1a1,#54a0ff)}
    </style>
</head>
<body>

<div class="hero">
    <div class="hero-content">
        <h1 class="title"><span class="stroke">Stroke</span><span class="guard">Guard</span> AI</h1>
        <p class="subtitle">국내 500만 명 + 미국 라벨링 데이터로 학습한<br>차세대 뇌졸중 예측 AI</p>
        <div class="badges">
            <span class="badge">ROC-AUC 0.796</span>
            <span class="badge">뇌졸중 검출 91.5%</span>
            <span class="badge">실시간 AI 주치의</span>
        </div>
        <p class="tagline"><span class="icon">Brain</span>작은 변화가 큰 미래를 만듭니다</p>
    </div>
</div>

<div class="container my-5" id="survey">
    <div class="row justify-content-center"><div class="col-lg-9">
        <div class="card p-5">
            <div class="progress mb-5"><div class="progress-bar" id="prog" style="width:12.5%"></div></div>
            <h2 class="text-center mb-5" id="question">1/8 성별을 선택해주세요</h2>

            <div id="step1" class="text-center mb-4">
                <button class="btn btn-opt active" data-value="1">남성</button>
                <button class="btn btn-opt" data-value="2">여성</button>
            </div>
            <div id="step2" class="d-none text-center"><input type="number" class="form-control form-control-lg text-center" id="age" placeholder="나이 (예: 65)"></div>
            <div id="step3" class="d-none text-center"><input type="number" step="0.1" class="form-control form-control-lg text-center" id="bmi" placeholder="BMI (예: 25.4)"></div>
            <div id="step4" class="d-none text-center"><input type="number" class="form-control form-control-lg text-center" id="sbp" placeholder="수축기 혈압 (예: 140)"></div>
            <div id="step5" class="d-none text-center"><input type="number" class="form-control form-control-lg text-center" id="dbp" placeholder="이완기 혈압 (예: 90)"></div>
            <div id="step6" class="d-none text-center"><input type="number" step="0.1" class="form-control form-control-lg text-center" id="glucose" placeholder="공복 혈당 (예: 120.0)"></div>
            <div id="step7" class="d-none text-center mb-4"><p class="fs-3 fw-bold mb-4">흡연하시나요?</p>
                <button class="btn btn-opt active" data-value="0">아니오</button><button class="btn btn-opt" data-value="1">예</button>
            </div>
            <div id="step8" class="d-none text-center mb-4"><p class="fs-3 fw-bold mb-4">음주하시나요?</p>
                <button class="btn btn-opt active" data-value="0">아니오</button><button class="btn btn-opt" data-value="1">예</button>
            </div>

            <div class="text-center mt-5">
                <button id="prev" class="btn btn-outline-secondary btn-lg me-4 d-none">이전</button>
                <button id="next" class="btn btn-step">다음</button>
                <button id="submit" class="btn btn-danger btn-lg d-none">지금 예측하기</button>
            </div>
        </div>
    </div></div>
</div>

<div class="container my-5 d-none" id="result">
    <div class="row justify-content-center"><div class="col-lg-8">
        <div class="card p-5 text-white text-center result-card" style="background:{{risk_class}}">
            <h1 class="display-1 fw-bold mb-3">{{prob}}%</h1>
            <h2 class="display-5 mb-5">{{risk_text}}군</h2>
            <div class="mt-5 fs-3 lh-lg px-4" style="text-shadow:0 2px 10px rgba(0,0,0,0.3)">{{advice|safe}}</div>
            <button class="btn btn-light btn-lg mt-5 px-5" onclick="location.reload()">다시 검사하기</button>
        </div>
    </div></div>
</div>

<script>
const questions = ["성별을 선택해주세요","나이를 입력해주세요","BMI를 입력해주세요","수축기 혈압을 입력해주세요","이완기 혈압을 입력해주세요","공복 혈당을 입력해주세요","흡연하시나요?","음주하시나요?"];
let step = 1;
const data = {gender:1, smoking:0, drinking:0};

document.getElementById("next").onclick = () => {
    if(step === 1) data.gender = document.querySelector("#step1 .active").dataset.value;
    if(step === 7) data.smoking = document.querySelector("#step7 .active").dataset.value;
    if(step === 8) data.drinking = document.querySelector("#step8 .active").dataset.value;
    if(step >= 2 && step <= 6){
        const keys = ["","age","bmi","sbp","dbp","glucose"];
        const val = document.getElementById(keys[step-1]).value.trim();
        if(!val || isNaN(val) || parseFloat(val) <= 0){ alert("정확한 값을 입력해주세요"); return; }
        data[keys[step-1]] = parseFloat(val);
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
    document.getElementById("submit").classList.add("d-none"); document.getElementById("next").classList.remove("d-none");
};

document.querySelectorAll(".btn-opt").forEach(b => b.onclick = function(){
    this.parentNode.querySelectorAll(".btn-opt").forEach(x => x.classList.remove("active"));
    this.classList.add("active");
});

document.getElementById("submit").onclick = () => {
    const required = ["age","bmi","sbp","dbp","glucose"];
    for(let k of required){
        const v = document.getElementById(k).value.trim();
        if(!v || isNaN(v) || parseFloat(v)<=0){ alert("모든 항목을 정확히 입력해주세요"); return; }
        data[k] = parseFloat(v);
    }
    document.querySelector(".container").innerHTML = `<div style="min-height:100vh;display:flex;align-items:center;justify-content:center;flex-direction:column"><h2>AI가 분석 중입니다...</h2><div class="spinner-border text-primary" style="width:5rem;height:5rem;"></div><p class="mt-4 fs-3 text-muted">잠시만 기다려주세요</p></div>`;
    fetch("/", {method:"POST", headers:{"Content-Type":"application/x-www-form-urlencoded"}, body:new URLSearchParams(data)})
    .then(r => r.text()).then(html => document.body.innerHTML = html)
    .catch(() => document.body.innerHTML = `<div class="text-center py-5"><h1>일시적인 오류가 발생했습니다</h1><button class="btn btn-primary btn-lg" onclick="location.reload()">다시 시도</button></div>`);
};
</script>
</body>
</html>
"""


# ===============================
# 라우트
# ===============================

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            d = {
                "gender": get_val(request.form, "gender"),
                "age": get_val(request.form, "age"),
                "bmi": get_val(request.form, "bmi"),
                "sbp": get_val(request.form, "sbp"),
                "dbp": get_val(request.form, "dbp"),
                "glucose": get_val(request.form, "glucose"),
                "smoking": get_val(request.form, "smoking"),
                "drinking": get_val(request.form, "drinking")
            }

            X = np.array([[d[k] for k in d]])
            prob = float(model.predict_proba(X)[0][1] * 100)

            # 등급
            if prob > 70:
                rc = "linear-gradient(135deg,#ff6b6b,#feca57)"
                rt = "고위험"
            elif prob > 30:
                rc = "linear-gradient(135deg,#feca57,#ff9ff3)"
                rt = "주의 필요"
            else:
                rc = "linear-gradient(135deg,#1dd1a1,#54a0ff)"
                rt = "안전"

            advice = get_llm_advice(d, prob)

            return render_template_string(
                HTML,
                prob=f"{prob:.1f}",
                risk_class=rc,
                risk_text=rt,
                advice=advice
            )

        except Exception as e:
            return f"<h1>오류 발생</h1><p>{e}</p>"

    return HTML


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
