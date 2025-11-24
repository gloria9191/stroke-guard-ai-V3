import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string
import requests
import threading
import gc
import time

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 10000))

# ============================
# 1) 모델 로드
# ============================
print("모델 불러오는 중...")
with open("stroke_model.pkl", "rb") as f:
    model = pickle.load(f)
gc.collect()
print("모델 로드 완료!")

# ============================
# 2) Groq LLM 설정
# ============================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# 인메모리 캐시
app.advice_cache = {}

# ============================
# 3) 비동기 LLM 호출
# ============================
def call_llm_async(data, prob, session_id):
    if not GROQ_API_KEY:
        app.advice_cache[session_id] = "혈압·혈당 관리와 규칙적인 운동을 추천드립니다."
        return

    prompt = f"""
    당신은 서울대병원 신경과 전문의입니다.
    환자 정보:
    나이: {data['age']}
    성별: {'남성' if data['gender']==1 else '여성'}
    BMI: {data['bmi']}
    혈압: {data['sbp']}/{data['dbp']}
    혈당: {data['glucose']}
    흡연: {data['smoking']}
    음주: {data['drinking']}
    위험도: {prob:.1f}%

    환자에게 따뜻하고 구체적인 조언을 4~6문장으로 전달하세요.
    """

    try:
        r = requests.post(
            GROQ_URL,
            json={
                "model": "llama-3.1-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 350,
            },
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=15,
        )

        if r.status_code == 200:
            advice = r.json()["choices"][0]["message"]["content"]
            app.advice_cache[session_id] = advice.replace("\n", "<br>")
        else:
            app.advice_cache[session_id] = "꾸준한 생활습관 관리가 중요합니다."
    except:
        app.advice_cache[session_id] = "규칙적인 운동과 충분한 수면이 도움이 됩니다."


# ============================
# 4) HTML (네 디자인 그대로 + 수정한 JS 포함)
# ============================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StrokeGuard AI</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
body {margin:0;background:#0f0f23;color:white;font-family:'Noto Sans KR',sans-serif;overflow-x:hidden}
.hero{min-height:100vh;background:linear-gradient(135deg,#0f0f23,#2d1b69);display:flex;align-items:center;justify-content:center;position:relative}
.hero::before{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:url('https://images.unsplash.com/photo-1638207213803-3c093c7cd506?auto=format&fit=crop&q=80') center/cover;opacity:0.15}
.hero-content{text-align:center;position:relative;z-index:2}
.title{font-size:clamp(4rem,10vw,7rem);font-weight:900;letter-spacing:-3px;margin-bottom:1.5rem}
.stroke{background:linear-gradient(90deg,#ff6b6b,#feca57);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.guard{color:#a29bfe;text-shadow:0 0 30px rgba(162,155,254,0.6)}
.subtitle{font-size:1.5rem;opacity:0.9;margin-bottom:3rem}
.badge{background:rgba(255,255,255,0.15);padding:10px 20px;border-radius:30px;margin:5px}
.card{background:white;color:#333;border-radius:20px;padding:40px}
</style>
</head>
<body>

<div class="hero">
    <div class="hero-content">
        <h1 class="title"><span class="stroke">Stroke</span><span class="guard">Guard</span> AI</h1>
        <p class="subtitle">국내 50만명 + 미국 NHANES 기반 차세대 뇌졸중 예측 모델</p>
        <div>
            <span class="badge">AUC 0.796</span>
            <span class="badge">실시간 AI 조언</span>
            <span class="badge">의료급 알고리즘</span>
        </div>
    </div>
</div>

<div class="container my-5">
    <div class="row justify-content-center">
    <div class="col-lg-8">
    <div class="card">

        <h3 class="text-center mb-4">뇌졸중 위험도 간편 검사</h3>

        <form id="survey">
            <div class="mb-3">
                <label>성별</label>
                <select class="form-control" name="gender">
                    <option value="1">남성</option>
                    <option value="2">여성</option>
                </select>
            </div>

            <div class="mb-3">
                <label>나이</label>
                <input type="number" class="form-control" name="age">
            </div>

            <div class="mb-3">
                <label>BMI</label>
                <input type="number" step="0.1" class="form-control" name="bmi">
            </div>

            <div class="mb-3">
                <label>수축기 혈압(SBP)</label>
                <input type="number" class="form-control" name="sbp">
            </div>

            <div class="mb-3">
                <label>이완기 혈압(DBP)</label>
                <input type="number" class="form-control" name="dbp">
            </div>

            <div class="mb-3">
                <label>공복 혈당</label>
                <input type="number" step="0.1" class="form-control" name="glucose">
            </div>

            <div class="mb-3">
                <label>흡연 여부</label>
                <select class="form-control" name="smoking">
                    <option value="0">아니오</option>
                    <option value="1">예</option>
                </select>
            </div>

            <div class="mb-3">
                <label>음주 여부</label>
                <select class="form-control" name="drinking">
                    <option value="0">아니오</option>
                    <option value="1">예</option>
                </select>
            </div>

            <button type="button" id="submitBtn" class="btn btn-primary w-100 mt-3">예측하기</button>
        </form>

    </div>
    </div>
    </div>
</div>

<script>
// =============================
// JS: 안정된 POST 방식 적용
// =============================
document.getElementById("submitBtn").onclick = function() {
    const form = document.getElementById("survey");
    const data = new URLSearchParams(new FormData(form));

    fetch("/predict", {
        method: "POST",
        body: data
    })
    .then(r => r.text())
    .then(html => {
        document.open();
        document.write(html);
        document.close();
    });
}
</script>

</body>
</html>
"""

# ============================
# 5) 라우팅
# ============================
@app.route("/")
def index():
    return HTML_PAGE

@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = {k: float(request.form[k]) for k in request.form}
        X = np.array([[d["gender"], d["age"], d["bmi"], d["sbp"], d["dbp"], d["glucose"], d["smoking"], d["drinking"]]])
        prob = model.predict_proba(X)[0][1] * 100

        session_id = str(time.time())
        app.advice_cache[session_id] = None

        threading.Thread(target=call_llm_async, args=(d, prob, session_id), daemon=True).start()

        rc = ("linear-gradient(135deg,#ff6b6b,#feca57)" if prob > 70 else
              "linear-gradient(135deg,#feca57,#ff9ff3)" if prob > 30 else
              "linear-gradient(135deg,#1dd1a1,#54a0ff)")
        rt = "고위험" if prob > 70 else "주의 필요" if prob > 30 else "안전"

        RESULT = f"""
        <div style="min-height:100vh;background:{rc};color:white;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:40px;">
            <h1 class="display-1 fw-bold">{prob:.1f}%</h1>
            <h2 class="mt-3">{rt}군</h2>
            <div id="advice" class="mt-5 fs-3">AI 조언 생성 중...</div>
            <button onclick="location.href='/'" class="btn btn-light mt-5 px-4">다시 검사하기</button>
        </div>

        <script>
        setTimeout(() => {{
            fetch("/advice?session={session_id}")
            .then(r => r.text())
            .then(t => {{
                if (t.length > 2)
                    document.getElementById("advice").innerHTML = t;
            }});
        }}, 3500);
        </script>
        """

        return RESULT

    except Exception as e:
        return f"<h1>에러 발생: {e}</h1>"

@app.route("/advice")
def advice():
    session = request.args.get("session")
    if app.advice_cache.get(session):
        msg = app.advice_cache.pop(session)
        return msg
    return ""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
