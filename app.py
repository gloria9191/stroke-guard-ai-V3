import os
import pickle
import numpy as np
from flask import Flask, request, make_response
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

# LLM 캐시
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

    환자에게 지금 당장 도움이 되는 따뜻하고 구체적인 조언을 4~6문장으로 전달하세요.
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
            text = r.json()["choices"][0]["message"]["content"]
            app.advice_cache[session_id] = text.replace("\n", "<br>")
        else:
            app.advice_cache[session_id] = "건강한 생활습관을 조금씩 실천하는 것이 중요합니다."
    except:
        app.advice_cache[session_id] = "규칙적인 운동과 충분한 수면이 도움이 됩니다."


# ============================
# 4) HTML 페이지
# ============================
def load_html():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


# ============================
# 5) 라우트
# ============================
@app.route("/")
def index():
    return load_html()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = {k: float(request.form[k]) for k in request.form}

        X = np.array([[
            d["gender"], d["age"], d["bmi"],
            d["sbp"], d["dbp"], d["glucose"],
            d["smoking"], d["drinking"]
        ]])
        prob = model.predict_proba(X)[0][1] * 100

        session_id = str(time.time())
        app.advice_cache[session_id] = None

        # 비동기 처리
        threading.Thread(
            target=call_llm_async,
            args=(d, prob, session_id),
            daemon=True
        ).start()

        rc = ("linear-gradient(135deg,#ff6b6b,#feca57)" if prob > 70 else
              "linear-gradient(135deg,#feca57,#ff9ff3)" if prob > 30 else
              "linear-gradient(135deg,#1dd1a1,#54a0ff)")
        rt = "고위험" if prob > 70 else "주의 필요" if prob > 30 else "안전"

        html = f"""
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
                if (t.length > 3)
                    document.getElementById("advice").innerHTML = t;
            }});
        }}, 3500);
        </script>
        """

        response = make_response(html)
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        return response

    except Exception as e:
        return f"<h1>에러 발생: {e}</h1>"


@app.route("/advice")
def advice():
    session = request.args.get("session")
    msg = app.advice_cache.get(session)

    if msg:
        app.advice_cache.pop(session, None)
        return msg
    return ""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
