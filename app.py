import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string, redirect, url_for
import requests
import threading
import gc
import time

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 10000))

# 모델 로드
with open("stroke_model.pkl", "rb") as f:
    model = pickle.load(f)
gc.collect()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

app.advice_cache = {}

# ------------------------------
# LLM 비동기 실행
# ------------------------------
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

    환자에게 지금 필요한 조언을 따뜻하게 4~6문장으로 해주세요.
    """

    try:
        r = requests.post(
            GROQ_URL,
            json={
                "model": "llama-3.1-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 300,
            },
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=15,
        )

        if r.status_code == 200:
            app.advice_cache[session_id] = r.json()["choices"][0]["message"]["content"]
        else:
            app.advice_cache[session_id] = "건강 관리에 더 신경써 주세요."
    except:
        app.advice_cache[session_id] = "규칙적인 운동과 수면이 도움이 됩니다."


# ------------------------------
# HTML 템플릿 (네 버전 그대로)
# ------------------------------
HTML = """{{html}}"""

# ------------------------------
# 메인 GET
# ------------------------------
@app.route("/")
def index():
    return HTML.replace("{{html}}", HTML_CONTENT)


# ------------------------------
# 예측 실행 (POST)
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        d = {k: float(request.form[k]) for k in request.form}
        X = np.array([[d[k] for k in ["gender","age","bmi","sbp","dbp","glucose","smoking","drinking"]]])
        prob = model.predict_proba(X)[0][1] * 100

        session_id = str(time.time())
        app.advice_cache[session_id] = None

        threading.Thread(target=call_llm_async, args=(d, prob, session_id), daemon=True).start()

        rc = (
            "linear-gradient(135deg,#ff6b6b,#feca57)"
            if prob > 70 else
            "linear-gradient(135deg,#feca57,#ff9ff3)"
            if prob > 30 else
            "linear-gradient(135deg,#1dd1a1,#54a0ff)"
        )
        rt = "고위험" if prob > 70 else "주의 필요" if prob > 30 else "안전"

        html = HTML_CONTENT
        html = html.replace("{{prob}}", f"{prob:.1f}")
        html = html.replace("{{risk_class}}", rc)
        html = html.replace("{{risk_text}}", rt)
        html = html.replace("{{session}}", session_id)

        return html
    except Exception as e:
        return f"<h1>에러 발생: {e}</h1>"


# ------------------------------
# LLM 결과 가져오기
# ------------------------------
@app.route("/advice")
def advice():
    session_id = request.args.get("session")
    if app.advice_cache.get(session_id):
        msg = app.advice_cache.pop(session_id)
        return msg
    return ""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
