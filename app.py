import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string
import requests
import json

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 10000))

# =================== 모델 로드 ===================
MODEL_PATH = "stroke_model.pkl"  # 당신 파일명 그대로!
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# =================== LLM 설정 (2가지 중 하나만 쓰세요) ===================
# 1. Groq (무료 + 초고속 + 한국어 최고) ← 강력 추천!
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # Render 환경변수에 넣기!
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# 2. OpenAI (GPT-4o) ← 돈 좀 들지만 완벽함
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_llm_advice(patient_data, prob):
    risk = "높은" if prob > 70 else "중간" if prob > 30 else "낮은"
    
    prompt = f"""
    당신은 서울대학교병원 신경과 전문의입니다. 환자에게 따뜻하고 전문적인 어조로 조언해주세요.
    환자 정보:
    - 성별: {'남성' if patient_data['gender']==1 else '여성'}
    - 나이: {patient_data['age']}세
    - BMI: {patient_data['bmi']:.1f}
    - 혈압: {patient_data['sbp']}/{patient_data['dbp']} mmHg
    - 공복혈당: {patient_data['glucose']:.1f} mg/dL
    - 흡연: {'흡연자' if patient_data['smoking']==1 else '비흡연자'}
    - 음주: {'음주자' if patient_data['drinking']==1 else '비음주자'}
    - 뇌졸중 예측 확률: {prob:.1f}%

    이 환자에게 줄 수 있는 가장 현실적이고 구체적인 생활습관 개선 조언 4~6줄을 작성해주세요.
    무조건 한국어로, '~하세요', '~하는 것이 좋습니다' 식으로 부드럽고 따뜻하게.
    과장된 경고보다는 실천 가능한 조언 위주로.
    """

    if GROQ_API_KEY:
        try:
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
            payload = {
                "model": "llama-3.1-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 300
            }
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=15)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except:
            pass

    # Groq 실패시 기본 조언
    return f"""
    현재 뇌졸중 위험이 {risk} 수준으로 평가되었습니다.
    혈압과 혈당을 꾸준히 관리하는 것이 가장 중요합니다.
    {'금연이 시급합니다.' if patient_data['smoking']==1 else '비흡연을 유지하는 것이 좋습니다.'}
    {'과도한 음주는 피해주세요.' if patient_data['drinking']==1 else '적정 음주나 금주를 권장합니다.'}
    매일 30분 이상 빠르게 걷기, 채소 중심 식사, 스트레스 관리를 실천해보세요.
    3~6개월 후 재검사를 권장드립니다.
    """

# =================== HTML + JS (당신이 원했던 완벽한 UI) ===================
HTML = """..."""  # (너무 길어서 아래에 별도 제공)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = {
            "gender": int(request.form["gender"]),
            "age": int(request.form["age"]),
            "bmi": float(request.form["bmi"]),
            "sbp": int(request.form["sbp"]),
            "dbp": int(request.form["dbp"]),
            "glucose": float(request.form["glucose"]),
            "smoking": int(request.form["smoking"]),
            "drinking": int(request.form["drinking"])
        }
        X = np.array([[
            data["gender"], data["age"], data["bmi"], data["sbp"],
            data["dbp"], data["glucose"], data["smoking"], data["drinking"]
        ]])
        prob = model.predict_proba(X)[0][1] * 100

        advice = get_llm_advice(data, prob)

        risk_class = "result-high" if prob > 70 else "result-medium" if prob > 30 else "result-low"
        risk_text = "고위험" if prob > 70 else "주의 필요" if prob > 30 else "안전"

        return render_template_string(HTML, prob=f"{prob:.1f}", risk_class=risk_class,
                                    risk_text=risk_text, advice=advice)

    return render_template_string(HTML)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
