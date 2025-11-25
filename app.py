from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
import requests

app = Flask(__name__)

# ------------------------------------------------
# 1) 모델 로드
# ------------------------------------------------
print("🔄 Loading stroke_model.pkl ...")
model = joblib.load("stroke_model.pkl")
print("✅ 모델 로드 완료")

THRESHOLD = 0.029698   # recall 기준 threshold

# ------------------------------------------------
# 2) GROQ API 설정 (LLM 안정화)
# ------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

def generate_advice(prob):
    # KEY 없으면 기본 조언 제공
    if not GROQ_API_KEY:
        return f"""생활습관 개선이 중요합니다.
- 혈압 및 혈당을 정기적으로 체크하세요.
- 과한 나트륨을 줄이고 채소·단백질 위주로 식사하세요.
- 가벼운 유산소 운동을 매일 유지하세요.
- 흡연자는 반드시 금연이 필요합니다."""

    prompt = f"""
    사용자의 뇌졸중 발병 확률은 {prob}% 입니다.

    한국인 생활습관 기준으로 다음 항목을 중심으로 4~5줄로 조언을 작성해주세요:
    - 식습관
    - 운동
    - 혈압/혈당 관리
    - 위험 신호 체크
    - 금연/절주

    따뜻하고 이해하기 쉬운 한국어 문장으로만 작성하세요.
    불필요한 특수문자, 영어, *, - 같은 기호를 넣지 마세요.
    """

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "max_tokens": 200
            },
            timeout=20
        )

        ans = r.json()
        return ans["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("LLM ERROR:", e)
        return (
            "AI 조언 생성 중 문제가 발생했습니다. "
            "기본 건강 관리 지침을 참고해 주세요.\n"
            "• 규칙적인 운동과 균형 잡힌 식사를 유지하세요.\n"
            "• 혈압과 혈당을 자주 확인하고 의사와 상담하세요.\n"
        )

# ------------------------------------------------
# 3) Routing
# ------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        gender = float(data["gender"])
        age = float(data["age"])
        bmi = float(data["bmi"])
        sbp = float(data["sbp"])
        dbp = float(data["dbp"])
        glucose = float(data["glucose"])
        smoking = float(data["smoking"])
        drinking = float(data["drinking"])

        X = np.array([[gender, age, bmi, sbp, dbp, glucose, smoking, drinking]])
        proba = model.predict_proba(X)[0][1]
        prob_percent = round(proba * 100, 1)

        # 위험군 판정
        if proba >= THRESHOLD:
            risk_class = "result-high"
            risk_text = "고위험"
        else:
            risk_class = "result-low"
            risk_text = "저위험"

        # LLM 조언 생성
        advice = generate_advice(prob_percent)

        # 모델 정보 추가 (UI 하단 표시용)
        model_info = f"AI 조언 생성 모델: Groq {GROQ_MODEL}"

        return jsonify({
            "prob": prob_percent,
            "risk_text": risk_text,
            "risk_class": risk_class,
            "advice": advice,
            "model_info": model_info
        })

    except Exception as e:
        print("SERVER ERROR:", e)
        return jsonify({"error": f"서버 오류: {str(e)}"})

# ------------------------------------------------
# Render 환경 → run() 금지
# ------------------------------------------------
if __name__ == "__main__":
    pass
