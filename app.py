from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import openai
import os

# ==========================================
# Flask Setup
# ==========================================
app = Flask(__name__)
app.secret_key = "stroke_secret_key_123"


# ==========================================
# Load Model
# ==========================================
MODEL_PATH = "model.pkl"   # 모델파일 경로
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ==========================================
# 12개 Input Questions
# ==========================================
QUESTIONS = [
    {"id": "Sex", "text": "성별을 입력해주세요 (0=여성, 1=남성)"},
    {"id": "AgeGroup", "text": "연령대를 입력해주세요 (예: 30, 40, 50 …)"},
    {"id": "Height", "text": "신장(5cm 단위)을 입력해주세요"},
    {"id": "Weight", "text": "체중(5kg 단위)을 입력해주세요"},
    {"id": "SBP", "text": "수축기 혈압(SBP)을 입력해주세요"},
    {"id": "DBP", "text": "이완기 혈압(DBP)을 입력해주세요"},
    {"id": "Glucose", "text": "공복혈당을 입력해주세요"},
    {"id": "Cholesterol", "text": "총 콜레스테롤을 입력해주세요"},
    {"id": "HDL", "text": "HDL 콜레스테롤을 입력해주세요"},
    {"id": "Triglyceride", "text": "Triglyceride 수치를 입력해주세요"},
    {"id": "Smoking", "text": "흡연 상태를 입력해주세요 (0/1)"},
    {"id": "Alcohol", "text": "음주 여부를 입력해주세요 (0/1)"},
]

# 모델에서 요구하는 Feature 순서
FEATURE_ORDER = [
    "Sex",
    "AgeGroup",
    "Height",
    "Weight",
    "SBP",
    "DBP",
    "Glucose",
    "Cholesterol",
    "HDL",
    "Triglyceride",
    "Smoking",
    "Alcohol"
]


# ==========================================
# Home → 첫 질문
# ==========================================
@app.route("/")
def home():
    session["answers"] = {}
    return redirect(url_for("question", q=0))


# ==========================================
# Question Page
# ==========================================
@app.route("/question/<int:q>", methods=["GET", "POST"])
def question(q):
    answers = session.get("answers", {})

    # POST → 저장하고 다음으로 이동
    if request.method == "POST":
        value = request.form.get("answer")
        qid = QUESTIONS[q]["id"]
        answers[qid] = value
        session["answers"] = answers

        if q == len(QUESTIONS) - 1:
            return redirect(url_for("result"))
        else:
            return redirect(url_for("question", q=q + 1))

    # GET → 질문 렌더링
    return render_template(
        "question.html",
        question=QUESTIONS[q]["text"],
        q_index=q,
        total=len(QUESTIONS)
    )


# ==========================================
# LLM Comment Generator (OpenAI API 연결 가능)
# ==========================================
def generate_llm_comment(values):
    if "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.getenv("OPENAI_API_KEY")

        prompt = f"""
        다음은 사용자의 건강 입력값입니다: 
        
        {values}

        이 정보를 바탕으로:
        1) 생활습관 위험요인
        2) 개선해야 할 점
        3) 뇌졸중과 관련된 개인 맞춤형 의견
        
        전문가 스타일로 3~4줄로 작성.
        """

        res = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message["content"]

    # Key 없으면 기본 메시지
    return "혈압, 혈당, 지질 수치를 기반으로 생활습관 개선이 필요할 수 있습니다. 규칙적인 검진과 관리가 중요합니다."


# ==========================================
# Result Page
# ==========================================
@app.route("/result")
def result():
    answers = session.get("answers", {})

    # 값 정렬 후 모델 입력
    x = np.array([[float(answers[k]) for k in FEATURE_ORDER]])

    # 예측 확률
    prob = model.predict_proba(x)[0][1]
    prob_percent = round(prob * 100, 2)

    # LLM 코멘트 생성
    comment = generate_llm_comment(answers)

    return render_template(
        "result.html",
        prob=prob_percent,
        comment=comment,
        raw=answers
    )


# ==========================================
# Server Run
# ==========================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
