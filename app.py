import os
import traceback
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import requests

app = Flask(__name__)

print("DEBUG: Flask starting…")
print("DEBUG: GROQ key exists?", os.getenv("GROQ_API_KEY") is not None)

# ---------------------------
# 모델 로드
# ---------------------------
try:
    print("🔄 Loading stroke_model.pkl ...")
    with open("stroke_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ 모델 로드 완료")
except Exception as e:
    print("❌ 모델 로드 실패:", e)
    print(traceback.format_exc())


# ---------------------------
# 메인 페이지
# ---------------------------
@app.route('/')
def index():
    return render_template("index.html")


# ---------------------------
# 예측 API
# ---------------------------
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        print("DEBUG: Received input:", data)

        # 필수 변수 체크
        required = ["age", "fbs", "smoking", "exercise", "drinking", "bp_high"]
        for r in required:
            if r not in data:
                return jsonify({"error": f"Missing input: {r}"}), 400

        # float 변환
        try:
            age = float(data["age"])
            fbs = float(data["fbs"])
            bp_high = float(data["bp_high"])
            smoking = int(data["smoking"])
            exercise = int(data["exercise"])
            drinking = int(data["drinking"])
        except Exception as e:
            print("❌ 변환 오류:", e)
            return jsonify({"error": "Invalid number format"}), 400

        # LightGBM 입력 (8개 feature만 임시)
        X_input = np.array([[age, fbs, bp_high, smoking, exercise, drinking, 0, 0]])
        print("DEBUG: Model Input:", X_input)

        prob = model.predict_proba(X_input)[0][1]
        print("DEBUG: Model Raw Probability:", prob)

        # Groq API 호출
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            print("❌ Groq API Key 없음")
            ai_comment = "AI 코멘트 생성 불가 (API key 없음)"
        else:
            groq_payload = {
                "model": "llama3-70b-8192",
                "messages": [
                    {
                        "role": "user",
                        "content": f"""
                        아래 사람의 생활습관 데이터 기반으로 건강 코멘트를 한국어로 3줄 생성해줘.

                        나이: {age}
                        공복혈당: {fbs}
                        음주: {drinking}
                        운동: {exercise}
                        흡연: {smoking}
                        혈압: {bp_high}
                        """
                    }
                ],
                "temperature": 0.4
            }

            groq_res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json=groq_payload,
                timeout=15
            )

            ai_comment = groq_res.json()["choices"][0]["message"]["content"]

        return jsonify({
            "probability": float(prob),
            "ai_comment": ai_comment
        })

    except Exception as e:
        print("❌ ERROR in /analyze:", e)
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ---------------------------
# 앱 실행
# ---------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
