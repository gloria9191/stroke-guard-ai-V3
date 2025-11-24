
import os
import pickle
import numpy as np
from flask import Flask, render_template, request

# -----------------------------
# Flask 기본 설정
# -----------------------------
app = Flask(__name__)

# -----------------------------
# 모델 로드
# -----------------------------
# Render에서는 리포지토리 루트에 stroke_model.pkl을 두면 됩니다.
MODEL_PATH = os.environ.get("MODEL_PATH", "stroke_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# 사용 변수 (학습 때 쓴 8개)
FEATS = ["Sex", "Age", "BMI", "SBP", "DBP", "Glucose", "Smoking", "Alcohol"]


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@app.route("/", methods=["GET", "POST"])
def index():
    pred_prob = None
    result_text = None
    mode = "A"
    threshold_used = None

    # 기본 예시 값 (placeholder 용)
    default_values = {
        "Sex": "1",
        "Age": "55",
        "BMI": "24.0",
        "SBP": "130",
        "DBP": "80",
        "Glucose": "95",
        "Smoking": "0",
        "Alcohol": "0",
    }

    if request.method == "POST":
        mode = request.form.get("mode", "A")
        vals = []
        for f in FEATS:
            vals.append(safe_float(request.form.get(f)))

        X = np.array([vals])
        prob = float(model.predict_proba(X)[0][1])  # 뇌졸중(1) 확률
        pred_prob = round(prob * 100, 2)  # %로 표현

        # -----------------------------
        # 모드별 threshold & 설명
        # -----------------------------
        if mode == "A":
            # 기본 모드: threshold=0.05 (기존 best)
            thr = 0.05
            threshold_used = thr
            if prob >= thr:
                level = "고위험(High risk)"
                detail = "질병(1) recall이 약 0.76 수준으로, 민감하게 고위험군을 잡아내는 설정입니다."
            else:
                level = "저위험(Lower risk)"
                detail = "정상(0) 쪽을 어느 정도 유지하면서, 위험 신호도 놓치지 않도록 설계된 기준입니다."
            mode_title = "모드 A – 기본 설정 (Threshold 0.05)"

        elif mode == "B":
            # 고감도 모드: threshold=0.01 (recall ~0.98, specificity 낮음)
            thr = 0.01
            threshold_used = thr
            if prob >= thr:
                level = "고위험(High risk, 매우 민감)"
                detail = "질병(1) recall이 매우 높은 대신, 정상도 많이 양성으로 보는 세팅입니다. 무조건 많이 잡아보는 전략."
            else:
                level = "저위험(Lower risk)"
                detail = "현재는 저위험으로 분류되지만, 이 모드는 거의 모든 위험을 잡는 방향이라 기준이 매우 빡셉니다."
            mode_title = "모드 B – 고감도(High Recall) 설정 (Threshold 0.01)"

        else:
            # 모드 C: 확률만 보여주는 '설명용' 모드 (이분법 말고 continuous)
            thr = None
            threshold_used = thr
            level = "확률 기반 설명 모드"
            detail = (
                "이 모드는 특정 기준으로 고위험/저위험을 자르지 않고, "
                "예측된 확률 값 자체를 그대로 보여줍니다. 임상의/전문의 판단과 함께 참고용으로 사용하는 것이 좋습니다."
            )
            mode_title = "모드 C – 확률 설명 모드 (No hard threshold)"

        # 최종 텍스트
        result_text = {
            "mode": mode,
            "mode_title": mode_title,
            "level": level,
            "detail": detail,
            "threshold": threshold_used,
        }

        # 입력값을 다시 폼에 채워주기
        for i, f in enumerate(FEATS):
            default_values[f] = str(vals[i])

    return render_template(
        "index.html",
        feats=FEATS,
        defaults=default_values,
        pred_prob=pred_prob,
        result=result_text,
        selected_mode=mode,
    )


if __name__ == "__main__":
    # 로컬에서 테스트용
    app.run(host="0.0.0.0", port=8000, debug=True)
