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
        .hero{min-height:100vh;background:linear-gradient(135deg,#0f0f23,#1a1a3a,#2d1b69);display:flex;align-items:center;justify-content:center;position:relative}
        .hero::before{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:url('https://images.unsplash.com/photo-1638207213803-3c093c7cd506?auto=format&fit=crop&q=80') center/cover;opacity:0.15}
        .hero-content{position:relative;z-index:2;max-width:900px;text-align:center;padding:0 20px}
        .title{font-size:clamp(4rem,10vw,7rem);font-weight:900;letter-spacing:-3px;margin:0 0 1.5rem;line-height:0.9;font-family:'Yoon Gothic 700',sans-serif}
        .stroke{background:linear-gradient(90deg,#ff6b6b,#feca57);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
        .guard{color:#a29bfe;text-shadow:0 0 30px rgba(162,155,254,0.6)}
        .subtitle{font-size:1.5rem;font-weight:300;margin-bottom:3rem;opacity:0.9;line-height:1.7}
        .badges{display:flex;flex-wrap:wrap;gap:14px;justify-content:center;margin-bottom:4rem}
        .badge{background:rgba(255,255,255,0.12);backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.2);padding:10px 24px;border-radius:50px;font-weight:500;transition:0.3s}
        .badge:hover{background:rgba(162,155,254,0.3);transform:translateY(-3px)}
        .tagline{font-size:1.6rem;font-weight:400;opacity:0.9}

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

<!-- HERO -->
<div class="hero">
    <div class="hero-content">
        <h1 class="title"><span class="stroke">Stroke</span><span class="guard">Guard</span> AI</h1>
        <p class="subtitle">국내 500만 명 + 미국 라벨링 데이터로 학습한<br>차세대 뇌졸중 예측 AI</p>
        <div class="badges">
            <span class="badge">ROC-AUC 0.796</span>
            <span class="badge">뇌졸중 검출 91.5%</span>
            <span class="badge">실시간 AI 주치의</span>
        </div>
        <p class="tagline">작은 변화가 큰 미래를 만듭니다</p>
    </div>
</div>

<!-- SURVEY -->
<div class="container my-5" id="survey">
    <div class="row justify-content-center">
        <div class="col-lg-9">
            <div class="card p-5">

                <div class="progress mb-5"><div class="progress-bar" id="prog" style="width:12.5%"></div></div>
                <h2 class="text-center mb-5" id="question">1/8 성별을 선택해주세요</h2>

                <div id="step1" class="text-center mb-4">
                    <button class="btn btn-opt active" data-value="1">남성</button>
                    <button class="btn btn-opt" data-value="2">여성</button>
                </div>

                <div id="step2" class="d-none text-center"><input type="number" id="age" class="form-control form-control-lg text-center" placeholder="나이 (예: 65)"></div>
                <div id="step3" class="d-none text-center"><input type="number" step="0.1" id="bmi" class="form-control form-control-lg text-center" placeholder="BMI (예: 25.4)"></div>
                <div id="step4" class="d-none text-center"><input type="number" id="sbp" class="form-control form-control-lg text-center" placeholder="수축기 혈압 (예: 140)"></div>
                <div id="step5" class="d-none text-center"><input type="number" id="dbp" class="form-control form-control-lg text-center" placeholder="이완기 혈압 (예: 90)"></div>
                <div id="step6" class="d-none text-center"><input type="number" step="0.1" id="glucose" class="form-control form-control-lg text-center" placeholder="공복 혈당 (예: 120.0)"></div>

                <div id="step7" class="d-none text-center mb-4">
                    <p class="fs-3 fw-bold mb-4">흡연하시나요?</p>
                    <button class="btn btn-opt active" data-value="0">아니오</button>
                    <button class="btn btn-opt" data-value="1">예</button>
                </div>

                <div id="step8" class="d-none text-center mb-4">
                    <p class="fs-3 fw-bold mb-4">음주하시나요?</p>
                    <button class="btn btn-opt active" data-value="0">아니오</button>
                    <button class="btn btn-opt" data-value="1">예</button>
                </div>

                <div class="text-center mt-5">
                    <button id="prev" class="btn btn-outline-secondary btn-lg me-4 d-none">이전</button>
                    <button id="next" class="btn btn-step">다음</button>
                    <button id="submit" class="btn btn-danger btn-lg d-none">지금 예측하기</button>
                </div>

            </div>
        </div>
    </div>
</div>

<!-- RESULT -->
<div class="container my-5 d-none" id="result">
    <div class="row justify-content-center">
        <div class="col-lg-8">
            <div id="resultCard" class="card p-5 text-white text-center"></div>
        </div>
    </div>
</div>

<script>
const questions = ["성별을 선택해주세요","나이를 입력해주세요","BMI를 입력해주세요","수축기 혈압을 입력해주세요","이완기 혈압을 입력해주세요","공복 혈당을 입력해주세요","흡연하시나요?","음주하시나요?"];
let step = 1;

const data = {gender:1, smoking:0, drinking:0};

document.querySelectorAll(".btn-opt").forEach(b => {
    b.onclick = function(){
        this.parentNode.querySelectorAll(".btn-opt").forEach(x => x.classList.remove("active"));
        this.classList.add("active");
    };
});

document.getElementById("next").onclick = () => {
    if(step === 1) data.gender = document.querySelector("#step1 .active").dataset.value;
    if(step === 7) data.smoking = document.querySelector("#step7 .active").dataset.value;
    if(step === 8) data.drinking = document.querySelector("#step8 .active").dataset.value;

    if(step >= 2 && step <= 6){
        const keys = ["","age","bmi","sbp","dbp","glucose"];
        const val = document.getElementById(keys[step-1]).value.trim();
        if(!val || isNaN(val) || parseFloat(val) <= 0){
            alert("정확한 값을 입력해주세요");
            return;
        }
        data[keys[step-1]] = parseFloat(val);
    }

    if(step < 8){
        document.getElementById("step"+step).classList.add("d-none");
        step++;
        document.getElementById("step"+step).classList.remove("d-none");

        document.getElementById("question").innerText = step + "/8 " + questions[step-1];
        document.getElementById("prog").style.width = (step/8*100) + "%";
        document.getElementById("prev").classList.remove("d-none");

        if(step === 8){
            document.getElementById("next").classList.add("d-none");
            document.getElementById("submit").classList.remove("d-none");
        }
    }
};

document.getElementById("prev").onclick = () => {
    document.getElementById("step"+step).classList.add("d-none");
    step--;
    document.getElementById("step"+step).classList.remove("d-none");

    document.getElementById("question").innerText = step + "/8 " + questions[step-1];
    document.getElementById("prog").style.width = (step/8*100) + "%";

    if(step === 1) document.getElementById("prev").classList.add("d-none");
    document.getElementById("submit").classList.add("d-none");
    document.getElementById("next").classList.remove("d-none");
};

document.getElementById("submit").onclick = () => {

    const required = ["age","bmi","sbp","dbp","glucose"];
    for(let k of required){
        const v = document.getElementById(k).value.trim();
        if(!v || isNaN(v) || parseFloat(v)<=0){
            alert("모든 항목을 정확히 입력해주세요");
            return;
        }
        data[k] = parseFloat(v);
    }

    document.getElementById("survey").classList.add("d-none");
    document.getElementById("result").classList.remove("d-none");

    document.getElementById("resultCard").innerHTML = `
        <div class="text-center py-5">
            <h2>AI가 분석 중입니다...</h2>
            <div class="spinner-border text-primary" style="width:4rem;height:4rem;"></div>
            <p class="mt-4 fs-4 text-muted">잠시만 기다려주세요</p>
        </div>
    `;

    fetch("/predict", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify(data)
    })
    .then(r => r.json())
    .then(res => {

        if(res.error){
            document.getElementById("resultCard").innerHTML = `
                <h1>예측 오류</h1><p>${res.error}</p>`;
            return;
        }

        document.getElementById("resultCard").className =
            "card p-5 text-white text-center " + res.risk_class;

        document.getElementById("resultCard").innerHTML = `
            <h1 class="display-1 fw-bold mb-3">${res.prob}%</h1>
            <h2 class="display-5 mb-5">${res.risk_text}군</h2>
            <div class="mt-5 fs-3 lh-lg px-4">${res.advice}</div>
            <button class="btn btn-light btn-lg mt-5 px-5" onclick="location.reload()">다시 검사하기</button>
        `;
    })
    .catch(() => {
        document.getElementById("resultCard").innerHTML = `
            <h1>서버 오류</h1><button onclick="location.reload()" class="btn btn-primary mt-3">다시 시도</button>`;
    });
};
</script>

</body>
</html>
