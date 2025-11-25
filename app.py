<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>StrokeGuard AI</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>

<body class="bg-light">

<div class="container py-5">

    <!-- ====================== PAGE 1 ===================== -->
    <div id="page1">

        <h1 class="mb-4 text-center">StrokeGuard AI</h1>
        <p class="text-center">뇌졸중 발병 위험 예측을 위한 간단한 문진을 시작합니다.</p>

        <div class="text-center mt-4">
            <button id="startBtn" class="btn btn-primary btn-lg">
                시작하기
            </button>
        </div>

    </div>

    <!-- ====================== PAGE 2 (질문/입력) ===================== -->
    <div id="page2" style="display:none;">

        <h2 class="mb-4">1. 기본 정보 입력</h2>

        <form id="predictForm">

            <div class="mb-3">
                <label class="form-label">나이</label>
                <input type="number" name="age" class="form-control" required>
            </div>

            <div class="mb-3">
                <label class="form-label">공복 혈당(FBS)</label>
                <input type="number" name="fbs" class="form-control" required>
            </div>

            <div class="text-center mt-4">
                <button type="submit" class="btn btn-success btn-lg">
                    결과 보기
                </button>
            </div>

        </form>

        <div class="text-center mt-3">
            <button id="backToPage1" class="btn btn-secondary">
                처음으로
            </button>
        </div>

    </div>

    <!-- ====================== PAGE 3 (결과 페이지) ===================== -->
    <div id="page3" style="display:none;">

        <h2 class="mb-3">예측 결과</h2>

        <p id="riskText" class="fs-4 fw-bold"></p>
        <p id="aiAdvice" class="mt-3"></p>

        <div class="text-center mt-4">
            <button id="restartBtn" class="btn btn-primary">
                다시 예측하기
            </button>
        </div>

    </div>

</div>

<!-- ====================== JS ===================== -->
<script>
    // 페이지 전환
    document.getElementById("startBtn").onclick = () => {
        showPage(2);
    };

    document.getElementById("backToPage1").onclick = () => {
        showPage(1);
    };

    document.getElementById("restartBtn").onclick = () => {
        showPage(1);
    };

    function showPage(pageNum) {
        document.getElementById("page1").style.display = "none";
        document.getElementById("page2").style.display = "none";
        document.getElementById("page3").style.display = "none";

        document.getElementById("page" + pageNum).style.display = "block";
    }

    // 예측 요청
    document.getElementById("predictForm").onsubmit = async (event) => {
        event.preventDefault();

        const formData = new FormData(event.target);
        const data = Object.fromEntries(formData.entries());

        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });

        const result = await response.json();

        document.getElementById("riskText").textContent =
            `뇌졸중 발병 확률: ${result.prob}%`;

        document.getElementById("aiAdvice").textContent =
            result.advice || "";

        showPage(3);
    };
</script>

</body>
</html>
