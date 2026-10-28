# ---------------------------------------------------------------
# BTS AI 이상 행동 탐지 서버 이미지
#   1) deps    : 파이썬 패키지 설치 (학습/서빙 공통)
#   2) trainer : 합성 데이터 생성 + 모델 학습  → models/behavior_model.joblib
#   3) runtime : 서빙에 필요한 코드와 학습된 모델만 담은 최종 이미지
#
# 모델을 이미지 안에서 학습하므로 학습 환경과 서빙 환경의 scikit-learn 버전이
# 항상 같고, 저장소를 받자마자 `docker compose up --build` 만으로 실행할 수 있다.
# (models/ 와 data/ 는 .gitignore 대상이라 저장소에 없다.)
# ---------------------------------------------------------------

FROM python:3.13-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt


FROM deps AS trainer

COPY main.py train_model.py preprocessing.py inference.py ./
RUN python main.py && python train_model.py


FROM deps AS runtime

# root가 아닌 사용자로 실행한다.
RUN useradd --system --create-home --uid 10001 app

COPY --chown=app:app app/ ./app/
COPY --chown=app:app preprocessing.py inference.py ./
COPY --from=trainer --chown=app:app /app/models ./models

ENV MODEL_PATH=/app/models/behavior_model.joblib

USER app

EXPOSE 8000

# slim 이미지에는 curl이 없어서 파이썬으로 /health를 확인한다.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# --host 0.0.0.0 이 없으면 컨테이너 안에서만 접속되어 포트 매핑(-p)이 동작하지 않는다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]