# ---------------------------------------------------------------
# BTS AI 이상 행동 탐지 서버 이미지 (최종 6개 Feature 모델)
#   1) deps    : 파이썬 패키지 설치 (학습/서빙 공통, 버전 고정)
#   2) trainer : 저장소의 학습용 Feature 표로 모델 학습 → models/behavior_model.joblib
#   3) runtime : 서빙에 필요한 코드와 학습된 모델만 담은 최종 이미지
#
# 모델 파일(models/)은 git 에 없다. 이미지 안에서 train_fds_model.py 로 만든다.
# 학습 데이터(ml_experiment/data/fds_v4_training_features.csv), 학습 스크립트, 패키지 버전이 같으므로
# 로컬에서 `python train_fds_model.py` 로 만든 모델과 같은 모델이 나온다.
# 이전 MVP 학습 경로(main.py + train_model.py)는 이 이미지에서 쓰지 않는다.
# ---------------------------------------------------------------

FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt


FROM deps AS trainer

COPY train_fds_model.py preprocessing.py inference.py ./
COPY ml_experiment/data/fds_v4_training_features.csv ml_experiment/data/fds_v4_training_features.meta.json ./ml_experiment/data/
RUN python train_fds_model.py


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
