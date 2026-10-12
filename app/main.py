"""BTS AI 이상 행동 탐지 서빙 API.

실행 (프로젝트 루트에서):
    uvicorn app1.main:app1 --port 8000

모델 파일 경로는 환경 변수 MODEL_PATH로 바꿀 수 있다 (기본 models/behavior_model.joblib).
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from inference import assess, assess_batch, load_bundle, set_inference_threads
from preprocessing import PayloadValidationError

logger = logging.getLogger("bts.ai")

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "behavior_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델이 없거나 깨져 있으면 서버가 뜨지 않게 해서, 요청 때 500이 나는 것을 막는다.
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    bundle = set_inference_threads(load_bundle(model_path))
    app.state.bundle = bundle
    logger.info(
        "모델 로드 완료: %s (features=%s)", bundle["model_version"], bundle["feature_columns"]
    )
    yield


app = FastAPI(
    title="BTS AI 이상 행동 탐지 API",
    description="사용자 행동 Feature를 받아 이상도 점수(0~1)를 반환합니다. "
    "점수는 암표상일 확률이 아니라 학습 사용자 대비 행동 패턴의 특이성입니다.",
    version="1.0.0",
    lifespan=lifespan,
)


# ========================================
# 오류 응답 형식: 항상 {"errors": ["...", ...]} (HTTP 422)
# ========================================

@app.exception_handler(PayloadValidationError)
async def payload_validation_handler(request: Request, error: PayloadValidationError):
    return JSONResponse(status_code=422, content={"errors": error.errors})


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, error: RequestValidationError):
    messages = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"] if part != "body")
        messages.append(f"{location}: {item['msg']}" if location else item["msg"])
    return JSONResponse(status_code=422, content={"errors": messages})


# ========================================
# 엔드포인트
# ========================================

@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    bundle = request.app.state.bundle
    return {
        "status": "ok",
        "modelVersion": bundle["model_version"],
        "features": bundle["feature_columns"],
        "anomalyThreshold": bundle["anomaly_threshold"],
        "trainedAt": bundle["trained_at"],
    }


# 예측은 CPU 연산이라 async가 아닌 일반 def로 둔다 (FastAPI가 스레드풀에서 실행).
@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request):
    return assess(body.model_dump(), request.app.state.bundle)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(body: BatchRequest, request: Request):
    """여러 사용자를 한 번에 예측한다. 형식 오류가 있는 항목은 status="INVALID"로
    표시하고 나머지는 정상 처리한다 (HTTP 200, 결과 순서 = 요청 순서)."""
    return {"results": assess_batch(body.requests, request.app.state.bundle)}