"""POST /predict 요청/응답 스키마 (BTS AI Feature 명세서 7, predict_feature_spec.md).

요청의 타입 검사는 Pydantic이, 범위·null 허용 여부 같은 도메인 규칙은
preprocessing.validate_payload가 담당한다 (학습 코드와 같은 규칙을 공유).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

MAX_BATCH_SIZE = 1000

# 문자열("12")이나 boolean은 숫자로 바꾸지 않고 오류로 처리한다.
Number = StrictInt | StrictFloat

EXAMPLE_REQUEST = {
    "userId": "user_001",
    "calculatedAt": "2026-10-08T15:00:00+09:00",
    "windowDays": 30,
    "features": {
        "bookingCount": 20,
        "avgBookingInterval": 12.5,
        "bookingIntervalVariance": 1.25,
        "cancelRate": 0.75,
        "cardReuseCount": 1,
        "ipBookingDensity": 2,
    },
}


class Features(BaseModel):
    """계산 불가 Feature는 null로 보낸다. 생략해도 null로 취급한다."""

    model_config = ConfigDict(extra="ignore")

    bookingCount: Number | None = Field(None, description="최근 30일 누적 예매 시도 수 (null 불가)")
    avgBookingInterval: Number | None = Field(None, description="평균 예매 간격(초). 예매 2건 미만이면 null")
    bookingIntervalVariance: Number | None = Field(None, description="예매 간격 분산(초^2). 예매 2건 미만이면 null")
    cancelRate: Number | None = Field(None, description="최근 30일 취소율(0~1). 예매 0건이면 null")
    cardReuseCount: Number | None = Field(1, description="동일 카드 결제 다계정 공유 수 (기본 1)")
    ipBookingDensity: Number | None = Field(1, description="동일 IP 내 집중 트랜잭션 수 (기본 1)")
    

class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_REQUEST})

    userId: str | StrictInt = Field(
        description="사용자 식별자. 문자열 또는 정수(Long). 응답에는 항상 문자열로 돌려준다"
    )
    calculatedAt: str = Field(description="ISO-8601, UTC 오프셋 필수")
    windowDays: StrictInt = Field(description="분석 기간(일). 현재 30만 허용")
    features: Features


class PredictResponse(BaseModel):
    userId: str
    status: Literal["OK", "INSUFFICIENT_DATA"]
    anomalyScore: float | None = Field(
        description="0~1, 높을수록 이상. INSUFFICIENT_DATA면 null"
    )
    modelVersion: str
    assessedAt: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"example": {"requests": [EXAMPLE_REQUEST]}}
    )

    # 한 건의 형식 오류가 전체를 막지 않도록 항목은 dict로 받고,
    # 항목별 검증은 서비스 로직(validate_payload)이 수행한다.
    requests: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchItemResult(BaseModel):
    userId: str | None
    status: Literal["OK", "INSUFFICIENT_DATA", "INVALID"]
    anomalyScore: float | None
    modelVersion: str
    assessedAt: str
    errors: list[str] | None = Field(None, description="status가 INVALID일 때만 채워짐")


class BatchResponse(BaseModel):
    results: list[BatchItemResult] = Field(description="요청 순서와 같은 순서")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    modelVersion: str
    features: list[str]
    anomalyThreshold: float
    trainedAt: str