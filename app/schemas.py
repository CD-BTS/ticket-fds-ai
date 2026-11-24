"""POST /predict 요청/응답 스키마 (docs/predict_feature_spec.md).

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
        "avgBookingInterval": 4.6,
        "bookingIntervalVariance": 0.12,
        "cancelRate": 0.25,
        "cardReuseCount": 1,
        "ipBookingDensity": 2,
    },
}


class Features(BaseModel):
    """계산 불가 Feature는 null로 보낸다. 생략해도 null로 취급한다.

    여기 없는 이름(이전 MVP 필드, 오타)은 preprocessing.validate_payload 가 422 로 거절한다.
    그 오류 문구를 단건·배치에서 같게 하려고 스키마는 추가 필드를 그대로 넘긴다 (extra="allow").
    """

    model_config = ConfigDict(extra="allow")

    bookingCount: Number | None = Field(None, description="최근 30일 고유 예매 시도 수 (null 불가). 0이면 INSUFFICIENT_DATA")
    avgBookingInterval: Number | None = Field(None, description="연속 시도 시작 간격의 평균(초). 시도 2건 미만이면 null")
    bookingIntervalVariance: Number | None = Field(None, description="같은 간격의 모분산(초^2). 간격 5개 미만(시도 6건 미만)이면 null")
    cancelRate: Number | None = Field(None, description="최근 30일 완료 예매 중 취소 비율(0~1). 완료 예매 0건이면 null")
    cardReuseCount: Number | None = Field(1, description="같은 결제수단을 쓴 고유 계정 수의 최댓값, 본인 포함 (기본 1)")
    ipBookingDensity: Number | None = Field(1, description="같은 IP로 예매한 고유 계정 수의 최댓값, 본인 포함 (기본 1)")


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_REQUEST})

    userId: str | StrictInt = Field(
        description="사용자 식별자. 문자열 또는 정수(Long). 응답에는 항상 문자열로 돌려준다"
    )
    calculatedAt: str = Field(description="ISO-8601, UTC 오프셋 필수")
    windowDays: StrictInt = Field(description="분석 기간(일). 현재 30만 허용")
    features: Features


RAW_SCORE = "Isolation Forest 원시 이상 점수(-score_samples). 높을수록 이상. INSUFFICIENT_DATA면 null"
NORMALIZED_SCORE = ("0~1 정규화 점수. 0.5 이상이면 isAnomaly=true 와 같은 뜻이다. 매크로일 확률이 아니다. "
                    "INSUFFICIENT_DATA면 null (정상이나 0점이 아니라 판정하지 않음)")
IS_ANOMALY = "원시 점수가 모델의 원시 임계값 이상이면 true. INSUFFICIENT_DATA면 null"


class PredictResponse(BaseModel):
    userId: str
    status: Literal["OK", "INSUFFICIENT_DATA"]
    rawAnomalyScore: float | None = Field(description=RAW_SCORE)
    anomalyScore: float | None = Field(description=NORMALIZED_SCORE)
    isAnomaly: bool | None = Field(description=IS_ANOMALY)
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
    rawAnomalyScore: float | None = Field(None, description=RAW_SCORE)
    anomalyScore: float | None = Field(None, description=NORMALIZED_SCORE)
    isAnomaly: bool | None = Field(None, description=IS_ANOMALY)
    modelVersion: str
    assessedAt: str
    errors: list[str] | None = Field(None, description="status가 INVALID일 때만 채워짐")


class BatchResponse(BaseModel):
    results: list[BatchItemResult] = Field(description="요청 순서와 같은 순서")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    modelVersion: str
    features: list[str] = Field(description="모델 입력 순서의 Feature 이름")
    featureDefinitionVersion: str
    anomalyThreshold: float = Field(description="정규화 점수 기준 임계값 (0.5)")
    rawThreshold: float = Field(description="원시 점수 기준 임계값. 판정은 rawAnomalyScore >= rawThreshold")
    scoreNormalization: dict[str, Any] = Field(description="정규화 방식과 파라미터")
    minBookingCount: int = Field(description="bookingCount 가 이 값보다 작으면 INSUFFICIENT_DATA")
    trainingDataVersion: str | None
    trainedAt: str