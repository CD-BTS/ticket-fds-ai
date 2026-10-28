"""BTS AI 전처리 모듈.

- Feature 스펙(단일 출처): 이름, 요청 필드명, 타입, 범위, 결측 허용 여부
- POST /predict 요청 검증
- 결측치 대체 / 변환 / 스케일링 파이프라인 구성
- 이상도 점수 0~1 정규화
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


WINDOW_DAYS = 30


# ========================================
# 1. Feature 스펙 (선행 연구 기반 6개 Feature로 개편)
# ========================================
# camel    : POST /predict 요청의 필드명
# integer  : 정수 여부
# min, max : 허용 범위 (max=None이면 상한 없음)
# nullable : null 허용 여부
# log      : log1p 변환 후보 여부

FEATURE_SPECS = {
    "avg_booking_interval": dict(
        camel="avgBookingInterval", integer=False, min=0, max=None,
        nullable=True, log=True,        # 예매 2건 미만이면 null 허용
    ),
    "booking_interval_variance": dict(
        camel="bookingIntervalVariance", integer=False, min=0, max=None,
        nullable=True, log=True,        # 예매 2건 미만이면 null 허용
    ),
    "cancel_rate": dict(
        camel="cancelRate", integer=False, min=0, max=1,
        nullable=True, log=False,       # 예매 0건이면 null 허용
    ),
    "card_reuse_count": dict(
        camel="cardReuseCount", integer=True, min=1, max=None,
        nullable=False, log=False,      # 기본값 1 (자기 자신)
    ),
    "ip_booking_density": dict(
        camel="ipBookingDensity", integer=True, min=1, max=None,
        nullable=False, log=False,      # 기본값 1
    ),
    "booking_count": dict(
        camel="bookingCount", integer=True, min=0, max=None,
        nullable=False, log=True,       # 필수 누적 시도 건수
    ),
}

CAMEL_TO_SNAKE = {spec["camel"]: name for name, spec in FEATURE_SPECS.items()}

# ========================================
# 2. 요청 검증 (POST /predict)
# ========================================

class PayloadValidationError(ValueError):
    """요청 형식 오류. API 계층에서 422로 변환한다."""

    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


def is_valid_user_id(value):
    """사용자 ID는 비어 있지 않은 문자열 또는 정수(Spring의 Long)를 허용한다. bool은 제외."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, str) and value.strip() != ""


def validate_payload(payload):
    """오류 메시지 목록을 반환한다. 비어 있으면 유효한 요청이다."""
    if not isinstance(payload, dict):
        return ["요청 본문은 JSON 객체여야 합니다."]

    errors = []

    if not is_valid_user_id(payload.get("userId")):
        errors.append("userId는 필수이며 비어 있지 않은 문자열 또는 정수여야 합니다.")

    calculated_at = payload.get("calculatedAt")
    if not isinstance(calculated_at, str):
        errors.append("calculatedAt은 필수이며 ISO-8601 문자열이어야 합니다.")
    else:
        # Java ZonedDateTime이 붙이는 지역 이름(예: "+09:00[Asia/Seoul]")은 오프셋이 이미
        # 있으므로 떼고 해석한다. 오프셋 자체가 없으면 아래에서 그대로 거절된다.
        calculated_at = re.sub(r"\[[^\]]*\]$", "", calculated_at.strip())
        try:
            ts = pd.Timestamp(calculated_at)
            if pd.isna(ts) or ts.tzinfo is None:
                errors.append("calculatedAt에는 UTC 오프셋이 포함되어야 합니다.")
        except (TypeError, ValueError):
            errors.append("calculatedAt이 ISO-8601 형식이 아닙니다.")

    if payload.get("windowDays") != WINDOW_DAYS:
        errors.append(f"windowDays는 {WINDOW_DAYS}이어야 합니다.")

    features = payload.get("features")
    if not isinstance(features, dict):
        errors.append("features는 필수이며 객체여야 합니다.")
        return errors

    for name, spec in FEATURE_SPECS.items():
        camel = spec["camel"]
        value = features.get(camel)

        if value is None:
            if not spec["nullable"]:
                errors.append(f"features.{camel}은 null일 수 없습니다.")
            continue

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            errors.append(f"features.{camel}은 유한한 숫자여야 합니다.")
            continue

        if spec["integer"] and value != int(value):
            errors.append(f"features.{camel}은 정수여야 합니다.")
        if value < spec["min"]:
            errors.append(f"features.{camel}은 {spec['min']} 이상이어야 합니다.")
        if spec["max"] is not None and value > spec["max"]:
            errors.append(f"features.{camel}은 {spec['max']} 이하여야 합니다.")

    return errors


def payload_to_frame(payload, feature_columns):
    """검증된 요청을 모델 입력(1행 DataFrame)으로 바꾼다. null은 NaN으로 둔다."""
    snake = {
        CAMEL_TO_SNAKE[key]: value
        for key, value in payload["features"].items()
        if key in CAMEL_TO_SNAKE
    }
    frame = pd.DataFrame([snake]).reindex(columns=feature_columns)
    return frame.apply(pd.to_numeric, errors="coerce")


# ========================================
# 3. 결측치 대체 / 변환 / 스케일링
# ========================================

class SelectiveLog1p(BaseEstimator, TransformerMixin):
    """지정한 열 위치에만 log1p를 적용한다 (꼬리가 긴 값의 분포를 완화)."""

    def __init__(self, indices=()):
        self.indices = indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        for i in self.indices:
            X[:, i] = np.log1p(np.clip(X[:, i], 0, None))
        return X


def build_preprocessor(
    feature_columns,
    imputation="median",
    scaling="standard",
    log_transform=False,
):
    """결측값 대체 → (선택) log1p → (선택) 스케일링.

    imputation: "median" | "mean" | "zero"
    scaling   : "standard" | "robust" | "none"
    """
    if imputation == "zero":
        imputer = SimpleImputer(strategy="constant", fill_value=0)
    elif imputation in ("median", "mean"):
        imputer = SimpleImputer(strategy=imputation)
    else:
        raise ValueError(f"지원하지 않는 imputation: {imputation}")

    steps = [("imputer", imputer)]

    if log_transform:
        log_indices = tuple(
            i for i, col in enumerate(feature_columns)
            if FEATURE_SPECS[col]["log"]
        )
        steps.append(("log1p", SelectiveLog1p(indices=log_indices)))

    if scaling == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaling == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scaling != "none":
        raise ValueError(f"지원하지 않는 scaling: {scaling}")

    return Pipeline(steps)


# ========================================
# 4. 이상도 점수 정규화 (0~1, 높을수록 이상)
# ========================================

class ScoreNormalizer:
    """Isolation Forest 원점수(s = -score_samples)를 0~1 이상도로 변환한다.

    method:
      "percentile": 학습 사용자 점수 분포에서의 백분위 (0.95 = 학습 사용자의 95%보다 특이)
      "minmax"    : 학습 점수의 최소~최대를 0~1로 선형 변환
      "raw"       : 원점수를 0~1로 clip (원 논문 점수, 정상 ≈ 0.5 이하)
    """

    def __init__(self, method="percentile"):
        if method not in ("percentile", "minmax", "raw"):
            raise ValueError(f"지원하지 않는 method: {method}")
        self.method = method

    def fit(self, raw_scores):
        raw_scores = np.asarray(raw_scores, dtype=float)
        self.sorted_ = np.sort(raw_scores)
        self.min_ = float(raw_scores.min())
        self.max_ = float(raw_scores.max())
        return self

    def transform(self, raw_scores):
        raw_scores = np.asarray(raw_scores, dtype=float)

        if self.method == "percentile":
            ranks = np.searchsorted(self.sorted_, raw_scores, side="right")
            normalized = ranks / len(self.sorted_)
        elif self.method == "minmax":
            span = max(self.max_ - self.min_, 1e-12)
            normalized = (raw_scores - self.min_) / span
        else:
            normalized = raw_scores

        return np.clip(normalized, 0.0, 1.0)