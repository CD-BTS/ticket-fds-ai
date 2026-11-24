"""모델 번들 저장/로드와 예측 로직 (학습 스크립트와 FastAPI 서버가 함께 사용).

번들(bundle) = 학습된 파이프라인 + 점수 정규화기 + 사용한 Feature 목록 + 버전 정보.
joblib(pickle 기반) 파일은 불러올 때 코드가 실행될 수 있으므로 신뢰할 수 있는 파일만 로드한다.
"""

import json
import logging
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from preprocessing import (
    FEATURE_SPECS,
    PayloadValidationError,
    is_valid_user_id,
    payload_to_frame,
    validate_payload,
)

logger = logging.getLogger(__name__)

BUNDLE_SCHEMA_VERSION = 2  # 2: 원시 임계값, Feature 정의 버전, 정규화·학습 정보 추가 (최종 6개 Feature 모델)

REQUIRED_KEYS = {
    "model",
    "normalizer",
    "feature_columns",
    "model_version",
    "min_booking_count",
    "anomaly_threshold",
    "raw_threshold",
    "feature_definition_version",
    "schema_version",
}


# ========================================
# 1. 번들 생성 / 저장 / 로드
# ========================================

def raw_anomaly_score(model, X):
    """Isolation Forest 원점수 s = -score_samples (높을수록 이상)."""
    return -model.score_samples(X)


def build_bundle(
    model,
    normalizer,
    feature_columns,
    preprocess,
    score_method,
    anomaly_threshold,
    model_version,
    min_booking_count,
    raw_threshold=None,
    feature_definition_version=None,
    normalization=None,
    training=None,
):
    """raw_threshold: 원시 점수 기준 탐지 임계값 (판정은 원시 점수 >= raw_threshold).
    anomaly_threshold: 같은 판정을 정규화 점수로 적은 값. 이전 MVP 스크립트는 raw_threshold 없이 호출하며,
    그렇게 만든 번들은 load_bundle 이 받지 않는다 (train_fds_model.py 로 다시 만들어야 한다)."""
    return {
        "model": model,
        "normalizer": normalizer,
        "feature_columns": list(feature_columns),
        "preprocess": dict(preprocess),
        "score_method": score_method,
        "anomaly_threshold": float(anomaly_threshold),
        "model_version": model_version,
        "min_booking_count": int(min_booking_count),
        "raw_threshold": None if raw_threshold is None else float(raw_threshold),
        "feature_definition_version": feature_definition_version,
        "normalization": dict(normalization or {}),
        "training": dict(training or {}),
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }


def meta_path_for(path):
    return Path(path).with_suffix(".meta.json")


def save_bundle(bundle, path):
    """번들을 .joblib로 저장하고, 사람이 읽을 수 있는 .meta.json을 옆에 함께 쓴다.

    임시 파일에 먼저 쓴 뒤 교체해서, 저장 도중 실패해도 기존 모델 파일이 깨지지 않는다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(path.name + ".tmp")
    joblib.dump(bundle, tmp_path, compress=3)
    os.replace(tmp_path, path)

    meta = {k: v for k, v in bundle.items() if k not in ("model", "normalizer")}
    meta_path_for(path).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_bundle(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {path} (python train_fds_model.py 로 만들거나 MODEL_PATH 를 확인하세요)"
        )

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise ValueError(
            "지원하지 않는 모델 파일 형식입니다. python train_fds_model.py 로 다시 만들어 주세요."
        )

    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"번들 스키마 버전이 다릅니다: 파일 {bundle.get('schema_version')}, "
            f"코드 {BUNDLE_SCHEMA_VERSION}. 이전 버전(MVP) 번들이면 python train_fds_model.py 로 다시 만들어 주세요."
        )

    missing = REQUIRED_KEYS - set(bundle)
    if missing:
        raise ValueError(f"모델 번들에 필요한 항목이 없습니다: {sorted(missing)}")

    if bundle["raw_threshold"] is None:
        raise ValueError("모델 번들에 원시 점수 임계값(raw_threshold)이 없습니다.")

    if list(bundle["feature_columns"]) != list(FEATURE_SPECS):
        raise ValueError(
            f"번들의 Feature 구성이 서버의 Feature 스펙과 다릅니다: {bundle['feature_columns']} / {list(FEATURE_SPECS)}"
        )

    trained_with = bundle.get("sklearn_version")
    if trained_with != sklearn.__version__:
        logger.warning(
            "scikit-learn 버전이 학습 시점(%s)과 다릅니다(현재 %s). 결과가 달라지거나 "
            "로드에 실패할 수 있으니 requirements.txt의 버전을 맞춰 주세요.",
            trained_with,
            sklearn.__version__,
        )

    return bundle


def set_inference_threads(bundle, n_jobs=1):
    """요청 1건을 처리할 때 스레드를 만드는 비용을 줄이기 위해 예측 병렬도를 낮춤. Isolation Forest는 학습 시점 n_jobs와 다르게 설정해도 됨."""
    model = bundle["model"]
    if hasattr(model, "named_steps") and "isolation_forest" in model.named_steps:
        model.named_steps["isolation_forest"].n_jobs = n_jobs
    elif hasattr(model, "n_jobs"):
        model.n_jobs = n_jobs
    return bundle


# ========================================
# 2. 예측 (POST /predict, 명세 7)
# ========================================

SCORE_DECIMALS = 6
_BELOW_HALF = 0.5 - 10 ** -SCORE_DECIMALS  # 반올림이 임계값 아래의 점수를 0.5 로 올리지 않게 한다


def _result(user_id, status, score, bundle, assessed_at, errors=None, raw_score=None, is_anomaly=None):
    result = {
        # 요청에서 숫자로 와도 응답은 항상 문자열로 통일한다.
        "userId": None if user_id is None else str(user_id),
        "status": status,
        "rawAnomalyScore": raw_score,
        "anomalyScore": score,
        "isAnomaly": is_anomaly,
        "modelVersion": bundle["model_version"],
        "assessedAt": assessed_at,
    }
    if errors is not None:
        result["errors"] = errors
    return result


def assess_batch(payloads, bundle):
    """요청 여러 건을 한 번에 예측한다. 결과 순서는 입력 순서와 같다.

    - 형식 오류가 있는 건은 전체를 실패시키지 않고 status="INVALID"와 errors를 담는다.
    - 최소 데이터 기준 미달(bookingCount < min_booking_count)은 status="INSUFFICIENT_DATA" 이고
      rawAnomalyScore, anomalyScore, isAnomaly 가 모두 None 이다. 정상 판정이나 점수 0 이 아니라 "판정하지 않음" 이다.
    """
    assessed_at = datetime.now(timezone.utc).isoformat()
    results = [None] * len(payloads)

    rows, row_positions = [], []
    for position, payload in enumerate(payloads):
        errors = validate_payload(payload)
        if errors:
            user_id = payload.get("userId") if isinstance(payload, dict) else None
            results[position] = _result(
                user_id if is_valid_user_id(user_id) else None,
                "INVALID", None, bundle, assessed_at, errors,
            )
            continue

        if payload["features"]["bookingCount"] < bundle["min_booking_count"]:
            results[position] = _result(
                payload["userId"], "INSUFFICIENT_DATA", None, bundle, assessed_at
            )
            continue

        rows.append(payload_to_frame(payload, bundle["feature_columns"]))
        row_positions.append(position)

    if rows:
        X = pd.concat(rows, ignore_index=True)
        raw_scores = raw_anomaly_score(bundle["model"], X)
        scores = bundle["normalizer"].transform(raw_scores)
        for position, raw, score in zip(row_positions, raw_scores, scores):
            # 판정은 반올림하지 않은 원시 점수와 원시 임계값으로 한다. 정규화 점수는 표시용으로만 반올림한다.
            is_anomaly = bool(raw >= bundle["raw_threshold"])
            score = round(float(score), SCORE_DECIMALS)
            score = max(score, 0.5) if is_anomaly else min(score, _BELOW_HALF)
            results[position] = _result(
                payloads[position]["userId"], "OK", score, bundle, assessed_at,
                raw_score=float(raw), is_anomaly=is_anomaly,
            )

    return results


def assess(payload, bundle):
    """요청 1건을 예측한다. 형식이 잘못되면 PayloadValidationError (API에서 422)."""
    errors = validate_payload(payload)
    if errors:
        raise PayloadValidationError(errors)

    result = assess_batch([payload], bundle)[0]
    result.pop("errors", None)
    return result