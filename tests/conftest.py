import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import train_fds_model as fds
from app.main import app
from preprocessing import FEATURE_SPECS


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    """실제 학습 스크립트와 실제 학습용 CSV로 만든 번들 (서버가 쓰는 것과 같은 모델)."""
    path = tmp_path_factory.mktemp("model") / "test_model.joblib"
    fds.train(fds.TRAINING_CSV, path)
    return path


@pytest.fixture(scope="session")
def client(model_path):
    os.environ["MODEL_PATH"] = str(model_path)
    with TestClient(app) as test_client:  # with 블록이 lifespan(모델 로드)을 실행한다
        yield test_client


def normal_payload(user_id="user_normal"):
    """평범한 사용자: 시도 3건, 간격이 길고 분산은 간격 5개 미만이라 null."""
    return {
        "userId": user_id,
        "calculatedAt": "2026-10-08T15:00:00+09:00",
        "windowDays": 30,
        "features": {
            "bookingCount": 3,
            "avgBookingInterval": 250_000.0,
            "bookingIntervalVariance": None,
            "cancelRate": 0.0,
            "cardReuseCount": 1,
            "ipBookingDensity": 1,
        },
    }


def extreme_payload(user_id="user_extreme"):
    """극단적인 사용자: 짧고 일정한 간격으로 많이 시도하고, 취소가 많고, 카드·IP를 여러 계정이 같이 쓴다."""
    payload = normal_payload(user_id)
    payload["features"].update(
        bookingCount=30, avgBookingInterval=3.0, bookingIntervalVariance=0.05,
        cancelRate=0.9, cardReuseCount=6, ipBookingDensity=8,
    )
    return payload


def table_payloads(frame):
    """학습용 CSV의 행(결측 포함)을 요청 형식으로 바꾼다."""
    payloads = []
    for row in frame.itertuples(index=False):
        payload = normal_payload(f"row-{row.row_id}")
        payload["features"] = {
            spec["camel"]: None if pd.isna(getattr(row, name)) else (int(getattr(row, name)) if spec["integer"] else float(getattr(row, name)))
            for name, spec in FEATURE_SPECS.items()
        }
        payloads.append(payload)
    return payloads
