import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import train_model as tm
from app.main import app
from inference import save_bundle


def make_features(n=400, seed=0):
    """테스트용 가짜 Feature. 대부분 평범하고 일부는 극단적인 사용자."""
    rng = np.random.default_rng(seed)
    booking_count = rng.integers(2, 12, n)
    cancel_count = rng.binomial(booking_count, 0.15)
    short_count = np.zeros(n, dtype=int)

    df = pd.DataFrame({
        "user_id": [f"U{i:04d}" for i in range(n)],
        "booking_count": booking_count,
        "cancel_count": cancel_count,
        "cancel_rate": cancel_count / booking_count,
        "avg_booking_interval": rng.uniform(20_000, 150_000, n),
        "short_interval_booking_count": short_count,
        "attendance_rate": rng.uniform(0.6, 1.0, n),
        "avg_booking_duration": rng.uniform(30, 300, n),
    })

    # 극단적인 사용자 20명: 많이 예매하고 간격이 매우 짧다
    extreme = df.index[:20]
    df.loc[extreme, "booking_count"] = 28
    df.loc[extreme, "cancel_count"] = 9
    df.loc[extreme, "cancel_rate"] = 9 / 28
    df.loc[extreme, "avg_booking_interval"] = 200.0
    df.loc[extreme, "short_interval_booking_count"] = 18
    return df


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    features = make_features()
    model, normalizer, _ = tm.train_and_score(features, tm.MVP_FEATURES)
    bundle = tm.make_bundle(model, normalizer, tm.MVP_FEATURES)
    return save_bundle(bundle, tmp_path_factory.mktemp("model") / "test_model.joblib")


@pytest.fixture(scope="session")
def client(model_path):
    import os

    os.environ["MODEL_PATH"] = str(model_path)
    with TestClient(app) as test_client:  # with 블록이 lifespan(모델 로드)을 실행한다
        yield test_client


def normal_payload(user_id="user_normal"):
    return {
        "userId": user_id,
        "calculatedAt": "2026-10-08T15:00:00+09:00",
        "windowDays": 30,
        "features": {
            "bookingCount": 6,
            "cancelCount": 1,
            "cancelRate": 0.17,
            "avgBookingInterval": 80_000.0,
            "shortIntervalBookingCount": 0,
            "attendanceRate": 0.9,
            "avgBookingDuration": 120.0,
        },
    }


def extreme_payload(user_id="user_extreme"):
    payload = normal_payload(user_id)
    payload["features"].update(
        bookingCount=29, cancelCount=9, cancelRate=0.31,
        avgBookingInterval=190.0, shortIntervalBookingCount=19,
    )
    return payload