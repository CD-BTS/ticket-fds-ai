"""Feature 계산 정답 샘플(docs/samples/golden) 검증.

HAND 표는 코드가 아니라 사람이 이벤트를 보고 직접 계산한 값이다 (계산 과정은 주석).
main.py의 계산이 이 값과 다르면 계산 규칙이 바뀐 것이므로, 백엔드(Spring) 구현과의
불일치를 막기 위해 의도한 변경인지 확인하고 문서를 함께 고쳐야 한다.
"""

import math
from pathlib import Path

import pandas as pd
import pytest

import main as pipeline  # 루트의 main.py (Feature 계산)
from tests.conftest import normal_payload

GOLDEN = Path(__file__).resolve().parent.parent / "docs" / "samples" / "golden"

COLUMNS = [
    "booking_count", "cancel_count", "cancel_rate", "avg_booking_interval",
    "short_interval_booking_count", "attendance_rate", "avg_booking_duration",
]

# 분석 기준 시각 2026-10-10T00:00Z, 분석 기간 2026-09-10T00:00Z ~ (30일). None = null
HAND = {
    # 예매 3건(9/15 10:00, 9/16 10:00, 9/20 12:00), b2만 취소, 관람은 b1만
    # 간격 86,400 / 352,800 → 평균 219,600 / 관람률 = b1 관람, b3 불참 → 1/2 / 소요 (60+120+30)/3 = 70
    "G01": [3, 1, 1 / 3, 219600.0, 0, 0.5, 70.0],
    # 예매 4건 10:00:00, :05, :11, :16 → 간격 5, 6, 5초. 5초 "이하"만 세므로 2회 / 공연 미종료라 관람률 null
    "G02": [4, 0, 0.0, 16 / 3, 2, None, 10.0],
    # 같은 external_event_id 중복과 재전송(3초 뒤)은 1건으로, 가장 이른 시각 사용 → 08:00:00 ~ 09:00:00 = 3,600초
    "G03": [2, 0, 0.0, 3600.0, 0, None, 30.0],
    # 9/9 23:59:59 예매는 분석 기간 밖(제외), 9/10 00:00:00 정각 예매는 포함. 밖 예매의 취소(9/12)는
    # cancel_count에는 1로 잡히지만, cancel_rate의 분자(분석 기간 내 예매 중 취소)는 0 → 0.0
    "G04": [2, 1, 0.0, 600.0, 0, 0.5, 30.0],
    # 예매 1건: 간격 null, 짧은 간격 0
    "G05": [1, 0, 0.0, None, 0, 1.0, 45.0],
    # 평가 시각 이후의 예매(10/10 00:00:01)와 취소(10/11)는 제외 → 예매 1건, 취소 0, 불참 1건 → 관람률 0.0
    "G06": [1, 0, 0.0, None, 0, 0.0, 20.0],
    # 예매 이벤트 없음, 취소 이벤트 1건: 예매 0건이라 cancel_rate null
    "G07": [0, 1, None, None, 0, None, None],
    # b1 공연 미종료(제외), b2 취소(제외), b3 관람 → 분모 1건 / 간격 86,400, 172,800 → 129,600
    "G08": [3, 1, 1 / 3, 129600.0, 0, 1.0, 30.0],
}


def assert_row(actual, expected, user):
    for column, want in zip(COLUMNS, expected):
        got = actual[column]
        if want is None:
            assert got is None or (isinstance(got, float) and math.isnan(got)), f"{user}.{column}: {got} (기대 null)"
        else:
            assert got == pytest.approx(want), f"{user}.{column}: {got} (기대 {want})"


@pytest.fixture
def computed(monkeypatch):
    monkeypatch.setattr(pipeline, "EVENT_PATH", GOLDEN / "events.csv")
    monkeypatch.setattr(pipeline, "TICKET_PATH", GOLDEN / "tickets.csv")
    return pipeline.compute_features(pipeline.load_events()).set_index("user_id")


def test_computation_matches_hand_calculation(computed):
    assert sorted(computed.index) == sorted(HAND)
    for user, expected in HAND.items():
        assert_row(computed.loc[user], expected, user)


def test_expected_features_file_matches_hand_calculation():
    expected = pd.read_csv(GOLDEN / "expected_features.csv", na_values=["null"]).set_index("user_id")
    assert sorted(expected.index) == sorted(HAND)
    for user, values in HAND.items():
        assert_row(expected.loc[user], values, user)


def test_golden_features_through_api(client):
    """정답 샘플의 Feature를 그대로 API에 보냈을 때의 상태 (백엔드가 보낼 값과 같은 형태)."""
    expected = pd.read_csv(GOLDEN / "expected_features.csv", na_values=["null"])
    camel = {
        "booking_count": "bookingCount", "cancel_count": "cancelCount", "cancel_rate": "cancelRate",
        "avg_booking_interval": "avgBookingInterval",
        "short_interval_booking_count": "shortIntervalBookingCount",
        "attendance_rate": "attendanceRate", "avg_booking_duration": "avgBookingDuration",
    }
    integer = {"booking_count", "cancel_count", "short_interval_booking_count"}

    payloads = []
    for _, row in expected.iterrows():
        payload = normal_payload(row["user_id"])
        payload["features"] = {
            camel[c]: None if pd.isna(row[c]) else (int(row[c]) if c in integer else float(row[c]))
            for c in COLUMNS
        }
        payloads.append(payload)

    results = client.post("/predict/batch", json={"requests": payloads}).json()["results"]
    status = {r["userId"]: r["status"] for r in results}

    # 예매 2건 이상은 점수를 내고, 그보다 적으면 INSUFFICIENT_DATA
    for user in ("G01", "G02", "G03", "G04", "G08"):
        assert status[user] == "OK", user
    for user in ("G05", "G06", "G07"):
        assert status[user] == "INSUFFICIENT_DATA", user