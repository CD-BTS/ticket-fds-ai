"""extract_features.py 가 BTS 계산식과 같은 값을 내는지 확인한다.

사례는 ticket-fds-admin feat/#40 의 FeatureAggregationServiceTest.java 를 그대로 옮긴 것이다
(같은 입력 → 같은 기대값). 아래 이벤트는 검증용 가짜 입력이며 실측 데이터가 아니다.

실행:  .venv/Scripts/python scripts/test_extract_features.py
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import extract  # noqa: E402

WINDOW_END = datetime(2026, 10, 31, 0, 0, 0)
T0 = datetime(2026, 10, 15, 10, 0, 0)


class Events:
    def __init__(self):
        self.rows = []

    def add(self, event_type, at, user="user_001", service="service_A", booking_id="", metadata=None):
        self.rows.append({
            "event_id": len(self.rows) + 1, "user_id": user, "external_user_id": f"ext-{user}", "group": "other",
            "service_id": service, "external_event_id": f"evt_{len(self.rows) + 1}", "event_type": event_type,
            "booking_id": booking_id, "ticket_id": "", "occurred_at": at, "meta": metadata or {},
        })
        return self

    def booking(self, booking_id, at, metadata=None, user="user_001"):
        return self.add("BOOKING", at, user=user, booking_id=booking_id, metadata=metadata)

    def cancel(self, booking_id, at):
        return self.add("CANCEL", at, booking_id=booking_id)

    def attempt(self, attempt_id, stage, at, service="service_A"):
        return self.add("BOOKING_ATTEMPT", at, service=service, metadata={"attemptId": attempt_id, "stage": stage})

    def features(self, user="user_001", gap_minutes=30):
        df = pd.DataFrame(self.rows).sort_values(["occurred_at", "event_id"]).reset_index(drop=True)
        features, intervals, _ = extract(df, WINDOW_END, gap_minutes)
        return features.set_index("user_id").loc[user], intervals


def near(actual, expected):
    return actual is not None and abs(actual - expected) < 1e-4


def sec(seconds):
    return T0 + timedelta(seconds=seconds)


def test_booking_count_uses_attempts():
    f, _ = Events().booking("b1", T0).booking("b2", sec(60)).attempt("a1", "STARTED", sec(-30)).features()
    assert f["booking_count"] == 1
    assert f["avg_booking_interval"] is None and f["booking_interval_variance"] is None


def test_attempt_stages_collapse():
    f, _ = (Events().attempt("a1", "STARTED", T0).attempt("a1", "QUEUE", sec(5)).attempt("a1", "SEAT_HOLD", sec(20))
            .attempt("a2", "STARTED", sec(30)).attempt("a2", "SEAT_HOLD", sec(31)).features())
    assert f["booking_count"] == 2
    assert near(f["avg_booking_interval"], 30.0)
    assert near(f["booking_interval_variance"], 0.0)
    assert bool(f["variance_degenerate"]) is True          # 간격 1개 → 분산 0은 계산상 결과
    assert f["session_interval_variance"] is None          # 세션 컬럼은 결측으로 둔다


def test_millisecond_intervals():
    f, _ = (Events().attempt("a1", "STARTED", T0).attempt("a2", "STARTED", sec(0.2))
            .attempt("a3", "STARTED", sec(0.6)).features())
    assert f["booking_count"] == 3
    assert near(f["avg_booking_interval"], 0.3)
    assert near(f["booking_interval_variance"], 0.01)       # 모분산
    assert near(f["session_interval_variance"], 0.02)       # 표본분산


def test_attempt_key_includes_service_and_requires_attempt_id():
    f, _ = (Events().attempt("a1", "STARTED", T0)
            .add("BOOKING_ATTEMPT", sec(5), metadata={"stage": "STARTED"})
            .attempt("a1", "STARTED", sec(10), service="service_B").features())
    assert f["booking_count"] == 2
    assert near(f["avg_booking_interval"], 10.0)


def test_cancel_rate_uses_distinct_bookings():
    f, _ = (Events().booking("b1", T0).booking("b1", sec(2)).booking("b2", sec(10))
            .booking("b3", sec(20)).cancel("b1", sec(30)).features())
    assert near(f["cancel_rate"], 1 / 3)
    assert f["booking_count"] == 0


def test_card_and_ip_use_max_shared_accounts():
    e = (Events().booking("b1", T0, {"paymentIdentifier": "card-A", "ipAddress": "10.0.0.1"})
         .booking("b2", sec(60), {"paymentIdentifier": "card-B", "ipAddress": "10.0.0.2"}))
    for n in range(3):   # card-B 를 쓰는 다른 계정 3명 → 본인 포함 4
        e.booking(f"x{n}", sec(100 + n), {"paymentIdentifier": "card-B"}, user=f"other_card_{n}")
    for n in range(5):   # 10.0.0.1 을 쓰는 다른 계정 5명 → 본인 포함 6
        e.booking(f"y{n}", sec(200 + n), {"ipAddress": "10.0.0.1"}, user=f"other_ip_{n}")
    e.booking("z", sec(300), {"ipAddress": "10.0.0.2"}, user="other_ip2")  # 10.0.0.2 → 2
    f, _ = e.features()
    assert f["card_reuse_count"] == 4
    assert f["ip_booking_density"] == 6


def test_missing_metadata_defaults_to_one():
    f, _ = Events().booking("b1", T0).booking("b2", sec(60), {"paymentIdentifier": " ", "ipAddress": ""}).features()
    assert f["card_reuse_count"] == 1 and f["ip_booking_density"] == 1


def test_window_boundaries_inclusive_and_future_excluded():
    start = WINDOW_END - timedelta(days=30)
    f, _ = (Events().attempt("before", "STARTED", start - timedelta(milliseconds=1))
            .attempt("at_start", "STARTED", start).attempt("at_end", "STARTED", WINDOW_END)
            .attempt("future", "STARTED", WINDOW_END + timedelta(milliseconds=1)).features())
    assert f["booking_count"] == 2


def test_session_split_keeps_30d_and_session_columns_separate():
    # 10초 간격 2번, 하루 뒤 다시 20초 간격 1번
    day = 86400
    f, intervals = (Events().attempt("a1", "S", T0).attempt("a2", "S", sec(10)).attempt("a3", "S", sec(20))
                    .attempt("a4", "S", sec(day)).attempt("a5", "S", sec(day + 20)).features())
    assert f["n_intervals_30d"] == 4 and f["n_session_intervals"] == 3 and f["n_sessions"] == 2
    assert near(f["avg_booking_interval"], (10 + 10 + (day - 20) + 20) / 4)   # 30일 전체: 긴 간격 포함
    assert near(f["session_interval_mean"], (10 + 10 + 20) / 3)               # 세션 내 간격만
    assert list(intervals["within_session"]) == [True, True, False, True]


def test_metadata_round_trip_from_csv(tmp_path=None):
    # export_events.py 가 쓰는 CSV 형식(문자열 metadata)을 load_events 로 읽을 수 있는지
    from extract_features import load_events
    import tempfile
    rows = [{"event_id": 1, "user_id": "u", "external_user_id": "human-01", "group": "human", "service_id": "s",
             "external_event_id": "e1", "event_type": "BOOKING_ATTEMPT", "booking_id": "", "ticket_id": "",
             "occurred_at_utc": "2026-10-15T10:00:00.123000",
             "metadata": json.dumps({"attemptId": "a1", "stage": "STARTED", "note": "탭\t줄바꿈\n\"따옴표\""})}]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "events.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        df = load_events(path)
    assert df.loc[0, "meta"]["attemptId"] == "a1"
    assert df.loc[0, "occurred_at"] == datetime(2026, 10, 15, 10, 0, 0, 123000)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}개 통과")
