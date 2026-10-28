"""Feature 계산 정답 샘플(golden)을 만든다.

백엔드(Spring)가 Feature를 Java/SQL로 다시 구현할 때, 같은 이벤트를 넣어서 같은 값이 나오는지
비교하기 위한 작은 데이터다. 합성 데이터에는 없는 경계 케이스(정확히 5초 간격, 30일 경계,
중복 이벤트, 미래 이벤트 등)를 일부러 넣었다.

출력 (docs/samples/golden/)
    events.csv             behavior_event와 같은 형식의 이벤트
    tickets.csv            ticket과 같은 형식 (공연 종료 시각)
    expected_features.csv  사용자별 기대 Feature (분석 기준 시각 2026-10-10T00:00:00Z, 30일)

기대값은 main.py의 계산 결과이며, 사람이 손으로 계산한 값과 일치하는지는
tests/test_golden_features.py가 검증한다.

실행:  python scripts/build_golden.py
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "docs" / "samples" / "golden"

UTC = timezone.utc


def t(text):
    """'2026-09-15 10:00:00' → UTC datetime"""
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)


class Builder:
    def __init__(self):
        self.events, self.tickets = [], []
        self._event_id = 0

    def _event(self, user, event_type, booking_id, ticket_id, perf, occurred_at,
               metadata=None, external_id=None):
        self._event_id += 1
        self.events.append({
            "event_id": self._event_id,
            "external_event_id": external_id or f"GEXT-{self._event_id:04d}",
            "user_id": user,
            "event_type": event_type,
            "booking_id": booking_id,
            "ticket_id": ticket_id,
            "performance_id": perf,
            "occurred_at": occurred_at.isoformat(),
            "metadata": json.dumps(metadata or {}),
        })

    def booking(self, user, n, completed, duration, show_ended, *, check_in=None,
                cancel=None, occurred_at=None, external_id=None, add_ticket=True):
        booking_id, ticket_id, perf = f"B-{user}-{n}", f"T-{user}-{n}", f"PERF-{user}"
        completed = t(completed)
        meta = {
            "booking_open_at": (completed - timedelta(hours=1)).isoformat(),
            "booking_started_at": (completed - timedelta(seconds=duration)).isoformat(),
            "booking_completed_at": completed.isoformat(),
            "booking_duration_seconds": duration,
        }
        self._event(user, "BOOKING", booking_id, ticket_id, perf,
                    t(occurred_at) if occurred_at else completed, meta, external_id)
        if add_ticket:
            self.tickets.append({
                "ticket_id": ticket_id, "user_id": user, "booking_id": booking_id,
                "performance_id": perf, "show_ended_at": t(show_ended).isoformat(),
            })
        if check_in:
            self._event(user, "CHECK_IN", booking_id, ticket_id, perf, t(check_in))
        if cancel:
            self._event(user, "CANCEL", booking_id, ticket_id, perf, t(cancel))

    def raw_event(self, user, event_type, booking_id, occurred_at, **kwargs):
        self._event(user, event_type, booking_id, f"T-{booking_id}", f"PERF-{user}",
                    t(occurred_at), **kwargs)


def build():
    b = Builder()

    # G01 기본: 예매 3건, 1건 취소, 관람 1건 / 불참 1건
    b.booking("G01", 1, "2026-09-15 10:00:00", 60, "2026-09-17 14:00:00", check_in="2026-09-17 11:00:00")
    b.booking("G01", 2, "2026-09-16 10:00:00", 120, "2026-09-18 14:00:00", cancel="2026-09-16 10:30:00")
    b.booking("G01", 3, "2026-09-20 12:00:00", 30, "2026-09-21 14:00:00")

    # G02 5초 경계: 간격 5초(포함), 6초(제외), 5초(포함). 공연이 아직 안 끝나 관람률 계산 불가
    for n, when in enumerate(["10:00:00", "10:00:05", "10:00:11", "10:00:16"], start=1):
        b.booking("G02", n, f"2026-09-15 {when}", 10, "2026-10-20 20:00:00")

    # G03 중복: 같은 external_event_id 2번 + 같은 예매가 다른 이벤트 ID로 재전송 (가장 이른 것 사용)
    b.booking("G03", 1, "2026-09-20 08:00:00", 40, "2026-10-20 20:00:00", external_id="GEXT-DUP")
    b.raw_event("G03", "BOOKING", "B-G03-1", "2026-09-20 08:00:00", external_id="GEXT-DUP",
                metadata={"booking_started_at": "2026-09-20T07:59:20+00:00",
                          "booking_completed_at": "2026-09-20T08:00:00+00:00"})
    b.raw_event("G03", "BOOKING", "B-G03-1", "2026-09-20 08:00:03",
                metadata={"booking_started_at": "2026-09-20T07:59:23+00:00",
                          "booking_completed_at": "2026-09-20T08:00:03+00:00"})
    b.booking("G03", 2, "2026-09-20 09:00:00", 20, "2026-10-20 20:00:00")

    # G04 30일 경계: 분석 시작 1초 전 예매(제외), 정확히 시작 시각 예매(포함). 제외된 예매의 취소는 cancel_count에만 반영
    b.booking("G04", 0, "2026-09-09 23:59:59", 30, "2026-09-11 20:00:00", cancel="2026-09-12 10:00:00")
    b.booking("G04", 1, "2026-09-10 00:00:00", 30, "2026-09-15 20:00:00", check_in="2026-09-15 18:00:00")
    b.booking("G04", 2, "2026-09-10 00:10:00", 30, "2026-09-15 21:00:00")

    # G05 예매 1건: 간격 계산 불가(null)
    b.booking("G05", 1, "2026-09-25 10:00:00", 45, "2026-09-26 14:00:00", check_in="2026-09-26 11:00:00")

    # G06 미래 이벤트: 평가 시각(2026-10-10) 이후의 예매와 취소는 제외
    b.booking("G06", 1, "2026-09-20 10:00:00", 20, "2026-09-21 10:00:00", cancel="2026-10-11 10:00:00")
    b.booking("G06", 2, "2026-10-10 00:00:01", 20, "2026-10-12 10:00:00")

    # G07 취소 이벤트만 있는 사용자 (예매 이벤트는 분석 기간 이전)
    b.raw_event("G07", "CANCEL", "B-G07-0", "2026-09-20 10:00:00")

    # G08 관람률: 공연 미종료(제외), 취소(제외), 관람 완료 1건만 분모
    b.booking("G08", 1, "2026-09-20 10:00:00", 30, "2026-10-12 20:00:00")
    b.booking("G08", 2, "2026-09-21 10:00:00", 30, "2026-09-22 20:00:00", cancel="2026-09-21 10:05:00")
    b.booking("G08", 3, "2026-09-23 10:00:00", 30, "2026-09-24 20:00:00", check_in="2026-09-24 17:00:00")

    return pd.DataFrame(b.events), pd.DataFrame(b.tickets)


def main():
    import main as pipeline  # 루트의 main.py (Feature 계산)

    events, tickets = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUT_DIR / "events.csv", index=False)
    tickets.to_csv(OUT_DIR / "tickets.csv", index=False)

    pipeline.EVENT_PATH = OUT_DIR / "events.csv"
    pipeline.TICKET_PATH = OUT_DIR / "tickets.csv"
    features = pipeline.compute_features(pipeline.load_events())
    features.to_csv(OUT_DIR / "expected_features.csv", index=False, na_rep="null")

    print(f"이벤트 {len(events)}건, 티켓 {len(tickets)}건, 사용자 {len(features)}명 → {OUT_DIR}")
    pd.set_option("display.width", 200)
    print(features.round(4).to_string(index=False))


if __name__ == "__main__":
    main()