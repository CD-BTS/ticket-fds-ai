"""스크립트 동작 점검용 가짜 이벤트를 만든다. 실측 데이터가 아니다.

출력: data/_selftest/events.csv  (export_events.py 와 같은 형식)
값은 임의로 정한 것이라 결과 수치에는 아무 의미가 없다. 파이프라인이 끝까지 도는지만 본다.

실행:  .venv/Scripts/python scripts/make_selftest_events.py
"""

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
OUT = EXP / "data" / "_selftest"
START = datetime(2026, 11, 19, 1, 0, 0)

# (그룹, 계정 접두사, 시도 간격 범위(초), 시도 수 목록)
GROUPS = [
    ("human", "human-", (40, 150), [5, 5, 4, 2, 1]),
    ("macro_fast", "macro-fast-", (6, 9), [5, 5, 5]),
    ("macro_jitter", "macro-jitter-", (9, 14), [5, 5, 5]),
    ("macro_human", "macro-human-", (25, 40), [5, 5, 5]),
]


def main():
    rng = random.Random(14)
    rows, event_id = [], 0

    def add(user, external, group, event_type, at, booking_id="", metadata=None):
        nonlocal event_id
        event_id += 1
        rows.append({"event_id": event_id, "user_id": user, "external_user_id": external, "group": group,
                     "service_id": "demo-ticketing", "external_event_id": f"selftest-{event_id}",
                     "event_type": event_type, "booking_id": booking_id, "ticket_id": "",
                     "occurred_at_utc": at.isoformat(timespec="microseconds"),
                     "metadata": json.dumps(metadata or {}, ensure_ascii=False)})

    clock = START
    for group, prefix, (low, high), attempt_counts in GROUPS:
        for index, n_attempts in enumerate(attempt_counts, start=1):
            external, user = f"{prefix}{index:02d}", f"uuid-{prefix}{index:02d}"
            at = clock
            for attempt in range(n_attempts):
                attempt_id = f"{external}-a{attempt + 1}"
                add(user, external, group, "BOOKING_ATTEMPT", at,
                    metadata={"attemptId": attempt_id, "concertId": "c1", "stage": "STARTED", "status": "ENTERED", "sequence": 1})
                done = at + timedelta(seconds=rng.uniform(3, 20))
                add(user, external, group, "BOOKING_ATTEMPT", done,
                    metadata={"attemptId": attempt_id, "concertId": "c1", "stage": "COMPLETED", "status": "SUCCEEDED", "sequence": 5})
                booking_id = f"b_{external}_{attempt + 1}"
                add(user, external, group, "BOOKING", done, booking_id,
                    {"concertId": "c1", "attemptId": attempt_id, "paymentIdentifier": "test-pay-shared-03",
                     "ipAddress": "127.0.0.1", "bookingDurationSeconds": 3.2})
                if group == "human" and attempt == 0 and index == 1:
                    add(user, external, group, "CANCEL", done + timedelta(seconds=30), booking_id, {"concertId": "c1"})
                at = at + timedelta(seconds=rng.uniform(low, high), milliseconds=rng.randint(0, 999))
            clock = at + timedelta(minutes=2)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "events.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"가짜 점검용 이벤트 {len(rows)}건 저장: {OUT / 'events.csv'}")


if __name__ == "__main__":
    main()
