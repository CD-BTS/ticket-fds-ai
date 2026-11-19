"""BTS DB의 behavior_event를 CSV로 내보낸다 (실험용, SELECT만 실행).

실행 (실험 폴더에서, e2e 스택이 떠 있는 상태):
    .venv/Scripts/python scripts/export_events.py --tag pilot

출력: data/<tag>/events.csv, data/<tag>/users.csv, data/<tag>/export_summary.json

기본 접속 정보는 demo-ticketing/e2e/compose.yaml의 일회성 테스트 DB 값이다.
다른 DB를 쓸 때는 --container/--user/--password/--database 로 바꾼다.

계정 그룹은 데모 로그인 ID(external_user_id) 접두사로 정한다.
    human-*         -> human
    macro-fast-*    -> macro_fast
    macro-jitter-*  -> macro_jitter
    macro-human-*   -> macro_human
    그 밖           -> other
"""

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent

GROUP_RULES = [
    (re.compile(r"^human-"), "human"),
    (re.compile(r"^macro-fast-"), "macro_fast"),
    (re.compile(r"^macro-jitter-"), "macro_jitter"),
    (re.compile(r"^macro-human-"), "macro_human"),
]

# metadata는 JSON 안의 탭·줄바꿈·따옴표가 깨지지 않도록 HEX로 받아서 복원한다.
EVENT_SQL = """
SELECT e.event_id, e.user_id, u.external_user_id, e.service_id, e.external_event_id,
       e.event_type, IFNULL(e.booking_id,''), IFNULL(e.ticket_id,''),
       DATE_FORMAT(e.occurred_at, '%Y-%m-%dT%H:%i:%s.%f'),
       IFNULL(HEX(CAST(e.metadata AS CHAR CHARACTER SET utf8mb4)), '')
FROM behavior_event e JOIN user_account u ON u.user_id = e.user_id
ORDER BY e.occurred_at, e.event_id
"""
USER_SQL = """
SELECT user_id, service_id, external_user_id, DATE_FORMAT(created_at, '%Y-%m-%dT%H:%i:%s'), account_status
FROM user_account ORDER BY created_at, user_id
"""


def group_of(external_user_id):
    for pattern, name in GROUP_RULES:
        if pattern.search(external_user_id):
            return name
    return "other"


def run_sql(args, sql):
    command = ["docker", "exec", "-e", f"MYSQL_PWD={args.password}", args.container, "mysql",
               f"-u{args.user}", args.database, "--batch", "--skip-column-names",
               "--default-character-set=utf8mb4", "-e", " ".join(sql.split())]
    done = subprocess.run(command, capture_output=True)
    if done.returncode != 0:
        raise SystemExit(f"SQL 실패: {done.stderr.decode('utf-8', 'replace')[:400]}")
    text = done.stdout.decode("utf-8")
    return [line.split("\t") for line in text.splitlines() if line]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="출력 폴더 이름 (예: pilot, main)")
    parser.add_argument("--container", default="cdbts-e2e-bts-mysql")
    parser.add_argument("--user", default="bts_e2e")
    parser.add_argument("--password", default="bts-e2e-only")
    parser.add_argument("--database", default="fairticket_db")
    args = parser.parse_args()

    out_dir = EXP / "data" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "events.csv").exists():
        raise SystemExit(f"{out_dir / 'events.csv'} 가 이미 있습니다. 덮어쓰지 않습니다. 다른 --tag 를 쓰세요.")

    events = []
    for row in run_sql(args, EVENT_SQL):
        (event_id, user_id, external_user_id, service_id, external_event_id,
         event_type, booking_id, ticket_id, occurred_at, metadata_hex) = row
        metadata = bytes.fromhex(metadata_hex).decode("utf-8") if metadata_hex else ""
        if metadata:
            json.loads(metadata)  # 복원 검증
        events.append({
            "event_id": event_id, "user_id": user_id, "external_user_id": external_user_id,
            "group": group_of(external_user_id), "service_id": service_id,
            "external_event_id": external_event_id, "event_type": event_type,
            "booking_id": booking_id, "ticket_id": ticket_id,
            "occurred_at_utc": occurred_at, "metadata": metadata,
        })

    users = [{"user_id": r[0], "service_id": r[1], "external_user_id": r[2], "group": group_of(r[2]),
              "created_at": r[3], "account_status": r[4]} for r in run_sql(args, USER_SQL)]

    for name, rows, fields in (
        ("events.csv", events, ["event_id", "user_id", "external_user_id", "group", "service_id",
                                "external_event_id", "event_type", "booking_id", "ticket_id",
                                "occurred_at_utc", "metadata"]),
        ("users.csv", users, ["user_id", "service_id", "external_user_id", "group", "created_at", "account_status"]),
    ):
        with open(out_dir / name, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "events": len(events), "users": len(users),
        "events_by_type": dict(Counter(e["event_type"] for e in events)),
        "users_by_group": dict(Counter(u["group"] for u in users)),
        "events_by_group_and_type": {f"{g}/{t}": n for (g, t), n in
                                     sorted(Counter((e["group"], e["event_type"]) for e in events).items())},
        "occurred_at_min": min((e["occurred_at_utc"] for e in events), default=None),
        "occurred_at_max": max((e["occurred_at_utc"] for e in events), default=None),
        "source": {"container": args.container, "database": args.database},
    }
    (out_dir / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
