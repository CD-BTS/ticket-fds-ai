"""정상 사용자 실측: 수집 전 점검과 이벤트 내보내기. DB 에는 SELECT 만 실행한다. 스택을 띄우거나 내리지 않는다.

    cd ai\\experiments\\issue17-event-based-dummy
    $py = "..\\issue14-real-validation\\.venv\\Scripts\\python"
    & $py measurement\\scripts\\collect.py check                       # 예매를 시작하기 전
    & $py measurement\\scripts\\collect.py export --run-id 20261125-u01 # 예매를 마친 뒤, 스택을 끄기 전

계정 ID 규칙 (데모 로그인 ID = BTS external_user_id)
    u<참가자 2자리>-<실험 a|b|c>-<회차 2자리>      예: u01-a-01, u01-b-01, u02-c-01
    helper-<2자리>                                 실험 C 에서 좌석을 먼저 잡아 두는 보조 계정. 분석에서 뺀다
  참가자 번호와 실제 사람의 대응은 이 폴더에 적지 않는다 (익명).

내보내기는 #14 의 SQL(export_events.py)을 그대로 쓴다. 출력은 #14·#16 과 다른 폴더에 저장한다:
    measurement/data/<run-id>/raw_events.csv, users.csv, export_summary.json, environment.json
이미 있는 run-id 는 덮어쓰지 않는다. measurement/data/ 는 git 에 올리지 않는다.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEASUREMENT = HERE.parent
EXP = MEASUREMENT.parent
AI = EXP.parents[1]
WORKSPACE = AI.parent
sys.path.insert(0, str(EXP.parent / "issue14-real-validation" / "scripts"))
import export_events as ex14  # noqa: E402  (#14 의 내보내기 SQL 과 docker exec 호출을 그대로 쓴다. 수정 없음)

DATA = MEASUREMENT / "data"
ACCOUNT = re.compile(r"^u(\d{2})-([abc])-(\d{2})$")
HELPER = re.compile(r"^helper-\d{2}$")
E2E_CONTAINERS = ["cdbts-e2e-bts-mysql", "cdbts-e2e-demo-mysql", "cdbts-e2e-demo-redis"]
BTS_DB = dict(container="cdbts-e2e-bts-mysql", user="bts_e2e", password="bts-e2e-only", database="fairticket_db")
DEMO_DB = dict(container="cdbts-e2e-demo-mysql", user="demo_e2e", password="demo-e2e-only", database="demo_ticketing_db")
EVENT_FIELDS = ["event_id", "user_id", "external_user_id", "group", "participant", "experiment", "round", "service_id", "external_event_id",
                "event_type", "booking_id", "ticket_id", "occurred_at_utc", "metadata"]


def parse_account(external_user_id):
    """계정 ID → (그룹, 참가자, 실험, 회차). 규칙에 맞지 않으면 other."""
    match = ACCOUNT.match(external_user_id)
    if match:
        return "participant", f"u{match.group(1)}", match.group(2).upper(), match.group(3)
    if HELPER.match(external_user_id):
        return "helper", "", "", ""
    return "other", "", "", ""


def sql(db, query):
    return ex14.run_sql(argparse.Namespace(**db), query)


def http_status(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except Exception as error:  # 연결 실패
        return f"연결 실패 ({type(error).__name__})"


def git_info(repo):
    def git(*args):
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8")
        return done.stdout.strip() if done.returncode == 0 else None
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "subject": git("log", "-1", "--format=%s"),
            "modified_tracked_files": [line for line in (status or "").splitlines() if not line.startswith("??")]}


def running_containers():
    done = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, encoding="utf-8")
    if done.returncode != 0:
        raise SystemExit("Docker 에 연결할 수 없습니다. Docker Desktop 을 켜고 e2e 스택을 띄운 뒤 다시 실행하세요.\n" + done.stderr.strip()[:300])
    return set(done.stdout.split())


def counts_by_account():
    rows = sql(BTS_DB, "SELECT u.external_user_id, e.event_type, COUNT(*) FROM behavior_event e JOIN user_account u ON u.user_id = e.user_id "
                       "GROUP BY u.external_user_id, e.event_type")
    table = {}
    for account, event_type, count in rows:
        table.setdefault(account, Counter())[event_type] = int(count)
    return table


def outbox_status():
    try:
        return {status: int(count) for status, count in sql(DEMO_DB, "SELECT delivery_status, COUNT(*) FROM outbox_events GROUP BY delivery_status")}
    except SystemExit as error:
        return {"조회 실패": str(error)[:200]}


def preflight(base):
    missing = [name for name in E2E_CONTAINERS if name not in running_containers()]
    if missing:
        raise SystemExit(f"e2e 컨테이너가 떠 있지 않습니다: {missing}\ndemo-ticketing\\e2e\\Start-E2E.ps1 -DemoPort 8080 을 먼저 실행하세요.")
    front, api = http_status(base), http_status(base + "/api/v1/concerts")
    return front, api


def command_check(args):
    front, api = preflight(args.base)
    print(f"프론트 {args.base}: {front}")
    print(f"프론트를 거친 Demo API (/api/v1/concerts): {api}")
    if front != 200 or api != 200:
        print("  → 프론트(npm run dev)와 Demo 백엔드가 떠 있는지, 프론트의 /api 프록시가 Demo 포트를 가리키는지 확인하세요.")
    env_path = WORKSPACE / "demo-ticketing" / "frontend" / ".env"
    flags = dict(line.split("=", 1) for line in env_path.read_text(encoding="utf-8").splitlines() if "=" in line and not line.startswith("#")) if env_path.exists() else {}
    print(f"프론트 .env: VITE_USE_MOCK_TICKETING={flags.get('VITE_USE_MOCK_TICKETING')}, VITE_USE_MOCK_BTS={flags.get('VITE_USE_MOCK_BTS')}")
    if flags.get("VITE_USE_MOCK_TICKETING", "true") != "false":
        print("  → VITE_USE_MOCK_TICKETING 이 false 가 아니면 예매가 브라우저 안에서만 처리되고 이벤트가 저장되지 않습니다.")
    table = counts_by_account()
    total = sum(sum(c.values()) for c in table.values())
    print(f"BTS behavior_event: 전체 {total}건, 계정 {len(table)}개")
    for account in sorted(table):
        group, participant, experiment, round_no = parse_account(account)
        print(f"  {account:<16} {group:<12} {dict(table[account])}")
    used = sorted(a for a in table if parse_account(a)[0] == "participant")
    print(f"이미 이벤트가 있는 참가자 계정: {used if used else '없음'}  (새 실험에는 쓰지 않은 계정 ID 를 쓰세요)")
    print(f"Demo outbox: {outbox_status()}")
    print("점검 끝. DB 는 읽기만 했습니다.")


def command_export(args):
    run_dir = DATA / args.run_id
    if run_dir.exists():
        raise SystemExit(f"{run_dir} 가 이미 있습니다. 덮어쓰지 않습니다. 다른 --run-id 를 쓰세요.")
    front, api = preflight(args.base)

    # Outbox 전송(5초 주기)이 끝날 때까지 기다린다: BTS 의 이벤트 수가 두 번 연속 같으면 끝난 것으로 본다
    previous, deadline = -1, time.time() + args.flush_wait
    while True:
        current = int(sql(BTS_DB, "SELECT COUNT(*) FROM behavior_event")[0][0])
        if current == previous or time.time() > deadline:
            break
        previous = current
        print(f"전송 대기 중 (BTS 이벤트 {current}건) ...", flush=True)
        time.sleep(8)
    flushed = current == previous

    events = []
    for row in sql(BTS_DB, ex14.EVENT_SQL):
        event_id, user_id, external_user_id, service_id, external_event_id, event_type, booking_id, ticket_id, occurred_at, metadata_hex = row
        metadata = bytes.fromhex(metadata_hex).decode("utf-8") if metadata_hex else ""
        if metadata:
            json.loads(metadata)  # 복원 검증
        group, participant, experiment, round_no = parse_account(external_user_id)
        events.append({"event_id": event_id, "user_id": user_id, "external_user_id": external_user_id, "group": group, "participant": participant,
                       "experiment": experiment, "round": round_no, "service_id": service_id, "external_event_id": external_event_id,
                       "event_type": event_type, "booking_id": booking_id, "ticket_id": ticket_id, "occurred_at_utc": occurred_at, "metadata": metadata})
    users = []
    for user_id, service_id, external_user_id, created_at, status in sql(BTS_DB, ex14.USER_SQL):
        group, participant, experiment, round_no = parse_account(external_user_id)
        users.append({"user_id": user_id, "service_id": service_id, "external_user_id": external_user_id, "group": group, "participant": participant,
                      "experiment": experiment, "round": round_no, "created_at": created_at, "account_status": status})
    if not any(e["group"] == "participant" for e in events):
        raise SystemExit("참가자 계정(u01-a-01 형식)의 이벤트가 없습니다. 계정 ID 규칙과 프론트 설정을 확인하세요. 아무것도 저장하지 않았습니다.")

    run_dir.mkdir(parents=True)
    for name, rows, fields in (("raw_events.csv", events, EVENT_FIELDS), ("users.csv", users, list(users[0].keys()))):
        with open(run_dir / name, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    mine = [e for e in events if e["group"] == "participant"]
    by_account = {}
    for event in mine:
        by_account.setdefault(event["external_user_id"], Counter())[event["event_type"]] += 1
    summary = {
        "run_id": args.run_id, "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "note": args.note,
        "events_total_in_db": len(events), "events_by_group": dict(Counter(e["group"] for e in events)),
        "participants": sorted({e["participant"] for e in mine}),
        "accounts_by_experiment": {x: sorted({e["external_user_id"] for e in mine if e["experiment"] == x}) for x in "ABC"},
        "events_by_account_and_type": {account: dict(counter) for account, counter in sorted(by_account.items())},
        "occurred_at_min": min(e["occurred_at_utc"] for e in mine), "occurred_at_max": max(e["occurred_at_utc"] for e in mine),
        "flush_stable": flushed, "demo_outbox_by_status": outbox_status(),
        "source": {"container": BTS_DB["container"], "database": BTS_DB["database"], "sql": "issue14-real-validation/scripts/export_events.py 의 EVENT_SQL, USER_SQL"},
        "accounts_outside_naming_rule": sorted({e["external_user_id"] for e in events if e["group"] == "other"}),
    }
    (run_dir / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env_path = WORKSPACE / "demo-ticketing" / "frontend" / ".env"
    environment = {
        "repos": {name: git_info(WORKSPACE / name) for name in ("demo-ticketing", "ticket-fds-admin")}, "ai": git_info(AI),
        "frontend_env": dict(line.split("=", 1) for line in env_path.read_text(encoding="utf-8").splitlines() if "=" in line and not line.startswith("#"))
        if env_path.exists() else None,
        "preflight": {"frontend_status": front, "demo_api_via_frontend_status": api, "base": args.base},
        "not_recorded_here": "Demo 백엔드의 HOLD_DURATION 등 실행 시 환경변수는 이 스크립트가 읽을 수 없다. 기본값(5분)과 다르게 띄웠다면 manifest.csv 에 적는다.",
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    template = MEASUREMENT / "manifest_template.csv"
    if template.exists() and not (run_dir / "manifest.csv").exists():
        (run_dir / "manifest.csv").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not flushed:
        print("주의: 대기 시간 안에 이벤트 수가 안정되지 않았습니다. 잠시 뒤 다른 --run-id 로 다시 내보내 비교하세요.")
    print(f"저장: {run_dir}\n다음: manifest.csv 를 채우고, measurement\\scripts\\analyze.py --run-id {args.run_id} 를 실행하세요.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "export"):
        part = sub.add_parser(name)
        part.add_argument("--base", default="http://localhost:5173", help="프론트 주소")
        if name == "export":
            part.add_argument("--run-id", required=True, help="저장 폴더 이름 (예: 20261125-u01). 날짜와 참가자 번호를 넣는다")
            part.add_argument("--flush-wait", type=int, default=60, help="Outbox 전송을 기다리는 최대 시간(초)")
            part.add_argument("--note", default="", help="이 수집에 대한 메모")
    args = parser.parse_args()
    (command_check if args.command == "check" else command_export)(args)


if __name__ == "__main__":
    main()
