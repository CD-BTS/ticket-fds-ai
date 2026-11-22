"""이슈 #16 — 매크로 예매 데이터를 수집하고, 끝나면 바로 분석(analyze.py)까지 실행한다.

실행 (이 실험 폴더에서, e2e 스택과 프론트가 떠 있는 상태):
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\collect.py
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\collect.py --run-id pilot --plan macro-fast-91:fast:2
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\collect.py --continuous --baseline-run 20261122-main

하는 일
  1. 사전 점검: e2e 일회성 BTS DB 컨테이너, 프론트/Demo 응답, 계획한 계정에 기존 이벤트가 없는지
  2. 실행 환경 기록: 세 레포의 커밋, 프론트 .env 의 mock 설정, macro.js 해시, node/Playwright 버전
  3. 계정마다 `node macro.js` 를 예매 1건당 1번씩 실행 (#14 와 같은 방식. --runs 는 계정을 바꾸므로 쓰지 않는다)
     --continuous 를 주면 계정마다 `node macro.js --repeat N` 을 1번 실행한다 (브라우저·로그인을 유지한 연속 예매).
     연속 예매 중 실패하면 macro.js 가 원인을 기록하고 그 계정을 중단한다. 남은 건은 실행하지 않은 것으로 남긴다.
  4. Outbox 전송을 기다린 뒤 BTS DB 의 behavior_event 를 SELECT 로 내보냄
  5. analyze.py 실행 → CSV, 그래프, REPORT.md

출력: runs/<run-id>/  (이미 있으면 덮어쓰지 않고 멈춘다)
    environment.json                 실행 환경·조건
    macro_runs.csv                   매크로 실행 1건당 한 줄 (macro.js 결과 + 실행 시작·종료 시각)
    macro-results-<계정>.csv         macro.js 가 직접 쓴 원본 결과
    macro_stdout.log                 macro.js 출력
    raw_events.csv, users.csv        BTS behavior_event / user_account 내보내기
    export_summary.json              이벤트 수, Outbox 상태

DB 에는 SELECT 만 실행한다. 스택을 띄우거나 내리지 않는다.
"""

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
AI = EXP.parent.parent
WORKSPACE = AI.parent
ISSUE14 = AI / "experiments" / "issue14-real-validation"
sys.path.insert(0, str(ISSUE14 / "scripts"))

import export_events as ex14  # noqa: E402  (#14 의 내보내기 SQL·그룹 규칙을 그대로 쓴다)

# #14 본 수집(data/count)과 같은 계정·모드·횟수
DEFAULT_PLAN = "macro-fast-01:fast:3,macro-fast-02:fast:20,macro-jitter-01:jitter:20,macro-human-01:human:20"
MACRO_COLUMNS = ["time", "user", "mode", "seats", "result", "error", "ms_login", "ms_to_seatmap",
                 "ms_first_seat", "ms_hold", "ms_paid", "ms_total", "ms_booking"]
# macro.js --repeat (연속 예매) 가 덧붙이는 컬럼. 1회 예매 방식에서는 빈 칸으로 둔다.
CONTINUOUS_COLUMNS = ["session_seq", "ms_browser_launch", "ms_context_open", "ms_prev_paid_to_begin", "hold_tries",
                      "start_utc", "begin_utc", "paid_utc", "fail_step", "fail_url"]
RUN_COLUMNS = ["seq", "account", "mode", "run_index", "card_arg", "invoke_start_utc", "invoke_end_utc",
               "invoke_wall_ms", "returncode"] + MACRO_COLUMNS + CONTINUOUS_COLUMNS


def utc_now():
    return datetime.now(timezone.utc)


def iso_ms(value):
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"


def git_info(repo):
    def git(*args):
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8")
        return done.stdout.strip() if done.returncode == 0 else None
    status = git("status", "--porcelain")
    return {"path": str(repo), "commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "subject": git("log", "-1", "--format=%s"),
            "dirty_files": status.splitlines() if status else [],
            "origin_dev": git("rev-parse", "--verify", "--quiet", "origin/dev"),
            "tree_equals_origin_dev": (subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "origin/dev"],
                                                      capture_output=True).returncode == 0) if git("rev-parse", "--verify", "--quiet", "origin/dev") else None}


def http_ok(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except Exception as error:  # noqa: BLE001
        return f"실패: {error}"


def read_env_flags(path):
    flags = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("VITE_") and "=" in line:
                key, value = line.split("=", 1)
                flags[key] = value
    return flags


def parse_plan(text):
    plan = []
    for item in text.split(","):
        account, mode, count = item.strip().split(":")
        if mode not in ("fast", "jitter", "human"):
            raise SystemExit(f"--plan 의 모드는 fast|jitter|human 이어야 합니다: {item}")
        plan.append({"account": account, "mode": mode, "runs": int(count)})
    return plan


def sql(db_args, query):
    return ex14.run_sql(db_args, query)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=None, help="결과 폴더 이름. 기본: UTC 시각 (예: 20261122-141500)")
    parser.add_argument("--plan", default=DEFAULT_PLAN, help="계정:모드:횟수 를 쉼표로 나열. 기본은 #14 와 같은 구성")
    parser.add_argument("--seats", default="1")
    parser.add_argument("--concert", default="0")
    parser.add_argument("--card", default="1", help="macro.js --card. 1=테스트 카드 A(기본), 2=테스트 카드 B, 15자리=직접 입력")
    parser.add_argument("--base", default="http://localhost:5173")
    parser.add_argument("--headless", action="store_true", help="#14 는 창을 띄운 상태로 실행했다. 기본은 같은 조건(창 표시)")
    parser.add_argument("--timeout", default="60", help="macro.js --timeout (초)")
    parser.add_argument("--flush-wait", type=int, default=90, help="Outbox 전송을 기다리는 최대 시간(초)")
    parser.add_argument("--allow-existing", action="store_true", help="계정에 기존 이벤트가 있어도 진행 (간격에 섞이므로 비권장)")
    parser.add_argument("--note", default="", help="이 실행에 대한 메모. notes.md 에 저장되어 보고서에 실린다")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--continuous", action="store_true",
                        help="연속 예매: 계정마다 브라우저·로그인을 유지한 채 이어서 예매 (macro.js --repeat)")
    parser.add_argument("--continue-on-fail", action="store_true",
                        help="--continuous 에서 예매가 실패해도 다음 건을 시도 (기본: 원인을 기록하고 그 계정 중단)")
    parser.add_argument("--macro-debug", action="store_true",
                        help="macro.js --debug: 단계와 브라우저 오류(HTTP 409 등)를 macro_stdout.log 에 남긴다")
    parser.add_argument("--baseline-run", default=None,
                        help="분석할 때 비교 기준으로 쓸 기존 실행의 run-id (예: 20261122-main). 생략하면 #14 와 비교")
    parser.add_argument("--macro-dir", default=str(WORKSPACE / "demo-ticketing" / "macro"))
    parser.add_argument("--admin-repo", default=str(WORKSPACE / "ticket-fds-admin"))
    parser.add_argument("--container", default="cdbts-e2e-bts-mysql")
    parser.add_argument("--user", default="bts_e2e")
    parser.add_argument("--password", default="bts-e2e-only")
    parser.add_argument("--database", default="fairticket_db")
    args = parser.parse_args()

    run_id = args.run_id or utc_now().strftime("%Y%m%d-%H%M%S")
    run_dir = EXP / "runs" / run_id
    if run_dir.exists():
        raise SystemExit(f"{run_dir} 가 이미 있습니다. 덮어쓰지 않습니다. 다른 --run-id 를 쓰세요.")
    plan = parse_plan(args.plan)
    macro_dir = Path(args.macro_dir)
    demo_repo = macro_dir.parent

    # ---- 1. 사전 점검 ----
    front = http_ok(args.base)
    api = http_ok(args.base + "/api/v1/concerts")
    if front != 200 or api != 200:
        raise SystemExit(f"사이트가 준비되지 않았습니다. 프론트 {args.base}: {front}, 프론트를 거친 Demo API: {api}")
    names = ",".join("'" + p["account"].replace("'", "") + "'" for p in plan)
    existing = sql(args, f"SELECT u.external_user_id, COUNT(*) FROM behavior_event e JOIN user_account u "
                         f"ON u.user_id = e.user_id WHERE u.external_user_id IN ({names}) GROUP BY u.external_user_id")
    if existing and not args.allow_existing:
        raise SystemExit(f"계획한 계정에 이미 이벤트가 있습니다 (연속 예매 간격에 섞입니다): {existing}\n"
                         "새 계정 이름을 쓰거나 --allow-existing 을 붙이세요.")
    events_before = int(sql(args, "SELECT COUNT(*) FROM behavior_event")[0][0])

    # ---- 2. 환경 기록 ----
    run_dir.mkdir(parents=True)
    macro_js = macro_dir / "macro.js"
    playwright_pkg = macro_dir / "node_modules" / "playwright" / "package.json"
    node_version = subprocess.run(["node", "-v"], capture_output=True, text=True).stdout.strip()
    environment = {
        "run_id": run_id, "started_utc": iso_ms(utc_now()), "plan": plan,
        "macro_options": {"seats": args.seats, "concert": args.concert, "card": args.card, "base": args.base,
                          "headless": args.headless, "timeout_s": args.timeout},
        "continuous": args.continuous,
        "macro_debug": args.macro_debug,
        "baseline_run": args.baseline_run,
        "launcher": ("python subprocess: 계정마다 `node macro.js --repeat N` 1회 (브라우저·로그인 유지, 연속 예매)" if args.continuous
                     else "python subprocess: 예매 1건마다 `node macro.js` 1회 (#14 는 PowerShell ForEach-Object 반복)"),
        "repos": {"ai": git_info(AI), "demo-ticketing": git_info(demo_repo), "ticket-fds-admin": git_info(Path(args.admin_repo))},
        "frontend_env": read_env_flags(demo_repo / "frontend" / ".env"),
        "macro_js_sha256": hashlib.sha256(macro_js.read_bytes()).hexdigest(),
        "node": node_version,
        "playwright": json.loads(playwright_pkg.read_text(encoding="utf-8"))["version"] if playwright_pkg.exists() else None,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "bts_db": {"container": args.container, "database": args.database, "events_before_run": events_before,
                   "existing_events_for_plan_accounts": existing},
        "preflight": {"frontend_status": front, "demo_api_via_frontend_status": api},
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.note:
        (run_dir / "notes.md").write_text(args.note.strip() + "\n", encoding="utf-8")

    # ---- 3. 매크로 실행 ----
    runs_path = run_dir / "macro_runs.csv"
    with open(runs_path, "w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=RUN_COLUMNS).writeheader()
    log = open(run_dir / "macro_stdout.log", "a", encoding="utf-8")
    seq = 0
    results = Counter()
    for item in plan:
        out_csv = run_dir / f"macro-results-{item['account']}.csv"
        if args.continuous:
            command = ["node", "macro.js", "--user", item["account"], "--mode", item["mode"], "--seats", args.seats,
                       "--concert", args.concert, "--card", args.card, "--base", args.base,
                       "--timeout", args.timeout, "--out", str(out_csv), "--repeat", str(item["runs"])]
            if args.headless:
                command.append("--headless")
            if args.macro_debug:
                command.append("--debug")
            if args.continue_on_fail:
                command.append("--continue-on-fail")
            started = utc_now()
            tick = time.perf_counter()
            done = subprocess.run(command, cwd=macro_dir, capture_output=True, text=True, encoding="utf-8", errors="replace")
            wall_ms = round((time.perf_counter() - tick) * 1000)
            ended = utc_now()
            log.write(f"--- {item['account']} --repeat {item['runs']} rc={done.returncode}\n{done.stdout}{done.stderr}")
            log.flush()
            written = []
            if out_csv.exists():
                with open(out_csv, encoding="utf-8", newline="") as stream:
                    written = list(csv.DictReader(stream))
            common = {"account": item["account"], "mode": item["mode"], "card_arg": args.card,
                      "invoke_start_utc": iso_ms(started), "invoke_end_utc": iso_ms(ended),
                      "invoke_wall_ms": wall_ms, "returncode": done.returncode}
            with open(runs_path, "a", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=RUN_COLUMNS)
                for index in range(1, item["runs"] + 1):
                    seq += 1
                    if index <= len(written):
                        row = written[index - 1]
                        macro_row = {key: row.get(key, "") for key in MACRO_COLUMNS}
                        extra = {key: row.get(key, "") for key in CONTINUOUS_COLUMNS if key != "session_seq"}
                        extra["session_seq"] = row.get("seq", "")
                    else:
                        # 앞 건의 실패로 중단되어 실행하지 않은 예매. 값을 지어내지 않고 미실행으로만 남긴다.
                        macro_row = dict.fromkeys(MACRO_COLUMNS, "")
                        macro_row.update(user=item["account"], mode=item["mode"], seats=args.seats,
                                         result="NOT_RUN" if written else "CRASH",
                                         error="앞 예매의 실패로 중단되어 실행하지 않음" if written
                                         else ((done.stderr or done.stdout).strip().splitlines() or [""])[-1][:200])
                        extra = dict.fromkeys(CONTINUOUS_COLUMNS, "")
                    results[macro_row["result"]] += 1
                    writer.writerow({"seq": seq, "run_index": index, **common, **macro_row, **extra})
                    print(f"[{seq}] {item['account']} {index}/{item['runs']} {macro_row['result']} "
                          f"(예매 {macro_row['ms_booking'] or '-'}ms, 앞 완료→시작 {extra.get('ms_prev_paid_to_begin') or '-'}ms) "
                          f"{macro_row['error']}", flush=True)
            continue
        for index in range(1, item["runs"] + 1):
            seq += 1
            command = ["node", "macro.js", "--user", item["account"], "--mode", item["mode"], "--seats", args.seats,
                       "--concert", args.concert, "--card", args.card, "--base", args.base,
                       "--timeout", args.timeout, "--out", str(out_csv)]
            if args.headless:
                command.append("--headless")
            if args.macro_debug:
                command.append("--debug")
            rows_before = sum(1 for _ in open(out_csv, encoding="utf-8")) if out_csv.exists() else 1
            started = utc_now()
            tick = time.perf_counter()
            done = subprocess.run(command, cwd=macro_dir, capture_output=True, text=True, encoding="utf-8", errors="replace")
            wall_ms = round((time.perf_counter() - tick) * 1000)
            ended = utc_now()
            log.write(f"--- #{seq} {item['account']} {index}/{item['runs']} rc={done.returncode}\n{done.stdout}{done.stderr}")
            log.flush()

            macro_row = dict.fromkeys(MACRO_COLUMNS, "")
            if out_csv.exists():
                with open(out_csv, encoding="utf-8", newline="") as stream:
                    written = list(csv.DictReader(stream))
                if len(written) + 1 > rows_before:
                    macro_row = {key: written[-1].get(key, "") for key in MACRO_COLUMNS}
            if not macro_row["result"]:
                # macro.js 가 결과 줄을 쓰지 못하고 끝난 경우. 값을 지어내지 않고 실패로만 남긴다.
                macro_row.update(user=item["account"], mode=item["mode"], seats=args.seats, result="CRASH",
                                 error=(done.stderr or done.stdout).strip().splitlines()[-1][:200] if (done.stderr or done.stdout).strip() else "")
            results[macro_row["result"]] += 1
            with open(runs_path, "a", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=RUN_COLUMNS).writerow({
                    "seq": seq, "account": item["account"], "mode": item["mode"], "run_index": index,
                    "card_arg": args.card, "invoke_start_utc": iso_ms(started), "invoke_end_utc": iso_ms(ended),
                    "invoke_wall_ms": wall_ms, "returncode": done.returncode, **macro_row})
            print(f"[{seq}] {item['account']} {index}/{item['runs']} {macro_row['result']} "
                  f"(예매 {macro_row['ms_booking'] or '-'}ms, 실행 {wall_ms}ms) {macro_row['error']}", flush=True)
    log.close()

    # ---- 4. Outbox 전송 대기 ----
    expected_bookings = results["OK"]
    deadline = time.time() + args.flush_wait
    stored = 0
    while True:
        stored = int(sql(args, f"SELECT COUNT(*) FROM behavior_event e JOIN user_account u ON u.user_id = e.user_id "
                               f"WHERE e.event_type = 'BOOKING' AND u.external_user_id IN ({names})")[0][0])
        if stored >= expected_bookings or time.time() > deadline:
            break
        time.sleep(3)
    time.sleep(6)  # 마지막 BOOKING 과 같은 주기의 BOOKING_ATTEMPT 가 함께 도착하도록 한 주기(5초) 더 기다린다

    # ---- 5. 내보내기 ----
    events = []
    for row in sql(args, ex14.EVENT_SQL):
        (event_id, user_id, external_user_id, service_id, external_event_id,
         event_type, booking_id, ticket_id, occurred_at, metadata_hex) = row
        metadata = bytes.fromhex(metadata_hex).decode("utf-8") if metadata_hex else ""
        if metadata:
            json.loads(metadata)
        events.append({"event_id": event_id, "user_id": user_id, "external_user_id": external_user_id,
                       "group": ex14.group_of(external_user_id), "service_id": service_id,
                       "external_event_id": external_event_id, "event_type": event_type, "booking_id": booking_id,
                       "ticket_id": ticket_id, "occurred_at_utc": occurred_at, "metadata": metadata})
    users = [{"user_id": r[0], "service_id": r[1], "external_user_id": r[2], "group": ex14.group_of(r[2]),
              "created_at": r[3], "account_status": r[4]} for r in sql(args, ex14.USER_SQL)]
    for name, rows in (("raw_events.csv", events), ("users.csv", users)):
        with open(run_dir / name, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else ["empty"])
            writer.writeheader()
            writer.writerows(rows)

    outbox = None
    try:
        demo_args = argparse.Namespace(container="cdbts-e2e-demo-mysql", user="demo_e2e", password="demo-e2e-only",
                                       database="demo_ticketing_db")
        outbox = {status: int(count) for status, count in
                  sql(demo_args, "SELECT delivery_status, COUNT(*) FROM outbox_events GROUP BY delivery_status")}
    except SystemExit as error:
        outbox = {"조회 실패": str(error)[:200]}

    plan_accounts = {p["account"] for p in plan}
    summary = {
        "finished_utc": iso_ms(utc_now()),
        "macro_results": dict(results), "macro_runs": seq,
        "expected_bookings": expected_bookings, "bookings_stored_in_bts": stored,
        "flush_complete": stored >= expected_bookings,
        "events_total_in_db": len(events),
        "events_for_plan_accounts_by_type": dict(Counter(e["event_type"] for e in events if e["external_user_id"] in plan_accounts)),
        "events_by_group_and_type": {f"{g}/{t}": n for (g, t), n in
                                     sorted(Counter((e["group"], e["event_type"]) for e in events).items())},
        "occurred_at_min": min((e["occurred_at_utc"] for e in events), default=None),
        "occurred_at_max": max((e["occurred_at_utc"] for e in events), default=None),
        "demo_outbox_by_status": outbox,
    }
    (run_dir / "export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["flush_complete"]:
        print(f"주의: 성공한 예매 {expected_bookings}건 중 BTS 에 저장된 BOOKING 은 {stored}건입니다.")

    # ---- 6. 분석 ----
    if not args.skip_analyze:
        analyze = [sys.executable, str(Path(__file__).with_name("analyze.py")), "--run-id", run_id]
        if args.baseline_run:
            analyze += ["--baseline-run", args.baseline_run]
        done = subprocess.run(analyze)
        if done.returncode != 0:
            raise SystemExit("분석 단계가 실패했습니다. 수집 파일은 남아 있으니 analyze.py 를 다시 실행하세요.")
    print(f"완료: {run_dir}")


if __name__ == "__main__":
    main()
