"""이슈 #16 — 수집한 실행(runs/<run-id>)을 #14 실측·더미데이터와 비교해 CSV·그래프·REPORT.md 를 만든다.

실행 (이 실험 폴더에서):
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\analyze.py --run-id <run-id>
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\analyze.py --run-id <run-id> --figures-only
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\analyze.py --run-id <run-id> --baseline-run <기존 run-id>

비교 기준 (둘 중 하나)
    기본                    : 기존 #14 실측 (issue14-real-validation/data/count)
    --baseline-run <run-id> : 이 폴더의 다른 실행 (예: 재실행 방식 20261122-main 대 연속 예매 실행).
                              생략해도 environment.json 에 baseline_run 이 적혀 있으면 그 값을 쓴다.
    기준 실행의 파일은 읽기만 한다. 결과는 항상 runs/<run-id>/ 안에만 쓴다.

입력 (모두 읽기 전용)
    runs/<run-id>/raw_events.csv, macro_runs.csv, environment.json      collect.py 출력 (#16 신규 실측)
    ../issue14-real-validation/data/count/events.csv, macro-results-*   #14 실측 (수정하지 않음)
    ../../ml_experiment/data/fairticket_synthetic_features.csv          기존 더미데이터 (수정하지 않음)

출력 (runs/<run-id>/ 안. 분석 산출물은 다시 실행하면 새로 만든다. 수집 원본은 건드리지 않는다)
    bookings.csv              예매 1건당 한 줄 (#14, #16): 시작·완료 시각, 단계별 소요 시간, 카드, 실패 여부
    booking_intervals.csv     같은 계정의 연속 예매 시도 간격 1개당 한 줄 + 간격 분해 + 긴 간격 표시
    feature_summary.csv       계정별 Feature (BTS 계산식) + 긴 간격 제외 민감도(가상)
    distribution_stats.csv    집단별 요약 통계 (평균·중앙값·표준편차·최소·최대·사분위수)
    comparison_14_vs_16.csv   모드별 기준 대 신규 차이 (--baseline-run 일 때는 comparison_vs_baseline.csv)
    figures/*.png             그래프
    REPORT.md                 요약 보고서

--figures-only: 위 CSV 가 이미 있을 때 그래프만 다시 그린다 (이벤트 재계산·보고서 재작성 없음).

용어 (혼동 주의)
- 예매 간격(gap_seconds): 같은 계정의 연속한 "예매 시도 시작 시각" 사이의 간격. BTS FeatureAggregationService 가
  avg_booking_interval / booking_interval_variance 계산에 쓰는 값이다 (attemptId 별 첫 BOOKING_ATTEMPT 의 occurredAt).
- 예매 소요 시간(ms_booking): 예매 1건 안에서 "예매하기" 클릭 → 결제 완료까지. 간격이 아니다.
- 결제 단계 소요 시간(payment_step_ms): ms_paid - ms_hold (결제 화면 진입 → 완료 화면).
  macro.js 의 ms_paid 자체는 실행 시작부터 결제 완료까지의 누적 시간이다.
"""

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

EXP = Path(__file__).resolve().parent.parent
AI = EXP.parent.parent
ISSUE14 = AI / "experiments" / "issue14-real-validation"
DUMMY_PATH = AI / "ml_experiment" / "data" / "fairticket_synthetic_features.csv"
sys.path.insert(0, str(ISSUE14 / "scripts"))

import extract_features as fx14  # noqa: E402  (BTS FeatureAggregationService.calculate 재현. #14 에서 단위 테스트로 검증된 코드)

# 비교하는 두 데이터셋의 키와 이름. 기본은 #14 대 #16 이고, --baseline-run 이면 configure_datasets 로 바꾼다.
BASE, NEW = "issue14", "issue16"
DATASET_LABEL = {BASE: "기존 #14", NEW: "신규 #16"}
DATASET_SHORT = {BASE: "#14", NEW: "#16"}
RUN_MODE = False  # True: 이 폴더의 다른 실행(--baseline-run)과 비교
MODES = ["fast", "jitter", "human"]
MODE_LABEL = {"fast": "Fast", "jitter": "Jitter", "human": "Human-like", "person": "사람"}
GROUP_MODE = {"macro_fast": "fast", "macro_jitter": "jitter", "macro_human": "human", "human": "person"}
MODE_MARKER = {"fast": "^", "jitter": "s", "human": "X", "person": "D"}
PAYMENT_CARD = {"test-pay-shared-01": "테스트 카드 A", "test-pay-shared-02": "테스트 카드 B"}

# 색: 데이터셋(#14 파랑, #16 주황, 더미 청록)은 모든 그래프에서 같은 뜻으로 쓴다.
C14, C16, C_DUMMY, C_ALERT = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
DATASET_COLOR = {BASE: C14, NEW: C16}


def configure_datasets(base_key, new_key, base_label, new_label, base_short, new_short):
    """비교 대상을 바꾼다 (색은 그대로: 기준 = 파랑, 신규 = 주황)."""
    global BASE, NEW, RUN_MODE
    BASE, NEW, RUN_MODE = base_key, new_key, True
    for table, values in ((DATASET_LABEL, (base_label, new_label)), (DATASET_SHORT, (base_short, new_short)),
                          (DATASET_COLOR, (C14, C16))):
        table.clear()
        table.update({base_key: values[0], new_key: values[1]})


def pair():
    return f"{DATASET_LABEL[BASE]} 대 {DATASET_LABEL[NEW]}"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COMPONENTS = [  # 간격 분해 구성요소, 시간 순서: (컬럼, 이름, 색)
    ("click_to_hold_s", "예매하기 클릭~좌석 선점", "#1baf7a"),
    ("payment_step_s", "결제 단계", "#eda100"),
    ("launch_overhead_s", "브라우저 종료·재기동", "#e87ba4"),
    ("next_before_click_s", "다음 예매 준비 (재실행: 로그인~예매하기 클릭 전 / 연속: 완료 화면~예매하기 클릭)", "#6250d6"),
]

# 결제 단계에서 macro.js 가 스스로 쉬는 시간의 기댓값 차이 (코드에서 읽은 값. 관측값이 아니다)
#   #14 (ee5a44e): 직접 입력 행 클릭 → pause → 카드번호 15자리 입력 → pause → 결제하기
#   #16 (9dd0146): 기본 테스트 카드가 선택된 상태에서 바로 결제하기
#   기대 차이 = 2 × E[pause] + 15 × E[type]  (pause/type 은 PROFILES 의 균등분포 중간값)
EXPECTED_PAYMENT_CUT_MS = {"fast": 2 * 10 + 0, "jitter": 2 * 360 + 15 * 37.5, "human": 2 * 1750 + 15 * 150}
ISSUE14_CONDITIONS = {
    "macro_commit": "ee5a44e (2026-11-19, 카드번호 15자리 직접 입력)",
    "bts": "ticket-fds-admin feat/#40/risk-assessment-credit-pipeline @46f8c67",
    "launcher": "PowerShell `1..N | ForEach-Object { node macro.js ... }`",
    "card": "직접 입력 123456789012345 → test-pay-shared-03",
}


# ---------------------------------------------------------------- 계산
def describe(values):
    s = pd.Series(values, dtype=float).dropna()
    if s.empty:
        return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, min=np.nan, max=np.nan, q1=np.nan, q3=np.nan)
    return dict(n=len(s), mean=s.mean(), median=s.median(), std=s.std(ddof=1) if len(s) > 1 else np.nan,
                min=s.min(), max=s.max(), q1=s.quantile(0.25), q3=s.quantile(0.75))


def flag_long_gaps(gaps):
    """긴 간격 표시 규칙 (표시용. 값을 지우지 않는다):
    같은 계정의 간격이 5개 이상일 때, Q3 + 3×IQR (Tukey 의 far-out 기준) 과 계정 중앙값의 2배를 모두 넘는 간격.
    두 번째 조건은 간격이 아주 고른 계정에서 1~2초 차이가 긴 간격으로 잡히는 것을 막는다."""
    s = pd.Series(gaps, dtype=float)
    if len(s) < 5:
        return [False] * len(s)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return list((s > q3 + 3 * (q3 - q1)) & (s > 2 * s.median()))


def num(value):
    try:
        return float(value) if value not in ("", None) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_runs_issue14():
    frames = []
    for path in sorted((ISSUE14 / "data").glob("*/macro-results*.csv")):
        frames.append(pd.read_csv(path, dtype=str, keep_default_na=False))
    runs = pd.concat(frames, ignore_index=True).drop_duplicates(["user", "time"])
    runs["account"] = runs["user"]
    return runs


def build_dataset(dataset, events_path, runs, calculated_at, accounts=None):
    """이벤트와 매크로 실행 결과로 예매·간격·Feature 표를 만든다."""
    events = fx14.load_events(events_path)
    if accounts is not None:
        events = events[events["external_user_id"].isin(accounts)].reset_index(drop=True)
    features, interval_table, summary = fx14.extract(events, calculated_at, 30.0)

    booking_rows, interval_rows = [], []
    for account, own in events.groupby("external_user_id", sort=False):
        group = own["group"].iloc[0]
        mode = GROUP_MODE.get(group, "other")
        attempts = {}
        for row in own.itertuples(index=False):
            attempt_id = fx14.metadata_string(row.meta, "attemptId")
            if attempt_id is None:
                continue
            a = attempts.setdefault(attempt_id, {"attempt_id": attempt_id, "start": None, "hold": None, "booking": None,
                                                 "booking_id": "", "payment_identifier": "", "n_events": 0,
                                                 "non_success": []})
            if row.event_type == "BOOKING_ATTEMPT":
                a["n_events"] += 1
                if a["start"] is None:
                    a["start"] = row.occurred_at
                stage, status = row.meta.get("stage"), row.meta.get("status")
                if stage == "SEAT_HOLD" and status == "SUCCEEDED" and a["hold"] is None:
                    a["hold"] = row.occurred_at
                if status not in ("ENTERED", "SUCCEEDED"):
                    a["non_success"].append(f"{stage}/{status}")
            elif row.event_type == "BOOKING" and a["booking"] is None:
                a["booking"] = row.occurred_at
                a["booking_id"] = row.booking_id
                a["payment_identifier"] = fx14.metadata_string(row.meta, "paymentIdentifier") or ""
        ordered = sorted((a for a in attempts.values() if a["start"] is not None), key=lambda a: a["start"])

        own_runs = runs[runs["account"] == account].copy() if runs is not None else pd.DataFrame()
        # 앞 예매의 실패로 실행하지 않은 줄(NOT_RUN)은 시도와 연결하지 않고 따로 남긴다
        not_run = int((own_runs["result"] == "NOT_RUN").sum()) if len(own_runs) else 0
        if not_run:
            own_runs = own_runs[own_runs["result"] != "NOT_RUN"].copy()
        if len(own_runs) and own_runs["mode"].iloc[0] in MODES:
            mode = own_runs["mode"].iloc[0]  # 계정 이름 규칙보다 실제 실행한 모드를 우선한다
        if len(own_runs):
            own_runs["end"] = pd.to_datetime(own_runs["time"].replace("", None), utc=True).dt.tz_localize(None)
            if "invoke_end_utc" in own_runs:
                own_runs["end"] = own_runs["end"].fillna(pd.to_datetime(own_runs["invoke_end_utc"], utc=True).dt.tz_localize(None))
            own_runs = own_runs.sort_values("end").reset_index(drop=True)
        # 시도 → 실행 연결: 시도 시작 시각 이후에 처음 끝난 실행
        run_of = {}
        for a in ordered:
            later = own_runs.index[own_runs["end"] >= a["start"]] if len(own_runs) else []
            run_of[a["attempt_id"]] = int(later[0]) if len(later) else None
        attempts_per_run = pd.Series(list(run_of.values())).value_counts().to_dict() if run_of else {}

        seen_runs = set()
        for a in ordered:
            k = run_of[a["attempt_id"]]
            run = own_runs.iloc[k] if k is not None else None
            first_of_run = k is not None and k not in seen_runs
            if k is not None:
                seen_runs.add(k)
            row = {"dataset": dataset, "account": account, "group": group, "mode": mode,
                   "run_index": (k + 1) if k is not None else "", "attempt_id": a["attempt_id"],
                   "attempts_in_run": attempts_per_run.get(k, "") if k is not None else "",
                   "attempt_start_utc": a["start"].isoformat(timespec="milliseconds"),
                   "seat_hold_utc": a["hold"].isoformat(timespec="milliseconds") if a["hold"] is not None else "",
                   "booking_completed_utc": a["booking"].isoformat(timespec="milliseconds") if a["booking"] is not None else "",
                   "completed": a["booking"] is not None, "booking_id": a["booking_id"],
                   "payment_identifier": a["payment_identifier"],
                   "card_label": PAYMENT_CARD.get(a["payment_identifier"], "직접 입력/기타" if a["payment_identifier"] else ""),
                   "n_attempt_events": a["n_events"], "non_success_stages": "|".join(a["non_success"]),
                   "server_attempt_to_booking_ms": (a["booking"] - a["start"]) / timedelta(milliseconds=1) if a["booking"] is not None else np.nan,
                   "server_hold_to_booking_ms": (a["booking"] - a["hold"]) / timedelta(milliseconds=1) if a["booking"] is not None and a["hold"] is not None else np.nan}
            for column in ("result", "error", "ms_login", "ms_to_seatmap", "ms_first_seat", "ms_hold", "ms_paid",
                           "ms_total", "ms_booking", "invoke_start_utc", "invoke_end_utc", "invoke_wall_ms",
                           "session_seq", "ms_browser_launch", "ms_context_open", "ms_prev_paid_to_begin", "hold_tries",
                           "fail_step", "fail_url"):
                row[column] = (run.get(column, "") if run is not None and first_of_run else "")
            # 연속 예매의 둘째 건부터는 로그인이 없다. 1회 예매 방식은 매번 로그인한다.
            row["login_in_this_booking"] = bool(row["ms_login"] != "")
            ms = {c: num(row[c]) for c in ("ms_hold", "ms_paid", "ms_total", "ms_booking")}
            row["payment_step_ms"] = ms["ms_paid"] - ms["ms_hold"]
            row["pre_click_ms"] = ms["ms_total"] - ms["ms_booking"]
            row["click_to_hold_ms"] = ms["ms_hold"] - row["pre_click_ms"]
            row["failed_or_timeout"] = bool(row["result"]) and row["result"] != "OK"
            booking_rows.append(row)
        for k in range(not_run):
            booking_rows.append({"dataset": dataset, "account": account, "group": group, "mode": mode,
                                 "run_index": len(own_runs) + k + 1, "attempt_id": "", "completed": False,
                                 "result": "NOT_RUN", "error": "앞 예매의 실패로 중단되어 실행하지 않음",
                                 "failed_or_timeout": True})
        # BTS 이벤트가 하나도 없는 실행(실패 등)도 빠뜨리지 않는다
        for k in range(len(own_runs)):
            if k not in seen_runs:
                run = own_runs.iloc[k]
                booking_rows.append({"dataset": dataset, "account": account, "group": group, "mode": mode,
                                     "run_index": k + 1, "attempt_id": "", "completed": False,
                                     **{c: run.get(c, "") for c in ("result", "error", "ms_total", "ms_booking",
                                                                    "invoke_start_utc", "invoke_end_utc", "invoke_wall_ms",
                                                                    "session_seq", "fail_step", "fail_url")},
                                     "failed_or_timeout": run.get("result", "") != "OK"})

        # 간격: extract_features 가 계산한 값(= BTS 식)을 그대로 쓰고, 분해 정보만 덧붙인다
        own_intervals = interval_table[interval_table["external_user_id"] == account] if len(interval_table) else interval_table
        gaps = list(own_intervals["gap_seconds"]) if len(own_intervals) else []
        flags = flag_long_gaps(gaps)
        for i, gap in enumerate(gaps):
            this_a, next_a = ordered[i], ordered[i + 1]
            row = {"dataset": dataset, "account": account, "group": group, "mode": mode, "interval_index": i + 1,
                   "from_attempt_utc": this_a["start"].isoformat(timespec="milliseconds"),
                   "to_attempt_utc": next_a["start"].isoformat(timespec="milliseconds"),
                   "gap_seconds": gap,
                   "completion_gap_seconds": ((next_a["booking"] - this_a["booking"]).total_seconds()
                                              if this_a["booking"] is not None and next_a["booking"] is not None else np.nan),
                   "long_gap": flags[i], "flow_after_click_s": np.nan, "click_to_hold_s": np.nan, "payment_step_s": np.nan,
                   "after_paid_to_next_begin_s": np.nan,
                   "next_before_click_s": np.nan, "launch_overhead_s": np.nan, "run_results": ""}
            k1, k2 = run_of[this_a["attempt_id"]], run_of[next_a["attempt_id"]]
            if k1 is not None and k2 is not None and k2 == k1 + 1:
                r1, r2 = own_runs.iloc[k1], own_runs.iloc[k2]
                row["run_results"] = f"{r1['result']}→{r2['result']}"
                if r1["result"] == "OK" and r2["result"] == "OK":
                    after = num(r1["ms_booking"]) / 1000
                    before = (num(r2["ms_total"]) - num(r2["ms_booking"])) / 1000
                    pay = (num(r1["ms_paid"]) - num(r1["ms_hold"])) / 1000
                    # after_paid_to_next_begin_s: 앞 예매 결제 완료 → 다음 예매하기 클릭 (= 재기동 + 다음 예매 준비)
                    row.update(flow_after_click_s=after, payment_step_s=pay, click_to_hold_s=after - pay,
                               after_paid_to_next_begin_s=gap - after,
                               next_before_click_s=before, launch_overhead_s=gap - after - before)
            interval_rows.append(row)

    features = features.rename(columns={"external_user_id": "account"})
    features.insert(0, "dataset", dataset)
    features["mode"] = features["group"].map(GROUP_MODE).fillna("other")
    mode_of = {r["account"]: r["mode"] for r in booking_rows}
    features["mode"] = features["account"].map(mode_of).fillna(features["mode"])
    return pd.DataFrame(booking_rows), pd.DataFrame(interval_rows), features, summary


def feature_summary(features, intervals, bookings):
    rows = []
    for f in features.itertuples(index=False):
        own = intervals[(intervals["dataset"] == f.dataset) & (intervals["account"] == f.account)]
        kept = own[~own["long_gap"].astype(bool)]["gap_seconds"]
        book = bookings[(bookings["dataset"] == f.dataset) & (bookings["account"] == f.account)]
        ok = book[book["result"] == "OK"]
        d = describe(own["gap_seconds"])
        rows.append({
            "dataset": f.dataset, "account": f.account, "group": f.group, "mode": f.mode,
            "booking_count": f.booking_count, "n_intervals": len(own),
            "avg_booking_interval": f.avg_booking_interval, "booking_interval_variance": f.booking_interval_variance,
            "interval_std_population": np.sqrt(f.booking_interval_variance) if pd.notna(f.booking_interval_variance) else np.nan,
            "interval_median": d["median"], "interval_min": d["min"], "interval_max": d["max"],
            "interval_q1": d["q1"], "interval_q3": d["q3"],
            "n_long_gaps": int(own["long_gap"].astype(bool).sum()),
            "avg_booking_interval_excl_long_gap_hypothetical": kept.mean() if len(kept) else np.nan,
            "booking_interval_variance_excl_long_gap_hypothetical": kept.var(ddof=0) if len(kept) else np.nan,
            "cancel_rate": f.cancel_rate, "card_reuse_count": f.card_reuse_count,
            "ip_booking_density": f.ip_booking_density, "n_completed_bookings": f.n_completed_bookings,
            "n_macro_runs": int((book["result"] != "").sum()), "n_macro_runs_failed": int(book["failed_or_timeout"].astype(bool).sum()),
            "ms_booking_mean": pd.to_numeric(ok["ms_booking"], errors="coerce").mean(),
            "ms_booking_median": pd.to_numeric(ok["ms_booking"], errors="coerce").median(),
            "payment_step_ms_mean": pd.to_numeric(ok["payment_step_ms"], errors="coerce").mean(),
            "payment_step_ms_median": pd.to_numeric(ok["payment_step_ms"], errors="coerce").median(),
            "payment_identifiers_used": f.payment_identifiers_used,
            "first_attempt_utc": f.first_attempt_utc, "last_attempt_utc": f.last_attempt_utc,
        })
    return pd.DataFrame(rows)


def load_dummy():
    dummy = pd.read_csv(DUMMY_PATH)
    return {"더미 정상": dummy[dummy["is_anomaly"] == 0], "더미 매크로": dummy[dummy["scenario"] == "macro"]}, dummy


def macro_only(df):
    return df[df["mode"].isin(MODES)]


def distribution_stats(bookings, intervals, feats, dummy_groups, dummy):
    rows = []

    def add(metric, unit, dataset, group, variant, values):
        rows.append({"metric": metric, "unit": unit, "dataset": dataset, "group": group, "variant": variant, **describe(values)})

    for dataset in (BASE, NEW):
        iv = intervals[intervals["dataset"] == dataset]
        bk = bookings[(bookings["dataset"] == dataset) & (bookings["result"] == "OK")]
        selections = [("전체 매크로", MODES)] + [(MODE_LABEL[m], [m]) for m in MODES + ["person"]]
        for name, modes in selections:
            sub = iv[iv["mode"].isin(modes)]
            if not len(sub):
                continue
            add("gap_seconds", "초", dataset, name, "원본", sub["gap_seconds"])
            if sub["long_gap"].astype(bool).any():
                add("gap_seconds", "초", dataset, name, "긴 간격 제외 (가상)", sub[~sub["long_gap"].astype(bool)]["gap_seconds"])
            for column in ("after_paid_to_next_begin_s", "launch_overhead_s", "next_before_click_s", "flow_after_click_s"):
                add(column, "초", dataset, name, "원본", sub[column])
            b = bk[bk["mode"].isin(modes)]
            for column in ("payment_step_ms", "click_to_hold_ms", "pre_click_ms", "ms_booking", "ms_total",
                           "server_hold_to_booking_ms"):
                add(column, "ms", dataset, name, "원본", pd.to_numeric(b[column], errors="coerce"))
        for account, sub in iv.groupby("account", sort=False):
            add("gap_seconds", "초", dataset, f"계정 {account}", "원본", sub["gap_seconds"])
            if sub["long_gap"].astype(bool).any():
                add("gap_seconds", "초", dataset, f"계정 {account}", "긴 간격 제외 (가상)", sub[~sub["long_gap"].astype(bool)]["gap_seconds"])
        fm = macro_only(feats[feats["dataset"] == dataset])
        for feature, unit in (("avg_booking_interval", "초"), ("booking_interval_variance", "초²")):
            add(feature, unit, dataset, "실측 매크로 계정", "원본", fm[feature])
            add(feature, unit, dataset, "실측 매크로 계정", "긴 간격 제외 (가상)", fm[f"{feature}_excl_long_gap_hypothetical"])
    for feature, unit in (("avg_booking_interval", "초"), ("booking_interval_variance", "초²")):
        for name, group in dummy_groups.items():
            add(feature, unit, "dummy", name, "원본", group[feature])
        for scenario, group in dummy.groupby("scenario"):
            add(feature, unit, "dummy", f"더미 시나리오 {scenario}", "원본", group[feature])
    return pd.DataFrame(rows)


def comparison_table(bookings, intervals):
    rows = []
    for mode in MODES:
        for metric, unit, source, variant in (
            ("gap_seconds", "초", "intervals", "원본"), ("gap_seconds", "초", "intervals", "긴 간격 제외 (가상)"),
            ("payment_step_ms", "ms", "bookings", "원본"), ("click_to_hold_ms", "ms", "bookings", "원본"),
            ("pre_click_ms", "ms", "bookings", "원본"), ("ms_booking", "ms", "bookings", "원본"),
            ("ms_total", "ms", "bookings", "원본"), ("launch_overhead_s", "초", "intervals", "원본"),
            ("after_paid_to_next_begin_s", "초", "intervals", "원본"), ("next_before_click_s", "초", "intervals", "원본"),
            ("click_to_hold_s", "초", "intervals", "원본"), ("payment_step_s", "초", "intervals", "원본"),
        ):
            values = {}
            for dataset in (BASE, NEW):
                if source == "intervals":
                    sub = intervals[(intervals["dataset"] == dataset) & (intervals["mode"] == mode)]
                    if variant != "원본":
                        sub = sub[~sub["long_gap"].astype(bool)]
                else:
                    sub = bookings[(bookings["dataset"] == dataset) & (bookings["mode"] == mode) & (bookings["result"] == "OK")]
                values[dataset] = pd.to_numeric(sub[metric], errors="coerce").dropna()
            a, b = values[BASE], values[NEW]
            if not len(a) or not len(b):
                continue
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) >= 3 and len(b) >= 3 else np.nan
            rows.append({"mode": mode, "metric": metric, "unit": unit, "variant": variant,
                         "n_base": len(a), "n_new": len(b), "mean_base": a.mean(), "mean_new": b.mean(),
                         "mean_diff": b.mean() - a.mean(), "mean_change_pct": (b.mean() / a.mean() - 1) * 100,
                         "median_base": a.median(), "median_new": b.median(), "median_diff": b.median() - a.median(),
                         "median_change_pct": (b.median() / a.median() - 1) * 100,
                         "std_base": a.std(ddof=1), "std_new": b.std(ddof=1),
                         "mannwhitney_p_exploratory": p})
    columns = ["mode", "metric", "unit", "variant", "n_base", "n_new", "mean_base", "mean_new", "mean_diff", "mean_change_pct",
               "median_base", "median_new", "median_diff", "median_change_pct", "std_base", "std_new", "mannwhitney_p_exploratory"]
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------- 그래프
def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plain(value, _pos=None):
    return f"{value:g}"


def log_axis(axis):
    axis.set_major_formatter(mticker.FuncFormatter(plain))
    axis.set_minor_formatter(mticker.NullFormatter())


def new_figure(width, height, title, subtitle, **kwargs):
    fig, axes = plt.subplots(figsize=(width, height), **kwargs)
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.012, y=0.985, ha="left", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.012, 0.935, subtitle, ha="left", va="top", fontsize=9, color=INK_2)
    return fig, axes


def save(fig, path):
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path.name


def dataset_handles(counts=None):
    return [Patch(facecolor=DATASET_COLOR[d], alpha=0.75, edgecolor="none",
                  label=DATASET_LABEL[d] + (f" (n={counts[d]})" if counts else "")) for d in (BASE, NEW)]


def fig_interval_distribution(intervals, path):
    iv = macro_only(intervals)
    fig, axes = new_figure(14, 5.4, f"매크로 예매 간격 분포: {pair()}",
                           "가로축 = 같은 계정의 연속한 예매 시도 시작 시각 사이의 간격(초, 선형 축). 막대 = 간격 개수, 아래 세로선 = 개별 간격.\n"
                           "모드마다 범위가 달라 패널별로 가로축이 다르다. 표시 범위를 넘는 긴 간격은 패널에 개수와 최댓값을 적었다 (로그 축 그래프 참고).",
                           ncols=3)
    for ax, mode in zip(axes, MODES):
        style(ax)
        sub = iv[iv["mode"] == mode]
        if not len(sub):
            ax.set_title(f"{MODE_LABEL[mode]} (데이터 없음)", loc="left", fontsize=11, color=INK)
            continue
        upper = sub[~sub["long_gap"].astype(bool)]["gap_seconds"].max() * 1.04
        lower = max(0, sub["gap_seconds"].min() * 0.95)
        bins = np.linspace(lower, upper, 22)
        counts, top = {}, 0
        for dataset in (BASE, NEW):
            values = sub[sub["dataset"] == dataset]["gap_seconds"]
            counts[dataset] = len(values)
            inside = values[values <= upper]
            heights, _, _ = ax.hist(inside, bins=bins, color=DATASET_COLOR[dataset], alpha=0.55, edgecolor=SURFACE, linewidth=1)
            top = max(top, heights.max() if len(heights) else 0)
        for dataset, level in ((BASE, -0.05), (NEW, -0.10)):
            values = sub[(sub["dataset"] == dataset) & (sub["gap_seconds"] <= upper)]["gap_seconds"]
            ax.vlines(values, level * top - 0.018 * top, level * top + 0.018 * top, color=DATASET_COLOR[dataset], linewidth=1.3)
            median = sub[sub["dataset"] == dataset]["gap_seconds"].median()
            if pd.notna(median):
                ax.axvline(median, color=DATASET_COLOR[dataset], linewidth=1.6, linestyle=(0, (4, 3)))
        outside = sub[sub["gap_seconds"] > upper]
        if len(outside):
            text = " / ".join(f"{DATASET_LABEL[d]} {len(g)}개(최대 {g['gap_seconds'].max():.1f}초)" for d, g in outside.groupby("dataset"))
            ax.text(0.98, 0.985, f"범위 밖 긴 간격: {text}", transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5), zorder=6)
        ax.set_ylim(-0.14 * top, top * 1.15)
        ax.set_xlim(lower, upper)
        ax.set_title(f"{MODE_LABEL[mode]}  ·  간격 수: {DATASET_SHORT[BASE]} n={counts[BASE]}, {DATASET_SHORT[NEW]} n={counts[NEW]}", loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel("예매 간격 (초)", color=INK_2, fontsize=10)
        ax.set_ylabel("간격 개수 (개)", color=INK_2, fontsize=10)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    handles = dataset_handles() + [Line2D([], [], color=INK_2, linestyle=(0, (4, 3)), linewidth=1.6, label="중앙값 (점선, 데이터 색)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.05, right=0.985, top=0.80, bottom=0.17, wspace=0.2)
    return save(fig, path)


def fig_interval_distribution_log(intervals, path):
    iv = macro_only(intervals)
    fig, axes = new_figure(11, 7.6, f"매크로 예매 간격 분포 (로그 축): {pair()}",
                           "가로축 = 예매 간격(초, 로그 축·값은 원래 단위). 긴 간격을 포함한 모든 간격을 그렸다. 세 패널의 가로축은 같다.",
                           nrows=3, sharex=True)
    if not len(iv):
        return save(fig, path)
    lo, hi = iv["gap_seconds"].min() / 1.25, iv["gap_seconds"].max() * 1.25
    bins = np.geomspace(lo, hi, 46)
    for ax, mode in zip(axes, MODES):
        style(ax)
        ax.set_xscale("log")
        counts = {}
        for dataset in (BASE, NEW):
            values = iv[(iv["mode"] == mode) & (iv["dataset"] == dataset)]["gap_seconds"]
            counts[dataset] = len(values)
            ax.hist(values, bins=bins, color=DATASET_COLOR[dataset], alpha=0.55, edgecolor=SURFACE, linewidth=1)
        flagged = iv[(iv["mode"] == mode) & iv["long_gap"].astype(bool)]
        for row in flagged.itertuples(index=False):
            ax.annotate(f"긴 간격 {row.gap_seconds:.1f}초\n({DATASET_LABEL[row.dataset]} {row.account})", (row.gap_seconds, 1),
                        xytext=(0, 22), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK,
                        arrowprops=dict(arrowstyle="-", color=C_ALERT, linewidth=1))
        ax.set_title(MODE_LABEL[mode], loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_ylabel("간격 개수 (개)", color=INK_2, fontsize=10)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(handles=dataset_handles(counts), frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
        ax.set_xlim(lo, hi)
    log_axis(axes[-1].xaxis)
    axes[-1].tick_params(labelbottom=True)
    axes[-1].set_xlabel("예매 간격 (초, 로그 축)", color=INK_2, fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.07, hspace=0.3)
    return save(fig, path)


def grouped_box(ax, frame, column, unit, log=False, decimals=1):
    """모드별로 #14, #16 상자를 나란히 그린다. 점 = 개별 값, 상자 = Q1~Q3, 굵은 선 = 중앙값."""
    style(ax)
    rng = np.random.default_rng(16)
    ticks, labels = [], []
    for i, mode in enumerate(MODES):
        ticks.append(i)
        parts = []
        for dataset, dx in ((BASE, -0.2), (NEW, 0.2)):
            values = pd.to_numeric(frame[(frame["mode"] == mode) & (frame["dataset"] == dataset)][column], errors="coerce").dropna().to_numpy()
            parts.append(f"{DATASET_SHORT[dataset]} n={len(values)}")
            if not len(values):
                continue
            x = i + dx
            color = DATASET_COLOR[dataset]
            ax.scatter(x + rng.uniform(-0.11, 0.11, len(values)), values, s=16, color=color, alpha=0.55, linewidths=0, zorder=2)
            if len(values) >= 4:
                q1, med, q3 = np.percentile(values, [25, 50, 75])
                ax.add_patch(Rectangle((x - 0.15, q1), 0.30, q3 - q1, fill=False, edgecolor=INK, linewidth=1.2, zorder=3))
                iqr = q3 - q1
                low, high = values[values >= q1 - 1.5 * iqr].min(), values[values <= q3 + 1.5 * iqr].max()
                ax.plot([x, x], [low, q1], color=INK, linewidth=1, zorder=3)
                ax.plot([x, x], [q3, high], color=INK, linewidth=1, zorder=3)
            else:
                med = np.median(values)
            ax.plot([x - 0.15, x + 0.15], [med, med], color=INK, linewidth=2.2, zorder=4)
            side = -1 if dataset == BASE else 1  # 숫자가 옆 상자와 겹치지 않도록 바깥쪽에 적는다
            ax.annotate(f"{med:.{decimals}f}", (x + side * 0.16, med), xytext=(side * 3, 0), textcoords="offset points",
                        ha="right" if side < 0 else "left", va="center", fontsize=8.5, color=INK, zorder=5)
        labels.append(f"{MODE_LABEL[mode]}\n" + " · ".join(parts))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, color=INK, fontsize=9)
    ax.set_xlim(-0.6, len(MODES) - 0.3)
    if log:
        ax.set_yscale("log")
        log_axis(ax.yaxis)
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    ax.set_ylabel(unit + (" · 로그 축" if log else ""), color=INK_2, fontsize=10)


def fig_interval_by_macro(intervals, path, log=False):
    fig, ax = new_figure(10.5, 6.2, f"매크로 유형별 예매 간격: {pair()}" + (" (로그 축)" if log else ""),
                         "점 하나 = 간격 하나(가로 흩뿌림은 겹침 방지용). 상자 = Q1~Q3, 굵은 선과 숫자 = 중앙값(초), 수염 = 1.5×IQR 안의 최소·최대.\n"
                         "n = 간격 개수. 긴 간격도 지우지 않고 그대로 그렸다." + ("" if log else " 선형 축이라 긴 간격이 있으면 상자가 눌려 보인다 (로그 축 그래프 참고)."))
    grouped_box(ax, macro_only(intervals), "gap_seconds", "예매 간격 (초)", log=log, decimals=2)
    fig.legend(handles=dataset_handles(), frameon=False, fontsize=9.5, labelcolor=INK, loc="upper right", ncol=2, bbox_to_anchor=(0.975, 0.99))
    fig.subplots_adjust(left=0.08, right=0.975, top=0.83, bottom=0.11)
    return save(fig, path)


def fig_feature(feats, dummy_groups, feature, unit, path):
    rng = np.random.default_rng(16)
    rows = [("더미 정상", None, None), ("더미 매크로", None, None)]
    if len(feats[(feats["dataset"] == BASE) & (feats["mode"] == "person")]):
        rows.append((f"{DATASET_LABEL[BASE]}\n실측 사람\n(재수집 없음)", BASE, ["person"]))
    rows += [(f"{DATASET_LABEL[BASE]}\n실측 매크로", BASE, MODES), (f"{DATASET_LABEL[NEW]}\n실측 매크로", NEW, MODES)]
    fig, ax = new_figure(11.5, 6.8, f"{feature}: 기존 더미데이터 대 실측 계정",
                         f"점 하나 = 계정 하나. 가로축 = {feature} ({unit}, 로그 축·값은 원래 단위). 더미의 세로 흩뿌림은 겹침 방지용.\n"
                         "상자 = 더미 Q1~Q3, 굵은 선 = 더미 중앙값. 실측 계정은 BTS 계산식 그대로(긴 간격 포함) 계산한 값이다.")
    style(ax, "x")
    ax.set_xscale("log")
    real = feats[feats[feature].notna()]
    positive = pd.concat([g[feature] for g in dummy_groups.values()] + [real[feature]])
    positive = positive[positive > 0]
    lo, hi = positive.min() / 1.8, positive.max() * 2.2
    ax.set_xlim(lo, hi)
    tick_labels = []
    for index, (label, dataset, modes) in enumerate(rows):
        y0 = len(rows) - 1 - index
        if dataset is None:
            values = dummy_groups[label][feature].to_numpy()
            ax.scatter(values, y0 + rng.uniform(-0.3, 0.3, len(values)), s=10, color=C_DUMMY, alpha=0.4, linewidths=0, zorder=2)
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            ax.add_patch(Rectangle((q1, y0 - 0.38), q3 - q1, 0.76, fill=False, edgecolor=INK, linewidth=1.3, zorder=3))
            ax.plot([med, med], [y0 - 0.38, y0 + 0.38], color=INK, linewidth=2.2, zorder=4)
            tick_labels.append(f"{label}\n(n={len(values)})")
            continue
        sub = real[(real["dataset"] == dataset) & (real["mode"].isin(modes))]
        tick_labels.append(f"{label}\n(n={len(sub)})")
        offsets = np.linspace(0.3, -0.3, len(sub)) if len(sub) > 1 else [0.0]
        for row, dy in zip(sub.itertuples(index=False), offsets):
            value = getattr(row, feature)
            shown = max(value, lo * 1.15)  # 0 은 로그 축에 그릴 수 없어 왼쪽 끝에 두고 값은 글자로 적는다
            ax.scatter([shown], [y0 + dy], s=110, color=DATASET_COLOR[dataset], marker=MODE_MARKER[row.mode],
                       edgecolor=SURFACE, linewidth=1.5, zorder=5)
            frac = (np.log(shown) - np.log(lo)) / (np.log(hi) - np.log(lo))
            right = frac < 0.7
            name = row.account.replace("human-", "사람-") if row.mode == "person" else row.account.replace("macro-", "")
            ax.annotate(f"{name} ({int(row.booking_count)}회): {value:.3g}", (shown, y0 + dy), xytext=(9 if right else -9, 0),
                        textcoords="offset points", ha="left" if right else "right", va="center", fontsize=8.5, color=INK, zorder=6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(reversed(tick_labels)), color=INK, fontsize=9.5)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axhline(len(rows) - 2.5, color=GRID, linewidth=1.2)
    log_axis(ax.xaxis)
    ax.set_xlabel(f"{feature} ({unit}, 로그 축)", color=INK_2, fontsize=10)
    handles = [Line2D([], [], linestyle="", marker="o", markersize=5, color=C_DUMMY, alpha=0.6, label="기존 더미데이터 (계정별)"),
               Line2D([], [], linestyle="", marker="o", markersize=8, color=C14, label=f"{DATASET_LABEL[BASE]} 실측"),
               Line2D([], [], linestyle="", marker="o", markersize=8, color=C16, label=f"{DATASET_LABEL[NEW]} 실측")]
    handles += [Line2D([], [], linestyle="", marker=MODE_MARKER[m], markersize=8, color=INK_2, label=f"모양: {MODE_LABEL[m]}")
                for m in MODES + (["person"] if len(rows) == 5 else [])]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8.5, labelcolor=INK, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.86, bottom=0.14)
    return save(fig, path)


def fig_payment_duration(bookings, path):
    ok = macro_only(bookings[bookings["result"] == "OK"])
    fig, axes = new_figure(14, 6.2, f"결제 단계·전체 예매 소요 시간: {pair()}",
                           "예매 1건 안의 소요 시간이다 (연속 예매 간격이 아님). macro.js 가 측정한 값, 성공(OK)한 예매만. 점 = 예매 1건, 상자 = Q1~Q3, 굵은 선과 숫자 = 중앙값(ms).\n"
                           "왼쪽: 결제 화면 진입 → 완료 화면 (ms_paid − ms_hold, 시연용 결제 지연 1,200ms 포함). 오른쪽: 예매하기 클릭 → 결제 완료 (ms_booking).",
                           ncols=2)
    grouped_box(axes[0], ok, "payment_step_ms", "결제 단계 소요 시간 (ms)", decimals=0)
    axes[0].set_title("결제 단계 소요 시간 (ms_paid − ms_hold)", loc="left", fontsize=10.5, color=INK)
    grouped_box(axes[1], ok, "ms_booking", "전체 예매 소요 시간 (ms)", decimals=0)
    axes[1].set_title("전체 예매 소요 시간 (ms_booking)", loc="left", fontsize=10.5, color=INK)
    fig.legend(handles=dataset_handles(), frameon=False, fontsize=9.5, labelcolor=INK, loc="upper right", ncol=2, bbox_to_anchor=(0.985, 0.99))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.11, wspace=0.16)
    return save(fig, path)


def fig_account_comparison(feats, path):
    fm = macro_only(feats)
    accounts = list(dict.fromkeys(fm.sort_values(["mode", "account"], key=lambda s: s.map({m: i for i, m in enumerate(MODES)}) if s.name == "mode" else s)["account"]))
    fig, axes = new_figure(14, 6.0, f"계정별 평균 예매 간격과 분산: {pair()}",
                           "BTS 계산식 그대로의 값 (긴 간격 포함, 분산은 모분산). 막대 위 숫자 = 값. 괄호 = 예매 시도 수.\n"
                           "분산은 값의 차이가 커서 로그 축으로 그렸다. 같은 이름의 계정끼리 짝지었다 (서로 다른 실행, 다른 DB 에서 수집).",
                           ncols=2)
    for ax, (feature, unit, log) in zip(axes, (("avg_booking_interval", "avg_booking_interval (초)", False),
                                               ("booking_interval_variance", "booking_interval_variance (초², 로그 축)", True))):
        style(ax)
        values_all = fm[feature].dropna()
        floor = values_all[values_all > 0].min() / 3 if log and (values_all > 0).any() else 0
        for i, account in enumerate(accounts):
            for dataset, dx in ((BASE, -0.19), (NEW, 0.19)):
                row = fm[(fm["dataset"] == dataset) & (fm["account"] == account)]
                if not len(row) or pd.isna(row[feature].iloc[0]):
                    ax.text(i + dx, floor if log else 0, "없음", ha="center", va="bottom", fontsize=8, color=INK_2)
                    continue
                value = row[feature].iloc[0]
                shown = max(value, floor * 1.2) if log else value
                ax.bar(i + dx, shown - floor if log else shown, bottom=floor if log else 0, width=0.34, color=DATASET_COLOR[dataset],
                       edgecolor=SURFACE, linewidth=1.5, zorder=2)
                ax.annotate(f"{value:.3g}\n({int(row['booking_count'].iloc[0])}회)", (i + dx, shown), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(accounts)))
        ax.set_xticklabels([a.replace("macro-", "") for a in accounts], color=INK, fontsize=10)
        ax.set_ylabel(unit, color=INK_2, fontsize=10)
        if log:
            ax.set_yscale("log")
            log_axis(ax.yaxis)
            ax.set_ylim(floor, values_all.max() * 6)
        else:
            ax.set_ylim(0, values_all.max() * 1.2)
        ax.set_title(feature, loc="left", fontsize=10.5, color=INK)
    axes[0].legend(handles=dataset_handles(), frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.08, wspace=0.16)
    return save(fig, path)


def fig_gap_breakdown(intervals, path):
    iv = macro_only(intervals).dropna(subset=["launch_overhead_s"])
    fig, ax = new_figure(12, 5.6, "예매 간격의 구성: 어느 구간이 달라졌나",
                         "막대 = 구성요소별 중앙값(초). 간격 = [이번 예매: 예매하기 클릭~좌석 선점 + 결제 단계] + [브라우저 종료·재기동] + [다음 예매 준비: 예매하기 클릭 전까지].\n"
                         "구성요소별 중앙값을 쌓은 것이라 합계가 간격의 중앙값과 정확히 같지는 않다. 오른쪽 숫자 = 실제 간격 중앙값. n = 분해한 간격 수.")
    style(ax, "x")
    labels, y = [], 0
    positions = {}
    for mode in MODES:
        for dataset in (BASE, NEW):
            positions[(mode, dataset)] = y
            y += 1
        y += 0.6
    for (mode, dataset), pos in positions.items():
        sub = iv[(iv["mode"] == mode) & (iv["dataset"] == dataset)]
        labels.append(f"{MODE_LABEL[mode]} · {DATASET_LABEL[dataset]} (n={len(sub)})")
        if not len(sub):
            continue
        left = 0
        for column, _name, color in COMPONENTS:
            width = max(0.0, sub[column].median())  # 연속 예매의 재기동 구간은 시계 오차 수준의 잔차라 음수일 수 있다
            ax.barh(pos, width, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2, zorder=2)
            if width > 0.45:
                ax.text(left + width / 2, pos, f"{width:.2f}", ha="center", va="center", fontsize=8.5, color=INK, zorder=3)
            left += width
        ax.text(left + 0.25, pos, f"간격 중앙값 {sub['gap_seconds'].median():.2f}초", ha="left", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(positions.values()))
    ax.set_yticklabels(labels, color=INK, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, ax.get_xlim()[1] * 1.18)
    ax.set_xlabel("시간 (초)", color=INK_2, fontsize=10)
    fig.legend(handles=[Patch(facecolor=c, label=n) for _c, n, c in COMPONENTS], loc="lower center", ncol=2, frameon=False,
               fontsize=9, labelcolor=INK, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.24 if RUN_MODE else 0.19, right=0.975, top=0.82, bottom=0.20)
    return save(fig, path)


def make_figures(run_dir, bookings, intervals, feats, dummy_groups):
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    return [
        fig_interval_distribution(intervals, out / "interval_distribution.png"),
        fig_interval_distribution_log(intervals, out / "interval_distribution_log.png"),
        fig_interval_by_macro(intervals, out / "interval_by_macro.png"),
        fig_interval_by_macro(intervals, out / "interval_by_macro_log.png", log=True),
        fig_feature(feats, dummy_groups, "avg_booking_interval", "초", out / "feature_mean_comparison.png"),
        fig_feature(feats, dummy_groups, "booking_interval_variance", "초²", out / "feature_variance_comparison.png"),
        fig_payment_duration(bookings, out / "payment_duration.png"),
        fig_account_comparison(feats, out / "account_comparison.png"),
        fig_gap_breakdown(intervals, out / "gap_breakdown.png"),
    ]


# ---------------------------------------------------------------- 보고서
def fmt(value, digits=2):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{value:,.{digits}f}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def signed(value, digits=2, unit=""):
    return "–" if pd.isna(value) else f"{value:+,.{digits}f}{unit}"


def write_report(run_dir, bookings, intervals, feats, stats, comp, dummy_groups, checks):
    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8")) if (run_dir / "environment.json").exists() else {}
    export = json.loads((run_dir / "export_summary.json").read_text(encoding="utf-8")) if (run_dir / "export_summary.json").exists() else {}
    notes = (run_dir / "notes.md").read_text(encoding="utf-8").strip() if (run_dir / "notes.md").exists() else ""
    iv, fm = macro_only(intervals), macro_only(feats)
    b16 = bookings[bookings["dataset"] == NEW]
    new_modes = [m for m in MODES if len(iv[(iv["dataset"] == NEW) & (iv["mode"] == m)])]

    def c(mode, metric, variant="원본"):
        row = comp[(comp["mode"] == mode) & (comp["metric"] == metric) & (comp["variant"] == variant)]
        return row.iloc[0] if len(row) else None

    L = []
    L.append(f"# 이슈 #16 예매 간격 재측정 보고서 — 실행 `{run_dir.name}`\n")
    L.append("생성 스크립트 `scripts/analyze.py` (자동 생성 문서. 직접 고치지 말고 `notes.md` 에 메모를 적은 뒤 다시 생성한다)\n")

    # ---- 0. 요약
    L.append("## 0. 요약: 기존 #14 대비 매크로 예매 간격\n")
    L.append("예매 간격 = 같은 계정의 연속한 예매 시도 시작 시각 사이의 간격 (BTS `avg_booking_interval` 의 재료). 개별 간격 기준.\n")
    rows = []
    for mode in new_modes:
        r, h = c(mode, "gap_seconds"), c(mode, "gap_seconds", "긴 간격 제외 (가상)")
        if r is None:
            continue
        rows.append([MODE_LABEL[mode], f"{fmt(r.mean_base)} → {fmt(r.mean_new)}", f"{signed(r.mean_diff)} ({signed(r.mean_change_pct, 1, '%')})",
                     f"{fmt(r.median_base)} → {fmt(r.median_new)}", f"{signed(r.median_diff)} ({signed(r.median_change_pct, 1, '%')})",
                     f"{fmt(h.mean_base)} → {fmt(h.mean_new)}" if h is not None else "–", f"{int(r.n_base)} / {int(r.n_new)}"])
    L.append(md_table(["모드", "평균 (초) #14 → #16", "평균 변화", "중앙값 (초) #14 → #16", "중앙값 변화", "평균, 긴 간격 제외 (가상)", "간격 수 #14 / #16"], rows))
    L.append("")
    dm = dummy_groups["더미 매크로"]["avg_booking_interval"]
    new_avg = fm[fm["dataset"] == NEW]["avg_booking_interval"].dropna()
    if len(new_avg):
        L.append(f"- 신규 #16 매크로 계정의 `avg_booking_interval` 은 {fmt(new_avg.min())}~{fmt(new_avg.max())}초다. "
                 f"기존 더미 매크로는 {fmt(dm.min())}~{fmt(dm.max())}초(중앙값 {fmt(dm.median())}초)로, "
                 f"신규 실측 {len(new_avg)}계정 중 {int((new_avg > dm.max()).sum())}계정이 더미 매크로의 최댓값보다 크다.")
    L.append("- 표본은 모드당 계정 1~2개, 간격 19~21개이며 한 PC 에서 한 번 수집한 결과다. 아래 차이는 경향으로만 읽는다.\n")

    # ---- 1. 환경
    L.append("## 1. 실험 환경\n")
    if env:
        rows = [[name, f"`{(info.get('commit') or '?')[:7]}`", info.get("branch"), info.get("subject"),
                 "없음" if not info.get("dirty_files") else "; ".join(info["dirty_files"])]
                for name, info in env["repos"].items()]
        L.append(md_table(["레포", "커밋", "브랜치", "커밋 제목", "커밋되지 않은 변경"], rows))
        admin = env["repos"].get("ticket-fds-admin", {})
        L.append("")
        if admin.get("origin_dev"):
            L.append(f"- BTS: 체크아웃한 커밋의 파일 내용이 `origin/dev`(`{admin['origin_dev'][:7]}`)와 "
                     f"{'같다' if admin.get('tree_equals_origin_dev') else '**다르다**'} (`git diff HEAD origin/dev` 기준).")
        fe = env.get("frontend_env", {})
        L.append(f"- 프론트 `.env`: " + ", ".join(f"`{k}={v}`" for k, v in fe.items()) + ".")
        L.append("  - `VITE_USE_MOCK_TICKETING=false`, `VITE_USE_MOCK_BTS=false` 이면 공연·좌석·예매·대기열·BTS 연동이 모두 실제 백엔드를 쓴다 "
                 "(대기열 mock 은 티켓팅 mock 이 켜져 있을 때만 동작).")
        L.append("  - `VITE_USE_MOCK_CARDS` 는 마이페이지 등록 카드 목록에만 해당한다. 이번 결제에 쓴 테스트 카드는 프론트 상수이고 결제 확정은 실제 Demo 백엔드로 간다.")
        L.append(f"- 스택: `demo-ticketing/e2e` 의 일회성 DB(tmpfs, `{env['bts_db']['container']}`) + BTS + Demo, 프론트 `{env['macro_options']['base']}`. 기존 개발 DB 는 쓰지 않았다.")
        L.append(f"- 매크로: `macro.js` sha256 `{env['macro_js_sha256'][:12]}…`, node {env['node']}, Playwright {env['playwright']}, {env['host']['platform']}.")
        L.append(f"- 실행 방식: {env['launcher']}. 창 표시: {'headless' if env['macro_options']['headless'] else '창을 띄움(headed)'}.")
        L.append(f"- 수집 시작 전 이 DB 의 이벤트 수: {env['bts_db']['events_before_run']}건"
                 + (" (같은 DB 에서 먼저 실행한 다른 계정의 이벤트. 이번 계정의 간격에는 섞이지 않는다)" if env['bts_db']['events_before_run'] else "") + ".")
    else:
        L.append("`environment.json` 이 없어 환경을 기록하지 못했다.")
    if notes:
        L.append(f"\n**실행 메모** (`notes.md`)\n\n{notes}")
    L.append("")

    # ---- 2. 실행 조건과 수집 건수
    L.append("## 2. 매크로별 실행 조건과 수집 건수\n")
    if env:
        o = env["macro_options"]
        L.append(f"공통 옵션: `--seats {o['seats']} --concert {o['concert']} --card {o['card']} --timeout {o['timeout_s']}`. "
                 "계정마다 `node macro.js` 를 예매 1건당 1번 실행했고, 실행 사이에 따로 쉬지 않았다.\n")
    rows = []
    for f in fm[fm["dataset"] == NEW].itertuples(index=False):
        own = b16[b16["account"] == f.account]
        results = own[own["result"] != ""]["result"].value_counts().to_dict()
        events = own["n_attempt_events"].sum() if "n_attempt_events" in own else 0
        retries = int((pd.to_numeric(own["attempts_in_run"], errors="coerce") > 1).sum())
        rows.append([f.account, MODE_LABEL[f.mode], f.n_macro_runs, ", ".join(f"{k} {v}" for k, v in results.items()) or "–",
                     int(f.booking_count), int(f.n_completed_bookings), int(events), f.n_intervals,
                     f.payment_identifiers_used or "–", retries])
    L.append(md_table(["계정", "모드", "매크로 실행", "결과", "예매 시도(attemptId)", "BOOKING", "BOOKING_ATTEMPT 이벤트", "간격 수", "결제 식별자", "한 실행에 시도 2건 이상"], rows))
    L.append("")
    if export:
        L.append(f"- Outbox: {export.get('demo_outbox_by_status')}. 성공한 예매 {export.get('expected_bookings')}건 중 BTS 에 저장된 BOOKING {export.get('bookings_stored_in_bts')}건"
                 f" ({'누락 없음' if export.get('flush_complete') else '**누락 있음**'}).")
    failed = b16[b16["failed_or_timeout"].astype(bool)]
    if len(failed):
        L.append(f"- 실패·타임아웃 {len(failed)}건:")
        for r in failed.itertuples(index=False):
            L.append(f"  - {r.account} {r.run_index}번째: {r.result} {r.error}")
    else:
        L.append("- 실패·재시도·타임아웃: 없음 (모든 실행이 OK).")
    non_success = b16[b16["non_success_stages"].astype("string").fillna("") != ""] if "non_success_stages" in b16 else []
    if len(non_success):
        L.append(f"- 성공이 아닌 단계 기록이 있는 시도 {len(non_success)}건: " + ", ".join(sorted(set(non_success["non_success_stages"]))))
    L.append("- 사람(실제 사용자) 데이터는 **이번에 새로 수집하지 않았다.** 그래프와 표의 사람 계정은 모두 기존 #14 의 human-01, human-02 다.\n")

    L.append("**기존 #14 와의 조건 차이** (같다고 가정하지 않고 기록)\n")
    cond = [["매크로 스크립트", ISSUE14_CONDITIONS["macro_commit"], f"`{(env.get('repos', {}).get('demo-ticketing', {}).get('commit') or '?')[:7]}` (기본 테스트 카드가 선택된 상태에서 바로 결제)"],
            ["결제 수단", ISSUE14_CONDITIONS["card"], "저장된 테스트 카드 A(`--card 1`) → test-pay-shared-01"],
            ["BTS", ISSUE14_CONDITIONS["bts"], f"`{(env.get('repos', {}).get('ticket-fds-admin', {}).get('commit') or '?')[:7]}` ({env.get('repos', {}).get('ticket-fds-admin', {}).get('branch')})"],
            ["Feature 계산식", "`FeatureAggregationService.calculate` @46f8c67", f"같은 파일 (46f8c67 이후 변경 없음: {checks.get('feature_service_unchanged', '확인 안 함')})"],
            ["반복 실행 방식", ISSUE14_CONDITIONS["launcher"], env.get("launcher", "–")],
            ["계정·횟수", "fast-01 3회, fast-02·jitter-01·human-01 각 20회", ", ".join(f"{p['account'].replace('macro-', '')} {p['runs']}회" for p in env.get("plan", []))],
            ["Demo 백엔드·프론트 커밋", "기록 없음 (#14 보고서에 남아 있지 않음)", f"`{(env.get('repos', {}).get('demo-ticketing', {}).get('commit') or '?')[:7]}`"],
            ["수집 순서·시점", "2026-11-19, 사람 예매·파일럿과 같은 DB 에서 fast-01 → (사람) → fast-02 → jitter-01 → human-01", f"{env.get('started_utc', '–')} 시작, 계획 순서대로 연속 실행"],
            ["Playwright / node", "node_modules 설치 시각 2026-11-19 (버전 기록 없음)", f"Playwright {env.get('playwright')} / node {env.get('node')}"]]
    L.append(md_table(["항목", "기존 #14", "신규 #16"], cond))
    L.append("\n- 요청서에는 테스트 카드 A/B/C 로 적혀 있으나, 현재 결제 화면에 있는 저장된 테스트 카드는 A·B 두 장이다 (`frontend/src/api/paymentId.ts`). 결제 식별자는 3종(01~03)이 있고 03 은 직접 입력으로만 나온다.\n")

    # ---- 3. #14 대비 변화
    L.append("## 3. 기존 #14 대비 예매 간격 변화\n")
    L.append("![예매 간격 분포](figures/interval_distribution.png)\n")
    L.append("![예매 간격 분포 (로그 축)](figures/interval_distribution_log.png)\n")
    L.append("![매크로 유형별 예매 간격](figures/interval_by_macro.png)\n")
    L.append("![매크로 유형별 예매 간격 (로그 축)](figures/interval_by_macro_log.png)\n")
    L.append("**개별 간격 통계 (초)**\n")
    rows = []
    for s in stats[(stats["metric"] == "gap_seconds") & (stats["dataset"] != "dummy")].itertuples(index=False):
        rows.append([DATASET_LABEL[s.dataset], s.group, s.variant, s.n, fmt(s.mean), fmt(s.median), fmt(s.std), fmt(s.min), fmt(s.q1), fmt(s.q3), fmt(s.max)])
    L.append(md_table(["데이터", "집단", "구분", "n", "평균", "중앙값", "표준편차", "최소", "Q1", "Q3", "최대"], rows))
    L.append("\n표준편차는 표본 표준편차다. \"긴 간격 제외 (가상)\" 은 관측값이 아니라 표시된 긴 간격을 빼고 다시 계산한 값이다.\n")
    L.append("**모드별 차이** (`comparison_14_vs_16.csv`)\n")
    rows = []
    for r in comp[comp["metric"] == "gap_seconds"].itertuples(index=False):
        rows.append([MODE_LABEL[r.mode], r.variant, f"{r.n_base} / {r.n_new}", fmt(r.mean_base), fmt(r.mean_new), signed(r.mean_diff), signed(r.mean_change_pct, 1, "%"),
                     fmt(r.median_base), fmt(r.median_new), signed(r.median_diff), fmt(r.mannwhitney_p_exploratory, 4)])
    L.append(md_table(["모드", "구분", "n #14 / #16", "평균 #14", "평균 #16", "평균 차", "변화율", "중앙값 #14", "중앙값 #16", "중앙값 차", "Mann–Whitney p (탐색용)"], rows))
    L.append("\nMann–Whitney p 는 참고용이다. 계정이 모드당 1~2개이고 한 계정 안의 간격은 서로 독립이 아니므로 유의성 판단에 쓰지 않는다.\n")

    L.append("**계정별 Feature** (BTS 계산식, 긴 간격 포함)\n")
    L.append("![계정별 평균 예매 간격과 분산](figures/account_comparison.png)\n")
    rows = []
    for f in feats.sort_values(["account", "dataset"]).itertuples(index=False):
        rows.append([f.account, DATASET_LABEL[f.dataset], int(f.booking_count), f.n_intervals, fmt(f.avg_booking_interval, 3), fmt(f.booking_interval_variance, 3),
                     fmt(f.interval_median, 3), fmt(f.interval_min, 2), fmt(f.interval_max, 2), f.n_long_gaps,
                     fmt(f.avg_booking_interval_excl_long_gap_hypothetical, 3) if f.n_long_gaps else "–",
                     fmt(f.booking_interval_variance_excl_long_gap_hypothetical, 3) if f.n_long_gaps else "–"])
    L.append(md_table(["계정", "데이터", "시도 수", "간격 수", "avg_booking_interval (초)", "booking_interval_variance (초²)", "간격 중앙값", "최소", "최대",
                       "긴 간격 수", "평균, 긴 간격 제외 (가상)", "분산, 긴 간격 제외 (가상)"], rows))
    L.append("")

    # ---- 4. 더미 대비
    L.append("## 4. 기존 더미데이터와 신규 실측 분포 차이\n")
    L.append("![avg_booking_interval](figures/feature_mean_comparison.png)\n")
    L.append("![booking_interval_variance](figures/feature_variance_comparison.png)\n")
    rows = []
    for s in stats[stats["metric"].isin(["avg_booking_interval", "booking_interval_variance"])
                   & (stats["group"].isin(["더미 정상", "더미 매크로", "실측 매크로 계정"]))].itertuples(index=False):
        rows.append([f"`{s.metric}` ({s.unit})", {"dummy": "기존 더미"}.get(s.dataset, DATASET_LABEL.get(s.dataset)), s.group, s.variant, s.n,
                     fmt(s.mean, 3), fmt(s.median, 3), fmt(s.std, 3), fmt(s.min, 3), fmt(s.q1, 3), fmt(s.q3, 3), fmt(s.max, 3)])
    L.append(md_table(["Feature", "데이터", "집단", "구분", "n (계정)", "평균", "중앙값", "표준편차", "최소", "Q1", "Q3", "최대"], rows))
    L.append("")
    dn = dummy_groups["더미 정상"]
    new = fm[fm["dataset"] == NEW]
    for feature, unit in (("avg_booking_interval", "초"), ("booking_interval_variance", "초²")):
        dmv, dnv, nv = dummy_groups["더미 매크로"][feature], dn[feature], new[feature].dropna()
        if not len(nv):
            continue
        L.append(f"- `{feature}`: 신규 실측 매크로 {fmt(nv.min(), 3)}~{fmt(nv.max(), 3)}{unit}. 더미 매크로 범위 {fmt(dmv.min(), 3)}~{fmt(dmv.max(), 3)}{unit} 안에 드는 계정 "
                 f"{int(((nv >= dmv.min()) & (nv <= dmv.max())).sum())}/{len(nv)}개, 더미 정상 범위 {fmt(dnv.min(), 3)}~{fmt(dnv.max(), 3)}{unit} 안에 드는 계정 "
                 f"{int(((nv >= dnv.min()) & (nv <= dnv.max())).sum())}/{len(nv)}개.")
    L.append("- 더미는 계정 단위 값(계정당 한 점)이고, 실측 계정은 모드당 1~2개뿐이다. 분포의 모양을 비교할 수 있는 표본이 아니므로 \"실측 값이 더미 분포의 어디에 놓이는가\"만 본다.\n")

    # ---- 5. 카드 UI 변경
    L.append("## 5. 카드번호 직접 입력 제거에 따른 변화\n")
    L.append("![결제 단계·전체 예매 소요 시간](figures/payment_duration.png)\n")
    L.append("![예매 간격의 구성](figures/gap_breakdown.png)\n")
    L.append("예매 1건 안의 소요 시간(ms, 성공한 예매)과 간격 구성요소. 차이 = #16 − #14.\n")
    rows = []
    for mode in new_modes:
        for metric, name in (("payment_step_ms", "결제 단계 (ms_paid − ms_hold)"), ("click_to_hold_ms", "예매하기 클릭~좌석 선점"),
                             ("pre_click_ms", "로그인~예매하기 클릭 전"), ("ms_booking", "전체 예매 소요 시간 (ms_booking)"),
                             ("ms_total", "실행 시작~결제 완료 (ms_total)"), ("launch_overhead_s", "브라우저 종료·재기동")):
            r = c(mode, metric)
            if r is None:
                continue
            k = 1000 if r.unit == "초" else 1
            rows.append([MODE_LABEL[mode], name, f"{r.n_base} / {r.n_new}", fmt(r.mean_base * k, 0), fmt(r.mean_new * k, 0), signed(r.mean_diff * k, 0),
                         fmt(r.median_base * k, 0), fmt(r.median_new * k, 0), signed(r.median_diff * k, 0)])
    L.append(md_table(["모드", "구간", "n #14 / #16", "평균 #14 (ms)", "평균 #16 (ms)", "평균 차 (ms)", "중앙값 #14 (ms)", "중앙값 #16 (ms)", "중앙값 차 (ms)"], rows))
    L.append("")
    L.append("**간격 변화 중 결제 단계가 차지하는 몫** (중앙값 기준)\n")
    rows = []
    for mode in new_modes:
        g, p, q, o, h = (c(mode, "gap_seconds"), c(mode, "payment_step_ms"), c(mode, "pre_click_ms"), c(mode, "launch_overhead_s"), c(mode, "click_to_hold_ms"))
        if g is None or p is None:
            continue
        share = (p.median_diff / 1000) / g.median_diff * 100 if abs(g.median_diff) >= 0.3 else np.nan
        rows.append([MODE_LABEL[mode], signed(g.median_diff), signed(p.median_diff / 1000), signed(h.median_diff / 1000) if h is not None else "–",
                     signed(q.median_diff / 1000) if q is not None else "–", signed(o.median_diff) if o is not None else "–",
                     (fmt(share, 0) + "%") if pd.notna(share) else "– (간격 차 0.3초 미만)", signed(-EXPECTED_PAYMENT_CUT_MS[mode] / 1000)])
    L.append(md_table(["모드", "간격 중앙값 차 (초)", "결제 단계 차 (초)", "클릭~선점 차 (초)", "클릭 전 구간 차 (초)", "브라우저 재기동 차 (초)",
                       "결제 단계 차 ÷ 간격 차", "코드상 기대되는 결제 단계 차 (초)"], rows))
    L.append("")
    L.append("- \"결제 단계 차 ÷ 간격 차\"가 100% 를 넘으면, 결제 단계가 줄어든 만큼 간격이 줄지 않았다는 뜻이다 (다른 구간이 늘어 일부를 상쇄).")
    L.append("- \"코드상 기대되는 결제 단계 차\"는 `macro.js` 의 두 버전을 비교해 계산한 값이다. 기존에는 결제 화면에서 `직접 입력 선택 → 쉼 → 15자리 입력 → 쉼 → 결제하기` 였고, "
             "지금은 바로 `결제하기` 다. 줄어든 쉼 2회와 글자 입력 시간의 기댓값(각 모드 지연 범위의 중간값)만 더했다. 관측값이 아니다.")
    L.append("- 결제 단계의 관측 변화가 이 기대값과 가까우면, 결제 단계의 변화는 입력 절차가 없어진 것과 방향·크기가 맞는다는 뜻이다. "
             "다만 이것만으로 간격 변화 전체의 원인을 카드 UI 변경으로 단정할 수는 없다. 아래 요인이 함께 바뀌었다.")
    L.append("  - 클릭 전 구간(로그인~공연 진입)과 브라우저 재기동 시간은 카드 UI 와 무관한데도 실행마다 달라진다. 위 표의 해당 열이 0 이 아니면 그만큼은 다른 요인이다.")
    L.append("  - BTS·Demo 백엔드 커밋, 반복 실행 방식(PowerShell → Python), 수집 일시와 PC 상태가 다르다 (2장 표).")
    L.append("  - 두 수집 모두 모드당 계정 1~2개, 1회 수집이라 실행 간 변동 폭을 알 수 없다.")
    L.append("- 시연용 결제 지연 1,200ms(`PAY_DELAY_MS`)는 두 버전 모두 결제 단계에 들어 있어, 결제 단계는 이 값 아래로 내려가지 않는다.\n")

    # ---- 6. 이상치와 한계
    L.append("## 6. 이상치와 측정 한계\n")
    flagged = iv[iv["long_gap"].astype(bool)]
    L.append("긴 간격 표시 기준: 같은 계정의 간격이 5개 이상일 때 `Q3 + 3×IQR` 와 계정 중앙값의 2배를 모두 넘는 간격. 표시만 하고 지우지 않았다. "
             "원본 통계가 기본이고, 제외한 값은 \"가상\"으로 따로 적었다.\n")
    if len(flagged):
        rows = []
        for r in flagged.itertuples(index=False):
            rows.append([DATASET_LABEL[r.dataset], r.account, r.interval_index, fmt(r.gap_seconds), fmt(r.flow_after_click_s), fmt(r.next_before_click_s),
                         fmt(r.launch_overhead_s), r.from_attempt_utc])
        L.append(md_table(["데이터", "계정", "간격 번호", "간격 (초)", "예매하기 클릭~결제 완료 (초)", "다음 실행 클릭 전 (초)", "브라우저 종료·재기동 (초)", "앞 시도 시작 (UTC)"], rows))
        L.append("")
        for dataset in (BASE, NEW):
            n = int((flagged["dataset"] == dataset).sum())
            L.append(f"- {DATASET_LABEL[dataset]}: 긴 간격 {n}개" + ("." if n else " (이번 기준으로는 없음)."))
    else:
        L.append("- 두 데이터 모두 기준을 넘는 긴 간격이 없다.")
    L.append("\n**긴 간격 민감도** (가상: 표시된 간격을 빼고 다시 계산)\n")
    rows = []
    for f in fm[fm["n_long_gaps"] > 0].itertuples(index=False):
        rows.append([DATASET_LABEL[f.dataset], f.account, f.n_long_gaps, fmt(f.avg_booking_interval, 3), fmt(f.avg_booking_interval_excl_long_gap_hypothetical, 3),
                     fmt(f.booking_interval_variance, 3), fmt(f.booking_interval_variance_excl_long_gap_hypothetical, 3)])
    L.append(md_table(["데이터", "계정", "긴 간격 수", "평균 원본", "평균 가상", "분산 원본", "분산 가상"], rows) if rows else "- 해당 계정 없음.")
    L.append("")
    over = stats[(stats["metric"] == "launch_overhead_s") & (stats["group"] == "전체 매크로")]
    for s in over.itertuples(index=False):
        L.append(f"- {DATASET_LABEL[s.dataset]} 브라우저 종료·재기동 구간: 중앙값 {fmt(s.median)}초, 최대 {fmt(s.max)}초 (간격 {s.n}개).")
    L.append("- 모든 간격에는 브라우저·node 재기동 시간이 들어 있다. 한 브라우저 안에서 연속 예매하는 매크로라면 간격이 이보다 짧을 수 있다.")
    L.append("- 간격은 \"시도 시작\" 기준이므로, 앞 예매의 소요 시간 전체와 다음 예매의 로그인·공연 진입 시간이 포함된다.")
    L.append("- 실측 계정 수가 모드당 1~2개다. 계정 단위 Feature(평균·분산)의 분포는 추정할 수 없고, 값의 위치만 볼 수 있다.")
    L.append("- 시도 2~3건인 계정(fast-01)은 간격이 1~2개라 평균·분산이 불안정하다.")
    L.append("- 사람 데이터는 재수집하지 않았다. 사람과 매크로의 비교는 이번 실행의 결과가 아니다.")
    L.append("- 검증: " + "; ".join(f"{k} = {v}" for k, v in checks.items()) + ".\n")

    # ---- 7. 다음 단계 참고
    L.append("## 7. 다음 단계에서 더미데이터 분포를 수정할 때 참고할 사항\n")
    L.append("이번 작업에서는 더미 생성 로직과 모델을 바꾸지 않았다. 아래는 수정할 때 쓸 수 있는 실측 수치다.\n")
    rows = []
    for mode in new_modes:
        s = stats[(stats["metric"] == "gap_seconds") & (stats["dataset"] == NEW) & (stats["group"] == MODE_LABEL[mode]) & (stats["variant"] == "원본")]
        if not len(s):
            continue
        s = s.iloc[0]
        cv = s["std"] / s["mean"] if s["mean"] else np.nan
        rows.append([MODE_LABEL[mode], s["n"], fmt(s["mean"]), fmt(s["median"]), fmt(s["std"]), fmt(cv, 3), f"{fmt(s['min'])}~{fmt(s['max'])}"])
    L.append(md_table(["모드 (신규 #16)", "간격 수", "평균 (초)", "중앙값 (초)", "표준편차 (초)", "변동계수", "범위 (초)"], rows))
    L.append("")
    L.append(f"- 기존 더미 매크로의 `avg_booking_interval` 은 `exponential(0.2) + 0.1` 로 {fmt(dm.min())}~{fmt(dm.max())}초다. "
             "브라우저를 조작하는 매크로의 실측 간격은 카드 입력을 없앤 뒤에도 위 표의 범위다. 더미 매크로 간격의 하한·범위를 정할 때 이 표를 기준으로 삼을 수 있다.")
    L.append("- 실측 간격의 하한은 `결제 지연 1.2초 + 페이지 이동·좌석 선점 + 브라우저 재기동`으로 정해진다 (구성 그래프 참고). "
             "더미에서 이보다 짧은 간격을 유지하려면 \"브라우저 없이 API 를 직접 호출하는 매크로\" 같은 별도 가정이 필요하며, 그 가정은 이번 실측으로 검증되지 않았다.")
    L.append("- 분산은 평균과 따로 뽑지 말고 개별 간격을 생성한 뒤 계산하는 편이 실측과 맞는다 (#14 결론과 같음). 모드 안에서의 흩어짐은 위 표의 표준편차·변동계수를 참고한다.")
    L.append("- 긴 간격 하나가 평균·분산을 크게 바꾼다 (6장 민감도 표). 더미에 드문 긴 간격을 넣을지, Feature 를 중앙값·IQR 처럼 덜 민감한 값으로 바꿀지는 별도 결정이 필요하다.")
    L.append("- Human-like 모드의 간격이 더미 정상 범위와 겹치는지는 4장의 범위 비교를 본다. 겹친다면 평균 간격만으로는 그 유형을 정상과 구분할 수 없다.")
    L.append("- 분포를 바꾸기 전에 같은 조건으로 2~3회 더 수집해 실행 간 변동을 확인하는 것이 좋다 (`collect.py` 를 다시 실행하면 `runs/` 에 새 폴더가 생긴다).\n")

    # ---- 파일
    L.append("## 파일\n")
    L.append(md_table(["파일", "내용"], [
        ["`environment.json`", "커밋, 옵션, 프론트 mock 설정, 버전"],
        ["`macro_runs.csv`, `macro-results-<계정>.csv`, `macro_stdout.log`", "매크로 실행 기록 (실행 시작·종료 시각 포함)"],
        ["`raw_events.csv`, `users.csv`, `export_summary.json`", "BTS `behavior_event` 내보내기 (BOOKING, BOOKING_ATTEMPT)"],
        ["`bookings.csv`", "예매 1건당 한 줄: 시작·완료 시각, 단계별 소요 시간, 카드, 실패 여부 (#14 포함)"],
        ["`booking_intervals.csv`", "연속 예매 간격 1개당 한 줄 + 구성요소 + 긴 간격 표시 (#14 포함)"],
        ["`feature_summary.csv`", "계정별 Feature 와 긴 간격 제외 민감도"],
        ["`distribution_stats.csv`, `comparison_14_vs_16.csv`", "요약 통계, 모드별 차이"],
        ["`figures/`", "그래프 PNG"],
    ]))
    L.append("")
    (run_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


def write_report_runs(run_dir, base_dir, bookings, intervals, feats, stats, comp, dummy_groups, checks):
    """--baseline-run: 이 폴더의 두 실행(예: 재실행 방식 대 연속 예매)을 비교하는 보고서."""
    def load(path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    env, env_b = load(run_dir / "environment.json"), load(base_dir / "environment.json")
    export = load(run_dir / "export_summary.json")
    notes = (run_dir / "notes.md").read_text(encoding="utf-8").strip() if (run_dir / "notes.md").exists() else ""
    iv, fm = macro_only(intervals), macro_only(feats)
    bn = bookings[bookings["dataset"] == NEW]
    new_modes = [m for m in MODES if len(bn[bn["mode"] == m])]
    LB, LN = DATASET_LABEL[BASE], DATASET_LABEL[NEW]

    def c(mode, metric, variant="원본"):
        row = comp[(comp["mode"] == mode) & (comp["metric"] == metric) & (comp["variant"] == variant)]
        return row.iloc[0] if len(row) else None

    def st(dataset, metric, group, variant="원본"):
        row = stats[(stats["dataset"] == dataset) & (stats["metric"] == metric) & (stats["group"] == group) & (stats["variant"] == variant)]
        return row.iloc[0] if len(row) else None

    L = []
    L.append(f"# 이슈 #16 연속 예매(브라우저 재사용) 측정 보고서 — 실행 `{run_dir.name}`\n")
    L.append(f"비교 기준: `runs/{base_dir.name}` ({LB}). 생성 스크립트 `scripts/analyze.py --baseline-run {base_dir.name}` "
             "(자동 생성 문서. 직접 고치지 말고 `notes.md` 에 메모를 적은 뒤 다시 생성한다)\n")

    # ---- 0. 요약
    L.append(f"## 0. 요약: {LB} 대 {LN}\n")
    L.append("예매 간격 = 같은 계정의 연속한 예매 시도 시작 시각 사이의 간격 (BTS `avg_booking_interval` 의 재료, 두 방식 모두 같은 계산식). 개별 간격 기준.\n")
    rows = []
    for mode in new_modes:
        r = c(mode, "gap_seconds")
        if r is None:
            continue
        rows.append([MODE_LABEL[mode], f"{fmt(r.mean_base)} → {fmt(r.mean_new)}", f"{signed(r.mean_diff)} ({signed(r.mean_change_pct, 1, '%')})",
                     f"{fmt(r.median_base)} → {fmt(r.median_new)}", f"{signed(r.median_diff)} ({signed(r.median_change_pct, 1, '%')})",
                     f"{fmt(r.std_base)} → {fmt(r.std_new)}", f"{int(r.n_base)} / {int(r.n_new)}"])
    L.append(md_table(["모드", f"평균 (초) {DATASET_SHORT[BASE]} → {DATASET_SHORT[NEW]}", "평균 변화", "중앙값 (초)", "중앙값 변화", "표준편차 (초)", "간격 수"], rows))
    L.append("")
    dm = dummy_groups["더미 매크로"]["avg_booking_interval"]
    new_avg = fm[fm["dataset"] == NEW]["avg_booking_interval"].dropna()
    if len(new_avg):
        L.append(f"- {LN} 계정의 `avg_booking_interval` 은 {fmt(new_avg.min())}~{fmt(new_avg.max())}초다. 기존 더미 매크로는 {fmt(dm.min())}~{fmt(dm.max())}초"
                 f"(중앙값 {fmt(dm.median())}초)로, {len(new_avg)}계정 중 {int((new_avg > dm.max()).sum())}계정이 더미 매크로의 최댓값보다 크다.")
    L.append("- 줄어든 시간의 출처는 4장의 구간별 표에 있다. 두 실행은 같은 PC·같은 날·같은 코드(매크로만 수정)에서 수집했지만 각각 1회 수집이다.\n")

    # ---- 1. 환경
    L.append("## 1. 실험 환경과 두 방식의 차이\n")
    if env:
        rows = []
        for name, info in env["repos"].items():
            before = env_b.get("repos", {}).get(name, {})
            rows.append([name, f"`{(info.get('commit') or '?')[:7]}`", info.get("branch"),
                         "없음" if not info.get("dirty_files") else "; ".join(info["dirty_files"]),
                         f"`{(before.get('commit') or '?')[:7]}`" + ("" if before.get("commit") == info.get("commit") else " (다름)")])
        L.append(md_table(["레포", "커밋", "브랜치", "커밋되지 않은 변경", f"기준 실행({base_dir.name})의 커밋"], rows))
        L.append("")
        admin = env["repos"].get("ticket-fds-admin", {})
        if admin.get("origin_dev"):
            L.append(f"- BTS: 체크아웃의 파일 내용이 `origin/dev`(`{admin['origin_dev'][:7]}`)와 {'같다' if admin.get('tree_equals_origin_dev') else '**다르다**'}.")
        L.append("- 프론트 `.env`: " + ", ".join(f"`{k}={v}`" for k, v in env.get("frontend_env", {}).items()) + " (mock 티켓팅·BTS 비활성).")
        L.append(f"- 스택: `demo-ticketing/e2e` 의 일회성 DB(tmpfs, `{env['bts_db']['container']}`) + BTS + Demo. 기존 개발 DB 는 쓰지 않았다. 수집 시작 전 이 DB 의 이벤트 수: {env['bts_db']['events_before_run']}건"
                 + (" (점검용 다른 계정의 이벤트. 이번 계정의 간격에는 섞이지 않는다)." if env['bts_db']['events_before_run'] else "."))
        same_macro = env.get("macro_js_sha256") == env_b.get("macro_js_sha256")
        L.append(f"- `macro.js` sha256: 기준 실행 `{(env_b.get('macro_js_sha256') or '?')[:12]}…`, 이번 실행 `{env['macro_js_sha256'][:12]}…`"
                 + (" (같은 파일)." if same_macro else " (다름: 연속 예매 모드 `--repeat` 를 추가한 수정본. 1회 예매 경로의 동작 순서와 지연 설정은 바꾸지 않았다)."))
        L.append(f"- node {env['node']}, Playwright {env['playwright']}, 창 표시: {'headless' if env['macro_options']['headless'] else '창을 띄움(headed)'}, "
                 f"옵션 `--seats {env['macro_options']['seats']} --concert {env['macro_options']['concert']} --card {env['macro_options']['card']}`.")
    L.append("")
    L.append(md_table(["항목", LB, LN], [
        ["매크로 실행", env_b.get("launcher", "–"), env.get("launcher", "–")],
        ["브라우저", "예매 1건마다 node·브라우저를 새로 실행하고 종료", "계정당 1번 실행해 끝까지 유지 (계정이 바뀌면 새 브라우저 컨텍스트)"],
        ["로그인", "예매 1건마다 로그인", "계정당 1번. 이후에는 같은 페이지에서 로그인 상태 유지"],
        ["다음 예매로 가는 길", "새 브라우저 → 로그인 → 공연 목록 → 공연 → 예매하기", "완료 화면 → `공연 목록으로` → 공연 → 예매하기"],
        ["클릭·입력 지연 (Fast/Jitter/Human-like 프로파일)", "PROFILES 그대로", "PROFILES 그대로 (변경 없음)"],
        ["결제", "기본 테스트 카드 A", "기본 테스트 카드 A (`--card` 옵션 동작 동일)"],
        ["BTS 이벤트 수집·Feature 계산", "`export_events` SQL, `extract_features.extract`", "같음 (변경 없음)"],
    ]))
    L.append("")
    L.append("- 연속 예매에서는 다음 예매로 넘어가기 전에 끝난 시도의 진행 기록(`sessionStorage` 의 `fds.attempt.v1.<계정>.<공연>`)을 매크로가 지운다. "
             "이 기록이 남아 있으면 같은 탭에서 두 번째 예매의 `예매하기` 가 409(요청이 현재 리소스 상태와 충돌)로 막힌다 (#14 README 에 적힌 현상과 같다). "
             "새 탭을 여는 것과 같은 효과이고 로그인 세션은 건드리지 않는다. 프론트엔드 코드는 수정하지 않았다.")
    if notes:
        L.append(f"\n**실행 메모** (`notes.md`)\n\n{notes}")
    L.append("")

    # ---- 2. 수집 건수
    L.append("## 2. 수집 건수와 연속 예매 가능 여부\n")
    rows = []
    for f in fm[fm["dataset"] == NEW].itertuples(index=False):
        own = bn[bn["account"] == f.account]
        planned = int((own["result"] != "").sum())
        ok = int((own["result"] == "OK").sum())
        not_run = int((own["result"] == "NOT_RUN").sum())
        failed = planned - ok - not_run
        retries = int((pd.to_numeric(own.get("hold_tries"), errors="coerce") > 1).sum()) if "hold_tries" in own else 0
        verdict = f"가능 ({ok}/{planned} 연속 성공)" if ok == planned else f"**중단됨** (성공 {ok}, 실패 {failed}, 미실행 {not_run})"
        rows.append([f.account, MODE_LABEL[f.mode], planned, ok, failed, not_run, int(f.booking_count), int(f.n_completed_bookings), f.n_intervals,
                     retries, f.payment_identifiers_used or "–", verdict])
    L.append(md_table(["계정", "모드", "계획", "성공", "실패", "미실행", "예매 시도(attemptId)", "BOOKING", "간격 수", "좌석 선점 재시도", "결제 식별자", "연속 예매"], rows))
    L.append("")
    if export:
        L.append(f"- Outbox: {export.get('demo_outbox_by_status')}. 성공한 예매 {export.get('expected_bookings')}건 중 BTS 에 저장된 BOOKING {export.get('bookings_stored_in_bts')}건"
                 f" ({'누락 없음' if export.get('flush_complete') else '**누락 있음**'}).")
    failed = bn[bn["failed_or_timeout"].astype(bool) & (bn["result"] != "NOT_RUN")]
    if len(failed):
        L.append(f"- 실패 {len(failed)}건 (원인 기록):")
        for r in failed.itertuples(index=False):
            L.append(f"  - {r.account} {r.run_index}번째: `{r.result}` 단계 `{getattr(r, 'fail_step', '')}` — {r.error} (주소 {getattr(r, 'fail_url', '')})")
        L.append("  - 실패가 나면 매크로는 그 계정의 남은 예매를 실행하지 않는다 (`--continue-on-fail` 을 주지 않은 경우). 미실행 건은 값 없이 `NOT_RUN` 으로만 남는다.")
    else:
        L.append("- 실패·타임아웃: 없음. 모든 계정이 계획한 횟수를 한 브라우저·한 로그인으로 끝까지 예매했다.")
    L.append("- 사람(실제 사용자) 데이터는 수집하지 않았다.\n")

    # ---- 3. 간격 비교
    L.append("## 3. 예매 간격 비교\n")
    L.append("![매크로 유형별 예매 간격](figures/interval_by_macro.png)\n")
    L.append("![매크로 유형별 예매 간격 (로그 축)](figures/interval_by_macro_log.png)\n")
    L.append("![예매 간격 분포](figures/interval_distribution.png)\n")
    L.append("![예매 간격 분포 (로그 축)](figures/interval_distribution_log.png)\n")
    L.append("**개별 간격 통계 (초)**\n")
    rows = []
    for s_ in stats[(stats["metric"] == "gap_seconds") & (stats["dataset"] != "dummy")].itertuples(index=False):
        rows.append([DATASET_LABEL[s_.dataset], s_.group, s_.variant, s_.n, fmt(s_.mean), fmt(s_.median), fmt(s_.std), fmt(s_.min), fmt(s_.q1), fmt(s_.q3), fmt(s_.max)])
    L.append(md_table(["데이터", "집단", "구분", "n", "평균", "중앙값", "표준편차", "최소", "Q1", "Q3", "최대"], rows))
    L.append("\n표준편차는 표본 표준편차다. \"긴 간격 제외 (가상)\" 은 관측값이 아니라 표시된 긴 간격을 빼고 다시 계산한 값이다.\n")

    # ---- 4. 구간별
    L.append("## 4. 구간별 소요 시간\n")
    L.append("![예매 간격의 구성](figures/gap_breakdown.png)\n")
    L.append("![결제 단계·전체 예매 소요 시간](figures/payment_duration.png)\n")
    L.append("**간격을 이루는 구간** (초, 간격 단위로 분해. 차이 = 신규 − 기준)\n")
    rows = []
    for mode in new_modes:
        for metric, name in (("gap_seconds", "예매 간격 (전체)"), ("click_to_hold_s", "① 예매하기 클릭~좌석 선점"), ("payment_step_s", "② 결제 단계"),
                             ("after_paid_to_next_begin_s", "③ 예매 완료 → 다음 예매하기 클릭 (= ③a + ③b)"),
                             ("launch_overhead_s", "③a 브라우저 종료·재기동"), ("next_before_click_s", "③b 다음 예매 준비 (예매하기 클릭 전까지)")):
            r = c(mode, metric)
            if r is None:
                continue
            rows.append([MODE_LABEL[mode], name, f"{r.n_base} / {r.n_new}", fmt(r.median_base), fmt(r.median_new), signed(r.median_diff),
                         fmt(r.mean_base), fmt(r.mean_new), signed(r.mean_diff)])
    L.append(md_table(["모드", "구간", "n", f"중앙값 {DATASET_SHORT[BASE]}", f"중앙값 {DATASET_SHORT[NEW]}", "중앙값 차", f"평균 {DATASET_SHORT[BASE]}", f"평균 {DATASET_SHORT[NEW]}", "평균 차"], rows))
    L.append("")
    L.append("- ③b 는 재실행 방식에서는 `로그인 → 공연 목록 → 공연 → 예매하기 클릭 전`, 연속 예매에서는 `완료 화면 → 공연 목록 → 공연 → 예매하기 클릭 전` 이다.")
    L.append("- 연속 예매의 ③a 는 브라우저를 다시 띄우지 않으므로 실제 구간이 아니라 계산 잔차다 (서버가 기록한 시도 시작 시각과 매크로가 잰 클릭 시각의 차이 수준).")
    L.append("- ①·② 는 두 방식에서 같은 화면·같은 지연 설정이다. 여기서 차이가 나면 방식 차이가 아니라 실행 간 변동으로 본다.\n")

    L.append("**브라우저 최초 실행과 로그인 시간**\n")
    first = bn[pd.to_numeric(bn.get("session_seq"), errors="coerce") == 1] if "session_seq" in bn else bn.iloc[0:0]
    if len(first):
        rows = [[r.account, MODE_LABEL[r.mode], fmt(num(r.ms_browser_launch), 0), fmt(num(r.ms_context_open), 0), fmt(num(r.ms_login), 0),
                 fmt(num(r.pre_click_ms), 0), fmt(num(r.invoke_wall_ms), 0)] for r in first.itertuples(index=False)]
        L.append(f"{LN} — 계정(세션)당 1번만 발생:\n")
        L.append(md_table(["계정", "모드", "브라우저 실행 (ms)", "컨텍스트·페이지 생성 (ms)", "로그인: 접속~공연 목록 표시 (ms)",
                           "첫 예매: 시작~예매하기 클릭 전 (ms)", "세션 전체 실행 시간 (ms)"], rows))
        L.append("")
    rows = []
    bb = bookings[(bookings["dataset"] == BASE) & (bookings["result"] == "OK") & bookings["mode"].isin(MODES)]
    for mode in new_modes:
        sub = bb[bb["mode"] == mode]
        if not len(sub):
            continue
        login, over = describe(pd.to_numeric(sub["ms_login"], errors="coerce")), st(BASE, "launch_overhead_s", MODE_LABEL[mode])
        rows.append([MODE_LABEL[mode], login["n"], fmt(login["median"], 0), fmt(login["min"], 0), fmt(login["max"], 0),
                     fmt(over["median"] * 1000, 0) if over is not None else "–", fmt(over["max"] * 1000, 0) if over is not None else "–"])
    if rows:
        L.append(f"{LB} — 예매 1건마다 발생:\n")
        L.append(md_table(["모드", "예매 수", "로그인 중앙값 (ms)", "로그인 최소", "로그인 최대", "브라우저 종료·재기동 중앙값 (ms)", "재기동 최대 (ms)"], rows))
        L.append("\n재실행 방식의 재기동 시간은 직접 잰 값이 아니라 `간격 − 예매 소요 시간 − 다음 실행의 클릭 전 구간` 으로 계산한 나머지다 (브라우저 종료 + node 시작 + 브라우저 실행).\n")

    L.append("**예매 완료 후 다음 예매 시작까지** (③, 초)\n")
    rows = []
    for dataset in (BASE, NEW):
        for mode in new_modes:
            r = st(dataset, "after_paid_to_next_begin_s", MODE_LABEL[mode])
            if r is not None and r["n"]:
                rows.append([DATASET_LABEL[dataset], MODE_LABEL[mode], r["n"], fmt(r["mean"]), fmt(r["median"]), fmt(r["std"]), fmt(r["min"]), fmt(r["q1"]), fmt(r["q3"]), fmt(r["max"])])
    L.append(md_table(["데이터", "모드", "n", "평균", "중앙값", "표준편차", "최소", "Q1", "Q3", "최대"], rows))
    L.append("\n`booking_intervals.csv` 의 `after_paid_to_next_begin_s`. 연속 예매는 `macro_runs.csv` 의 `ms_prev_paid_to_begin` (매크로가 직접 잰 값) 과 같은 구간이다.\n")

    # ---- 5. Feature
    L.append("## 5. 계정별 Feature 와 더미데이터 대비 위치\n")
    L.append("![계정별 평균 예매 간격과 분산](figures/account_comparison.png)\n")
    rows = []
    for f in fm.sort_values(["account", "dataset"]).itertuples(index=False):
        rows.append([f.account, DATASET_LABEL[f.dataset], int(f.booking_count), f.n_intervals, fmt(f.avg_booking_interval, 3), fmt(f.booking_interval_variance, 3),
                     fmt(f.interval_std_population, 3), fmt(f.interval_median, 3), fmt(f.interval_min, 2), fmt(f.interval_max, 2), f.n_long_gaps])
    L.append(md_table(["계정", "데이터", "시도 수", "간격 수", "avg_booking_interval (초)", "booking_interval_variance (초²)", "표준편차 (초)", "간격 중앙값", "최소", "최대", "긴 간격 수"], rows))
    L.append("\nBTS 계산식 그대로 (긴 간격 포함, 분산은 모분산).\n")
    L.append("![avg_booking_interval](figures/feature_mean_comparison.png)\n")
    L.append("![booking_interval_variance](figures/feature_variance_comparison.png)\n")
    dn = dummy_groups["더미 정상"]
    for dataset in (BASE, NEW):
        sub = fm[fm["dataset"] == dataset]
        for feature, unit in (("avg_booking_interval", "초"), ("booking_interval_variance", "초²")):
            dmv, dnv, nv = dummy_groups["더미 매크로"][feature], dn[feature], sub[feature].dropna()
            if not len(nv):
                continue
            L.append(f"- {DATASET_LABEL[dataset]} `{feature}`: {fmt(nv.min(), 3)}~{fmt(nv.max(), 3)}{unit}. 더미 매크로 범위({fmt(dmv.min(), 3)}~{fmt(dmv.max(), 3)}) 안 "
                     f"{int(((nv >= dmv.min()) & (nv <= dmv.max())).sum())}/{len(nv)}계정, 더미 정상 범위({fmt(dnv.min(), 3)}~{fmt(dnv.max(), 3)}) 안 "
                     f"{int(((nv >= dnv.min()) & (nv <= dnv.max())).sum())}/{len(nv)}계정.")
    L.append("- 실측 계정은 모드당 1~2개다. 분포의 모양이 아니라 값의 위치만 본다.\n")

    # ---- 6. 이상치·한계
    L.append("## 6. 이상치와 측정 한계\n")
    L.append("긴 간격 표시 기준: 같은 계정의 간격이 5개 이상일 때 `Q3 + 3×IQR` 와 계정 중앙값의 2배를 모두 넘는 간격. 표시만 하고 지우지 않았다.\n")
    flagged = iv[iv["long_gap"].astype(bool)]
    if len(flagged):
        rows = [[DATASET_LABEL[r.dataset], r.account, r.interval_index, fmt(r.gap_seconds), fmt(r.flow_after_click_s), fmt(r.after_paid_to_next_begin_s), r.from_attempt_utc]
                for r in flagged.itertuples(index=False)]
        L.append(md_table(["데이터", "계정", "간격 번호", "간격 (초)", "예매하기 클릭~결제 완료 (초)", "완료~다음 예매하기 클릭 (초)", "앞 시도 시작 (UTC)"], rows))
        L.append("")
        rows = [[DATASET_LABEL[f.dataset], f.account, f.n_long_gaps, fmt(f.avg_booking_interval, 3), fmt(f.avg_booking_interval_excl_long_gap_hypothetical, 3),
                 fmt(f.booking_interval_variance, 3), fmt(f.booking_interval_variance_excl_long_gap_hypothetical, 3)] for f in fm[fm["n_long_gaps"] > 0].itertuples(index=False)]
        L.append("민감도 (가상: 표시된 간격을 빼고 다시 계산)\n")
        L.append(md_table(["데이터", "계정", "긴 간격 수", "평균 원본", "평균 가상", "분산 원본", "분산 가상"], rows))
        L.append("")
    else:
        L.append("- 두 실행 모두 기준을 넘는 긴 간격이 없다.")
    L.append("- 연속 예매의 간격에는 브라우저 실행·로그인이 들어가지 않지만, 예매 1건의 전체 흐름(대기열 → 좌석 → 선점 → 결제 지연 1.2초)과 완료 화면에서 공연으로 돌아가는 이동은 들어 있다.")
    L.append("- 두 실행은 각각 1회 수집이고 모드당 계정 1~2개다. 실행 간 변동 폭은 알 수 없으므로 수백 ms 수준의 차이는 방식 차이로 해석하지 않는다.")
    L.append("- 연속 예매의 첫 건만 로그인을 포함한다. 간격은 시도 시작 시각의 차이이므로 첫 건의 로그인 시간은 어떤 간격에도 들어가지 않는다.")
    L.append("- 공연의 남은 좌석이 줄어드는 정도(96석 중 약 70석 사용)는 두 실행이 비슷하다. Fast 는 항상 목록의 첫 좌석을, Jitter·Human-like 는 무작위 좌석을 고른다.")
    L.append("- 검증: " + "; ".join(f"{k} = {v}" for k, v in checks.items()) + ".\n")

    # ---- 7. 참고
    L.append("## 7. 더미데이터 분포를 수정할 때 참고할 사항\n")
    L.append("이번 작업에서는 더미 생성 로직과 모델을 바꾸지 않았다.\n")
    rows = []
    for dataset in (BASE, NEW):
        for mode in new_modes:
            r = st(dataset, "gap_seconds", MODE_LABEL[mode])
            if r is None or not r["n"]:
                continue
            rows.append([DATASET_LABEL[dataset], MODE_LABEL[mode], r["n"], fmt(r["mean"]), fmt(r["median"]), fmt(r["std"]),
                         fmt(r["std"] / r["mean"], 3) if r["mean"] else "–", f"{fmt(r['min'])}~{fmt(r['max'])}"])
    L.append(md_table(["방식", "모드", "간격 수", "평균 (초)", "중앙값 (초)", "표준편차 (초)", "변동계수", "범위 (초)"], rows))
    L.append("")
    L.append(f"- 기존 더미 매크로의 `avg_booking_interval` 은 {fmt(dm.min())}~{fmt(dm.max())}초다. 브라우저와 로그인을 유지하는 매크로의 실측 간격은 위 표의 {LN} 줄이다.")
    L.append("- 브라우저를 매번 다시 띄우는 매크로와 유지하는 매크로는 둘 다 있을 수 있는 유형이다. 더미의 매크로 간격 범위를 정할 때 두 방식의 실측을 하한(연속)과 상한(재실행) 쪽 참고값으로 함께 쓸 수 있다.")
    L.append("- 연속 예매의 하한은 `예매하기 클릭~좌석 선점 + 결제 지연 1.2초 + 화면 이동` 이다 (구성 그래프). 이보다 짧은 간격은 화면을 거치지 않고 API 를 직접 호출하는 경우이며 이번 실측으로 확인되지 않았다.")
    L.append("- 간격의 흩어짐(표준편차·변동계수)은 위 표를 참고한다. 분산을 평균과 따로 뽑지 말고 개별 간격을 생성해 계산하는 편이 실측과 맞는다.")
    L.append("- 분포를 바꾸기 전에 같은 조건으로 2~3회 더 수집해 실행 간 변동을 확인하는 것이 좋다.\n")

    L.append("## 파일\n")
    L.append(md_table(["파일", "내용"], [
        ["`booking_intervals.csv`", "예매 간격 1개당 한 줄 (두 실행 모두): 간격, 구간 분해, 완료→다음 시작, 긴 간격 표시"],
        ["`feature_summary.csv`", "계정별 Feature (BTS 계산식) 와 민감도"],
        ["`bookings.csv`", "예매 1건당 한 줄: 시작·완료 시각, 단계별 소요 시간, 브라우저 실행·로그인 시간, 실패 원인"],
        ["`distribution_stats.csv`, `comparison_vs_baseline.csv`", "요약 통계, 모드별 차이 (`_base` = 기준 실행, `_new` = 이번 실행)"],
        ["`macro_runs.csv`, `macro-results-<계정>.csv`, `macro_stdout.log`", "매크로 실행 기록 원본"],
        ["`raw_events.csv`, `users.csv`, `export_summary.json`, `environment.json`", "BTS 이벤트 내보내기, 실행 환경"],
        ["`figures/interval_by_macro.png`, `interval_distribution.png` (+ `_log`)", "매크로 유형별 예매 간격 분포"],
        ["`figures/account_comparison.png`", "기준 실행 대비 계정별 평균·분산"],
        ["`figures/gap_breakdown.png`, `payment_duration.png`", "구간별 소요 시간 비교"],
        ["`figures/feature_mean_comparison.png`, `feature_variance_comparison.png`", "더미데이터 대비 위치"],
    ]))
    L.append("")
    (run_dir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")



# ---------------------------------------------------------------- 실행
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="runs/ 아래 폴더 이름")
    parser.add_argument("--figures-only", action="store_true", help="기존 CSV 로 그래프만 다시 그린다")
    parser.add_argument("--baseline-tag", default="count", help="#14 실측 데이터 태그 (issue14-real-validation/data/<tag>)")
    parser.add_argument("--baseline-run", default=None,
                        help="비교 기준으로 쓸 이 폴더의 다른 실행 (runs/<run-id>). 생략하면 environment.json 의 baseline_run, 그것도 없으면 #14")
    args = parser.parse_args()

    run_dir = EXP / "runs" / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"{run_dir} 가 없습니다.")
    env_file = run_dir / "environment.json"
    run_env = json.loads(env_file.read_text(encoding="utf-8")) if env_file.exists() else {}
    baseline_run = args.baseline_run or run_env.get("baseline_run")
    base_run_dir = None
    if baseline_run:
        base_run_dir = EXP / "runs" / baseline_run
        if baseline_run == args.run_id or not (base_run_dir / "raw_events.csv").exists():
            raise SystemExit(f"기준 실행 {base_run_dir} 를 쓸 수 없습니다 (없거나 자기 자신).")
        base_env_file = base_run_dir / "environment.json"
        base_env = json.loads(base_env_file.read_text(encoding="utf-8")) if base_env_file.exists() else {}
        kind = lambda e: ("연속 예매", "연속") if e.get("continuous") else ("재실행 방식", "재실행")  # noqa: E731
        (bl, bs), (nl, ns) = kind(base_env), kind(run_env)
        if bl == nl:  # 같은 방식끼리 비교하면 실행 이름으로 구분한다
            bl, bs, nl, ns = f"기준 {baseline_run}", "기준", f"신규 {args.run_id}", "신규"
        configure_datasets(baseline_run, args.run_id, bl, nl, bs, ns)
    dummy_groups, dummy = load_dummy()
    names = ("bookings.csv", "booking_intervals.csv", "feature_summary.csv")

    checks = {}
    if not args.figures_only:
        if not (run_dir / "raw_events.csv").exists():
            raise SystemExit(f"{run_dir / 'raw_events.csv'} 가 없습니다. collect.py 를 먼저 실행하세요.")
        if base_run_dir is not None:
            # 기준 실행: 그 폴더의 수집 원본을 읽기만 한다. 저장돼 있는 Feature 와 재계산이 같은지 확인한다.
            base_runs = pd.read_csv(base_run_dir / "macro_runs.csv", dtype=str, keep_default_na=False)
            base_events = fx14.load_events(base_run_dir / "raw_events.csv")
            base_at = base_events["occurred_at"].max() + timedelta(seconds=60)
            b14, i14, f14, _ = build_dataset(BASE, base_run_dir / "raw_events.csv", base_runs, base_at, sorted(set(base_runs["account"])))
            saved_path = base_run_dir / "feature_summary.csv"
            if saved_path.exists():
                saved = pd.read_csv(saved_path, encoding="utf-8-sig")
                saved = saved[saved["account"].isin(f14["account"]) & (saved["dataset"] != "issue14")].set_index("account")
                again = f14.set_index("account")
                checks[f"기준 실행 Feature 재계산이 runs/{baseline_run}/feature_summary.csv 와 일치"] = bool(len(saved) == len(again) and all(
                    np.allclose(again.loc[saved.index, col].astype(float), saved[col].astype(float), equal_nan=True)
                    for col in ("booking_count", "avg_booking_interval", "booking_interval_variance")))
        else:
            # #14: 기존 파일을 읽기만 한다. 기존 계산 시각을 그대로 써서 Feature 가 재현되는지 확인한다.
            base_dir = ISSUE14 / "data" / args.baseline_tag
            base_at = fx14.parse_utc(json.loads((base_dir / "extraction_summary.json").read_text(encoding="utf-8"))["calculated_at_utc"])
            b14, i14, f14, _ = build_dataset(BASE, base_dir / "events.csv", load_runs_issue14(), base_at)
            old = pd.read_csv(base_dir / "real_features.csv").set_index("external_user_id")
            again = f14.set_index("account")
            checks["#14 Feature 재계산이 기존 real_features.csv 와 일치"] = bool(all(
                np.allclose(again.loc[old.index, col].astype(float), old[col].astype(float), equal_nan=True)
                for col in ("booking_count", "avg_booking_interval", "booking_interval_variance")))

        runs16 = pd.read_csv(run_dir / "macro_runs.csv", dtype=str, keep_default_na=False)
        events16 = fx14.load_events(run_dir / "raw_events.csv")
        accounts = sorted(set(runs16["account"]))
        new_at = events16["occurred_at"].max() + timedelta(seconds=60)
        b16, i16, f16, summary16 = build_dataset(NEW, run_dir / "raw_events.csv", runs16, new_at, accounts)
        checks["이번 실행 Feature 계산 기준 시각(UTC)" if RUN_MODE else "#16 Feature 계산 기준 시각(UTC)"] = new_at.isoformat(timespec="seconds")
        if "seq" in runs16:
            ok_runs = int((runs16["result"] == "OK").sum())
            checks["성공한 예매 수 = BOOKING 수" if RUN_MODE else "#16 성공한 실행 수 = BOOKING 수"] = f"{ok_runs} = {int(b16['completed'].astype(bool).sum())}"
        env_path = run_dir / "environment.json"
        if env_path.exists():
            import subprocess
            admin = json.loads(env_path.read_text(encoding="utf-8"))["repos"]["ticket-fds-admin"]
            done = subprocess.run(["git", "-C", admin["path"], "diff", "--quiet", "46f8c67", admin["commit"], "--",
                                   "backend/src/main/java/com/bts/fairticket/risk/service/FeatureAggregationService.java"], capture_output=True)
            checks["feature_service_unchanged"] = {0: "변경 없음", 1: "변경 있음 — 계산식 확인 필요"}.get(done.returncode, "확인 실패")

        bookings = pd.concat([b14, b16], ignore_index=True)
        intervals = pd.concat([i14, i16], ignore_index=True)
        feats = feature_summary(pd.concat([f14, f16], ignore_index=True), intervals, bookings)
        for name, frame in zip(names, (bookings, intervals, feats)):
            frame.to_csv(run_dir / name, index=False, encoding="utf-8-sig")
    elif not all((run_dir / n).exists() for n in names):
        raise SystemExit("--figures-only 에 필요한 CSV 가 없습니다. 먼저 --figures-only 없이 실행하세요.")

    # 그래프·보고서는 항상 저장된 CSV 에서 다시 읽어 만든다 (CSV 만으로 재현 가능하도록)
    bookings = pd.read_csv(run_dir / names[0], dtype={"result": str, "error": str}, keep_default_na=True, encoding="utf-8-sig")
    bookings["result"] = bookings["result"].fillna("")
    bookings["failed_or_timeout"] = bookings["failed_or_timeout"].fillna(False).astype(bool)
    intervals = pd.read_csv(run_dir / names[1], encoding="utf-8-sig")
    feats = pd.read_csv(run_dir / names[2], encoding="utf-8-sig")
    feats["payment_identifiers_used"] = feats["payment_identifiers_used"].fillna("")

    figures = make_figures(run_dir, bookings, intervals, feats, dummy_groups)
    print("그래프:", ", ".join(figures))
    if args.figures_only:
        return
    stats = distribution_stats(bookings, intervals, feats, dummy_groups, dummy)
    comp = comparison_table(bookings, intervals)
    stats.to_csv(run_dir / "distribution_stats.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(run_dir / ("comparison_vs_baseline.csv" if RUN_MODE else "comparison_14_vs_16.csv"), index=False, encoding="utf-8-sig")
    if RUN_MODE:
        write_report_runs(run_dir, base_run_dir, bookings, intervals, feats, stats, comp, dummy_groups, checks)
    else:
        write_report(run_dir, bookings, intervals, feats, stats, comp, dummy_groups, checks)
    pd.set_option("display.width", 220)
    print(comp[comp["metric"] == "gap_seconds"][["mode", "variant", "n_base", "n_new", "mean_base", "mean_new", "mean_diff", "median_base", "median_new"]].round(3).to_string(index=False))
    print(f"보고서: {run_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
