"""#16 실측 매크로 이벤트에서 '예매 시도 1건의 단계별 소요 시간 + 다음 시도까지의 간격'을 뽑아 참조 파일로 저장한다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\build_reference_timings.py

입력 (읽기만 함): ../issue16-booking-interval/runs/<run>/raw_events.csv  — git 에 올리지 않는 실측 원본
출력: reference/macro_attempt_cycles.csv, reference/reference_summary.json

이벤트 생성기는 이 참조 파일만 읽는다. 실측 원본이 없는 PC 에서는 이 스크립트를 건너뛰고 저장된 참조 파일을 그대로 쓴다.
계정은 #16 v2 가 쓴 것과 같다 (간격 5개 이상인 Fast 3계정, Jitter 3계정).
간격은 #16 의 `dummy_adjusted/v2/real_interval_pool.csv` 와 같은 값인지 확인한다.
"""
import json
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parent.parent
ISSUE16 = EXP.parent / "issue16-booking-interval"
RUNS = ["20261122-continuous", "20261123-remeasure"]
MODES = {"macro_fast": "fast", "macro_jitter": "jitter"}
MIN_INTERVALS = 5
STAGES = ["STARTED", "QUEUE", "SEAT_SELECTION", "SEAT_HOLD", "PAYMENT", "BOOKING", "COMPLETED"]


def attempts_of(run):
    events = pd.read_csv(ISSUE16 / "runs" / run / "raw_events.csv", dtype=str, keep_default_na=False)
    events = events[events["group"].isin(MODES)].copy()
    events["t"] = pd.to_datetime(events["occurred_at_utc"])
    meta = events["metadata"].map(json.loads)
    events["attempt_id"] = meta.map(lambda m: m.get("attemptId"))
    events["stage"] = meta.map(lambda m: m.get("stage"))
    events.loc[events["event_type"] == "BOOKING", "stage"] = "BOOKING"
    wide = (events.pivot_table(index=["group", "external_user_id", "attempt_id"], columns="stage", values="t", aggfunc="min")
            .reset_index().sort_values(["external_user_id", "STARTED"]).reset_index(drop=True))
    # 20261122 실행 당시의 데모는 PAYMENT 단계 이벤트를 보내지 않았다. 그 계정은 PAYMENT 를 빈 칸으로 두고 (지어내지 않음)
    # 생성기도 그 계정에서 나온 시도에는 PAYMENT 이벤트를 만들지 않는다.
    if "PAYMENT" not in wide.columns:
        wide["PAYMENT"] = pd.NaT
    required = [s for s in STAGES if s != "PAYMENT"]
    if wide[required].isna().any().any() or wide.groupby("external_user_id")["PAYMENT"].apply(lambda v: v.isna().nunique() > 1).any():
        sys.exit(f"{run}: 단계 이벤트가 빠진 시도가 있습니다. 값을 지어내지 않고 중단합니다.")
    rows = pd.DataFrame({"dataset": run, "account": wide["external_user_id"], "mode": wide["group"].map(MODES)})
    rows["attempt_index"] = wide.groupby("external_user_id").cumcount() + 1
    for name, start, end in [("queue_ms", "STARTED", "QUEUE"), ("pre_seat_ms", "QUEUE", "SEAT_SELECTION"),
                             ("seat_ms", "SEAT_SELECTION", "SEAT_HOLD"), ("hold_to_payment_ms", "SEAT_HOLD", "PAYMENT"),
                             ("hold_to_booking_ms", "SEAT_HOLD", "BOOKING"), ("completed_ms", "BOOKING", "COMPLETED")]:
        rows[name] = ((wide[end] - wide[start]).dt.total_seconds() * 1000).round().astype("Int64")
    gap = wide.groupby("external_user_id")["STARTED"].diff().shift(-1)
    gap[wide["external_user_id"] != wide["external_user_id"].shift(-1)] = pd.NaT
    rows["gap_to_next_ms"] = (gap.dt.total_seconds() * 1000).round().astype("Int64")
    return rows


def main():
    cycles = pd.concat([attempts_of(run) for run in RUNS], ignore_index=True)
    n_gaps = cycles.groupby(["dataset", "account"])["gap_to_next_ms"].transform("count")
    cycles = cycles[n_gaps >= MIN_INTERVALS].reset_index(drop=True)

    all_cols = [c for c in cycles.columns if c.endswith("_ms") and c != "gap_to_next_ms"]
    stage_cols = [c for c in all_cols if c != "hold_to_payment_ms"]  # 합하면 시도 시작~완료
    if (cycles[all_cols] <= 0).any().any() or (cycles["hold_to_payment_ms"] >= cycles["hold_to_booking_ms"]).any():
        sys.exit("단계 소요 시간이 0 이하이거나 순서가 맞지 않는 시도가 있습니다.")
    with_gap = cycles.dropna(subset=["gap_to_next_ms"])
    if (with_gap["gap_to_next_ms"] < with_gap[stage_cols].sum(axis=1)).any():
        sys.exit("다음 시도가 이전 시도 완료 전에 시작한 기록이 있습니다. 순차 예매 가정이 깨집니다.")

    # #16 v2 가 쓴 간격과 같은 값인지 대조
    pool = pd.read_csv(ISSUE16 / "dummy_adjusted" / "v2" / "real_interval_pool.csv", encoding="utf-8-sig")
    pool = pool[pool["account"].isin(cycles["account"])].sort_values(["account", "interval_index"])
    mine = with_gap.sort_values(["account", "attempt_index"])
    same = (len(pool) == len(mine)
            and (pool["account"].to_numpy() == mine["account"].to_numpy()).all()
            and (abs(pool["gap_seconds"].to_numpy() * 1000 - mine["gap_to_next_ms"].to_numpy(dtype=float)) < 0.5).all())
    if not same:
        sys.exit("간격이 #16 real_interval_pool.csv 와 다릅니다.")

    out = EXP / "reference"
    out.mkdir(exist_ok=True)
    cycles.to_csv(out / "macro_attempt_cycles.csv", index=False, encoding="utf-8")
    summary = {
        "source_runs": RUNS, "source": "issue16-booking-interval/runs/<run>/raw_events.csv (BTS behavior_event 내보내기)",
        "accounts": cycles.groupby("mode")["account"].unique().map(list).to_dict(),
        "attempts": int(len(cycles)), "intervals": int(len(with_gap)),
        "intervals_match_issue16_v2_pool": bool(same),
        "median_ms_by_mode": cycles.groupby("mode")[all_cols + ["gap_to_next_ms"]].median().astype(float).round(0).to_dict("index"),
        "accounts_without_payment_stage_event": sorted(cycles.loc[cycles["hold_to_payment_ms"].isna(), "account"].unique()),
        "environment_note": "격리 E2E 환경(대기열 없음, 좌석 경쟁 없음, 시연용 결제 지연 1.2초). 모든 시도가 예매 완료까지 갔다.",
    }
    (out / "reference_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
