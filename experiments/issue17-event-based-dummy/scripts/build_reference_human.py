"""#14 실측의 'human' 프로필 매크로 1계정에서 예매 시도 1건의 단계별 소요 시간을 뽑아 참조 파일로 저장한다 (v3 용).

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\build_reference_human.py

입력 (읽기만 함): ../issue14-real-validation/data/count/events.csv  — git 에 올리지 않는 실측 원본
출력: reference/macro_human_attempt_cycles.csv, reference/reference_human_summary.json

- 기존 참조 파일(reference/macro_attempt_cycles.csv, Fast/Jitter)은 건드리지 않는다.
- #14 는 시도마다 매크로 프로세스를 새로 띄우고 로그인했다. 그래서 '다음 시도까지의 간격'에는 실행·로그인 시간이 들어 있어
  연속 실행의 간격으로 쓸 수 없다. 이 파일에는 시도 안의 단계 시간만 넣고, 간격은 참고용 컬럼(gap_with_launch_ms)으로만 남긴다.
- #14 측정은 #16 의 사이트 수정 전이고 계정이 1개다. 근거의 강도는 'weak_measured' 다.
"""
import json
import sys
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parent.parent
SOURCE = EXP.parent / "issue14-real-validation" / "data" / "count" / "events.csv"
GROUP = "macro_human"
STAGES = ["STARTED", "QUEUE", "SEAT_SELECTION", "SEAT_HOLD", "BOOKING", "COMPLETED"]


def main():
    events = pd.read_csv(SOURCE, dtype=str, keep_default_na=False)
    events = events[events["group"] == GROUP].copy()
    events["t"] = pd.to_datetime(events["occurred_at_utc"])
    meta = events["metadata"].map(json.loads)
    events["attempt_id"] = meta.map(lambda m: m.get("attemptId"))
    events["stage"] = meta.map(lambda m: m.get("stage"))
    events.loc[events["event_type"] == "BOOKING", "stage"] = "BOOKING"
    wide = (events.pivot_table(index=["external_user_id", "attempt_id"], columns="stage", values="t", aggfunc="min")
            .reset_index().sort_values(["external_user_id", "STARTED"]).reset_index(drop=True))
    if wide[STAGES].isna().any().any():
        sys.exit("단계 이벤트가 빠진 시도가 있습니다. 값을 지어내지 않고 중단합니다.")
    rows = pd.DataFrame({"dataset": "issue14-count", "account": wide["external_user_id"], "mode": "human"})
    rows["attempt_index"] = wide.groupby("external_user_id").cumcount() + 1
    for name, start, end in [("queue_ms", "STARTED", "QUEUE"), ("pre_seat_ms", "QUEUE", "SEAT_SELECTION"), ("seat_ms", "SEAT_SELECTION", "SEAT_HOLD"),
                             ("hold_to_booking_ms", "SEAT_HOLD", "BOOKING"), ("completed_ms", "BOOKING", "COMPLETED")]:
        rows[name] = ((wide[end] - wide[start]).dt.total_seconds() * 1000).round().astype("Int64")
    gap = wide.groupby("external_user_id")["STARTED"].diff().shift(-1)
    rows["gap_with_launch_ms"] = (gap.dt.total_seconds() * 1000).round().astype("Int64")
    stage_cols = ["queue_ms", "pre_seat_ms", "seat_ms", "hold_to_booking_ms", "completed_ms"]
    if (rows[stage_cols] <= 0).any().any():
        sys.exit("단계 소요 시간이 0 이하인 시도가 있습니다.")

    out = EXP / "reference"
    rows.to_csv(out / "macro_human_attempt_cycles.csv", index=False, encoding="utf-8")
    summary = {
        "source": "issue14-real-validation/data/count/events.csv (BTS behavior_event 내보내기), group = macro_human",
        "accounts": sorted(rows["account"].unique()), "attempts": int(len(rows)),
        "median_ms": rows[stage_cols].median().astype(float).round(0).to_dict(),
        "cycle_ms_min_median_max": [float(v) for v in rows[stage_cols].sum(axis=1).quantile([0, 0.5, 1]).round(0)],
        "gap_with_launch_ms_median": float(rows["gap_with_launch_ms"].median()),
        "notes": ["시도마다 프로세스를 새로 띄우고 로그인해서 간격에는 실행·로그인 시간이 들어 있다. 연속 실행의 간격으로 쓰지 않는다.",
                  "#16 의 사이트 수정 전에 측정했다. 계정 1개, 시도 20건.",
                  "당시 데모는 PAYMENT 단계 이벤트를 보내지 않았다. 생성기도 이 시도에서 나온 예매에는 PAYMENT 이벤트를 만들지 않는다."],
    }
    (out / "reference_human_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
