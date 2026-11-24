"""v4 생성기가 쓰는 참조 파일 두 개를 실측에서 만든다. 실측 원본은 읽기만 한다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\build_reference_v4.py

1) reference/macro_human_continuous_cycles.csv
   #16 연속 실행(20261122-continuous)의 human 프로필 매크로 1계정 20건. 시도 1건의 단계 시간과 다음 시도까지의 간격.
   v3 는 #14 의 재실행 측정값에 추정 대기를 더해 썼다 (평균 17.3초). 연속 실행 실측은 12.8초다 (REPORT_INTERVAL_VALIDATION.md).
   컬럼은 reference/macro_attempt_cycles.csv 와 같다.

2) reference/measured_person_fast_end.json
   2026-11-24 에 실측한 정상 사용자 1명(8계정, 완료 40건)의 단계별 중앙값. 집계값만 저장하고 원본은 저장하지 않는다.
   v4 에서 이 값은 정상 사용자 속도 범위의 '빠른 쪽 끝점'으로만 쓴다. 분포의 중심이나 표본으로 쓰지 않는다.

실측 원본이 없는 PC 에서는 이 스크립트를 건너뛰고 저장된 참조 파일을 그대로 쓴다.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_reference_timings as base  # noqa: E402  (#16 실측에서 시도 표를 만드는 함수를 그대로 쓴다)

EXP = HERE.parent
RUN = "20261122-continuous"
PERSON_ATTEMPTS = EXP / "measurement" / "data" / "_analysis" / "20261124-all" / "attempts.csv"
OUT = EXP / "reference"


def human_cycles():
    base.MODES = {"macro_human": "human"}
    cycles = base.attempts_of(RUN)
    stage_cols = ["queue_ms", "pre_seat_ms", "seat_ms", "hold_to_booking_ms", "completed_ms"]
    if (cycles[stage_cols] <= 0).any().any():
        sys.exit("단계 소요 시간이 0 이하인 시도가 있습니다.")
    with_gap = cycles.dropna(subset=["gap_to_next_ms"])
    if (with_gap["gap_to_next_ms"] < with_gap[stage_cols].sum(axis=1)).any():
        sys.exit("다음 시도가 이전 시도 완료 전에 시작한 기록이 있습니다.")
    return cycles, with_gap


def person_fast_end():
    attempts = pd.read_csv(PERSON_ATTEMPTS)
    people = sorted(attempts["participant"].unique())
    if len(people) != 1:
        sys.exit(f"참가자가 1명이 아닙니다: {people}. 빠른 끝점은 한 사람의 값으로 정의했습니다.")
    done = attempts[attempts["outcome"] == "completed"]
    after = attempts.dropna(subset=["previous_end_to_start_s"])
    by_previous = after.groupby("previous_outcome")["previous_end_to_start_s"].agg(["count", "median", "min", "max"]).round(3)
    return {
        "source": "measurement/data/_analysis/20261124-all/attempts.csv (git 제외). 집계값만 저장한다",
        "people": 1, "accounts": int(attempts["external_user_id"].nunique()), "attempts": int(len(attempts)), "completed": int(len(done)),
        "use": "정상 사용자 속도 범위의 빠른 쪽 끝점. 분포의 중심이나 표본으로 쓰지 않는다. 사용자 사이의 차이는 이 값으로 추정하지 않는다",
        "stage_median_s": {"queue": round(float(done["queue_s"].median()), 3), "pre_seat": round(float(done["to_seat_page_s"].median()), 3),
                           "seat": round(float(done["seat_to_hold_s"].median()), 3), "hold_to_payment": round(float(done["hold_to_payment_page_s"].median()), 3),
                           "payment": round(float(done["payment_s"].median()), 3)},
        "start_to_booking_median_s": round(float(done["start_to_booking_s"].median()), 3),
        "previous_end_to_next_start_s": {name: {"n": int(row["count"]), "median": float(row["median"]), "min": float(row["min"]), "max": float(row["max"])}
                                         for name, row in by_previous.iterrows()},
        "activity_gap_median_s": round(float(attempts.loc[attempts["in_session_gap_only"], "gap_s"].median()), 3),
        "environment": "격리 E2E (대기열 대기 0, 좌석 경쟁 없음, 결제는 저장된 수단 클릭 + 시연용 지연 1.2초). 사이트에 익숙한 사람",
    }


def main():
    OUT.mkdir(exist_ok=True)
    cycles, with_gap = human_cycles()
    cycles.to_csv(OUT / "macro_human_continuous_cycles.csv", index=False, encoding="utf-8")
    person = person_fast_end()
    (OUT / "measured_person_fast_end.json").write_text(json.dumps(person, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"human_profile": {"run": RUN, "accounts": sorted(cycles["account"].unique()), "attempts": int(len(cycles)), "intervals": int(len(with_gap)),
                                        "gap_s_mean_min_max": [round(float(v) / 1000, 2) for v in (with_gap["gap_to_next_ms"].mean(), with_gap["gap_to_next_ms"].min(),
                                                                                                   with_gap["gap_to_next_ms"].max())],
                                        "payment_stage_event": bool(cycles["hold_to_payment_ms"].notna().any())},
                      "person": person}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
