"""매크로 연속 시도 간격에 브라우저 실행 시간이 얼마나 들어 있는지 분해한다 (실험용, 기록 목적).

macro.js 는 실행마다 node 와 브라우저를 새로 띄운다. 그래서 같은 계정의 "시도 시작 간격"에는
매크로의 조작 속도뿐 아니라 프로세스·브라우저 기동 시간이 섞인다. 이 스크립트는 그 비중을 계산해 남긴다.

    간격(k→k+1) = [k번째: 예매하기 클릭 → 결제 완료]            = ms_booking(k)
                + [브라우저 종료 + node·브라우저 재기동]         = launch_overhead  (나머지로 계산)
                + [k+1번째: 로그인 → 공연 진입 → 예매하기 클릭]  = ms_total(k+1) - ms_booking(k+1)

실행 (실험 폴더에서):
    .venv/Scripts/python scripts/macro_gap_breakdown.py --tag count --results data/count/macro-results-fast-02.csv

입력 : data/<tag>/events.csv, macro.js 가 쓴 결과 CSV (--results, 한 계정 한 파일)
출력 : data/<tag>/macro_gap_breakdown__<계정>.csv  (이미 있으면 덮어쓰지 않음)

실패한 실행(result != OK)이 끼어 있으면 그 앞뒤 간격은 분해하지 않고 비워 둔다.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results", required=True, help="macro.js --out 으로 저장한 CSV 경로")
    parser.add_argument("--no-save", action="store_true", help="파일로 저장하지 않고 화면에만 출력")
    args = parser.parse_args()

    runs = pd.read_csv(args.results)
    accounts = runs["user"].unique()
    if len(accounts) != 1:
        raise SystemExit(f"결과 파일에 계정이 {len(accounts)}개 있습니다. 계정별 파일을 지정하세요: {list(accounts)}")
    account, mode = accounts[0], "/".join(runs["mode"].unique())

    events = pd.read_csv(EXP / "data" / args.tag / "events.csv", dtype=str, keep_default_na=False)
    own = events[(events["external_user_id"] == account) & (events["event_type"] == "BOOKING_ATTEMPT")].copy()
    own["attempt_id"] = own["metadata"].map(lambda s: json.loads(s).get("attemptId"))
    starts = pd.to_datetime(own.groupby("attempt_id")["occurred_at_utc"].min().sort_values()).reset_index(drop=True)

    if len(starts) != len(runs):
        print(f"주의: BTS 시도 {len(starts)}건, 매크로 실행 {len(runs)}건으로 수가 다릅니다. 앞에서부터 순서대로 맞춥니다.")
    n = min(len(starts), len(runs))
    rows = []
    for k in range(n - 1):
        gap = (starts[k + 1] - starts[k]).total_seconds()
        this_run, next_run = runs.iloc[k], runs.iloc[k + 1]
        row = {"account": account, "mode": mode, "interval_index": k + 1, "gap_seconds": round(gap, 3),
               "flow_after_click_s": None, "next_before_click_s": None, "launch_overhead_s": None,
               "launch_share": None, "run_results": f"{this_run['result']}→{next_run['result']}"}
        if this_run["result"] == "OK" and next_run["result"] == "OK":
            after = float(this_run["ms_booking"]) / 1000
            before = (float(next_run["ms_total"]) - float(next_run["ms_booking"])) / 1000
            overhead = gap - after - before
            row.update(flow_after_click_s=round(after, 3), next_before_click_s=round(before, 3),
                       launch_overhead_s=round(overhead, 3), launch_share=round(overhead / gap, 3) if gap > 0 else None)
        rows.append(row)
    table = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    print(table.to_string(index=False))
    valid = table.dropna(subset=["launch_overhead_s"])
    if len(valid):
        print(f"\n{account} ({mode}) 간격 {len(valid)}개 요약: 간격 평균 {valid['gap_seconds'].mean():.2f}s, "
              f"기동 시간 평균 {valid['launch_overhead_s'].mean():.2f}s (간격의 {valid['launch_share'].mean():.0%}), "
              f"기동 시간 표준편차 {valid['launch_overhead_s'].std(ddof=1) if len(valid) > 1 else float('nan'):.2f}s")
    if not args.no_save:
        out = EXP / "data" / args.tag / f"macro_gap_breakdown__{account}.csv"
        if out.exists():
            raise SystemExit(f"{out} 가 이미 있습니다. 덮어쓰지 않습니다.")
        table.to_csv(out, index=False)
        print(f"저장: {out}")


if __name__ == "__main__":
    main()
