"""한 번 내보낸 이벤트에서 "누적 N회 시점"의 스냅샷 태그를 만든다 (실험용).

매크로를 중간에 멈추지 않고 끝까지 돌린 뒤, 특정 계정의 N번째 예매 시도가 끝난 시점까지의
이벤트만 잘라 별도 태그 폴더에 저장한다. 중간에 멈춰서 내보내면 그 대기 시간이 시도 간격에
섞이므로, 사후에 잘라내는 방식을 쓴다.

실행 (실험 폴더에서):
    .venv/Scripts/python scripts/make_snapshots.py --source-tag count --account macro-fast-02 --counts 3 10 20

출력: data/<source-tag>__<account>__n03/ 등
        events.csv          기준 시각 이전 이벤트 (모든 계정 포함: 카드·IP 공유 수도 그 시점 기준이 된다)
        users.csv           원본 복사
        snapshot_info.json  기준 계정, N, 기준 시각(cutoff)

기준 시각 = 그 계정의 1~N번째 시도에 속한 이벤트(단계 이벤트, 연결된 BOOKING/CANCEL) 중 마지막 시각.
N은 실행 횟수가 아니라 BTS에 기록된 시도(attemptId) 수로 센다 (실패한 실행도 시도로 남을 수 있다).

원본 태그는 읽기만 하고, 이미 있는 스냅샷 폴더는 덮어쓰지 않는다.
이어서 실행할 명령(Feature 추출 등)을 화면에 출력한다.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--account", required=True, help="기준 계정의 데모 로그인 ID (예: macro-fast-02)")
    parser.add_argument("--counts", type=int, nargs="+", required=True, help="누적 시도 수 목록 (예: 3 10 20)")
    args = parser.parse_args()

    source = EXP / "data" / args.source_tag
    events = pd.read_csv(source / "events.csv", dtype=str, keep_default_na=False)
    events["attempt_id"] = events["metadata"].map(lambda s: (json.loads(s) if s else {}).get("attemptId"))

    own = events[events["external_user_id"] == args.account]
    starts = (own[own["event_type"] == "BOOKING_ATTEMPT"].dropna(subset=["attempt_id"])
              .groupby("attempt_id")["occurred_at_utc"].min().sort_values())
    if starts.empty:
        raise SystemExit(f"{args.account} 의 예매 시도가 {source / 'events.csv'} 에 없습니다.")
    print(f"{args.account}: 기록된 시도 {len(starts)}건")

    commands = []
    for n in sorted(set(args.counts)):
        if n > len(starts):
            print(f"  n={n}: 시도가 {len(starts)}건뿐이라 건너뜀")
            continue
        attempt_ids = list(starts.index[:n])
        cutoff = own[own["attempt_id"].isin(attempt_ids)]["occurred_at_utc"].max()
        if n < len(starts) and cutoff >= starts.iloc[n]:
            # 다음 시도가 이전 시도 종료 전에 시작된 경우(동시 실행 등): 다음 시도 시작 직전으로 자른다
            cutoff = (pd.Timestamp(starts.iloc[n]) - pd.Timedelta(milliseconds=1)).isoformat(timespec="microseconds")
        tag = f"{args.source_tag}__{args.account}__n{n:02d}"
        out = EXP / "data" / tag
        if out.exists():
            print(f"  n={n}: {out} 가 이미 있어 건너뜀 (덮어쓰지 않음)")
            continue
        out.mkdir(parents=True)
        subset = events[events["occurred_at_utc"] <= cutoff].drop(columns=["attempt_id"])
        subset.to_csv(out / "events.csv", index=False)
        if (source / "users.csv").exists():
            shutil.copy2(source / "users.csv", out / "users.csv")
        info = {"source_tag": args.source_tag, "account": args.account, "n_attempts": n,
                "cutoff_utc": cutoff, "events_in_snapshot": int(len(subset)), "events_in_source": int(len(events)),
                "accounts_with_bookings_at_cutoff": int(subset.loc[subset["event_type"] == "BOOKING", "user_id"].nunique()),
                "note": "calculated_at 을 cutoff_utc 로 지정해 Feature 를 계산한다"}
        (out / "snapshot_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  n={n}: 기준 시각 {cutoff} / 이벤트 {len(subset)}건 → data/{tag}")
        commands += [f".venv\\Scripts\\python scripts\\extract_features.py --tag {tag} --calculated-at {cutoff}",
                     f".venv\\Scripts\\python scripts\\score_model.py --tag {tag}",
                     f".venv\\Scripts\\python scripts\\compare_with_dummy.py --tag {tag}"]

    if commands:
        print("\n이어서 실행할 명령:")
        print("\n".join(commands))


if __name__ == "__main__":
    main()
