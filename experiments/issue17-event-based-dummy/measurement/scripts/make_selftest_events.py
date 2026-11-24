"""분석 스크립트 점검용 가짜 이벤트를 만든다. 실측이 아니다. 결과에는 의미가 없다.

    & $py measurement\\scripts\\make_selftest_events.py
    & $py measurement\\scripts\\analyze.py --run-id _selftest

v3 합성 이벤트(data/v3/events.csv, 읽기만 함)에서 계정 몇 개를 골라 참가자 계정 ID 로 이름만 바꾼다.
완료, 좌석 이탈, 결제 이탈, 선점 거절, 선점 만료, 취소가 모두 들어 있어 analyze.py 의 모든 경로를 한 번씩 지나간다.
출력: measurement/data/_selftest/raw_events.csv  (git 제외)
"""
import csv
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MEASUREMENT = HERE.parent
EXP = MEASUREMENT.parent
sys.path.insert(0, str(HERE))
from collect import EVENT_FIELDS, parse_account  # noqa: E402

OUT = MEASUREMENT / "data" / "_selftest"
WANTED = {"u01-a-01": ["abandon_seat"], "u01-b-01": ["complete"], "u01-c-01": ["hold_rejected", "abandon_payment"], "u02-a-01": ["hold_expired_wait"],
          "u02-b-01": ["complete"], "u02-c-01": ["abandon_seat", "abandon_payment"], "helper-01": ["complete"]}


def main():
    attempts = pd.read_csv(EXP / "data" / "v3" / "attempts.csv")
    accounts = pd.read_csv(EXP / "data" / "v3" / "accounts.csv", keep_default_na=False)
    normal = set(accounts.loc[accounts["scenario"].isin(["normal", "fast_normal", "shared_ip", "frequent_cancel"]), "user_id"])
    per_user = attempts[attempts["user_id"].isin(normal)].groupby("user_id").agg(n=("attempt_id", "size"), outcomes=("outcome", set),
                                                                              rejected=("hold_rejections", "sum"), cancelled=("cancel_utc", "count"))
    chosen, used = {}, set()
    for new_id, needed in WANTED.items():
        for user, row in per_user.iterrows():
            ok = user not in used and 4 <= row["n"] <= 9 and set(needed) <= row["outcomes"]
            if new_id == "u01-c-01":
                ok = ok and row["rejected"] > 0
            if new_id == "u01-b-01":
                ok = ok and row["cancelled"] > 0
            if ok:
                chosen[user] = new_id
                used.add(user)
                break
        else:
            sys.exit(f"{new_id} 에 맞는 합성 계정을 찾지 못했습니다.")
    events = pd.read_csv(EXP / "data" / "v3" / "events.csv", dtype=str, keep_default_na=False)
    events = events[events["user_id"].isin(chosen)]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "raw_events.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for row in events.to_dict("records"):
            new_id = chosen[row["user_id"]]
            group, participant, experiment, round_no = parse_account(new_id)
            writer.writerow({**{k: row[k] for k in EVENT_FIELDS if k in row}, "user_id": "bts-" + new_id, "external_user_id": new_id, "group": group,
                             "participant": participant, "experiment": experiment, "round": round_no})
    (OUT / "export_summary.json").write_text(json.dumps({"selftest": True, "note": "v3 합성 이벤트의 이름만 바꾼 것. 실측이 아니다.",
                                                         "accounts": chosen}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"가짜 이벤트 {len(events)}건 저장: {OUT}  (계정 {sorted(chosen.values())})")


if __name__ == "__main__":
    main()
