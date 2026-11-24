"""v4 의 민감도 조건을 생성 seed 별로 만들고 Feature 를 추출해 계정 1행 표로 저장한다 (이벤트는 저장하지 않는다).

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\sweep_v4.py                 # 기준 + 한 번에 한 축 (D1·D2·D3 각각)
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\sweep_v4.py --combine       # 변화가 큰 축 두 개의 조합 (위 실행 뒤)

조건: config/scenarios_v4.json 의 baseline 과 sensitivity_axes. 평가 전에 고정한 값이고 여기서 바꾸지 않는다.
  S-A(정상 사용자 사이의 속도 분포)는 축이 아니다. D1·D2·D3 각각을 기준으로 삼아 같은 조건을 세 벌 만든다.
생성 seed: 0~9. Feature 는 기존 BTS 추출기(#14)로 뽑고, 간격 2개의 다른 정의(extract_v4.interval_definitions)를 같이 계산한다.

조합할 축을 고르는 기준 (모델 성능을 보지 않는다. 이 스크립트를 처음 돌리기 전에 정했다)
  축마다 '기준 조건 대비 간격 평균의 단일 Feature AUC 변화'의 절댓값 중 가장 큰 값
  (세 경계 d30 / concert / activity, 세 가정 D1~D3, 그 축의 모든 값에 대해. 생성 seed 10개의 평균으로 비교)
  이 값이 큰 순서로 두 축을 고르고, 두 축의 모든 값 조합을 D 마다 만든다.

출력: data/v4/_runs/<D>__<조건>__seed<k>.csv.gz   계정 1행 (git 제외)
      output/v4/sensitivity/conditions.json, runs.csv (조건·seed 별 생성 요약), (--combine) combination_choice.json
"""
import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))

RUNS = EXP / "data" / "v4" / "_runs"
OUT = EXP / "output" / "v4" / "sensitivity"
SEEDS = list(range(10))
TRAITS = ["ip_group_size", "card_group_size", "tool_profile", "n_attempts", "n_sessions", "speed_s", "goal_bookings", "pause_ms", "account_level_s"]


def value_name(value):
    return "+".join(value) + "off" if isinstance(value, list) else str(value).lower() if isinstance(value, bool) else str(value)


def conditions(v4):
    """[(조건 이름, 축, 덮어쓸 params)]. 첫 항목이 기준."""
    out = [("base", "", {})]
    for axis, spec in v4["sensitivity_axes"].items():
        if axis.startswith("_"):
            continue
        for value in spec["values"]:
            out.append((f"{axis}={value_name(value)}", axis, {spec["param"]: value}))
    return out


def run_path(dist, name, seed):
    return RUNS / f"{dist}__{name.replace('=', '_')}__seed{seed}.csv.gz"


def one(task):
    dist, name, overrides, seed = task
    path = run_path(dist, name, seed)
    if path.exists():
        return None
    from extract_v4 import build_all
    from generate_events_v4 import generate, load_v4
    configs = load_v4()
    events, accounts, attempts, params = generate(dist, overrides, seed, configs)
    dataset, full, definitions, _, _ = build_all(events, accounts, configs[0])
    frame = dataset.merge(definitions, on="user_id", validate="one_to_one").merge(
        accounts[["user_id"] + [c for c in TRAITS if c in accounts.columns]], on="user_id", validate="one_to_one")
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", compression="gzip")
    normal = accounts[accounts["is_anomaly"] == 0]
    return {"dist": dist, "condition": name, "seed": seed, "events": int(len(events)), "attempts": int(len(attempts)),
            "normal_attempts_per_account": float(normal["n_attempts"].mean()), "normal_one_attempt_share": float((normal["n_attempts"] == 1).mean()),
            "normal_speed_median": float(normal["speed_s"].median()), "bookings": int((attempts["booking_id"] != "").sum()),
            "hold_rejections": int(attempts["hold_rejections"].sum()), "normal_attempts_on_other_concert": int((attempts["concert"] != attempts["open_concert"]).sum())}


def choose_axes(v4):
    """조합할 두 축. 간격 평균의 단일 Feature AUC 변화가 큰 순서."""
    from sklearn.metrics import roc_auc_score
    from extract_v4 import BOUNDARIES, definition_columns
    from generate_events_v4 import DISTS
    rows = []
    for dist in DISTS:
        for name, axis, _ in conditions(v4):
            for seed in SEEDS:
                frame = pd.read_csv(run_path(dist, name, seed))
                for boundary in BOUNDARIES:
                    part = frame.dropna(subset=[definition_columns(boundary, "mean_var")[0]])
                    auc = roc_auc_score(part["is_anomaly"], -part[definition_columns(boundary, "mean_var")[0]])
                    rows.append({"dist": dist, "condition": name, "axis": axis, "boundary": boundary, "seed": seed, "auc": auc})
    table = pd.DataFrame(rows).groupby(["dist", "condition", "axis", "boundary"], as_index=False)["auc"].mean()
    base = table[table["condition"] == "base"].set_index(["dist", "boundary"])["auc"]
    table["delta"] = table["auc"] - base.reindex(pd.MultiIndex.from_frame(table[["dist", "boundary"]])).to_numpy()
    ranking = table[table["axis"] != ""].assign(size=lambda d: d["delta"].abs()).groupby("axis")["size"].max().sort_values(ascending=False)
    return list(ranking.index[:2]), ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    from generate_events import sha256
    from generate_events_v4 import CONFIG_PATH, DISTS, load_v4
    _, v4 = load_v4()
    RUNS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    todo = conditions(v4)
    record = {"config_sha256": sha256(CONFIG_PATH), "generation_seeds": SEEDS, "speed_assumptions": list(DISTS),
              "one_at_a_time": [{"condition": name, "axis": axis, "overrides": overrides} for name, axis, overrides in todo]}
    if args.combine:
        axes, ranking = choose_axes(v4)
        specs = [v4["sensitivity_axes"][axis] for axis in axes]
        todo = [(f"{axes[0]}={value_name(a)}&{axes[1]}={value_name(b)}", "+".join(axes), {specs[0]["param"]: a, specs[1]["param"]: b})
                for a, b in itertools.product(specs[0]["values"], specs[1]["values"])]
        choice = {"criterion": "축마다 max |간격 평균의 단일 Feature AUC − 기준| (경계 3 × 가정 3 × 그 축의 값, seed 10개 평균). 모델 성능은 보지 않는다",
                  "ranking": {axis: round(float(v), 4) for axis, v in ranking.items()}, "chosen": axes,
                  "combinations": [{"condition": name, "overrides": overrides} for name, _, overrides in todo]}
        (OUT / "combination_choice.json").write_text(json.dumps(choice, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(choice, ensure_ascii=False, indent=2))
    else:
        (OUT / "conditions.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tasks = [(dist, name, overrides, seed) for dist in DISTS for name, _, overrides in todo for seed in SEEDS]
    with Pool(args.workers) as pool:
        rows = [row for row in pool.imap_unordered(one, tasks, chunksize=2) if row is not None]
    if rows:
        path = OUT / "runs.csv"
        new = pd.DataFrame(rows)
        if path.exists():
            new = pd.concat([pd.read_csv(path), new], ignore_index=True).drop_duplicates(["dist", "condition", "seed"], keep="last")
        new.sort_values(["dist", "condition", "seed"]).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"조건 {len(todo)}개 × 가정 {len(DISTS)}개 × seed {len(SEEDS)}개 = {len(tasks)}건 (새로 만든 것 {len(rows)}건)")


if __name__ == "__main__":
    main()
