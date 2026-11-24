"""사후 보충: evaluate_v4.py 의 결과를 본 뒤에 추가한 확인이다 (사전에 계획한 실험이 아니다).

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\evaluate_v4_posthoc.py

E2 (D1·D2·D3 × 생성 seed 10개)에서 '30일 간격 + 분산은 간격 5개 이상' 정의가 세 가정 모두에서 가장 고르게 나왔다.
이 정의가 민감도 조건(대기열 대기, 오픈 참여 횟수 등)에서도 현재 BTS 정의보다 나은지를 같은 절차로 확인한다.
생성 데이터, 모델 설정, 분할, 임계값 규칙은 evaluate_v4.py 와 같다. 생성 파라미터는 바꾸지 않는다.

주의: 이 정의는 test 결과를 보고 고른 것이다. 여기서 나온 값도 독립적인 최종 검증 성능이 아니다.

출력: output/v4/model_evaluation/posthoc_condition_runs.csv, posthoc_condition_summary.csv
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_v4 import BTS, CONDITION_MODEL_SEEDS, GENERATION_SEEDS, METRICS, OUT, RATES, SENS, SPLIT_SEED, all_rows, definition_name, run, splits, summarize, with_definition  # noqa: E402
from generate_events_v4 import DISTS  # noqa: E402
from sweep_v4 import run_path  # noqa: E402

CANDIDATES = [BTS, ("d30", "mean_var", True), ("d30", "end_start", True)]


def for_assumption(dist):
    listed = [c["condition"] for c in json.loads((SENS / "conditions.json").read_text(encoding="utf-8"))["one_at_a_time"]]
    listed += [c["condition"] for c in json.loads((SENS / "combination_choice.json").read_text(encoding="utf-8"))["combinations"]]
    rows = []
    for condition in listed:
        for generation_seed in GENERATION_SEEDS:
            data = pd.read_csv(run_path(dist, condition, generation_seed))
            _, train, val, test = splits(data, SPLIT_SEED)
            for definition in CANDIDATES:
                frame = with_definition(data, definition)
                for seed in CONDITION_MODEL_SEEDS:
                    row, *_ = run(frame, train, val, test, all_rows, seed)
                    rows.append({"assumption": dist, "condition": condition, "definition": definition_name(definition), "training": "혼합",
                                 "generation_seed": generation_seed, "model_seed": seed, **row})
    return pd.DataFrame(rows)


def main():
    with Pool(len(DISTS)) as pool:
        runs = pd.concat(pool.map(for_assumption, DISTS), ignore_index=True)
    runs.to_csv(OUT / "posthoc_condition_runs.csv", index=False, encoding="utf-8-sig")
    per_seed = runs.groupby(["assumption", "condition", "definition", "generation_seed"], sort=False)[METRICS + RATES + ["recall_macro"]].mean().reset_index()
    summary = summarize(per_seed, ["assumption", "condition", "definition"])
    summary.to_csv(OUT / "posthoc_condition_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 300)
    for metric in ("f1_mean", "recall_mean", "fpr_mean"):
        print(metric, "\n", summary.pivot(index="condition", columns=["definition", "assumption"], values=metric).round(3).to_string())


if __name__ == "__main__":
    main()
