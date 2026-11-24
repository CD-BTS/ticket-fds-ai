"""최종 학습 데이터 구성을 고른다: D1 / D2 / D3 / Mixed 를 같은 평가 데이터에서 비교한다.

    ..\\..\\.venv\\Scripts\\python scripts\\select_training_data.py

데이터   #17 이 만든 기준 조건의 계정 표를 다시 쓴다 (issue17/data/v4/_runs, 생성 seed 0~9, 가정당 10개). 새로 생성하지 않는다.
         같은 생성 seed 의 D1·D2·D3 는 같은 계정 1,000개이고 이상 계정의 이벤트가 같다 (정상 계정의 속도만 다르다).
분할     계정 단위로 한 번 (test 300 / train 490 / validation 210). 네 후보가 같은 계정 분할을 쓴다.
후보     D1, D2, D3  train·validation 계정의 값을 그 가정의 데이터에서 가져온다
         Mixed       train·validation 계정을 시나리오별로 3등분해 D1·D2·D3 에서 하나씩 가져온다 (계정은 한 번만 들어간다)
         네 후보의 학습 계정 수와 이상 계정 비율은 같다. 임계값은 후보마다 같은 방식으로 만든 validation 에서 정한다.
평가     같은 test 계정을 D1·D2·D3 의 값으로 각각 평가한다 (후보 4 × 평가 분포 3).
모델     train_fds_model.fit (서빙 번들과 같은 코드). 모델 seed 0, 1, 2
학습 수  후보 4 × 생성 seed 10 × 모델 seed 3 = 120

선정 기준 (실행 전에 고정)
  1. 세 평가 분포의 F1 중 가장 낮은 값(생성 seed 평균)이 가장 높은 후보
  2. 그 값이 가장 좋은 후보와 0.01 이내인 후보끼리는, 세 평가 분포의 정상 오탐률 중 가장 높은 값이 낮은 쪽
  3. 그래도 같으면(소수 셋째 자리까지) 특정 분포를 가정하지 않는 Mixed
  매크로 유형별 Recall 과 정상 유형별 오탐률은 함께 저장해 근거로 남긴다.

D1·D2·D3 는 정상 사용자 속도에 대한 가정이다. 어느 것도 실제 분포가 아니다.
출력: output/selection_runs.csv, selection_summary.csv, selection_worst_case.csv, selection_decision.json
"""
import json

import numpy as np
import pandas as pd

from common import ANOMALY, DISTS, ISSUE17, NORMAL, OUT, feature_table, metrics, mixed_assignment, split_accounts, take
from sweep_v4 import run_path  # noqa: E402  (#17 의 저장 위치)
from train_fds_model import FEATURES, TARGET_FPR, fit, raw_anomaly_score  # noqa: E402

GENERATION_SEEDS = list(range(10))
MODEL_SEEDS = [0, 1, 2]
CANDIDATES = ["D1", "D2", "D3", "Mixed"]
TIE = 0.01


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    for generation_seed in GENERATION_SEEDS:
        tables = {dist: feature_table(pd.read_csv(run_path(dist, "base", generation_seed))) for dist in DISTS}
        base = tables["D1"]
        train, validation, test = split_accounts(base)
        assignment = mixed_assignment(base, generation_seed)
        anomaly = (base["is_anomaly"] == 1).to_numpy()
        check = {
            "generation_seed": generation_seed,
            "same_accounts_and_labels_in_D1_D2_D3": all(tables[d][["user_id", "scenario", "is_anomaly"]].equals(base[["user_id", "scenario", "is_anomaly"]]) for d in DISTS),
            "anomaly_rows_identical_in_D1_D2_D3": all(tables[d].loc[anomaly, FEATURES].equals(base.loc[anomaly, FEATURES]) for d in DISTS),
            "train_validation_test_disjoint": not (set(train) & set(validation) or set(train) & set(test) or set(validation) & set(test)),
            "accounts_train_validation_test": [len(train), len(validation), len(test)],
        }
        sets = {}
        for candidate in CANDIDATES:
            source = assignment if candidate == "Mixed" else candidate
            sets[candidate] = (take(tables, train, source), take(tables, validation, source))
            fit_rows = sets[candidate][0]
            check[f"{candidate}_train_accounts"] = int(len(fit_rows))
            check[f"{candidate}_train_accounts_unique"] = bool(fit_rows["user_id"].is_unique)
            check[f"{candidate}_train_anomaly_share"] = float(fit_rows["is_anomaly"].mean())
        mixed_train = sets["Mixed"][0]
        check["Mixed_train_by_source"] = mixed_train["source_assumption"].value_counts().sort_index().to_dict()
        check["Mixed_train_anomaly_by_source"] = mixed_train.loc[mixed_train["is_anomaly"] == 1, "source_assumption"].value_counts().sort_index().to_dict()
        checks.append(check)

        tests = {dist: tables[dist].iloc[test] for dist in DISTS}
        for candidate, (fit_rows, validation_rows) in sets.items():
            for model_seed in MODEL_SEEDS:
                model, threshold, _ = fit(fit_rows, validation_rows, model_seed)
                normal = validation_rows[validation_rows["is_anomaly"] == 0]
                validation_fpr = float((raw_anomaly_score(model, normal[FEATURES]) >= threshold).mean())
                for dist, frame in tests.items():
                    rows.append({"candidate": candidate, "evaluated_on": dist, "generation_seed": generation_seed, "model_seed": model_seed,
                                 "raw_threshold": threshold, "target_fpr": TARGET_FPR, "validation_fpr": validation_fpr,
                                 **metrics(frame, raw_anomaly_score(model, frame[FEATURES]), threshold)})
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "selection_runs.csv", index=False, encoding="utf-8-sig")

    numeric = [c for c in runs.columns if c not in ("candidate", "evaluated_on", "generation_seed", "model_seed", "TP", "FP", "FN", "TN")]
    per_seed = runs.groupby(["candidate", "evaluated_on", "generation_seed"], sort=False)[numeric].mean().reset_index()
    grouped = per_seed.groupby(["candidate", "evaluated_on"], sort=False)[numeric]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1)[["f1", "recall", "fpr"]].add_suffix("_sd")).join(
        grouped.min()[["f1", "recall"]].add_suffix("_min_seed")).join(grouped.max()[["fpr"]].add_suffix("_max_seed")).reset_index()
    summary.to_csv(OUT / "selection_summary.csv", index=False, encoding="utf-8-sig")

    worst_rows = []
    for candidate in CANDIDATES:
        part = summary[summary["candidate"] == candidate].set_index("evaluated_on")
        worst_rows.append({
            "candidate": candidate, "f1_D1": part.loc["D1", "f1_mean"], "f1_D2": part.loc["D2", "f1_mean"], "f1_D3": part.loc["D3", "f1_mean"],
            "worst_f1": part["f1_mean"].min(), "worst_f1_on": part["f1_mean"].idxmin(), "mean_f1": part["f1_mean"].mean(),
            "worst_recall": part["recall_mean"].min(), "worst_fpr": part["fpr_mean"].max(), "worst_fpr_on": part["fpr_mean"].idxmax(),
            "worst_single_seed_f1": part["f1_min_seed"].min(), "roc_auc_min": part["roc_auc_mean"].min(),
            **{f"worst_fpr_{name}": part[f"fpr_{name}_mean"].max() for name in NORMAL},
            **{f"worst_recall_{name}": part[f"recall_{name}_mean"].min() for name in ANOMALY},
        })
    worst = pd.DataFrame(worst_rows)
    worst.to_csv(OUT / "selection_worst_case.csv", index=False, encoding="utf-8-sig")

    best = worst["worst_f1"].max()
    tied = worst[worst["worst_f1"] >= best - TIE]
    lowest = tied["worst_fpr"].round(3).min()
    finalists = tied[tied["worst_fpr"].round(3) == lowest]
    chosen = "Mixed" if "Mixed" in set(finalists["candidate"]) else finalists.sort_values("worst_f1", ascending=False)["candidate"].iloc[0]
    decision = {
        "rule": ["1. 세 평가 분포 중 가장 낮은 F1(생성 seed 평균)이 가장 높은 후보", f"2. 그 값이 최고와 {TIE} 이내인 후보끼리는 최악 분포의 정상 오탐률이 낮은 쪽",
                 "3. 그래도 같으면(소수 셋째 자리) Mixed"],
        "best_worst_f1": float(best), "within_tie_of_best": tied["candidate"].tolist(),
        "worst_fpr_of_tied": {r.candidate: round(float(r.worst_fpr), 4) for r in tied.itertuples()}, "chosen": chosen,
        "data_reused": str((ISSUE17 / "data" / "v4" / "_runs").relative_to(ISSUE17.parent.parent).as_posix()) + " (base, 생성 seed 0~9)",
        "generation_seeds": GENERATION_SEEDS, "model_seeds": MODEL_SEEDS, "model_fits": len(CANDIDATES) * len(GENERATION_SEEDS) * len(MODEL_SEEDS),
        "new_datasets_generated": 0, "checks_all_seeds": {
            "same_accounts_and_labels_in_D1_D2_D3": all(c["same_accounts_and_labels_in_D1_D2_D3"] for c in checks),
            "anomaly_rows_identical_in_D1_D2_D3": all(c["anomaly_rows_identical_in_D1_D2_D3"] for c in checks),
            "train_validation_test_disjoint": all(c["train_validation_test_disjoint"] for c in checks),
            "every_candidate_has_same_train_accounts": all(len({c[f"{k}_train_accounts"] for k in CANDIDATES}) == 1 for c in checks),
            "every_candidate_has_same_train_anomaly_share": all(len({round(c[f"{k}_train_anomaly_share"], 12) for k in CANDIDATES}) == 1 for c in checks),
            "no_account_repeated_within_a_candidate": all(c[f"{k}_train_accounts_unique"] for c in checks for k in CANDIDATES)},
        "checks_by_seed": checks,
        "note": "선정에 쓴 데이터(생성 seed 0~9)는 독립 검증(생성 seed 100~109)에 쓰지 않는다. D1·D2·D3 는 가정이며 실제 분포가 아니다.",
    }
    (OUT / "selection_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pd.set_option("display.width", 250, "display.max_columns", 40)
    shown = ["candidate", "evaluated_on", "precision_mean", "recall_mean", "f1_mean", "f1_sd", "fpr_mean", "roc_auc_mean", "validation_fpr_mean"]
    print(summary[shown].round(3).to_string(index=False))
    print(worst.round(3).T.to_string())
    print(json.dumps({k: v for k, v in decision.items() if k != "checks_by_seed"}, ensure_ascii=False, indent=2))
    print("check (seed 0):", {k: v for k, v in checks[0].items() if "Mixed" in k or "accounts" in k})


if __name__ == "__main__":
    main()
