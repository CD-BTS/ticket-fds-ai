"""Feature 조합 × 결측치 대체 방식별 이상치 민감도 실험.

A. 단일 Feature: 각 Feature 하나만 썼을 때 얼마나 구분하는지
B. 다중 조합: MVP / 전체 / 하나씩 뺀 조합 / 의미 단위 묶음
C. 결측 대체 × 조합: 값이 일부 비었을 때(10%, 30%) median / mean / zero 대체 비교
   - AUC: 비정상 유형을 얼마나 위로 올리는지
   - spearman_vs_clean: 결측이 없을 때의 점수 순위가 얼마나 유지되는지 (1에 가까울수록 안정)
   - false_flag: 값만 비었을 뿐인 정상 사용자가 이상치로 분류되는 비율

정답 라벨(합성 데이터 user_type)은 평가에만 사용한다. 시드 3회 평균.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import train_model as tm

SEEDS = [0, 1, 2]
MISSING_FRACTIONS = [0.10, 0.30]
NULLABLE = ["cancel_rate", "avg_booking_interval", "attendance_rate", "avg_booking_duration"]
OUT_PATH = tm.BASE_DIR / "data" / "feature_experiments.csv"

FULL = tm.FEATURE_SETS["full"]
MVP = tm.FEATURE_SETS["mvp"]


def build_combinations():
    combos = {}
    for column in FULL:                                   # A. 단일
        combos[f"single:{column}"] = [column]
    combos["mvp(5)"] = MVP                                # B. 다중
    combos["full(7)"] = FULL
    for column in FULL:
        combos[f"full-{column}"] = [c for c in FULL if c != column]
    combos["cancel_group"] = ["cancel_count", "cancel_rate"]
    combos["timing_group"] = ["avg_booking_interval", "short_interval_booking_count"]
    return combos


def run_once(features_df, columns, preprocess, seed):
    _, _, results = tm.train_and_score(
        features_df, columns, preprocess=preprocess, random_state=seed
    )
    return results


def metrics_for(results, labels):
    merged = results.merge(labels, on="user_id")
    abnormal = merged["user_type"].isin(["rapid_booking", "frequent_cancel"])
    from sklearn.metrics import roc_auc_score
    return {
        "auc_rapid": roc_auc_score(merged["user_type"] == "rapid_booking", merged["anomaly_score"]),
        "auc_frequent_cancel": roc_auc_score(
            merged["user_type"] == "frequent_cancel", merged["anomaly_score"]
        ),
        "auc_abnormal": roc_auc_score(abnormal, merged["anomaly_score"]),
        "detect_rapid": merged.loc[merged["user_type"] == "rapid_booking", "is_anomaly"].mean(),
        "detect_frequent_cancel": merged.loc[
            merged["user_type"] == "frequent_cancel", "is_anomaly"
        ].mean(),
        "detect_normal": merged.loc[merged["user_type"] == "normal", "is_anomaly"].mean(),
    }


def corrupt(features_df, columns, fraction, seed):
    """선택된 Feature 중 null 허용 항목에만 무작위(MCAR) 결측을 만든다."""
    rng = np.random.default_rng(1000 + seed)
    corrupted = features_df.copy()
    injected = np.zeros(len(corrupted), dtype=bool)
    for column in columns:
        if column in NULLABLE:
            mask = rng.random(len(corrupted)) < fraction
            corrupted.loc[mask, column] = np.nan
            injected |= mask
    return corrupted, injected


def main():
    features_df = tm.load_features()
    labels = tm.load_labels()
    user_type = labels.set_index("user_id")["user_type"]
    is_normal = (features_df["user_id"].map(user_type) == "normal").to_numpy()

    rows = []
    combos = build_combinations()

    # ---- A, B: 결측 없는 데이터, median 대체 ----
    for name, columns in combos.items():
        seed_rows = [
            metrics_for(run_once(features_df, columns, dict(imputation="median"), s), labels)
            for s in SEEDS
        ]
        rows.append({
            "experiment": "combination", "combo": name, "n_features": len(columns),
            "imputation": "median", "missing": 0.0,
            **pd.DataFrame(seed_rows).mean().to_dict(),
        })

    # ---- C: 결측 주입 × 대체 방식 ----
    for name, columns in combos.items():
        if not any(c in NULLABLE for c in columns):
            continue  # 결측이 생기지 않는 조합은 대체 방식 비교 대상이 아니다
        for fraction in MISSING_FRACTIONS:
            for imputation in ["median", "mean", "zero"]:
                seed_rows = []
                for seed in SEEDS:
                    clean = run_once(features_df, columns, dict(imputation=imputation), seed)
                    corrupted_df, injected = corrupt(features_df, columns, fraction, seed)
                    results = run_once(corrupted_df, columns, dict(imputation=imputation), seed)

                    row = metrics_for(results, labels)
                    row["spearman_vs_clean"] = spearmanr(
                        clean["anomaly_score"], results["anomaly_score"]
                    ).statistic
                    hit = injected & is_normal
                    row["false_flag"] = results["is_anomaly"].to_numpy()[hit].mean() if hit.any() else np.nan
                    seed_rows.append(row)

                rows.append({
                    "experiment": "imputation", "combo": name, "n_features": len(columns),
                    "imputation": imputation, "missing": fraction,
                    **pd.DataFrame(seed_rows).mean().to_dict(),
                })

    result = pd.DataFrame(rows)
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    comb = result[result["experiment"] == "combination"]
    cols = ["combo", "auc_rapid", "auc_frequent_cancel", "auc_abnormal",
            "detect_rapid", "detect_frequent_cancel", "detect_normal"]

    print("\n===== A. 단일 Feature (median, 결측 없음) =====")
    print(comb[comb["combo"].str.startswith("single:")][cols].round(3).to_string(index=False))

    print("\n===== B. 다중 조합 (median, 결측 없음) =====")
    print(comb[~comb["combo"].str.startswith("single:")][cols].round(3).to_string(index=False))

    imp = result[result["experiment"] == "imputation"]

    print("\n===== C-1. 대체 방식별 평균 (결측이 생길 수 있는 모든 조합 평균) =====")
    print(
        imp.groupby(["missing", "imputation"])[
            ["auc_abnormal", "auc_rapid", "spearman_vs_clean", "false_flag"]
        ].mean().round(3)
    )

    focus = imp[imp["combo"].isin(["mvp(5)", "full(7)"]) & (imp["missing"] == 0.30)]
    print("\n===== C-2. MVP / 전체, 결측 30% 상세 =====")
    print(
        focus[["combo", "imputation", "auc_abnormal", "auc_rapid",
               "spearman_vs_clean", "false_flag"]].round(3).to_string(index=False)
    )
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()