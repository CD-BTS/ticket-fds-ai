"""전처리 전략 실험: 스케일링/로그 변환 비교, 결측 주입 강건성 비교.

정답 라벨(합성 데이터 user_type)은 평가에만 사용한다.
Isolation Forest는 랜덤성이 있어 여러 시드의 평균으로 비교한다.
"""

import numpy as np
import pandas as pd

import train_model as tm

SEEDS = [0, 1, 2, 3, 4]
OUT_PATH = tm.BASE_DIR / "data" / "preprocessing_experiments.csv"


def mean_over_seeds(features_df, columns, preprocess, extra_eval=None):
    rows = []
    for seed in SEEDS:
        _, _, results = tm.train_and_score(
            features_df, columns, preprocess=preprocess, random_state=seed
        )
        metrics = tm.evaluate(results, tm.load_labels())
        if extra_eval:
            metrics.update(extra_eval(results))
        rows.append(metrics)
    return pd.DataFrame(rows).mean()


# ---------- A. 스케일링 / 로그 변환 비교 ----------

def experiment_scaling(features_df):
    configs = {
        "none": dict(scaling="none", log_transform=False),
        "standard": dict(scaling="standard", log_transform=False),
        "robust": dict(scaling="robust", log_transform=False),
        "log1p+standard": dict(scaling="standard", log_transform=True),
    }
    rows = []
    for set_name, columns in tm.FEATURE_SETS.items():
        for name, preprocess in configs.items():
            summary = mean_over_seeds(features_df, columns, preprocess)
            rows.append({
                "experiment": "scaling",
                "feature_set": set_name,
                "config": name,
                **summary[[
                    "auc_rapid_vs_rest",
                    "auc_abnormal_vs_normal",
                    "detect_rate_rapid_booking",
                    "detect_rate_frequent_cancel",
                    "detect_rate_normal",
                ]].to_dict(),
            })
    return pd.DataFrame(rows)


# ---------- B. 결측 주입 강건성 (full 7개 Feature) ----------

def experiment_imputation(features_df, missing_fraction=0.10):
    rng = np.random.default_rng(123)
    columns = tm.FEATURE_SETS["full"]
    labels = tm.load_labels().set_index("user_id")["user_type"]

    corrupted = features_df.copy()
    injected = np.zeros(len(corrupted), dtype=bool)
    for column in ["attendance_rate", "avg_booking_duration"]:
        mask = rng.random(len(corrupted)) < missing_fraction
        corrupted.loc[mask, column] = np.nan
        injected |= mask

    injected_normal = injected & (
        corrupted["user_id"].map(labels).to_numpy() == "normal"
    )

    def false_flag(results):
        # 값이 누락됐을 뿐인 정상 사용자가 이상치로 분류되는 비율
        flagged = results["is_anomaly"].to_numpy()
        return {"false_flag_on_missing_normal": flagged[injected_normal].mean()}

    rows = []
    for imputation in ["median", "mean", "zero"]:
        summary = mean_over_seeds(
            corrupted, columns, dict(imputation=imputation), extra_eval=false_flag
        )
        rows.append({
            "experiment": f"imputation (결측 {missing_fraction:.0%} 주입)",
            "feature_set": "full",
            "config": imputation,
            **summary[[
                "auc_rapid_vs_rest",
                "auc_abnormal_vs_normal",
                "detect_rate_normal",
                "false_flag_on_missing_normal",
            ]].to_dict(),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    features_df = tm.load_features()

    scaling_df = experiment_scaling(features_df)
    imputation_df = experiment_imputation(features_df)

    result = pd.concat([scaling_df, imputation_df], ignore_index=True)
    result.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print("\n===== A. 스케일링 / 로그 변환 (시드 5회 평균) =====")
    print(scaling_df.drop(columns="experiment").round(3).to_string(index=False))
    print("\n===== B. 결측치 대체 전략 (attendance_rate, avg_booking_duration에 10% 결측 주입) =====")
    print(imputation_df.drop(columns="experiment").round(3).to_string(index=False))
    print(f"\n저장: {OUT_PATH}")