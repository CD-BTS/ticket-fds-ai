"""확정된 최종 모델 번들을 새 생성 seed 의 합성 데이터에서 한 번 평가한다 (독립 검증).

    ..\\..\\.venv\\Scripts\\python scripts\\independent_validation.py

- 모델: models/behavior_model.joblib (train_fds_model.py 가 만든 서빙 번들 그대로). 여기서 학습하지 않는다.
- 추론: 서버와 같은 코드 경로 (inference.assess_batch: 요청 검증 → Feature 순서 맞추기 → 번들의 전처리·모델 → 원시 점수 → 판정·정규화).
- 데이터: 생성 seed 100~109 × D1·D2·D3 = 30개 데이터셋, 각 1,000계정 전부. 모델 선정(seed 0~9)과 학습(seed 2026)에 쓰지 않은 seed 다.
- 임계값과 정규화 파라미터는 번들에 저장된 값이다. 이 데이터로 다시 계산하지 않는다.
- 이 결과를 보고 Feature 정의, 모델 설정, 임계값을 바꾸지 않는다.

합성 데이터에서의 검증이다. 실제 운영 성능이 아니다. D1·D2·D3 는 가정이며 결과를 따로 적는다.
출력: output/independent_runs.csv, independent_summary.csv, independent_score_checks.json
"""
import hashlib
import json

import numpy as np
import pandas as pd

from common import ANOMALY, DISTS, NORMAL, OUT, metrics
from build_datasets import INDEPENDENT_SEEDS, dataset_path
from inference import assess_batch, load_bundle
from train_fds_model import FEATURES, MODEL_PATH, TARGET_FPR, TRAINING_CSV
from preprocessing import FEATURE_SPECS

POLICY_CUTS = (0.40, 0.70, 0.85)  # 현재 BTS PolicyDecisionService 의 이상 점수 기준 (이번 이슈에서 바꾸지 않는다)


def payloads(table):
    out = []
    for row in table.itertuples(index=False):
        features = {}
        for name, spec in FEATURE_SPECS.items():
            value = getattr(row, name)
            features[spec["camel"]] = None if pd.isna(value) else (int(value) if spec["integer"] else float(value))
        out.append({"userId": str(row.user_id), "calculatedAt": "2026-11-23T00:00:00Z", "windowDays": 30, "features": features})
    return out


def score(table, bundle):
    results = assess_batch(payloads(table), bundle)
    assert all(r["status"] == "OK" for r in results), "모든 합성 계정은 시도 1건 이상이다"
    return (np.array([r["rawAnomalyScore"] for r in results]), np.array([r["anomalyScore"] for r in results]),
            np.array([r["isAnomaly"] for r in results]))


def main():
    bundle = load_bundle(MODEL_PATH)
    threshold, normalizer = bundle["raw_threshold"], bundle["normalizer"]
    rows, pooled = [], []
    for seed in INDEPENDENT_SEEDS:
        for dist in DISTS:
            table = pd.read_csv(dataset_path(dist, seed))
            table["tool_profile"] = table["tool_profile"].fillna("")
            raw, normalized, flagged = score(table, bundle)
            normal = (table["is_anomaly"] == 0).to_numpy()
            rows.append({"evaluated_on": dist, "generation_seed": seed, "accounts": len(table), "target_fpr": TARGET_FPR,
                         **metrics(table, raw, threshold),
                         **{f"normal_share_score_ge_{cut:.2f}": float((normalized[normal] >= cut).mean()) for cut in POLICY_CUTS},
                         **{f"anomaly_share_score_ge_{cut:.2f}": float((normalized[~normal] >= cut).mean()) for cut in POLICY_CUTS},
                         "share_clipped_to_0": float((raw <= normalizer.low).mean()), "share_clipped_to_1": float((raw >= normalizer.high).mean())})
            pooled.append(pd.DataFrame({"evaluated_on": dist, "raw": raw, "score": normalized, "flagged": flagged, "is_anomaly": table["is_anomaly"].to_numpy(),
                                        "scenario": table["scenario"].to_numpy()}))
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "independent_runs.csv", index=False, encoding="utf-8-sig")
    numeric = [c for c in runs.columns if c not in ("evaluated_on", "generation_seed", "accounts", "TP", "FP", "FN", "TN", "target_fpr")]
    grouped = runs.groupby("evaluated_on", sort=False)[numeric]
    summary = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1)[["precision", "recall", "f1", "fpr", "roc_auc"]].add_suffix("_sd")).join(
        grouped.min()[["precision", "recall", "f1", "roc_auc"]].add_suffix("_min")).join(grouped.max()[["fpr"]].add_suffix("_max")).reset_index()
    summary.insert(1, "datasets", grouped.size().to_numpy())
    summary.insert(2, "target_fpr", TARGET_FPR)
    summary.to_csv(OUT / "independent_summary.csv", index=False, encoding="utf-8-sig")

    # ---- 학습용 CSV 의 test 분할 (학습·임계값에 쓰지 않은 300계정). 참고용 ----
    holdout = pd.read_csv(TRAINING_CSV).query("split == 'test'").assign(user_id=lambda d: d["row_id"].astype(str), tool_profile="")
    h_raw, _, _ = score(holdout, bundle)
    holdout_metrics = metrics(holdout, h_raw, threshold)

    # ---- 점수 체계 확인 (30개 데이터셋 3만 계정 전체) ----
    everything = pd.concat(pooled, ignore_index=True)
    raw, normalized, flagged = everything["raw"].to_numpy(), everything["score"].to_numpy(), everything["flagged"].to_numpy()
    order = np.argsort(raw, kind="stable")
    inside = (raw > normalizer.low) & (raw < normalizer.high)
    inside_sorted = np.sort(raw[inside])
    strict = normalizer.transform(inside_sorted)
    normal = (everything["is_anomaly"] == 0).to_numpy()
    checks = {
        "bundle": {"path": "models/behavior_model.joblib", "sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(), "model_version": bundle["model_version"],
                   "trained_at": bundle["trained_at"], "feature_definition_version": bundle["feature_definition_version"],
                   "training_data_sha256": bundle["training"]["data_sha256"], "raw_threshold": threshold, "normalization": bundle["normalization"]},
        "accounts_scored": int(len(everything)), "datasets": int(len(runs)),
        "is_anomaly_equals_raw_ge_threshold": bool((flagged == (raw >= threshold)).all()),
        "is_anomaly_equals_score_ge_0_5": bool((flagged == (normalized >= 0.5)).all()),
        "score_is_non_decreasing_in_raw": bool((np.diff(normalized[order]) >= 0).all()),
        "score_is_strictly_increasing_inside_low_high (반올림 전)": bool((np.diff(strict[np.r_[True, np.diff(inside_sorted) > 0]]) > 0).all()),
        "share_clipped_to_0 (raw <= low)": float((raw <= normalizer.low).mean()), "share_clipped_to_1 (raw >= high)": float((raw >= normalizer.high).mean()),
        "distinct_raw_scores": int(len(np.unique(raw))), "distinct_scores_after_rounding": int(len(np.unique(normalized))),
        "raw_score_range": [float(raw.min()), float(raw.max())],
        "closest_raw_below_threshold": float(raw[raw < threshold].max()), "closest_raw_at_or_above_threshold": float(raw[raw >= threshold].min()),
        "normal_accounts_at_or_above_policy_cut": {
            dist: {f"{cut:.2f}": float((normalized[normal & (everything["evaluated_on"] == dist).to_numpy()] >= cut).mean()) for cut in POLICY_CUTS} for dist in DISTS},
        "anomaly_accounts_at_or_above_policy_cut": {
            dist: {f"{cut:.2f}": float((normalized[~normal & (everything["evaluated_on"] == dist).to_numpy()] >= cut).mean()) for cut in POLICY_CUTS} for dist in DISTS},
        "normal_score_quantiles": {f"p{int(q * 100):02d}": float(np.quantile(normalized[normal], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "anomaly_score_quantiles": {f"p{int(q * 100):02d}": float(np.quantile(normalized[~normal], q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "holdout_of_training_csv (seed 2026 의 test 300계정, 참고)": {k: holdout_metrics[k] for k in ("precision", "recall", "f1", "fpr", "roc_auc")},
        "note": "합성 데이터에서의 값이다. 실제 운영 성능이 아니다. 정책 기준(0.40/0.70/0.85)은 기록만 하고 바꾸지 않았다.",
    }
    (OUT / "independent_score_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pd.set_option("display.width", 250, "display.max_columns", 60)
    main_cols = ["evaluated_on", "datasets"] + [f"{m}_mean" for m in ("precision", "recall", "f1", "fpr", "roc_auc")] + ["f1_sd", "f1_min", "recall_min", "fpr_sd", "fpr_max"]
    print(summary[main_cols].round(3).to_string(index=False))
    print(summary[["evaluated_on"] + [f"fpr_{n}_mean" for n in NORMAL] + [f"recall_{n}_mean" for n in ANOMALY] + [f"recall_tool_{n}_mean" for n in ("fast", "jitter", "human")]].round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in checks.items() if k != "bundle"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
