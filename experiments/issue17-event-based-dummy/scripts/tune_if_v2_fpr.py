"""이벤트 기반 v2 데이터셋(C)에서 정상 사용자 오탐률(FPR) 5% 이하를 조건으로 Isolation Forest 의 판정 임계값과 학습 데이터 구성을 고른다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\tune_if_v2_fpr.py

scripts/tune_if_v2.py 의 점수 계산(Scorer)과 분할을 그대로 쓴다. 그 스크립트와 결과(output/v2/model_tuning/)는 건드리지 않는다.
바꾸지 않는 것: Feature 6개, v2 데이터셋·이벤트 생성 규칙, 트리 수(100)·max_samples(기본값). 새 Feature 는 만들지 않는다.

분할: test 300 (기존과 같은 계정) / train 490 / validation 210  — tune_if_v2.py 와 같은 계정
  - 모델·결측 대체는 train 에만 fit 한다. 임계값은 그 모델의 validation 점수로만 정한다.
  - 임계값: validation 정상 계정의 오탐률이 TARGET_FPR 이하가 되는 가장 낮은 점수 (= 조건 안에서 Recall 이 가장 높은 임계값).
    임계값은 모델에 묶여 있으므로 test 는 train 으로 학습한 같은 모델, 같은 임계값으로 평가한다 (재학습 없음).
  - 설정(전처리 4가지 × 학습 데이터 3가지)은 validation Recall(모델 seed 10개 평균, 동률이면 F1)로 고른다. test 는 선택에 쓰지 않는다.

학습 데이터 구성
  - 혼합: train 전체 (정상 + 이상, 레이블 미사용)
  - 정상만: train 중 normal, fast_normal, shared_ip, frequent_cancel (레이블로 이상 계정을 뺀다)
  - 혼합(크기 맞춤): 혼합 train 에서 시나리오 비율을 유지한 채 "정상만"과 같은 건수만 뽑는다 (크기 차이 통제)
  - 그 밖에 학습 건수를 바꿔 가며 두 구성을 비교한다 (크기 자체의 영향)
  - "정상만"에 이상 계정을 0~20% 섞어 본다 (실제로는 이상 계정을 완전히 걸러낼 수 없으므로)

출력: output/v2/model_fpr/
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tune_if_v2 import (BASELINE, BEST_CONTAMINATION, C_ANOM, C_NORMAL, C_PATH, EXP, GRID, INK, MODEL_SEED, MUTED, NORMAL_SCENARIOS,  # noqa: E402
                        ORIGINAL_PATH, REPO, SCENARIOS, SPLIT_SEED, SURFACE, TEST_SIZE, VAL_SIZE, F, Scorer, evaluate, flag_rates,
                        grouped_bars, mean_std, policy_name, save, sha256, style)

OUT = EXP / "output" / "v2" / "model_fpr"
TARGET_FPR = 0.05
SELECT_SEEDS = list(range(10))
MODEL_SEEDS = list(range(20))
SPLIT_SEEDS = list(range(20))
NESTED_SELECT_SEEDS = list(range(5))
TUNED = (1, "median", True)  # REPORT_V2_TUNING.md 에서 고른 전처리
POLICIES = [BASELINE, (1, "median", False), TUNED, (2, "median", True)]
MIXED, MATCHED, NORMAL_ONLY = "혼합", "혼합 (크기 맞춤)", "정상만"
TRAININGS = [MIXED, MATCHED, NORMAL_ONLY]
SIZES = [100, 200, 300]
PURITY_SHARES = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]  # "정상만" 학습 데이터에 섞는 이상 계정의 비율
C_A, C_BASE, C_SEL = "#2a78d6", "#1baf7a", "#6250d6"
METRICS = ["precision", "recall", "f1", "fpr", "roc_auc"]
RATES = [f"flag_rate_{s}" for s in SCENARIOS]
# 기존 문서의 Feature 기준 후보 (REPORT_V2.md 1.1~1.3절: 데모 프론트 FEATURE_SPECS 의 후보와 Credit 규칙). booking_count 의 기준은 문서에 없다.
CRITERIA = {
    "간격 < 10초 (후보)": lambda d: d["avg_booking_interval"] < 10,
    "간격 < 5초 (후보)": lambda d: d["avg_booking_interval"] < 5,
    "분산 < 1.0초² (후보)": lambda d: d["booking_interval_variance"] < 1.0,
    "분산 < 1.0초², 시도 5건 이상 (Credit)": lambda d: (d["booking_interval_variance"] < 1.0) & (d["booking_count"] >= 5),
    "취소율 ≥ 30% (후보)": lambda d: d["cancel_rate"] >= 0.3,
    "취소율 ≥ 50% (후보)": lambda d: d["cancel_rate"] >= 0.5,
    "취소율 ≥ 70% (후보)": lambda d: d["cancel_rate"] >= 0.7,
    "카드 ≥ 2 (Credit 약한 감점)": lambda d: d["card_reuse_count"] >= 2,
    "카드 ≥ 4 (Credit)": lambda d: d["card_reuse_count"] >= 4,
    "IP ≥ 5 (Credit)": lambda d: d["ip_booking_density"] >= 5,
    "결측 Feature 있음": lambda d: d[F].isna().any(axis=1),
}


def training_rows(training, train_idx, scenario, seed, size=None):
    normal_idx = train_idx[scenario.loc[train_idx].isin(NORMAL_SCENARIOS).to_numpy()]
    if training == NORMAL_ONLY:
        pool, size = normal_idx, size
    else:
        pool, size = train_idx, (len(normal_idx) if training == MATCHED and size is None else size)
    if size is None or size >= len(pool):
        return pool
    picked, _ = train_test_split(pool, train_size=size, random_state=seed, stratify=scenario.loc[pool])
    return picked


def fpr_threshold(y, scores, target=TARGET_FPR):
    """정상 계정의 오탐률이 target 이하가 되는 가장 낮은 임계값 (score > 임계값 이면 이상). 분리된 계정(NaN)은 이상 아님으로 센다."""
    normal = y == 0
    allowed = int(np.floor(target * normal.sum() + 1e-9))
    ranked = np.sort(scores[normal & ~np.isnan(scores)])[::-1]
    return -np.inf if len(ranked) <= allowed else float(ranked[allowed])


def run(frame, y, scenario, train_idx, val_idx, test_idx, policy, training, seed, split_seed, size=None):
    fit_idx = training_rows(training, train_idx, scenario, split_seed, size)
    scorer = Scorer(frame, fit_idx, policy, seed)
    val_scores, test_scores = scorer.score(val_idx), scorer.score(test_idx)
    y_val, y_test = y.loc[val_idx].to_numpy(), y.loc[test_idx].to_numpy()
    threshold = fpr_threshold(y_val, val_scores)
    val, _ = evaluate(y_val, val_scores, threshold)
    test, pred = evaluate(y_test, test_scores, threshold)
    row = {"policy": policy_name(policy), "training": training, "seed": seed, "n_fit": len(fit_idx), "threshold": threshold,
           **{f"val_{k}": val[k] for k in METRICS}, **{f"test_{k}": test[k] for k in METRICS + ["TP", "FP", "TN", "FN", "insufficient"]},
           **{f"test_{k}": v for k, v in flag_rates(pred, scenario.loc[test_idx].to_numpy()).items()}}
    return row, scorer, threshold


def select(summary):
    """validation 에서 오탐률 조건을 맞춘 임계값의 Recall 이 가장 높은 설정 (동률이면 F1)."""
    top = summary.sort_values(["val_recall", "val_f1"], ascending=False).iloc[0]
    return top["policy"], top["training"]


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 400)

    hash_before = {"A": sha256(ORIGINAL_PATH), "C": sha256(C_PATH)}
    data, original = pd.read_csv(C_PATH), pd.read_csv(ORIGINAL_PATH)
    y, scenario = data["is_anomaly"], data["scenario"]
    by_name = {policy_name(p): p for p in POLICIES}

    def split(seed):
        dev_idx, test_idx = train_test_split(data.index, test_size=TEST_SIZE, random_state=seed, stratify=y)
        train_idx, val_idx = train_test_split(dev_idx, test_size=VAL_SIZE, random_state=seed, stratify=scenario.loc[dev_idx])
        return train_idx, val_idx, test_idx

    train_idx, val_idx, test_idx = split(SPLIT_SEED)
    y_val, y_test = y.loc[val_idx].to_numpy(), y.loc[test_idx].to_numpy()
    s_val, s_test = scenario.loc[val_idx].to_numpy(), scenario.loc[test_idx].to_numpy()

    def csv(frame, name):
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    # ---- 실험 1·2: 전처리 4 × 학습 데이터 3, 임계값은 validation 오탐률 5% 이하 ----
    rows = []
    for policy in POLICIES:
        for training in TRAININGS:
            for seed in sorted(set(MODEL_SEEDS) | {MODEL_SEED}):
                row, scorer, _ = run(data, y, scenario, train_idx, val_idx, test_idx, policy, training, seed, SPLIT_SEED)
                # 참고: 레이블이 있는 validation 없이 정하는 임계값 (학습 점수의 분위수 = contamination)
                for c in (TARGET_FPR, BEST_CONTAMINATION):
                    ref, _ = evaluate(y_test, scorer.score(test_idx), scorer.threshold(c))
                    row.update({f"test_c{c:g}_{k}": ref[k] for k in ("precision", "recall", "f1", "fpr")})
                rows.append(row)
    runs = pd.DataFrame(rows)
    csv(runs, "runs.csv")
    keys = ["policy", "training"]
    numeric = [c for c in runs.columns if c not in keys + ["seed"]]
    selection = runs[runs["seed"].isin(SELECT_SEEDS)].groupby(keys, sort=False)[numeric].mean().reset_index()
    csv(selection[keys + ["n_fit"] + [f"val_{k}" for k in METRICS]], "validation_selection.csv")
    chosen_policy, chosen_training = select(selection)
    chosen = by_name[chosen_policy]
    # 혼합 학습 안에서만 고른 설정 (레이블 없이 학습할 수 있는 구성)
    mixed_policy, _ = select(selection[selection["training"] == MIXED])
    print(f"validation 선택: {chosen_policy} / {chosen_training}   (혼합 학습 안에서: {mixed_policy})", flush=True)

    model_seed_summary = mean_std(runs[runs["seed"].isin(MODEL_SEEDS)], keys, [c for c in numeric if c.startswith(("val_", "test_")) or c == "threshold"])
    model_seed_summary["test_fpr_within_target_share"] = (runs[runs["seed"].isin(MODEL_SEEDS)].assign(ok=lambda d: d["test_fpr"] <= TARGET_FPR)
                                                          .groupby(keys, sort=False)["ok"].mean().to_numpy())
    csv(model_seed_summary, "model_seed_summary.csv")
    seed42 = runs[runs["seed"] == MODEL_SEED].drop(columns="seed").reset_index(drop=True)
    csv(seed42, "test_metrics_seed42.csv")

    # 참고: A(기존 Feature 기반)를 같은 방식(기존 전처리, 혼합 학습, validation 오탐률 5% 임계값)으로
    rows = []
    for seed in sorted(set(MODEL_SEEDS) | {MODEL_SEED}):
        row, _, _ = run(original, y, scenario, train_idx, val_idx, test_idx, BASELINE, MIXED, seed, SPLIT_SEED)
        rows.append({"dataset": "A 기존 Feature 기반", **row})
    reference_a = pd.DataFrame(rows)
    csv(reference_a, "reference_a_runs.csv")

    # ---- 실험 2: 학습 건수의 영향 ----
    rows = []
    n_normal = int(scenario.loc[train_idx].isin(NORMAL_SCENARIOS).sum())
    for policy in dict.fromkeys([BASELINE, chosen]):
        for training in (MIXED, NORMAL_ONLY):
            for size in SIZES + [n_normal] + ([len(train_idx)] if training == MIXED else []):
                for seed in SELECT_SEEDS:
                    row, _, _ = run(data, y, scenario, train_idx, val_idx, test_idx, policy, training, seed, seed, size)
                    rows.append({**row, "requested_size": size})
    sizes = pd.DataFrame(rows)
    size_summary = (sizes.groupby(["policy", "training", "requested_size"], sort=False)[["n_fit"] + [f"{p}_{k}" for p in ("val", "test") for k in METRICS]]
                    .agg(["mean", "std"]))
    size_summary.columns = [f"{a}_{b}" for a, b in size_summary.columns]
    size_summary = size_summary.reset_index()
    csv(size_summary, "training_size.csv")

    # ---- 실험 2: 정상 학습 데이터에 이상 계정이 섞여 있을 때 (실제로는 이상 계정을 완전히 걸러낼 수 없다) ----
    rows = []
    normal_train = train_idx[scenario.loc[train_idx].isin(NORMAL_SCENARIOS).to_numpy()]
    anomaly_train = train_idx[~scenario.loc[train_idx].isin(NORMAL_SCENARIOS).to_numpy()]
    for share in PURITY_SHARES:
        count = min(int(round(share / (1 - share) * len(normal_train))), len(anomaly_train))
        for seed in SELECT_SEEDS:
            extra = anomaly_train if count == len(anomaly_train) else (
                train_test_split(anomaly_train, train_size=count, random_state=seed, stratify=scenario.loc[anomaly_train])[0] if count else anomaly_train[:0])
            scorer = Scorer(data, normal_train.append(extra), chosen, seed)
            val_scores, test_scores = scorer.score(val_idx), scorer.score(test_idx)
            threshold = fpr_threshold(y_val, val_scores)
            val, test = evaluate(y_val, val_scores, threshold)[0], evaluate(y_test, test_scores, threshold)[0]
            free = evaluate(y_test, test_scores, scorer.threshold(TARGET_FPR))[0]
            rows.append({"policy": chosen_policy, "anomaly_share_in_training": share, "anomalies_in_training": count, "n_fit": len(normal_train) + count,
                         "seed": seed, **{f"val_{k}": val[k] for k in METRICS}, **{f"test_{k}": test[k] for k in METRICS},
                         **{f"test_c0.05_{k}": free[k] for k in ("precision", "recall", "f1", "fpr")}})
    purity = (pd.DataFrame(rows).groupby(["policy", "anomaly_share_in_training", "anomalies_in_training", "n_fit"], sort=False)
              .mean(numeric_only=True).drop(columns="seed").reset_index())
    csv(purity, "training_purity.csv")
    print("\n[학습 데이터의 이상 계정 비율]\n", purity.drop(columns="policy").round(3).to_string(index=False), flush=True)

    # ---- 실험 3: 오탐 원인 (validation, 모델 seed 0 의 상세 + seed 10개 평균 비율) ----
    settings = [("기존 모델", BASELINE, MIXED), ("개선 모델", chosen, chosen_training)]
    variance_p90 = data.loc[train_idx, "booking_interval_variance"].quantile(0.9)
    details, scenario_rows, criteria_rows = [], [], []
    for label, policy, training in settings:
        rates = []
        for seed in SELECT_SEEDS:
            _, scorer, threshold = run(data, y, scenario, train_idx, val_idx, test_idx, policy, training, seed, SPLIT_SEED)
            scores = scorer.score(val_idx)
            _, pred = evaluate(y_val, scores, threshold)
            rates.append(flag_rates(pred, s_val))
            if seed != SELECT_SEEDS[0]:
                continue
            x = data.loc[val_idx, F].to_numpy(dtype=float)
            ok = ~np.isnan(scores)
            contribution = np.full((len(val_idx), len(F)), np.nan)
            for j in range(len(F)):
                swapped = x[ok].copy()
                swapped[:, j] = scorer.fill[j]
                contribution[ok, j] = scores[ok] - scorer.raw(swapped)
            part = data.loc[val_idx].reset_index(names="row_index")
            part["setting"], part["score"], part["threshold"], part["margin"] = label, scores, threshold, scores - threshold
            part["pred_anomaly"], part["insufficient"] = pred, ~ok
            part["top_feature"] = [F[int(np.nanargmax(r))] if not np.isnan(r).all() else "" for r in contribution]
            for j, feature in enumerate(F):
                part[f"contribution_{feature}"] = contribution[:, j]
            for name, rule in CRITERIA.items():
                part[name] = rule(part).fillna(False).to_numpy()
            part["분산 ≥ 학습 상위 10%"] = (part["booking_interval_variance"] >= variance_p90).to_numpy()
            part["시도 2건 (분산 0)"] = (part["booking_count"] == 2).to_numpy()
            details.append(part)
        rates, part = pd.DataFrame(rates).mean(), details[-1]
        for name in SCENARIOS:
            sub = part[part["scenario"] == name]
            wrong = sub[sub["pred_anomaly"] != sub["is_anomaly"]]
            row = {"setting": label, "scenario": name, "actual": "정상" if name in NORMAL_SCENARIOS else "이상", "n_val": len(sub),
                   "flag_rate_mean_10seeds": rates[f"flag_rate_{name}"], "kind": "오탐" if name in NORMAL_SCENARIOS else "미탐",
                   "errors_seed0": len(wrong), "score_median": sub["score"].median(), "margin_median": sub["margin"].median(),
                   "errors_margin_median": wrong["margin"].median()}
            row.update({f"top_{feature}": int((wrong["top_feature"] == feature).sum()) for feature in F})
            scenario_rows.append(row)
        normal = part[part["is_anomaly"] == 0]
        for name in list(CRITERIA) + ["분산 ≥ 학습 상위 10%", "시도 2건 (분산 0)"]:
            met = normal[normal[name]]
            criteria_rows.append({"setting": label, "criterion": name, "normal_meeting": len(met), "normal_meeting_share": len(met) / len(normal),
                                  "flagged_among_meeting": int(met["pred_anomaly"].sum()),
                                  "fpr_among_meeting": met["pred_anomaly"].mean() if len(met) else np.nan,
                                  "false_positives_total": int(normal["pred_anomaly"].sum()),
                                  "share_of_false_positives": met["pred_anomaly"].sum() / max(normal["pred_anomaly"].sum(), 1),
                                  "anomaly_meeting_share": float(part.loc[part["is_anomaly"] == 1, name].mean()),
                                  **{f"fp_{s}": int(met.loc[met["scenario"] == s, "pred_anomaly"].sum()) for s in NORMAL_SCENARIOS}})
    detail = pd.concat(details)
    errors, criteria = pd.DataFrame(scenario_rows), pd.DataFrame(criteria_rows)
    none_met = (detail[(detail["is_anomaly"] == 0) & (detail["pred_anomaly"] == 1)]
                .assign(any_criterion=lambda d: d[list(CRITERIA)].any(axis=1)).groupby("setting", sort=False)["any_criterion"].agg(["size", "sum"]))
    csv(detail, "error_detail_validation.csv")
    csv(errors, "error_analysis_validation.csv")
    csv(criteria, "false_positive_criteria_validation.csv")
    correlation = pd.DataFrame([{"setting": label, "group": group, "feature": feature,
                                 "spearman_with_score": sub[feature].corr(sub["score"], method="spearman")}
                                for label, _, _ in settings for group, flag in (("정상", 0), ("이상", 1)) for feature in F
                                for sub in [detail[(detail["setting"] == label) & (detail["is_anomaly"] == flag)]]])
    csv(correlation, "feature_score_correlation_validation.csv")

    # ---- 안정성: 분할 seed (분할마다 validation 에서 임계값과 설정을 다시 정한다) ----
    rows, picks = [], []
    for split_seed in SPLIT_SEEDS:
        tr, va, te = split(split_seed)
        nested = pd.DataFrame([run(data, y, scenario, tr, va, te, policy, training, seed, split_seed)[0]
                               for policy in POLICIES for training in TRAININGS for seed in NESTED_SELECT_SEEDS])
        nested_summary = nested.groupby(keys, sort=False)[["val_recall", "val_f1"]].mean().reset_index()
        picked = select(nested_summary)
        picks.append({"split_seed": split_seed, "policy": picked[0], "training": picked[1]})
        for label, policy, training in (("C 기존 모델 (기존 전처리, 혼합)", BASELINE, MIXED), (f"C 혼합 학습 ({mixed_policy})", by_name[mixed_policy], MIXED),
                                        ("C 개선 모델 (고정: seed 42 분할에서 고른 설정)", chosen, chosen_training),
                                        ("C 개선 모델 (분할마다 다시 선택)", by_name[picked[0]], picked[1])):
            row, _, _ = run(data, y, scenario, tr, va, te, policy, training, MODEL_SEED, split_seed)
            rows.append({"setting": label, "split_seed": split_seed, **row})
        row, _, _ = run(original, y, scenario, tr, va, te, BASELINE, MIXED, MODEL_SEED, split_seed)
        rows.append({"setting": "A 기존 Feature 기반 (기존 전처리, 혼합)", "split_seed": split_seed, **row})
        print(f"분할 seed {split_seed}: {picked[0]} / {picked[1]}", flush=True)
    split_runs, picks = pd.DataFrame(rows), pd.DataFrame(picks)
    csv(split_runs, "split_seed_runs.csv")
    csv(picks, "split_seed_selected.csv")
    split_cols = [f"{p}_{k}" for p in ("val", "test") for k in METRICS] + [f"test_{r}" for r in RATES]
    split_summary = mean_std(split_runs, ["setting"], split_cols)
    split_summary["test_fpr_within_target_share"] = split_runs.assign(ok=lambda d: d["test_fpr"] <= TARGET_FPR).groupby("setting", sort=False)["ok"].mean().to_numpy()
    csv(split_summary, "split_seed_summary.csv")

    # ---------------- 그래프 ----------------
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, (label, policy, training) in zip(axes, settings):
        style(ax)
        part = detail[detail["setting"] == label]
        for position, name in enumerate(SCENARIOS):
            values = part.loc[part["scenario"] == name, "score"].dropna().to_numpy()
            ax.scatter(position + rng.uniform(-0.25, 0.25, len(values)), values, s=10, alpha=0.6, linewidth=0,
                       color=C_NORMAL if name in NORMAL_SCENARIOS else C_ANOM)
        ax.axhline(part["threshold"].iloc[0], color=INK, linewidth=1.2, linestyle="--")
        ax.text(len(SCENARIOS) - 0.5, part["threshold"].iloc[0], " 임계값", va="bottom", ha="right", fontsize=8, color=INK)
        ax.set_xticks(range(len(SCENARIOS)))
        ax.set_xticklabels(SCENARIOS, fontsize=8, rotation=20)
        ax.set_title(f"{label}: {policy_name(policy)}, {training} 학습", loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("이상 점수 (클수록 이상)", fontsize=9, color=MUTED)
    fig.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color=C_NORMAL, label="정상 시나리오"),
                        plt.Line2D([], [], marker="o", linestyle="", color=C_ANOM, label="이상 시나리오")],
               loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("시나리오별 이상 점수와 오탐률 5% 임계값 (validation, 모델 seed 0)", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "score_by_scenario.png")

    for label, name in (("기존 모델", "feature_vs_score_baseline.png"), ("개선 모델", "feature_vs_score_selected.png")):
        part = detail[detail["setting"] == label]
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharey=True, constrained_layout=True)
        for ax, feature in zip(axes.ravel(), F):
            style(ax)
            for flag, color in ((0, C_NORMAL), (1, C_ANOM)):
                sub = part[part["is_anomaly"] == flag]
                jitter = rng.uniform(-0.12, 0.12, len(sub)) if feature in ("card_reuse_count", "ip_booking_density") else 0
                ax.scatter(sub[feature] + jitter, sub["score"], s=10, alpha=0.55, linewidth=0, color=color)
            ax.axhline(part["threshold"].iloc[0], color=INK, linewidth=1.2, linestyle="--")
            if feature in ("avg_booking_interval", "booking_interval_variance", "booking_count"):
                ax.set_xscale("symlog", linthresh=1.0)
                ax.set_xlim(left=0)
            ax.set_title(feature, loc="left", fontsize=10, color=INK)
        for ax in axes[:, 0]:
            ax.set_ylabel("이상 점수", fontsize=9, color=MUTED)
        fig.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color=C_NORMAL, label="정상"),
                            plt.Line2D([], [], marker="o", linestyle="", color=C_ANOM, label="이상"),
                            plt.Line2D([], [], color=INK, linestyle="--", linewidth=1.2, label="오탐률 5% 임계값")],
                   loc="outside lower center", ncol=3, frameon=False, fontsize=9)
        fig.suptitle(f"Feature 값과 이상 점수 — {label} (validation, 모델 seed 0, 결측 계정은 표시되지 않음)", fontsize=11, color=INK, x=0.01, ha="left")
        save(fig, OUT / "figures" / name)

    def pick(frame, policy, training):
        return frame[(frame["policy"] == policy) & (frame["training"] == training)].iloc[0]

    base42, sel42 = pick(seed42, policy_name(BASELINE), MIXED), pick(seed42, chosen_policy, chosen_training)
    a42 = reference_a[reference_a["seed"] == MODEL_SEED].iloc[0]
    bars = [("A 기존 Feature 기반", a42, C_A), ("C 기존 모델", base42, C_BASE), ("C 개선 모델", sel42, C_SEL)]
    names = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("fpr", "오탐률(FPR)")]
    fig, ax = plt.subplots(figsize=(10.5, 4.4), constrained_layout=True)
    style(ax)
    grouped_bars(ax, [n for _, n in names], [(label, [row[f"test_{key}"] for key, _ in names], color) for label, row, color in bars], fmt="{:.3f}")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    ax.set_title("validation 오탐률 5% 이하 임계값을 그대로 적용한 test 300건 (모델 seed 42)", loc="left", fontsize=10, color=INK)
    save(fig, OUT / "figures" / "overall_metrics.png")

    counts = pd.Series(s_test).value_counts()
    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
    style(ax)
    grouped_bars(ax, [f"{s}\n({'정상' if s in NORMAL_SCENARIOS else '이상'}, n={counts[s]})" for s in SCENARIOS],
                 [(label, [row[f"test_flag_rate_{s}"] for s in SCENARIOS], color) for label, row, color in bars])
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    ax.set_ylabel("이상 판정 비율", fontsize=9, color=MUTED)
    ax.set_title("시나리오별 이상 판정 비율 (정상 = 오탐률, 이상 = Recall), test, 모델 seed 42", loc="left", fontsize=10, color=INK)
    save(fig, OUT / "figures" / "scenario_detection.png")

    model_panel = pd.concat([runs[(runs["policy"] == policy_name(BASELINE)) & (runs["training"] == MIXED) & runs["seed"].isin(MODEL_SEEDS)].assign(setting="기존 모델"),
                             runs[(runs["policy"] == chosen_policy) & (runs["training"] == chosen_training) & runs["seed"].isin(MODEL_SEEDS)].assign(setting="개선 모델")])
    split_panel = split_runs[split_runs["setting"].isin(["C 기존 모델 (기존 전처리, 혼합)", "C 개선 모델 (분할마다 다시 선택)"])].assign(
        setting=lambda d: d["setting"].map({"C 기존 모델 (기존 전처리, 혼합)": "기존 모델", "C 개선 모델 (분할마다 다시 선택)": "개선 모델\n(분할마다 다시 선택)"}))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharey="row", constrained_layout=True)
    for col, (title, frame) in enumerate(((f"모델 seed {len(MODEL_SEEDS)}개 (분할 고정)", model_panel), (f"분할 seed {len(SPLIT_SEEDS)}개", split_panel))):
        for row_i, (key, ylabel) in enumerate((("test_fpr", "test 오탐률"), ("test_recall", "test Recall"))):
            ax = axes[row_i, col]
            style(ax)
            order = list(dict.fromkeys(frame["setting"]))
            for position, name in enumerate(order):
                values = frame.loc[frame["setting"] == name, key].to_numpy()
                ax.scatter(position + rng.uniform(-0.16, 0.16, len(values)), values, s=22, color=C_BASE if name == "기존 모델" else C_SEL,
                           edgecolor=SURFACE, linewidth=0.6)
                ax.hlines(values.mean(), position - 0.3, position + 0.3, color=INK, linewidth=1.4)
                ax.text(position + 0.34, values.mean(), f"{values.mean():.3f}", va="center", fontsize=8, color=INK)
            if key == "test_fpr":
                ax.axhline(TARGET_FPR, color=INK, linewidth=1, linestyle="--")
                ax.set_ylim(0, 0.16)
                ax.set_title(title, loc="left", fontsize=10, color=INK)
            else:
                ax.set_ylim(0, 1.05)
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, fontsize=9)
            ax.set_xlim(-0.6, len(order) - 0.2)
            if col == 0:
                ax.set_ylabel(ylabel + (" (점선 = 목표 5%)" if key == "test_fpr" else ""), fontsize=9, color=MUTED)
    fig.suptitle("seed 를 바꿨을 때의 test 오탐률과 Recall (점 = 실행 1회, 가로선 = 평균)", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "seed_stability.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True, constrained_layout=True)
    for ax, policy in zip(axes, dict.fromkeys([policy_name(BASELINE), chosen_policy])):
        style(ax)
        for training, color in ((MIXED, C_BASE), (NORMAL_ONLY, C_SEL)):
            part = size_summary[(size_summary["policy"] == policy) & (size_summary["training"] == training)]
            ax.plot(part["n_fit_mean"], part["val_recall_mean"], color=color, linewidth=2, marker="o", markersize=5, label=f"{training} 학습")
            ax.fill_between(part["n_fit_mean"], part["val_recall_mean"] - part["val_recall_std"], part["val_recall_mean"] + part["val_recall_std"],
                            color=color, alpha=0.12, linewidth=0)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("학습 계정 수", fontsize=9, color=MUTED)
        ax.set_title(policy, loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("validation Recall (오탐률 5% 이하)", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(f"학습 계정 수에 따른 Recall (모델 seed {len(SELECT_SEEDS)}개 평균 ± 표준편차)", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "training_size.png")

    record = {
        "script": "scripts/tune_if_v2_fpr.py", "dataset": {"path": C_PATH.relative_to(REPO).as_posix(), "sha256": hash_before["C"]},
        "datasets_unchanged_after_run": hash_before == {"A": sha256(ORIGINAL_PATH), "C": sha256(C_PATH)},
        "features": F, "target_fpr": TARGET_FPR,
        "split": {"test": len(test_idx), "train": len(train_idx), "validation": len(val_idx), "random_state": SPLIT_SEED,
                  "train_normal": n_normal, "validation_normal": int((y_val == 0).sum()), "validation_allowed_false_positives": int(np.floor(TARGET_FPR * (y_val == 0).sum()))},
        "search_space": {"preprocessing": [policy_name(p) for p in POLICIES], "training": TRAININGS},
        "threshold_rule": "validation 정상 계정의 오탐률이 target 이하가 되는 가장 낮은 점수. test 에는 같은 모델·같은 임계값",
        "selection": {"criterion": f"validation Recall (모델 seed {len(SELECT_SEEDS)}개 평균), 동률이면 F1", "selected": [chosen_policy, chosen_training],
                      "selected_within_mixed_training": mixed_policy},
        "model_seeds": MODEL_SEEDS, "split_seeds": SPLIT_SEEDS, "nested_selection_model_seeds": NESTED_SELECT_SEEDS,
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]},
        "models_saved": False,
    }
    (OUT / "experiment_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cols = [f"{p}_{k}_{s}" for p in ("val", "test") for k in METRICS for s in ("mean", "std")]
    print("\n[선택용] validation (seed 10개 평균)\n", selection[keys + ["n_fit"] + [f"val_{k}" for k in METRICS]].round(3).to_string(index=False))
    print("\n[test seed 42]\n", seed42[keys + ["n_fit", "threshold"] + [f"test_{k}" for k in METRICS + ["TP", "FP", "TN", "FN", "insufficient"]]].round(3).to_string(index=False))
    print(seed42[keys + [f"test_{r}" for r in RATES]].round(2).to_string(index=False))
    print("\n[모델 seed 20개]\n", model_seed_summary[keys + cols + ["test_fpr_within_target_share"]].round(3).to_string(index=False))
    print(model_seed_summary[keys + [f"test_{r}_mean" for r in RATES]].round(2).to_string(index=False))
    print(model_seed_summary[keys + [f"test_c{c:g}_{k}_mean" for c in (TARGET_FPR, BEST_CONTAMINATION) for k in ("precision", "recall", "f1", "fpr")]].round(3).to_string(index=False))
    a_cols = [f"test_{k}" for k in METRICS]
    print("\nA 참고 seed 42:", a42[a_cols + [f"test_{r}" for r in RATES]].astype(float).round(3).to_dict())
    print("A 참고 seed 20개 평균:", reference_a[reference_a["seed"].isin(MODEL_SEEDS)][a_cols].mean().round(3).to_dict(),
          "std", reference_a[reference_a["seed"].isin(MODEL_SEEDS)][a_cols].std().round(3).to_dict())
    print("\n[학습 건수]\n", size_summary[["policy", "training", "requested_size", "n_fit_mean", "val_recall_mean", "val_recall_std", "val_f1_mean",
                                        "test_recall_mean", "test_fpr_mean", "test_f1_mean", "test_roc_auc_mean"]].round(3).to_string(index=False))
    print("\n[오탐 원인] validation\n", errors.round(3).T.to_string())
    print(criteria.round(3).to_string(index=False))
    print("오탐 중 기준 후보를 하나라도 충족:", none_met.to_dict("index"))
    print(correlation.pivot(index="feature", columns=["setting", "group"], values="spearman_with_score").round(2).to_string())
    print("\n[분할 seed]\n", split_summary[["setting"] + cols + ["test_fpr_within_target_share"]].round(3).to_string(index=False))
    print(split_summary[["setting"] + [f"test_{r}_mean" for r in RATES]].round(2).to_string(index=False))
    print(picks.value_counts(["policy", "training"]).to_string())
    print(record["datasets_unchanged_after_run"], f"저장: {OUT}")


if __name__ == "__main__":
    main()
