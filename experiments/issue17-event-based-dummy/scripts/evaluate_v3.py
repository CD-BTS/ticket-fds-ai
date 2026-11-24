"""이벤트 기반 v2 와 v3 에서 Isolation Forest 를 같은 조건으로 평가한다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\evaluate_v3.py

Feature 6개, 트리 수 100, max_samples 기본값은 그대로다. Ablation 은 하지 않고 새 Feature 도 만들지 않는다.
점수 계산·분할·임계값 규칙은 scripts/tune_if_v2.py, scripts/tune_if_v2_fpr.py 의 것을 import 해서 쓴다 (그 스크립트와 결과는 건드리지 않는다).

하는 일
  1. 기존 설정 고정: 결측 그대로, contamination 0.20, 700건 학습 / 같은 test 300건
  2. 오탐률 5% 조건 (train 490 / validation 210 / test 300, 임계값은 validation 정상 계정의 오탐률 5% 이하, test 에는 그대로 적용)
       - 기존 모델 (결측 그대로, 혼합 학습)
       - 전처리 개선 (중앙값 + log1p, 혼합 학습)
       - 정상만 학습: 정상 레이블을 알고 학습한다. 실제로는 쓸 수 없는 성능 상한이다
       - 저위험 계정 선별: 레이블 없이 학습 데이터를 고르는 방법 (Credit 규칙 통과 계정 / 기준 후보까지 통과한 계정 / Isolation Forest 자기 정제)
  3. 학습 데이터에 이상 계정이 섞이는 비율에 따른 민감도
  4. 모델 seed, 분할 seed, 데이터 생성 seed
  5. v3 의 규칙을 하나씩 껐을 때 (어떤 가정이 결과를 바꾸는지)

출력: output/v3/model_evaluation/
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tune_if_v2 import (BASELINE, BEST_CONTAMINATION, EXP, GRID, INK, MODEL_SEED, MUTED, NORMAL_SCENARIOS, SCENARIOS, SPLIT_SEED, SURFACE,  # noqa: E402
                        TEST_SIZE, VAL_SIZE, F, Scorer, evaluate, fit_and_score, flag_rates, mean_std, metrics, policy_name, save, sha256, style)
from tune_if_v2_fpr import TARGET_FPR, fpr_threshold  # noqa: E402
from extract_dataset import build_dataset  # noqa: E402
from generate_events import generate as generate_v2, load_config  # noqa: E402
from generate_events_v3 import CONFIG_PATH as V3_CONFIG, RULES, generate as generate_v3  # noqa: E402

OUT = EXP / "output" / "v3" / "model_evaluation"
V2_CONFIG = EXP / "config" / "scenarios_v2.json"
PATHS = {"v2": EXP / "data" / "v2", "v3": EXP / "data" / "v3"}
TUNED = (1, "median", True)
SELECT_SEEDS = list(range(10))
MODEL_SEEDS = list(range(20))
SPLIT_SEEDS = list(range(20))
GENERATION_SEEDS = list(range(10))
PURITY_SHARES = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
METRICS = ["precision", "recall", "f1", "fpr", "roc_auc"]
RATES = [f"flag_rate_{s}" for s in SCENARIOS]
C_V2, C_V3, C_BASE, C_UPPER, C_LOW = "#1baf7a", "#6250d6", "#6b7280", "#2a78d6", "#eb6834"


# ---------- 학습 데이터를 고르는 방법 ----------
def all_rows(frame, idx, seed):
    return idx


def normal_only(frame, idx, seed):
    return idx[(frame.loc[idx, "is_anomaly"] == 0).to_numpy()]


def credit_flags(frame, wide=False):
    """기존 문서의 Credit 규칙 중 6개 Feature 로 계산할 수 있는 3개 (REPORT_V2.md 1.1절). wide 는 기준 후보 2개를 더한다. 레이블을 쓰지 않는다."""
    flagged = (((frame["booking_interval_variance"] < 1.0) & (frame["booking_count"] >= 5)) | (frame["card_reuse_count"] >= 4)
               | (frame["ip_booking_density"] >= 5))
    if wide:
        flagged |= (frame["avg_booking_interval"] < 10) | (frame["cancel_rate"] >= 0.5)
    return flagged


def credit_pass(frame, idx, seed):
    return idx[~credit_flags(frame.loc[idx]).to_numpy()]


def credit_wide_pass(frame, idx, seed):
    return idx[~credit_flags(frame.loc[idx], wide=True).to_numpy()]


def self_trim(share):
    def select(frame, idx, seed):
        """혼합 데이터로 한 번 학습해 점수가 높은 계정을 빼고 다시 학습한다. 레이블을 쓰지 않는다."""
        first = Scorer(frame, idx, TUNED, seed)
        return idx[first.fit_scores <= np.quantile(first.fit_scores, 1 - share)]
    return select


SETTINGS = [  # (이름, 전처리, 학습 데이터 선택, 레이블 사용 여부)
    ("기존 모델 (결측 그대로, 혼합)", BASELINE, all_rows, False),
    ("전처리 개선 (혼합)", TUNED, all_rows, False),
    ("정상만 학습 (레이블 사용, 상한)", TUNED, normal_only, True),
    ("저위험: Credit 규칙 통과", TUNED, credit_pass, False),
    ("저위험: Credit + 기준 후보 통과", TUNED, credit_wide_pass, False),
    ("저위험: IF 자기 정제 (상위 10% 제외)", TUNED, self_trim(0.10), False),
    ("저위험: IF 자기 정제 (상위 20% 제외)", TUNED, self_trim(0.20), False),
    ("저위험: IF 자기 정제 (상위 30% 제외)", TUNED, self_trim(0.30), False),
]
KEY = [SETTINGS[i] for i in (0, 1, 2, 3, 6)]  # seed 반복에 쓰는 설정


def splits(frame, seed):
    y, scenario = frame["is_anomaly"], frame["scenario"]
    dev, test = train_test_split(frame.index, test_size=TEST_SIZE, random_state=seed, stratify=y)
    train, val = train_test_split(dev, test_size=VAL_SIZE, random_state=seed, stratify=scenario.loc[dev])
    return dev, train, val, test


def fpr_run(frame, train, val, test, policy, selector, seed):
    y, scenario = frame["is_anomaly"], frame["scenario"]
    fit_idx = selector(frame, train, seed)
    scorer = Scorer(frame, fit_idx, policy, seed)
    val_scores, test_scores = scorer.score(val), scorer.score(test)
    threshold = fpr_threshold(y.loc[val].to_numpy(), val_scores)
    v, _ = evaluate(y.loc[val].to_numpy(), val_scores, threshold)
    t, pred = evaluate(y.loc[test].to_numpy(), test_scores, threshold)
    train_normal, train_anomaly = (y.loc[train] == 0).sum(), (y.loc[train] == 1).sum()
    fit_anomaly = int(y.loc[fit_idx].sum())
    row = {"n_fit": len(fit_idx), "fit_anomaly_share": fit_anomaly / len(fit_idx), "train_normal_kept": (len(fit_idx) - fit_anomaly) / train_normal,
           "train_anomaly_removed": 1 - fit_anomaly / train_anomaly, **{f"val_{k}": v[k] for k in METRICS},
           **{f"test_{k}": t[k] for k in METRICS + ["TP", "FP", "TN", "FN"]}, **flag_rates(pred, scenario.loc[test].to_numpy())}
    return row, pred


def fixed_run(frame, dev, test, seed=MODEL_SEED):
    y, scenario = frame["is_anomaly"], frame["scenario"]
    pred, _, normalized = fit_and_score(frame, dev, test, F, BEST_CONTAMINATION, seed)
    return {**metrics(y.loc[test].to_numpy(), pred, normalized), **flag_rates(pred, scenario.loc[test].to_numpy())}


def single_feature_auc(frame, feature):
    part = frame.dropna(subset=[feature])
    auc = roc_auc_score(part["is_anomaly"], part[feature])
    return max(auc, 1 - auc)


def summary_row(frame, model_seed=MODEL_SEED, split_seed=SPLIT_SEED):
    """데이터셋 하나를 요약한다 (생성 seed·규칙 끄기 반복용): 단일 Feature AUC, 기존 설정, 오탐률 5% 조건의 주요 설정."""
    dev, train, val, test = splits(frame, split_seed)
    fixed = fixed_run(frame, dev, test, model_seed)
    row = {"auc_avg_booking_interval": single_feature_auc(frame, "avg_booking_interval"), "auc_booking_count": single_feature_auc(frame, "booking_count"),
           "null_rows": int(frame[F].isna().any(axis=1).sum()), **{f"fixed_{k}": fixed[k] for k in METRICS}}
    for name, policy, selector, _ in KEY:
        result, _ = fpr_run(frame, train, val, test, policy, selector, model_seed)
        row.update({f"{name} | {k}": result[f"test_{k}"] for k in METRICS})
        row[f"{name} | fit_anomaly_share"] = result["fit_anomaly_share"]
    return row


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_rows", 500)
    files = {name: folder / "fairticket_event_based_features.csv" for name, folder in PATHS.items()}
    before = {name: sha256(path) for name, path in files.items()}
    data = {name: pd.read_csv(path) for name, path in files.items()}
    assert data["v2"]["user_id"].equals(data["v3"]["user_id"]) and data["v2"]["is_anomaly"].equals(data["v3"]["is_anomaly"])
    accounts = pd.read_csv(PATHS["v3"] / "accounts.csv", keep_default_na=False)
    dev, train, val, test = splits(data["v3"], SPLIT_SEED)  # 계정과 레이블이 같으므로 두 데이터셋에 같은 분할

    def csv(frame, name):
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    # ---- 1) 기존 설정 고정 ----
    rows = [{"dataset": name, "seed": seed, **fixed_run(frame, dev, test, seed)} for name, frame in data.items() for seed in sorted(set(MODEL_SEEDS) | {MODEL_SEED})]
    fixed = pd.DataFrame(rows)
    csv(fixed, "fixed_setting_runs.csv")
    fixed42 = fixed[fixed["seed"] == MODEL_SEED].drop(columns="seed").reset_index(drop=True)
    fixed_summary = mean_std(fixed[fixed["seed"].isin(MODEL_SEEDS)], ["dataset"], METRICS + RATES)
    csv(fixed_summary, "fixed_setting_summary.csv")
    print("[1] 기존 설정 (seed 42)\n", fixed42[["dataset"] + METRICS + ["TP", "FP", "TN", "FN"] + RATES].round(3).to_string(index=False), flush=True)

    # ---- 2) 오탐률 5% 조건 ----
    rows, predictions = [], {}
    for name, frame in data.items():
        for label, policy, selector, uses_labels in SETTINGS:
            for seed in sorted(set(MODEL_SEEDS) | {MODEL_SEED}):
                row, pred = fpr_run(frame, train, val, test, policy, selector, seed)
                rows.append({"dataset": name, "setting": label, "uses_labels": uses_labels, "preprocessing": policy_name(policy), "seed": seed, **row})
                if seed in SELECT_SEEDS:
                    predictions.setdefault((name, label), []).append(pred)
    runs = pd.DataFrame(rows)
    csv(runs, "fpr5_runs.csv")
    numeric = [c for c in runs.columns if c not in ("dataset", "setting", "uses_labels", "preprocessing", "seed")]
    fpr5 = mean_std(runs[runs["seed"].isin(MODEL_SEEDS)], ["dataset", "setting", "uses_labels"], numeric)
    fpr5["test_fpr_within_target_share"] = (runs[runs["seed"].isin(MODEL_SEEDS)].assign(ok=lambda d: d["test_fpr"] <= TARGET_FPR)
                                            .groupby(["dataset", "setting"], sort=False)["ok"].mean().to_numpy())
    csv(fpr5, "fpr5_summary.csv")
    csv(runs[runs["seed"] == MODEL_SEED].drop(columns="seed"), "fpr5_seed42.csv")
    show = ["dataset", "setting", "n_fit_mean", "fit_anomaly_share_mean", "train_normal_kept_mean", "train_anomaly_removed_mean"]
    print("\n[2] 오탐률 5% 조건 (모델 seed 20개 평균)\n", fpr5[show + [f"test_{k}_{s}" for k in METRICS for s in ("mean", "std")] + ["test_fpr_within_target_share"]].round(3).to_string(index=False))
    print(fpr5[["dataset", "setting"] + [f"{r}_mean" for r in RATES]].round(2).to_string(index=False), flush=True)

    # 시나리오 밖의 특성별 오탐·미탐 (v3, test, 모델 seed 10개 평균)
    info = accounts.set_index("user_id").loc[data["v3"].loc[test, "user_id"]]
    part = data["v3"].loc[test]
    traits = pd.DataFrame({"is_anomaly": part["is_anomaly"].to_numpy(), "scenario": part["scenario"].to_numpy(),
                           "오픈 참여": np.where(info["n_sessions"].to_numpy() >= 2, "2회 이상", "1회"),
                           "시도 수": pd.cut(part["booking_count"], [0, 1, 4, 9, 1000], labels=["1건", "2~4건", "5~9건", "10건 이상"]).astype(str).to_numpy(),
                           "매크로 도구 프로필": info["tool_profile"].replace("", "(도구 아님)").to_numpy()})
    rows = []
    for label in [s[0] for s in KEY]:
        rate = np.mean(predictions[("v3", label)], axis=0)
        for column in ("오픈 참여", "시도 수", "매크로 도구 프로필"):
            for (flag, value), index in traits.groupby(["is_anomaly", column]).indices.items():
                rows.append({"setting": label, "group": "이상 (Recall)" if flag else "정상 (오탐률)", "trait": column, "value": value, "n_test": len(index),
                             "flag_rate": float(rate[index].mean())})
    trait_table = pd.DataFrame(rows)
    csv(trait_table, "v3_error_by_trait.csv")
    print("\n[2b] v3 특성별 이상 판정 비율\n", trait_table.pivot_table(index=["group", "trait", "value", "n_test"], columns="setting", values="flag_rate").round(2).to_string(), flush=True)

    # ---- 3) 학습 데이터의 이상 계정 비율 ----
    rows = []
    for name, frame in data.items():
        y = frame["is_anomaly"]
        normal_train, anomaly_train = train[(y.loc[train] == 0).to_numpy()], train[(y.loc[train] == 1).to_numpy()]
        for share in PURITY_SHARES:
            count = min(int(round(share / (1 - share) * len(normal_train))), len(anomaly_train))
            for seed in SELECT_SEEDS:
                extra = anomaly_train if count == len(anomaly_train) else (
                    train_test_split(anomaly_train, train_size=count, random_state=seed, stratify=frame["scenario"].loc[anomaly_train])[0] if count else anomaly_train[:0])
                row, _ = fpr_run(frame, train, val, test, TUNED, lambda f, i, s, rows_=normal_train.append(extra): rows_, seed)
                rows.append({"dataset": name, "anomaly_share_in_training": share, "anomalies_in_training": count, "seed": seed, **row})
    purity = pd.DataFrame(rows).groupby(["dataset", "anomaly_share_in_training", "anomalies_in_training"], sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
    csv(purity, "training_purity.csv")
    print("\n[3] 학습 데이터의 이상 계정 비율\n", purity[["dataset", "anomaly_share_in_training", "anomalies_in_training", "n_fit"] + [f"test_{k}" for k in METRICS]].round(3).to_string(index=False), flush=True)

    # ---- 4a) 분할 seed ----
    rows = []
    for split_seed in SPLIT_SEEDS:
        for name, frame in data.items():
            d, tr, va, te = splits(frame, split_seed)
            rows.append({"dataset": name, "setting": "기존 설정 고정 (c=0.20)", "split_seed": split_seed, **{f"test_{k}": v for k, v in fixed_run(frame, d, te).items() if k in METRICS}})
            for label, policy, selector, _ in KEY:
                row, _ = fpr_run(frame, tr, va, te, policy, selector, MODEL_SEED)
                rows.append({"dataset": name, "setting": label, "split_seed": split_seed, **row})
    split_runs = pd.DataFrame(rows)
    csv(split_runs, "split_seed_runs.csv")
    split_summary = mean_std(split_runs, ["dataset", "setting"], [f"test_{k}" for k in METRICS])
    split_summary["test_fpr_within_target_share"] = split_runs.assign(ok=lambda d: d["test_fpr"] <= TARGET_FPR).groupby(["dataset", "setting"], sort=False)["ok"].mean().to_numpy()
    csv(split_summary, "split_seed_summary.csv")
    print("\n[4a] 분할 seed 20개\n", split_summary[["dataset", "setting"] + [f"test_{k}_{s}" for k in METRICS for s in ("mean", "std")] + ["test_fpr_within_target_share"]].round(3).to_string(index=False), flush=True)

    # ---- 4b) 데이터 생성 seed ----
    config2, config3 = load_config(V2_CONFIG), load_config(V3_CONFIG)
    rows = []
    for seed in GENERATION_SEEDS:
        events, generated_accounts, _ = generate_v2(config2, "single_session", seed)
        rows.append({"dataset": "v2", "generation_seed": seed, **summary_row(build_dataset(events, generated_accounts, config2)[0])})
        events, generated_accounts, _ = generate_v3(config3, seed)
        rows.append({"dataset": "v3", "generation_seed": seed, **summary_row(build_dataset(events, generated_accounts, config3)[0])})
        print(f"생성 seed {seed} 완료", flush=True)
    generation = pd.DataFrame(rows)
    csv(generation, "generation_seed_runs.csv")
    value_cols = [c for c in generation.columns if c not in ("dataset", "generation_seed")]
    generation_summary = generation.groupby("dataset", sort=False)[value_cols].agg(["mean", "std", "min", "max"]).stack(level=0, future_stack=True).reset_index().rename(columns={"level_1": "quantity"})
    csv(generation_summary, "generation_seed_summary.csv")
    print("\n[4b] 데이터 생성 seed 10개\n", generation_summary.round(3).to_string(index=False), flush=True)

    # ---- 5) v3 의 규칙을 하나씩 껐을 때 (seed 42) ----
    rules = {c["id"]: c["change"] for c in config3["changes_from_v2"]}
    variants = [("v3 (규칙 전부)", ())] + [(f"{rule} 끔", (rule,)) for rule in RULES] + [("전부 끔 (v2 의 규칙)", RULES)]
    rows = []
    for label, off in variants:
        events, generated_accounts, _ = generate_v3(config3, config3["seed"], off=off)
        rows.append({"variant": label, "rule_switched_off": "; ".join(rules[r] for r in off) if len(off) == 1 else "", **summary_row(build_dataset(events, generated_accounts, config3)[0])})
        print(f"{label} 완료", flush=True)
    rule_table = pd.DataFrame(rows)
    csv(rule_table, "rule_switch_off.csv")
    print("\n[5] 규칙 끄기\n", rule_table.drop(columns="rule_switched_off").round(3).T.to_string(), flush=True)

    # ---------------- 그래프 ----------------
    names = [s[0] for s in SETTINGS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for ax, (key, title) in zip(axes, (("test_recall", "Recall"), ("test_f1", "F1"), ("test_fpr", "test 오탐률 (점선 = 목표 5%)"))):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        position = np.arange(len(names))[::-1]
        for offset, dataset, color in ((0.2, "v2", C_V2), (-0.2, "v3", C_V3)):
            part = fpr5[fpr5["dataset"] == dataset].set_index("setting").loc[names]
            ax.barh(position + offset, part[f"{key}_mean"], height=0.37, color=color, label=f"이벤트 기반 {dataset}")
            for p, value in zip(position + offset, part[f"{key}_mean"]):
                ax.text(value + (0.003 if key == "test_fpr" else 0.012), p, f"{value:.2f}", va="center", fontsize=7, color=INK)
        if key == "test_fpr":
            ax.axvline(TARGET_FPR, color=INK, linewidth=1, linestyle="--")
            ax.set_xlim(0, 0.14)
        else:
            ax.set_xlim(0, 1.12)
        ax.set_yticks(position)
        ax.set_yticklabels(names if ax is axes[0] else [], fontsize=8)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("오탐률 5% 조건: validation 에서 정한 임계값을 test 300건에 적용 (모델 seed 20개 평균)", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "fpr5_v2_v3.png")

    counts = data["v3"].loc[test, "scenario"].value_counts()
    chosen = [("기존 모델 (결측 그대로, 혼합)", C_BASE), ("정상만 학습 (레이블 사용, 상한)", C_UPPER), ("저위험: Credit 규칙 통과", C_LOW)]
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8), constrained_layout=True)
    for ax, dataset in zip(axes, ("v2", "v3")):
        style(ax)
        x = np.arange(len(SCENARIOS))
        for i, (label, color) in enumerate(chosen):
            values = fpr5[(fpr5["dataset"] == dataset) & (fpr5["setting"] == label)][[f"{r}_mean" for r in RATES]].to_numpy().ravel()
            ax.bar(x + (i - 1) * 0.27, values, width=0.25, color=color, label=label)
            for xi, value in zip(x + (i - 1) * 0.27, values):
                ax.text(xi, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n({'정상' if s in NORMAL_SCENARIOS else '이상'}, n={counts[s]})" for s in SCENARIOS], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"이벤트 기반 {dataset}: 시나리오별 이상 판정 비율 (정상 = 오탐률, 이상 = Recall)", loc="left", fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    save(fig, OUT / "figures" / "scenario_detection.png")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True, constrained_layout=True)
    for ax, (key, title) in zip(axes, (("test_recall", "Recall"), ("test_f1", "F1"))):
        style(ax)
        for dataset, color in (("v2", C_V2), ("v3", C_V3)):
            part = purity[purity["dataset"] == dataset]
            ax.plot(part["anomaly_share_in_training"] * 100, part[key], color=color, linewidth=2, marker="o", markersize=5, label=f"이벤트 기반 {dataset}")
        ax.set_xlabel("학습 데이터 중 이상 계정 (%)", fontsize=9, color=MUTED)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("정상 학습 데이터에 이상 계정이 섞일 때 (오탐률 5% 조건, 모델 seed 10개 평균)", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "training_purity.png")

    rng = np.random.default_rng(0)
    panels = [("auc_avg_booking_interval", "평균 간격 하나의 AUC"), ("fixed_f1", "기존 설정 고정의 F1"),
              ("정상만 학습 (레이블 사용, 상한) | f1", "정상만 학습의 F1 (오탐률 5%)"), ("저위험: Credit 규칙 통과 | f1", "Credit 규칙 통과 학습의 F1 (오탐률 5%)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4.2), sharey=True, constrained_layout=True)
    for ax, (key, title) in zip(axes, panels):
        style(ax)
        for position, (dataset, color) in enumerate((("v2", C_V2), ("v3", C_V3))):
            values = generation.loc[generation["dataset"] == dataset, key].to_numpy()
            ax.scatter(position + rng.uniform(-0.15, 0.15, len(values)), values, s=24, color=color, edgecolor=SURFACE, linewidth=0.6)
            ax.hlines(values.mean(), position - 0.3, position + 0.3, color=INK, linewidth=1.4)
            ax.text(position + 0.33, values.mean(), f"{values.mean():.3f}", va="center", fontsize=8, color=INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["v2", "v3"], fontsize=9)
        ax.set_xlim(-0.6, 1.8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.suptitle(f"데이터 생성 seed {len(GENERATION_SEEDS)}개 (점 = 데이터셋 1개, 가로선 = 평균)", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "generation_seeds.png")

    panels = [("auc_avg_booking_interval", "평균 간격 하나의 AUC"), ("auc_booking_count", "시도 횟수 하나의 AUC"), ("fixed_f1", "기존 설정 고정의 F1"),
              ("정상만 학습 (레이블 사용, 상한) | recall", "정상만 학습의 Recall (오탐률 5%)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4.4), sharey=True, constrained_layout=True)
    for ax, (key, title) in zip(axes, panels):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        position = np.arange(len(rule_table))[::-1]
        colors = [C_V3] + [MUTED] * len(RULES) + [C_V2]
        ax.barh(position, rule_table[key], height=0.6, color=colors)
        for p, value in zip(position, rule_table[key]):
            ax.text(value + 0.012, p, f"{value:.2f}", va="center", fontsize=8, color=INK)
        ax.set_yticks(position)
        ax.set_yticklabels(rule_table["variant"], fontsize=9)
        ax.set_xlim(0, 1.12)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.suptitle("v3 의 규칙을 하나씩 껐을 때 (seed 42). S1 오픈 참여, S2 매크로 시도 수, S3 정상 시도 수, S4 링의 도구 프로필, S5 선점 거절, S6 결제 오류",
                 fontsize=10, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "rule_switch_off.png")

    record = {
        "script": "scripts/evaluate_v3.py", "datasets": {name: {"path": path.relative_to(EXP).as_posix(), "sha256": before[name]} for name, path in files.items()},
        "datasets_unchanged_after_run": before == {name: sha256(path) for name, path in files.items()},
        "features": F, "model": {"class": "sklearn.ensemble.IsolationForest", "n_estimators": 100, "max_samples": "auto"},
        "fixed_setting": {"preprocessing": "없음 (결측 그대로)", "contamination": BEST_CONTAMINATION, "train": len(dev), "test": len(test)},
        "fpr5_protocol": {"target_fpr": TARGET_FPR, "train": len(train), "validation": len(val), "test": len(test),
                          "threshold": "validation 정상 계정의 오탐률이 target 이하가 되는 가장 낮은 점수. test 에는 같은 모델·같은 임계값"},
        "settings": [{"name": s[0], "preprocessing": policy_name(s[1]), "uses_labels_for_training_data": s[3]} for s in SETTINGS],
        "credit_rules_used": ["간격 분산 < 1.0초² 이고 시도 5건 이상", "카드 공유 4계정 이상", "IP 공유 5계정 이상"],
        "credit_candidates_added_in_wide": ["평균 간격 < 10초", "취소율 ≥ 50%"],
        "model_seeds": MODEL_SEEDS, "split_seeds": SPLIT_SEEDS, "generation_seeds": GENERATION_SEEDS, "purity_shares": PURITY_SHARES,
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]},
        "models_saved": False,
    }
    (OUT / "experiment_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(record["datasets_unchanged_after_run"], f"저장: {OUT}")


if __name__ == "__main__":
    main()
