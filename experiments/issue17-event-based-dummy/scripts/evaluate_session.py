"""v3 이벤트에서 간격 Feature 2개를 세션 기준으로 바꿨을 때의 분포와 Isolation Forest 성능을 기존 추출 결과와 비교한다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\evaluate_session.py

- 기준 모델: 혼합 학습(레이블 미사용) + 결측 중앙값 대체 + log1p, n_estimators 100, max_samples 256 (기존 기본값 'auto' 와 같다)
- 임계값: validation 정상 계정의 오탐률이 5% 이하가 되는 가장 낮은 점수. test 에는 같은 모델·같은 임계값
- 비교: Feature 추출 방식 5가지 (기존 30일 전체 / 세션 pooled / median / min / latest) × n_estimators (100, 200, 300) × max_samples (64, 128, 256)
- 설정은 validation Recall(모델 seed 10개 평균, 동률이면 F1, 그래도 같으면 기준 모델에 가까운 쪽)로만 고른다. test 는 고른 뒤의 평가에만 쓴다.
- Feature 는 6개 그대로다. 간격 2개의 집계 방식만 다르다. Ablation 은 하지 않는다.
  "간격만 / 분산만 세션 기준" 은 Feature 를 빼는 것이 아니라 두 변경 중 어느 쪽이 기여했는지 나눠 보는 것이다.
- 보조 분석: 같은 Feature 를 학습 데이터 구성만 바꿔 평가한다 (혼합 / 레이블 없이 고른 저위험 계정 / 정상만 = 상한).
  기준 모델의 조건(혼합 학습)에서 세션 Feature 가 왜 효과가 없는지 원인을 나눠 보기 위한 것이며, 기준 모델의 선택에는 쓰지 않는다.

출력: output/v3_session/
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
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tune_if_v2 import (EXP, GRID, INK, MODEL_SEED, MUTED, NORMAL_SCENARIOS, SCENARIOS, SPLIT_SEED, SURFACE, TEST_SIZE, VAL_SIZE, F, evaluate,  # noqa: E402
                        flag_rates, mean_std, save, sha256, style, transformers)
from tune_if_v2_fpr import TARGET_FPR, fpr_threshold  # noqa: E402
from evaluate_v3 import all_rows, credit_pass, credit_wide_pass, normal_only, self_trim  # noqa: E402
from extract_dataset import build_dataset  # noqa: E402
from extract_session_features import AGGREGATIONS, INTERVAL, VARIANCE, build as build_session  # noqa: E402
from generate_events import load_config  # noqa: E402
from generate_events_v3 import CONFIG_PATH, generate  # noqa: E402

OUT = EXP / "output" / "v3_session"
EXISTING = "기존 (30일 전체)"
NAMES = {"pooled": "세션: 모든 간격", "median": "세션 평균의 중앙값", "min": "세션 평균의 최솟값", "latest": "최근 세션"}
TUNED = (1, "median", True)
N_ESTIMATORS, MAX_SAMPLES = [100, 200, 300], [64, 128, 256]
DEFAULT_HP = (100, 256)
SELECT_SEEDS, MODEL_SEEDS, SPLIT_SEEDS, GENERATION_SEEDS = list(range(10)), list(range(20)), list(range(20)), list(range(10))
METRICS = ["precision", "recall", "f1", "fpr", "roc_auc"]
RATES = [f"flag_rate_{s}" for s in SCENARIOS]
GROUPS = ["recall_single_open", "recall_multi_open", "recall_macro_single_open", "recall_macro_multi_open", "fpr_single_open", "fpr_multi_open"]
MACRO = ["fast", "jitter", "multi_account"]
C_OLD, C_NEW, C_NORMAL, C_ANOM, C_LOW = "#6b7280", "#6250d6", "#2a78d6", "#eb6834", "#1baf7a"
TRAININGS = [("혼합", all_rows, False), ("저위험: Credit 규칙 통과", credit_pass, False), ("저위험: Credit + 기준 후보 통과", credit_wide_pass, False),
             ("저위험: IF 자기 정제 (상위 20% 제외)", self_trim(0.20), False), ("정상만 (레이블 사용, 상한)", normal_only, True)]


class Scorer:
    """혼합 학습 + 중앙값 대체 + log1p (tune_if_v2 의 전처리 그대로) 에 트리 수와 max_samples 만 바꿀 수 있게 한 것."""

    def __init__(self, frame, fit_idx, n_estimators, max_samples, seed):
        self.frame = frame
        self.imputer, self.log1p = transformers(TUNED)
        x = frame.loc[fit_idx, F].to_numpy(dtype=float)
        self.imputer.fit(x)  # 결측 대체값은 학습 데이터에서만 계산한다
        self.model = IsolationForest(contamination="auto", random_state=seed, n_estimators=n_estimators, max_samples=max_samples).fit(self.prepare(x))

    def prepare(self, x):
        return self.log1p.transform(self.imputer.transform(np.asarray(x, dtype=float)))

    def score(self, idx):
        return -self.model.score_samples(self.prepare(self.frame.loc[idx, F].to_numpy(dtype=float)))


def splits(frame, seed):
    y, scenario = frame["is_anomaly"], frame["scenario"]
    dev, test = train_test_split(frame.index, test_size=TEST_SIZE, random_state=seed, stratify=y)
    train, val = train_test_split(dev, test_size=VAL_SIZE, random_state=seed, stratify=scenario.loc[dev])
    return train, val, test


def run(frame, opens, train, val, test, hp, seed, selector=all_rows):
    y, scenario = frame["is_anomaly"], frame["scenario"]
    fit_idx = selector(frame, train, seed)
    scorer = Scorer(frame, fit_idx, hp[0], hp[1], seed)
    val_scores, test_scores = scorer.score(val), scorer.score(test)
    threshold = fpr_threshold(y.loc[val].to_numpy(), val_scores)
    v, _ = evaluate(y.loc[val].to_numpy(), val_scores, threshold)
    t, pred = evaluate(y.loc[test].to_numpy(), test_scores, threshold)
    yt, st, multi = y.loc[test].to_numpy(), scenario.loc[test].to_numpy(), opens.loc[test].to_numpy() >= 2
    macro = np.isin(st, MACRO)
    rate = lambda mask: float(pred[mask].mean()) if mask.any() else np.nan
    groups = {"recall_single_open": rate((yt == 1) & ~multi), "recall_multi_open": rate((yt == 1) & multi),
              "recall_macro_single_open": rate(macro & ~multi), "recall_macro_multi_open": rate(macro & multi),
              "fpr_single_open": rate((yt == 0) & ~multi), "fpr_multi_open": rate((yt == 0) & multi)}
    fit_anomaly = int(y.loc[fit_idx].sum())
    row = {"n_fit": len(fit_idx), "fit_anomaly_share": fit_anomaly / len(fit_idx), "train_anomaly_removed": 1 - fit_anomaly / int(y.loc[train].sum()),
           "train_normal_kept": (len(fit_idx) - fit_anomaly) / int((y.loc[train] == 0).sum()),
           **{f"val_{k}": v[k] for k in METRICS}, **{f"test_{k}": t[k] for k in METRICS + ["TP", "FP", "TN", "FN"]}, **flag_rates(pred, st), **groups}
    return row, pred


def single_auc(frame, feature, anomaly_mask=None):
    part = frame.dropna(subset=[feature])
    if anomaly_mask is not None:
        part = part[(part["is_anomaly"] == 0) | anomaly_mask.loc[part.index]]
    if part["is_anomaly"].nunique() < 2:
        return np.nan
    auc = roc_auc_score(part["is_anomaly"], part[feature])
    return max(auc, 1 - auc)


def select(summary):
    ranked = summary.assign(is_default=(summary["n_estimators"] == DEFAULT_HP[0]) & (summary["max_samples"] == DEFAULT_HP[1])).sort_values(
        ["val_recall", "val_f1", "is_default"], ascending=False)
    top = ranked.iloc[0]
    return top["features"], (int(top["n_estimators"]), int(top["max_samples"]))


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_rows", 500)
    paths = {EXISTING: EXP / "data" / "v3" / "fairticket_event_based_features.csv",
             **{NAMES[a]: EXP / "data" / "v3_session" / f"fairticket_event_based_features_{a}.csv" for a in AGGREGATIONS}}
    before = {name: sha256(path) for name, path in paths.items()}
    data = {name: pd.read_csv(path) for name, path in paths.items()}
    base = data[EXISTING]
    opens = pd.read_csv(EXP / "data" / "v3" / "accounts.csv", keep_default_na=False)["n_sessions"]
    train, val, test = splits(base, SPLIT_SEED)

    def csv(frame, name):
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    # ---- 1) 분포 ----
    rows = []
    for name, frame in data.items():
        for feature in (INTERVAL, VARIANCE):
            for label, flag in (("정상", 0), ("이상", 1)):
                for group, mask in (("전체", opens >= 0), ("오픈 1회", opens == 1), ("오픈 2회 이상", opens >= 2)):
                    part = frame.loc[(frame["is_anomaly"] == flag) & mask, feature]
                    q = part.dropna().quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_numpy()
                    rows.append({"features": name, "feature": feature, "label": label, "opens": group, "accounts": len(part), "null": int(part.isna().sum()),
                                 "null_share": float(part.isna().mean()), **dict(zip(["p05", "p25", "median", "p75", "p95"], q))})
    distribution = pd.DataFrame(rows)
    csv(distribution, "distribution.csv")
    rows = []
    for name, frame in data.items():
        for feature in (INTERVAL, VARIANCE):
            row = {"features": name, "feature": feature, "auc_all": single_auc(frame, feature),
                   "auc_anomaly_single_open": single_auc(frame, feature, opens == 1), "auc_anomaly_multi_open": single_auc(frame, feature, opens >= 2)}
            row.update({f"auc_{s}": single_auc(frame, feature, frame["scenario"] == s) for s in SCENARIOS[4:]})
            rows.append(row)
    separability = pd.DataFrame(rows)
    csv(separability, "separability.csv")
    print("[1] 분포\n", distribution[distribution["opens"] != "오픈 1회"].round(3).to_string(index=False))
    print(separability.round(3).to_string(index=False), flush=True)

    # ---- 2) 격자: validation 으로만 고른다 ----
    rows = []
    for name, frame in data.items():
        for n_estimators in N_ESTIMATORS:
            for max_samples in MAX_SAMPLES:
                for seed in SELECT_SEEDS:
                    row, _ = run(frame, opens, train, val, test, (n_estimators, max_samples), seed)
                    rows.append({"features": name, "n_estimators": n_estimators, "max_samples": max_samples, "seed": seed, **row})
    grid_runs = pd.DataFrame(rows)
    csv(grid_runs, "grid_runs.csv")
    keys = ["features", "n_estimators", "max_samples"]
    grid = grid_runs.groupby(keys, sort=False)[[f"val_{k}" for k in METRICS]].mean().reset_index()
    grid["val_recall_std"] = grid_runs.groupby(keys, sort=False)["val_recall"].std().to_numpy()
    csv(grid, "validation_grid.csv")
    chosen_features, chosen_hp = select(grid)
    _, existing_hp = select(grid[grid["features"] == EXISTING])
    best_hp = {name: select(grid[grid["features"] == name])[1] for name in data}
    print(f"\n[2] validation 선택: {chosen_features}, n_estimators {chosen_hp[0]}, max_samples {chosen_hp[1]}  (기존 Feature 안에서: {existing_hp})")
    print(grid.pivot_table(index=["features"], columns=["n_estimators", "max_samples"], values="val_recall").reindex(list(data)).round(3).to_string(), flush=True)

    # 세션 집계 가운데 validation 이 가장 높은 것. 전체 선택과 별개로, 기존 Feature 와 비교할 세션 Feature 로 쓴다
    session_name, session_hp = select(grid[grid["features"] != EXISTING])
    only_interval, only_variance = base.copy(), base.copy()
    only_interval[INTERVAL] = data[session_name][INTERVAL]
    only_variance[VARIANCE] = data[session_name][VARIANCE]
    swapped = {"간격만 세션 기준 (분산은 기존)": only_interval, "분산만 세션 기준 (간격은 기존)": only_variance}
    frames = {**data, **swapped}
    print(f"세션 집계 중 validation 최고: {session_name}, n_estimators {session_hp[0]}, max_samples {session_hp[1]}", flush=True)

    # ---- 3) 기준 모델의 조건(혼합 학습)에서의 최종 평가 (test, 모델 seed 20개) ----
    final = [("기준 모델: 기존 Feature, 기본 설정", EXISTING, DEFAULT_HP), ("기존 Feature, 트리 설정만 선택", EXISTING, existing_hp)]
    final += [(f"{name}, 기본 설정", name, DEFAULT_HP) for name in frames if name != EXISTING]
    final += [(f"{session_name}, n_estimators {session_hp[0]}, max_samples {session_hp[1]}", session_name, session_hp)]
    rows, predictions = [], {}
    for label, name, hp in final:
        for seed in sorted(set(MODEL_SEEDS) | {MODEL_SEED}):
            row, pred = run(frames[name], opens, train, val, test, hp, seed)
            rows.append({"setting": label, "features": name, "n_estimators": hp[0], "max_samples": hp[1], "seed": seed, **row})
            if seed in MODEL_SEEDS:
                predictions.setdefault(label, []).append(pred)
    test_runs = pd.DataFrame(rows)
    csv(test_runs, "test_runs.csv")
    numeric = [c for c in test_runs.columns if c not in ("setting", "features", "n_estimators", "max_samples", "seed")]
    summary = mean_std(test_runs[test_runs["seed"].isin(MODEL_SEEDS)], ["setting", "features", "n_estimators", "max_samples"], numeric)
    summary["test_fpr_within_target_share"] = (test_runs[test_runs["seed"].isin(MODEL_SEEDS)].assign(ok=lambda d: d["test_fpr"] <= TARGET_FPR)
                                               .groupby("setting", sort=False)["ok"].mean().to_numpy())
    csv(summary, "test_summary.csv")
    csv(test_runs[test_runs["seed"] == MODEL_SEED].drop(columns="seed"), "test_seed42.csv")
    baseline_label, session_label = final[0][0], final[-1][0]
    cols = [f"test_{k}_{s}" for k in METRICS for s in ("mean", "std")]
    print("\n[3] 혼합 학습, test (모델 seed 20개 평균)\n", summary[["setting", "val_recall_mean"] + cols + ["test_fpr_within_target_share"]].round(3).to_string(index=False))
    print(summary[["setting"] + [f"{r}_mean" for r in RATES]].round(2).to_string(index=False))
    print(summary[["setting"] + [f"{g}_mean" for g in GROUPS]].round(3).to_string(index=False), flush=True)

    # ---- 3b) 보조: 학습 데이터 구성 × Feature (기본 트리 설정, 모델 seed 20개) ----
    rows = []
    for name, frame in frames.items():
        for training, selector, uses_labels in TRAININGS:
            for seed in MODEL_SEEDS:
                row, pred = run(frame, opens, train, val, test, DEFAULT_HP, seed, selector)
                rows.append({"features": name, "training": training, "uses_labels": uses_labels, "seed": seed, **row})
                predictions.setdefault((name, training), []).append(pred)
    composition_runs = pd.DataFrame(rows)
    csv(composition_runs, "training_composition_runs.csv")
    composition = mean_std(composition_runs, ["features", "training", "uses_labels"], numeric)
    composition["test_fpr_within_target_share"] = (composition_runs.assign(ok=lambda d: d["test_fpr"] <= TARGET_FPR)
                                                   .groupby(["features", "training"], sort=False)["ok"].mean().to_numpy())
    csv(composition, "training_composition_summary.csv")
    print("\n[3b] 학습 데이터 구성 × Feature\n", composition[["features", "training", "n_fit_mean", "fit_anomaly_share_mean", "train_anomaly_removed_mean",
                                                    "train_normal_kept_mean", "val_recall_mean"] + cols + ["test_fpr_within_target_share"]].round(3).to_string(index=False))
    print(composition[["features", "training"] + [f"{r}_mean" for r in RATES]].round(2).to_string(index=False))
    print(composition[["features", "training"] + [f"{g}_mean" for g in GROUPS]].round(3).to_string(index=False), flush=True)

    # ---- 3c) 참고: 평균 간격 하나에 임계값 하나를 둔 규칙 (Isolation Forest 가 아니다. 같은 절차로 validation 에서 임계값을 정한다) ----
    rows = []
    for name in (EXISTING, session_name):
        frame = data[name]
        score = -frame[INTERVAL].fillna(1e12)  # 간격이 짧을수록 이상. 결측 계정은 판정하지 않는다
        for split_seed in sorted(set(SPLIT_SEEDS) | {SPLIT_SEED}):
            tr, va, te = splits(frame, split_seed)
            threshold = fpr_threshold(frame["is_anomaly"].loc[va].to_numpy(), score.loc[va].to_numpy())
            result, pred = evaluate(frame["is_anomaly"].loc[te].to_numpy(), score.loc[te].to_numpy(), threshold)
            rows.append({"features": name, "split_seed": split_seed, "threshold_seconds": -threshold, **{f"test_{k}": result[k] for k in METRICS},
                         **flag_rates(pred, frame["scenario"].loc[te].to_numpy())})
    rule_runs = pd.DataFrame(rows)
    csv(rule_runs, "single_feature_rule_runs.csv")
    rule_summary = pd.concat([rule_runs[rule_runs["split_seed"] == SPLIT_SEED].assign(splits=f"seed {SPLIT_SEED} 분할").drop(columns="split_seed"),
                              rule_runs[rule_runs["split_seed"].isin(SPLIT_SEEDS)].groupby("features", sort=False).mean(numeric_only=True).reset_index()
                              .assign(splits=f"분할 seed {len(SPLIT_SEEDS)}개 평균").drop(columns="split_seed")], ignore_index=True)
    csv(rule_summary, "single_feature_rule_summary.csv")
    print("\n[3c] 참고: 평균 간격 하나의 임계값 규칙\n", rule_summary.round(3).to_string(index=False), flush=True)

    # 오픈 참여 × 시나리오, 시도 수별
    credit = TRAININGS[1][0]
    compared = [("기준 모델 (기존 Feature, 혼합)", predictions[baseline_label]), ("세션 Feature, 혼합", predictions[session_label]),
                ("기존 Feature, Credit 규칙 통과 학습", predictions[(EXISTING, credit)]), ("세션 Feature, Credit 규칙 통과 학습", predictions[(session_name, credit)])]
    part = base.loc[test]
    traits = pd.DataFrame({"is_anomaly": part["is_anomaly"].map({0: "정상", 1: "이상"}).to_numpy(), "scenario": part["scenario"].to_numpy(),
                           "opens": np.where(opens.loc[test].to_numpy() >= 2, "오픈 2회 이상", "오픈 1회"),
                           "attempts": pd.cut(part["booking_count"], [0, 1, 4, 9, 1000], labels=["1건", "2~4건", "5~9건", "10건 이상"]).astype(str).to_numpy()})
    rows = []
    for label, preds in compared:
        rate = np.mean(preds, axis=0)
        for columns in (["scenario", "opens"], ["is_anomaly", "attempts"], ["is_anomaly", "opens"]):
            for key, index in traits.groupby(columns).indices.items():
                rows.append({"setting": label, "by": " × ".join(columns), "group": " / ".join(map(str, key)), "n_test": len(index), "flag_rate": float(rate[index].mean())})
    trait_table = pd.DataFrame(rows)
    csv(trait_table, "error_by_trait.csv")
    print(trait_table.pivot_table(index=["by", "group", "n_test"], columns="setting", values="flag_rate", sort=False)[[c[0] for c in compared]].round(2).to_string(), flush=True)

    # ---- 4) 분할 seed, 데이터 생성 seed ----
    quint = [("기준 모델 (기존 Feature, 혼합)", EXISTING, DEFAULT_HP, all_rows), ("세션 Feature, 혼합", session_name, session_hp, all_rows),
             ("기존 Feature, Credit 규칙 통과 학습", EXISTING, DEFAULT_HP, credit_pass), ("세션 Feature, Credit 규칙 통과 학습", session_name, DEFAULT_HP, credit_pass),
             ("세션 Feature, 정상만 (상한)", session_name, DEFAULT_HP, normal_only)]
    shown = ["fit_anomaly_share"] + [f"test_{k}" for k in METRICS] + GROUPS
    rows = []
    for split_seed in SPLIT_SEEDS:
        tr, va, te = splits(base, split_seed)
        for label, name, hp, selector in quint:
            row, _ = run(data[name], opens, tr, va, te, hp, MODEL_SEED, selector)
            rows.append({"setting": label, "split_seed": split_seed, **row})
    split_runs = pd.DataFrame(rows)
    csv(split_runs, "split_seed_runs.csv")
    split_summary = mean_std(split_runs, ["setting"], shown)
    split_summary["test_fpr_within_target_share"] = split_runs.assign(ok=lambda d: d["test_fpr"] <= TARGET_FPR).groupby("setting", sort=False)["ok"].mean().to_numpy()
    paired = split_runs.pivot(index="split_seed", columns="setting", values="test_recall")
    split_summary["splits_with_higher_recall_than_baseline"] = [int((paired[s] > paired[quint[0][0]]).sum()) for s in split_summary["setting"]]
    csv(split_summary, "split_seed_summary.csv")
    print("\n[4a] 분할 seed 20개\n", split_summary.round(3).T.to_string(), flush=True)

    config = load_config(CONFIG_PATH)
    calculated_at = pd.Timestamp(config["calculated_at_utc"])
    rows = []
    for seed in GENERATION_SEEDS:
        events, accounts, _ = generate(config, seed)
        dataset, _, _ = build_dataset(events, accounts, config)
        session_sets, _, _, _ = build_session(events, dataset, calculated_at)
        sets = {EXISTING: dataset, **{NAMES[a]: frame for a, frame in session_sets.items()}}
        tr, va, te = splits(dataset, SPLIT_SEED)
        for label, name, hp, selector in quint:
            row, _ = run(sets[name], accounts["n_sessions"], tr, va, te, hp, MODEL_SEED, selector)
            rows.append({"setting": label, "generation_seed": seed, "auc_interval": single_auc(sets[name], INTERVAL),
                         "interval_null_share": float(sets[name][INTERVAL].isna().mean()), **row})
        print(f"생성 seed {seed} 완료", flush=True)
    generation = pd.DataFrame(rows)
    csv(generation, "generation_seed_runs.csv")
    generation_summary = mean_std(generation, ["setting"], ["auc_interval", "interval_null_share"] + shown)
    paired = generation.pivot(index="generation_seed", columns="setting", values="test_recall")
    generation_summary["datasets_with_higher_recall_than_baseline"] = [int((paired[s] > paired[quint[0][0]]).sum()) for s in generation_summary["setting"]]
    csv(generation_summary, "generation_seed_summary.csv")
    print("\n[4b] 데이터 생성 seed 10개\n", generation_summary.round(3).T.to_string(), flush=True)

    # ---------------- 그래프 ----------------
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    high = max(data[name][INTERVAL].max() for name in (EXISTING, session_name))
    for ax, name in zip(axes, (EXISTING, session_name)):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        frame = data[name]
        for position, scenario in enumerate(SCENARIOS[::-1]):
            values = frame.loc[frame["scenario"] == scenario, INTERVAL].dropna().to_numpy()
            ax.scatter(values, position + rng.uniform(-0.3, 0.3, len(values)), s=6, alpha=0.4, linewidth=0, color=C_NORMAL if scenario in NORMAL_SCENARIOS else C_ANOM)
            ax.vlines(np.median(values), position - 0.4, position + 0.4, color=INK, linewidth=1.6)
        ax.set_yticks(range(len(SCENARIOS)))
        ax.set_yticklabels(SCENARIOS[::-1] if ax is axes[0] else [], fontsize=9)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(0, high * 1.3)
        ax.set_title(f"avg_booking_interval — {name}", loc="left", fontsize=10, color=INK)
    fig.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color=C_NORMAL, label="정상 시나리오"),
                        plt.Line2D([], [], marker="o", linestyle="", color=C_ANOM, label="이상 시나리오"), plt.Line2D([], [], color=INK, linewidth=1.6, label="중앙값")],
               loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("시나리오별 평균 예매 간격 (초, 로그 눈금): 30일 전체와 세션 기준. 결측 계정은 표시되지 않음", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "interval_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True, constrained_layout=True)
    for ax, feature in zip(axes, (INTERVAL, VARIANCE)):
        style(ax)
        part = separability[separability["feature"] == feature].set_index("features").loc[list(data)]
        x = np.arange(len(part))
        for offset, column, color, label in ((-0.27, "auc_all", INK, "이상 전체"), (0, "auc_anomaly_single_open", C_NORMAL, "오픈 1회 참여한 이상 계정"),
                                             (0.27, "auc_anomaly_multi_open", C_ANOM, "오픈 2회 이상 참여한 이상 계정")):
            ax.bar(x + offset, part[column], width=0.25, color=color, label=label)
            for xi, value in zip(x + offset, part[column]):
                ax.text(xi, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" (", "\n(").replace(": ", ":\n") for n in part.index], fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"{feature} 하나의 AUC (정상 전체와 비교)", loc="left", fontsize=10, color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    save(fig, OUT / "figures" / "single_feature_auc.png")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True, constrained_layout=True)
    for ax, max_samples in zip(axes, MAX_SAMPLES):
        style(ax)
        for name in data:
            part = grid[(grid["features"] == name) & (grid["max_samples"] == max_samples)]
            emphasis = name in (EXISTING, session_name)
            ax.plot(part["n_estimators"], part["val_recall"], marker="o", markersize=5, linewidth=2 if emphasis else 1,
                    color=C_OLD if name == EXISTING else (C_NEW if name == session_name else GRID), label=name, zorder=3 if emphasis else 1)
        ax.set_xticks(N_ESTIMATORS)
        ax.set_xlabel("n_estimators", fontsize=9, color=MUTED)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"max_samples {max_samples}", loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("validation Recall (오탐률 5% 이하)", fontsize=9, color=MUTED)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle(f"혼합 학습: 트리 설정에 따른 validation Recall (모델 seed {len(SELECT_SEEDS)}개 평균). 회색 가는 선은 나머지 세션 집계", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "validation_grid.png")

    by_setting, by_pair = summary.set_index("setting"), composition.set_index(["features", "training"])
    series = [("기준 모델 (기존 Feature, 혼합)", by_setting.loc[baseline_label], C_OLD), ("세션 Feature, 혼합", by_setting.loc[session_label], C_NEW),
              ("세션 Feature, Credit 규칙 통과 학습", by_pair.loc[(session_name, credit)], C_LOW)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.6]}, constrained_layout=True)
    for ax, (columns, labels, title) in zip(axes, (
            ([f"test_{k}_mean" for k in METRICS], ["Precision", "Recall", "F1", "오탐률", "ROC-AUC"], "전체 성능"),
            ([f"{r}_mean" for r in RATES], SCENARIOS, "시나리오별 이상 판정 비율 (앞 4개 = 오탐률, 뒤 4개 = Recall)"))):
        style(ax)
        x = np.arange(len(columns))
        for i, (label, row, color) in enumerate(series):
            values = row[columns].to_numpy(dtype=float)
            ax.bar(x + (i - 1) * 0.27, values, width=0.25, color=color, label=label)
            for xi, value in zip(x + (i - 1) * 0.27, values):
                ax.text(xi, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"오탐률 5% 조건, test 300건 (모델 seed {len(MODEL_SEEDS)}개 평균)", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "final_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    names = [t[0] for t in TRAININGS]
    for ax, (key, title) in zip(axes, (("test_recall_mean", "Recall"), ("test_f1_mean", "F1"), ("test_fpr_mean", "test 오탐률 (점선 = 5%)"))):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        position = np.arange(len(names))[::-1]
        for offset, name, color in ((0.2, EXISTING, C_OLD), (-0.2, session_name, C_NEW)):
            values = by_pair.loc[[(name, t) for t in names], key].to_numpy()
            ax.barh(position + offset, values, height=0.37, color=color, label=name)
            for p_, value in zip(position + offset, values):
                ax.text(value + (0.003 if key == "test_fpr_mean" else 0.012), p_, f"{value:.2f}", va="center", fontsize=7, color=INK)
        if key == "test_fpr_mean":
            ax.axvline(TARGET_FPR, color=INK, linewidth=1, linestyle="--")
            ax.set_xlim(0, 0.14)
        else:
            ax.set_xlim(0, 1.12)
        ax.set_yticks(position)
        ax.set_yticklabels(names if ax is axes[0] else [], fontsize=8)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"학습 데이터 구성에 따른 성능: 기존 Feature 와 세션 Feature (오탐률 5% 조건, 모델 seed {len(MODEL_SEEDS)}개 평균)", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "training_composition.png")

    by_open = trait_table[trait_table["by"] == "scenario × opens"]
    order = [f"{s} / {o}" for s in SCENARIOS[4:] for o in ("오픈 1회", "오픈 2회 이상")]
    picked = [(compared[0][0], C_OLD), (compared[1][0], C_NEW), (compared[3][0], C_LOW)]
    fig, ax = plt.subplots(figsize=(13, 4.6), constrained_layout=True)
    style(ax)
    x = np.arange(len(order))
    for i, (label, color) in enumerate(picked):
        part = by_open[by_open["setting"] == label].set_index("group").reindex(order)
        ax.bar(x + (i - 1) * 0.27, part["flag_rate"], width=0.25, color=color, label=label)
        for xi, value in zip(x + (i - 1) * 0.27, part["flag_rate"]):
            if not np.isnan(value):
                ax.text(xi, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    counts = by_open.drop_duplicates("group").set_index("group")["n_test"].reindex(order)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g.replace(' / ', chr(10))}\n(n={0 if np.isnan(n) else int(n)})" for g, n in counts.items()], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.2))
    ax.set_title(f"이상 시나리오 × 오픈 참여 횟수별 Recall (test, 모델 seed {len(MODEL_SEEDS)}개 평균)", loc="left", fontsize=10, color=INK)
    save(fig, OUT / "figures" / "recall_by_open_participation.png")

    panels = [("test_recall", "Recall"), ("recall_macro_multi_open", "여러 오픈에 참여한 매크로의 Recall"), ("test_fpr", "test 오탐률 (점선 = 5%)")]
    ticks = ["기준 모델\n(기존, 혼합)", "세션\n혼합", "기존\nCredit 통과", "세션\nCredit 통과", "세션\n정상만 (상한)"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.6), constrained_layout=True)
    for r, (frame, title) in enumerate(((split_runs, f"분할 seed {len(SPLIT_SEEDS)}개"), (generation, f"데이터 생성 seed {len(GENERATION_SEEDS)}개"))):
        for c, (key, name) in enumerate(panels):
            ax = axes[r, c]
            style(ax)
            for position, (label, features, _, _) in enumerate(quint):
                values = frame.loc[frame["setting"] == label, key].dropna().to_numpy()
                ax.scatter(position + rng.uniform(-0.15, 0.15, len(values)), values, s=20, color=C_OLD if features == EXISTING else C_NEW, edgecolor=SURFACE, linewidth=0.6)
                ax.hlines(values.mean(), position - 0.3, position + 0.3, color=INK, linewidth=1.4)
                ax.text(position, 1.07 if key != "test_fpr" else 0.158, f"{values.mean():.3f}", ha="center", va="top", fontsize=8, color=INK)
            ax.set_xticks(range(len(quint)))
            ax.set_xticklabels(ticks, fontsize=8)
            ax.set_xlim(-0.6, len(quint) - 0.4)
            if key == "test_fpr":
                ax.axhline(TARGET_FPR, color=INK, linewidth=1, linestyle="--")
                ax.set_ylim(0, 0.16)
            else:
                ax.set_ylim(0, 1.08)
            ax.set_title(f"{title}: {name}", loc="left", fontsize=10, color=INK)
    fig.suptitle("seed 를 바꿨을 때 (점 = 실행 1회, 가로선과 위의 숫자 = 평균). 회색 = 기존 Feature, 보라 = 세션 Feature", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "seed_stability.png")

    record = {
        "script": "scripts/evaluate_session.py", "datasets": {name: {"path": path.relative_to(EXP).as_posix(), "sha256": before[name]} for name, path in paths.items()},
        "datasets_unchanged_after_run": before == {name: sha256(path) for name, path in paths.items()},
        "features": F, "changed_features": [INTERVAL, VARIANCE],
        "baseline": {"training": "혼합 (레이블 미사용)", "preprocessing": "결측 중앙값 대체 + log1p", "n_estimators": DEFAULT_HP[0], "max_samples": DEFAULT_HP[1]},
        "split": {"train": len(train), "validation": len(val), "test": len(test), "random_state": SPLIT_SEED},
        "threshold": f"validation 정상 계정의 오탐률이 {TARGET_FPR} 이하가 되는 가장 낮은 점수. test 에는 같은 모델·같은 임계값",
        "search_space": {"features": list(data), "n_estimators": N_ESTIMATORS, "max_samples": MAX_SAMPLES},
        "selection": {"criterion": f"validation Recall (모델 seed {len(SELECT_SEEDS)}개 평균), 동률이면 F1", "selected_features": chosen_features,
                      "selected_n_estimators": chosen_hp[0], "selected_max_samples": chosen_hp[1], "selected_within_existing_features": list(existing_hp),
                      "best_session_aggregation": session_name, "best_session_tree_setting": list(session_hp),
                      "best_tree_setting_per_features": {k: list(v) for k, v in best_hp.items()}},
        "supplementary_training_compositions": [{"name": t[0], "uses_labels": t[2]} for t in TRAININGS],
        "model_seeds": MODEL_SEEDS, "split_seeds": SPLIT_SEEDS, "generation_seeds": GENERATION_SEEDS,
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]},
        "models_saved": False,
    }
    (OUT / "experiment_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(record["datasets_unchanged_after_run"], f"저장: {OUT}")


if __name__ == "__main__":
    main()
