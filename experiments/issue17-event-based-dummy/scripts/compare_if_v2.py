"""A(기존 직접 샘플링) · B(이벤트 기반 v1) · C(이벤트 기반 v2)로 Isolation Forest 를 같은 조건에서 학습·평가해 비교한다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\compare_if_v2.py

조건은 scripts/compare_if.py 와 같다 (= ml_experiment/train_and_evaluate_if.py 의 조건, #16 compare_if_models.py 의 함수·상수 재사용).
v2 의 생성 규칙은 이 스크립트를 돌리기 전에 확정했고, 이 결과를 보고 다시 바꾸지 않았다.

추가로 하는 일
  - 분할 seed 20개 (모델 seed 고정), 모델 seed 20개 (분할 고정), 데이터 생성 seed 10개
    (A·B 의 생성 seed 결과는 scripts/compare_if.py 가 저장한 output/model_comparison/generation_seed_runs.csv 를 그대로 읽는다)
  - INSUFFICIENT_DATA 분리: 시도 2건 미만(booking_count < 2)인 계정을 점수 대상에서 뺐을 때의 지표.
    기준은 docs/predict_feature_spec.md 와 train_model.py(MIN_BOOKING_COUNT = 2)다.
    현재 6개 Feature 번들을 만드는 pack_bundle.py 는 min_booking_count=1 이라 그 설정에서는 분리되는 계정이 없다 (= 기본 결과).
    결측값은 어떤 경우에도 숫자로 채우지 않는다.

출력: output/v2/model_comparison/
"""
import argparse
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
EXP = HERE.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP.parent / "issue16-booking-interval" / "scripts"))
from compare_if import SCENARIOS, cross_fit, evaluate_own_split  # noqa: E402
from compare_if_models import (BEST_CONTAMINATION, BEST_FEATURES, CONFIGS, INK, MODEL_SEED, MUTED, N_ESTIMATORS, PRIMARY,  # noqa: E402
                               SENSITIVITY_SEEDS, SPLIT_SEED, SURFACE, TEST_SIZE, fit_and_score, grouped_bars, metrics, save, style)
from extract_dataset import build_dataset  # noqa: E402
from generate_events import ORIGINAL_PATH, V2_PATH, generate, load_config, sha256  # noqa: E402

OUT = EXP / "output" / "v2" / "model_comparison"
L_A, L_B, L_C = "A 원본", "B 이벤트 v1", "C 이벤트 v2"
PATHS = {L_A: ORIGINAL_PATH, L_B: EXP / "data" / "single_session" / "fairticket_event_based_features.csv",
         L_C: EXP / "data" / "v2" / "fairticket_event_based_features.csv"}
COLOR = {L_A: "#2a78d6", L_B: "#eb6834", L_C: "#1baf7a"}
MIN_ATTEMPTS = 2  # docs/predict_feature_spec.md, train_model.py MIN_BOOKING_COUNT
KEEP = ("precision", "recall", "f1", "accuracy", "fpr", "roc_auc", "TP", "FP", "TN", "FN", "flagged")


def summarize(frame, by, columns):
    return (frame.groupby(by, sort=False)[columns].agg(["mean", "std", "min", "max"]).stack(level=0, future_stack=True)
            .reset_index().rename(columns=lambda name: "metric" if str(name).startswith("level_") else name))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation-seeds", type=int, default=10)
    args = parser.parse_args()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    hashes = {label: sha256(path) for label, path in PATHS.items()}
    data = {label: pd.read_csv(path) for label, path in PATHS.items()}
    original = data[L_A]
    scenario = pd.read_csv(V2_PATH)["scenario"]
    for label, frame in data.items():
        if not (frame["user_id"].equals(original["user_id"]) and frame["is_anomaly"].equals(original["is_anomaly"])
                and list(frame.columns) == list(original.columns)):
            sys.exit(f"{label}: 행 순서·레이블·컬럼이 원본과 다릅니다.")
    labels = list(data)
    y = original["is_anomaly"]
    train_idx, test_idx = train_test_split(original.index, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)
    y_test = y.loc[test_idx].to_numpy()
    test_scenario = scenario.loc[test_idx].to_numpy()

    # 1) 구성 × 데이터셋
    rows, predictions = [], {}
    for config, features, contamination in CONFIGS:
        for label, frame in data.items():
            pred, decision, normalized = fit_and_score(frame, train_idx, test_idx, features, contamination)
            rows.append({"config": config, "dataset": label, **metrics(y_test, pred, normalized)})
            if config == PRIMARY:
                predictions[label] = pd.DataFrame({
                    "row_index": test_idx, "user_id": original.loc[test_idx, "user_id"].to_numpy(), "scenario": test_scenario,
                    "is_anomaly": y_test, "pred_anomaly": pred, "decision_function": decision, "anomaly_score_normalized": normalized,
                    "booking_count": frame.loc[test_idx, "booking_count"].to_numpy(),
                    "has_null_model_input": frame.loc[test_idx, BEST_FEATURES].isna().any(axis=1).to_numpy(),
                    "cancel_rate_null": frame.loc[test_idx, "cancel_rate"].isna().to_numpy()})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "metrics_all_configs.csv", index=False, encoding="utf-8-sig")
    primary = results[results["config"] == PRIMARY].reset_index(drop=True)
    primary.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    primary[["dataset", "TP", "FP", "TN", "FN", "flagged"]].to_csv(OUT / "confusion_matrix.csv", index=False, encoding="utf-8-sig")
    pd.concat([frame.assign(dataset=label) for label, frame in predictions.items()]).to_csv(
        OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # 2) 시나리오별
    rows = []
    for label, frame in predictions.items():
        for name in SCENARIOS:
            part = frame[frame["scenario"] == name]
            actual = int(part["is_anomaly"].iloc[0])
            rows.append({"scenario": name, "actual": "이상" if actual else "정상", "dataset": label, "n_test": len(part),
                         "flagged": int(part["pred_anomaly"].sum()), "flag_rate": part["pred_anomaly"].mean(),
                         "meaning": "탐지율" if actual else "오탐률", "decision_median": part["decision_function"].median(),
                         "null_rows": int(part["has_null_model_input"].sum()),
                         "null_rows_flagged": int((part["has_null_model_input"] & (part["pred_anomaly"] == 1)).sum())})
    by_scenario = pd.DataFrame(rows)
    by_scenario.to_csv(OUT / "scenario_detection.csv", index=False, encoding="utf-8-sig")

    # 3) seed 흔들림
    rows = []
    for seed in SENSITIVITY_SEEDS:
        for label, frame in data.items():
            pred, _, normalized = fit_and_score(frame, train_idx, test_idx, BEST_FEATURES, BEST_CONTAMINATION, seed)
            row = {"varied": "모델 seed", "seed": seed, "dataset": label, **metrics(y_test, pred, normalized)}
            for name in SCENARIOS:
                row[f"flag_rate_{name}"] = pred[test_scenario == name].mean()
            rows.append(row)
        tr, te = train_test_split(original.index, test_size=TEST_SIZE, random_state=seed, stratify=y)
        for label, frame in data.items():
            pred, _, normalized = fit_and_score(frame, tr, te, BEST_FEATURES, BEST_CONTAMINATION)
            row = {"varied": "분할 seed", "seed": seed, "dataset": label, **metrics(y.loc[te].to_numpy(), pred, normalized)}
            for name in SCENARIOS:
                row[f"flag_rate_{name}"] = pred[scenario.loc[te].to_numpy() == name].mean()
            rows.append(row)
    seeds = pd.DataFrame(rows)
    seeds.to_csv(OUT / "seed_runs.csv", index=False, encoding="utf-8-sig")
    seed_cols = ["f1", "precision", "recall", "fpr", "roc_auc"] + [f"flag_rate_{s}" for s in SCENARIOS]
    seed_summary = summarize(seeds, ["varied", "dataset"], seed_cols)
    seed_summary.to_csv(OUT / "seed_summary.csv", index=False, encoding="utf-8-sig")

    generation = pd.DataFrame()
    previous = EXP / "output" / "model_comparison" / "generation_seed_runs.csv"
    if args.generation_seeds > 0 and previous.exists():
        old = pd.read_csv(previous)
        old["dataset"] = old["method"].map({"Feature 직접 샘플링 (기존 생성 코드)": L_A, "이벤트 기반": L_B})
        config = load_config(EXP / "config" / "scenarios_v2.json")
        rows = []
        for seed in range(args.generation_seeds):
            events, accounts, _ = generate(config, "single_session", seed)
            dataset, _, _ = build_dataset(events, accounts, config)
            rows.append({"generation_seed": seed, "dataset": L_C, **evaluate_own_split(dataset)})
            print(f"v2 생성 seed {seed} 완료", flush=True)
        generation = pd.concat([old[old["generation_seed"] < args.generation_seeds].drop(columns="method"), pd.DataFrame(rows)], ignore_index=True)
        generation.to_csv(OUT / "generation_seed_runs.csv", index=False, encoding="utf-8-sig")
        generation_summary = summarize(generation, ["dataset"], ["f1", "precision", "recall", "fpr", "roc_auc"])
        generation_summary.to_csv(OUT / "generation_seed_summary.csv", index=False, encoding="utf-8-sig")

    # 4) INSUFFICIENT_DATA 분리 (시도 2건 미만). 결측은 채우지 않는다
    rows = []
    for label, frame in data.items():
        enough = frame["booking_count"] >= MIN_ATTEMPTS
        test_keep = test_idx[enough.loc[test_idx].to_numpy()]
        y_keep = y.loc[test_keep].to_numpy()
        base = predictions[label].set_index("row_index")
        variants = {
            "분리 없음 (기본 결과, pack_bundle 의 min_booking_count=1)": (train_idx, test_idx),
            "평가에서만 분리 (학습은 그대로)": (train_idx, test_keep),
            "학습과 평가 모두에서 분리": (train_idx[enough.loc[train_idx].to_numpy()], test_keep),
        }
        for name, (tr, te) in variants.items():
            pred, _, normalized = fit_and_score(frame, tr, te, BEST_FEATURES, BEST_CONTAMINATION)
            remaining_null = frame.loc[te, BEST_FEATURES].isna().any(axis=1).to_numpy()
            row = {"dataset": label, "policy": name, "train_rows": len(tr), "scored_test_rows": len(te),
                   "insufficient_test_rows": len(test_idx) - len(te),
                   "insufficient_test_rows_normal": int((y.loc[test_idx].to_numpy() == 0).sum() - (y.loc[te].to_numpy() == 0).sum()),
                   "insufficient_test_rows_anomaly": int(y_test.sum() - y.loc[te].sum()),
                   **{k: v for k, v in metrics(y.loc[te].to_numpy(), pred, normalized).items() if k in KEEP},
                   "scored_rows_still_null": int(remaining_null.sum()), "still_null_flagged": int(pred[remaining_null].sum()),
                   "still_null_actual_anomaly": int(y.loc[te].to_numpy()[remaining_null].sum())}
            for scen in SCENARIOS:
                mask = scenario.loc[te].to_numpy() == scen
                row[f"flag_rate_{scen}"] = pred[mask].mean() if mask.any() else np.nan
            rows.append(row)
        del base, y_keep
    policy = pd.DataFrame(rows)
    policy.to_csv(OUT / "insufficient_data_policy.csv", index=False, encoding="utf-8-sig")

    # 5) 교차 평가
    rows = []
    for train_label in labels:
        for test_label in labels:
            pred, normalized = cross_fit(data[train_label], data[test_label], train_idx, test_idx)
            row = {"train": train_label, "test": test_label, **metrics(y_test, pred, normalized)}
            for name in SCENARIOS:
                row[f"flag_rate_{name}"] = pred[test_scenario == name].mean()
            rows.append(row)
    cross = pd.DataFrame(rows)
    cross.to_csv(OUT / "cross_evaluation.csv", index=False, encoding="utf-8-sig")

    # 6) hold_cancel 의 test 계정을 데이터셋별로 나란히 (탐지 변화의 원인 확인용)
    hold = scenario.loc[test_idx] == "hold_cancel"
    hold_rows = []
    for label, frame in data.items():
        part = frame.loc[test_idx[hold.to_numpy()], ["user_id"] + BEST_FEATURES].copy()
        part.insert(1, "dataset", label)
        part["pred_anomaly"] = predictions[label].set_index("row_index").loc[part.index, "pred_anomaly"].to_numpy()
        part["decision_function"] = predictions[label].set_index("row_index").loc[part.index, "decision_function"].to_numpy()
        hold_rows.append(part)
    hold_table = pd.concat(hold_rows)
    hold_table.to_csv(OUT / "hold_cancel_test_accounts.csv", index=False, encoding="utf-8-sig")

    # ---------------- 그래프 ----------------
    names = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("fpr", "오탐률(FPR)"), ("roc_auc", "ROC-AUC")]
    fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    style(ax)
    grouped_bars(ax, [n for _, n in names], [(label, [primary.loc[primary["dataset"] == label, key].squeeze() for key, _ in names],
                                              COLOR[label]) for label in labels], fmt="{:.3f}")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=len(labels))
    ax.set_title(f"전체 성능 — {PRIMARY}, test {len(test_idx)}건 (합성 데이터 안에서의 값)", loc="left", fontsize=10, color=INK)
    save(fig, OUT / "figures" / "overall_metrics.png")

    counts = by_scenario.drop_duplicates("scenario").set_index("scenario")
    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
    style(ax)
    grouped_bars(ax, [f"{s}\n({counts.loc[s, 'actual']}, n={counts.loc[s, 'n_test']})" for s in SCENARIOS],
                 [(label, [by_scenario.loc[(by_scenario["dataset"] == label) & (by_scenario["scenario"] == s), "flag_rate"].squeeze()
                           for s in SCENARIOS], COLOR[label]) for label in labels])
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=len(labels))
    ax.set_ylabel("이상 판정 비율", fontsize=9, color=MUTED)
    ax.set_title("시나리오별 이상 판정 비율 (정상 = 오탐률, 이상 = 탐지율)", loc="left", fontsize=10, color=INK)
    save(fig, OUT / "figures" / "scenario_detection.png")

    rng = np.random.default_rng(0)  # 점을 좌우로 흩는 용도
    panels = [("모델 seed", seeds[seeds["varied"] == "모델 seed"]), ("분할 seed", seeds[seeds["varied"] == "분할 seed"])]
    if len(generation):
        panels.append(("데이터 생성 seed", generation))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.3), sharey=True, constrained_layout=True)
    for ax, (title, frame) in zip(np.atleast_1d(axes), panels):
        style(ax)
        for position, label in enumerate(labels):
            values = frame.loc[frame["dataset"] == label, "f1"].to_numpy()
            ax.scatter(position + rng.uniform(-0.18, 0.18, len(values)), values, s=18, color=COLOR[label], edgecolor=SURFACE, linewidth=0.5)
            ax.hlines(values.mean(), position - 0.3, position + 0.3, color=INK, linewidth=1.4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_title(f"{title} {frame['dataset'].value_counts().iloc[0]}개", loc="left", fontsize=10, color=INK)
    np.atleast_1d(axes)[0].set_ylabel("F1", fontsize=9, color=MUTED)
    fig.suptitle("seed 를 바꿨을 때의 F1 (점 = 실행 1회, 가로선 = 평균)", fontsize=11, color=INK)
    save(fig, OUT / "figures" / "seed_sensitivity.png")

    short = {"분리 없음 (기본 결과, pack_bundle 의 min_booking_count=1)": "분리 없음", "평가에서만 분리 (학습은 그대로)": "평가에서만 분리",
             "학습과 평가 모두에서 분리": "학습·평가 모두 분리"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    for ax, (key, title) in zip(axes, (("f1", "F1"), ("fpr", "오탐률(FPR)"), ("recall", "Recall"))):
        style(ax)
        grouped_bars(ax, list(short.values()),
                     [(label, [policy.loc[(policy["dataset"] == label) & (policy["policy"] == name), key].squeeze() for name in short],
                       COLOR[label]) for label in labels], fmt="{:.3f}", ymax=1.0 if key != "fpr" else 0.2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    fig.suptitle(f"INSUFFICIENT_DATA 분리(시도 {MIN_ATTEMPTS}건 미만)에 따른 지표", fontsize=11, color=INK)
    save(fig, OUT / "figures" / "insufficient_data_policy.png")

    record = {
        "script": "scripts/compare_if_v2.py", "reference_training_script": "ml_experiment/train_and_evaluate_if.py (수정하지 않음)",
        "datasets": {label: {"path": path.relative_to(REPO).as_posix(), "sha256": hashes[label],
                             "unchanged_after_run": hashes[label] == sha256(path)} for label, path in PATHS.items()},
        "primary": {"features": BEST_FEATURES, "contamination": BEST_CONTAMINATION}, "preprocessing": "없음 (결측도 그대로)",
        "split": {"test_size": TEST_SIZE, "random_state": SPLIT_SEED, "stratify": "is_anomaly", "train": len(train_idx), "test": len(test_idx)},
        "model": {"class": "sklearn.ensemble.IsolationForest", "n_estimators": N_ESTIMATORS, "random_state": MODEL_SEED},
        "model_seeds": SENSITIVITY_SEEDS, "split_seeds": SENSITIVITY_SEEDS, "generation_seeds": list(range(args.generation_seeds)),
        "generation_seed_results_for_A_and_B": "output/model_comparison/generation_seed_runs.csv (scripts/compare_if.py 의 결과)",
        "insufficient_data_rule": f"booking_count < {MIN_ATTEMPTS} (docs/predict_feature_spec.md, train_model.py). pack_bundle.py 는 min_booking_count=1",
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]},
        "models_saved": False,
    }
    (OUT / "experiment_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pd.set_option("display.width", 280)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 400)
    print(results.round(3).to_string(index=False))
    print(by_scenario.pivot(index="scenario", columns="dataset", values="flagged").reindex(SCENARIOS)[labels].assign(
        n_test=[counts.loc[s, "n_test"] for s in SCENARIOS]).to_string())
    print(by_scenario[by_scenario["null_rows"] > 0][["scenario", "dataset", "n_test", "null_rows", "null_rows_flagged"]].to_string(index=False))
    print(seed_summary[seed_summary["metric"].isin(["f1", "precision", "recall", "fpr", "roc_auc"])].round(3).to_string(index=False))
    print(seeds.groupby(["varied", "dataset"], sort=False)[[f"flag_rate_{s}" for s in SCENARIOS]].mean().round(2).T.to_string())
    if len(generation):
        print(generation_summary.round(3).to_string(index=False))
    print(policy.round(3).T.to_string())
    print(cross[["train", "test", "precision", "recall", "f1", "fpr", "roc_auc", "TP", "FP", "FN"]].round(3).to_string(index=False))
    print(hold_table.round(3).sort_values(["user_id", "dataset"]).to_string(index=False))
    print({k: v["unchanged_after_run"] for k, v in record["datasets"].items()})
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
