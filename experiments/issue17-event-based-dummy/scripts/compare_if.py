"""기존 Feature 직접 샘플링 데이터셋과 이벤트 기반 데이터셋으로 Isolation Forest 를 같은 조건에서 학습·평가해 비교한다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\compare_if.py
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\compare_if.py --generation-seeds 0     # 생성 seed 반복 생략 (빠름)

조건은 `ml_experiment/train_and_evaluate_if.py` 와 같다. 그 조건을 옮겨 둔 #16 `compare_if_models.py` 의 함수·상수
(fit_and_score, metrics, CONFIGS, 분할·모델 seed)를 import 해서 그대로 쓴다.
  - 전처리 없음, train_test_split(test_size=0.3, random_state=42, stratify=is_anomaly), 모든 데이터셋에 같은 행 번호
  - IsolationForest(n_estimators=100, random_state=42, contamination=c), 기준 구성은 Feature 6개 · contamination 0.20
  - 결측(시도 1건의 간격 등)은 채우지 않고 그대로 넣는다. 기존 코드에도 대체 단계가 없고, 이 scikit-learn 버전의 Isolation Forest 는 결측을 받는다.

하는 일
  1. 기준 구성과 기존 스크립트의 다른 구성으로 원본 · #16 v2 · 이벤트 기반 · 이벤트 기반(다회 세션)을 학습·평가
  2. 시나리오별 이상 판정 비율, 혼동행렬, 결측 행을 뺀 지표
  3. 모델 seed 20개, 데이터 생성 seed N개(두 생성 방식 모두)로 흔들림 확인
  4. 교차 평가(한 데이터셋으로 학습 → 다른 데이터셋의 test 로 평가), Feature 묶음 교체(원본에 이벤트 기반 값을 한 묶음씩 넣어 봄)

모델 파일은 저장하지 않는다. 기존 학습·서빙 코드와 데이터셋은 건드리지 않는다.
출력: output/model_comparison/
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP.parent / "issue16-booking-interval" / "scripts"))
from compare_if_models import (ALL_FEATURES, BEST_CONTAMINATION, BEST_FEATURES, CONFIGS, INK, MODEL_SEED, MUTED,  # noqa: E402
                               N_ESTIMATORS, PRIMARY, SENSITIVITY_SEEDS, SPLIT_SEED, SURFACE, TEST_SIZE, fit_and_score,
                               grouped_bars, metrics, save, style)
from extract_dataset import build_dataset  # noqa: E402
from generate_events import ORIGINAL_PATH, V2_PATH, generate, load_config, sha256  # noqa: E402

OUT = EXP / "output" / "model_comparison"
L_ORIG, L_V2, L_EVENT, L_MULTI = "원본", "#16 v2", "이벤트 기반", "이벤트 기반(다회 세션)"
COLOR = {L_ORIG: "#2a78d6", L_V2: "#6250d6", L_EVENT: "#eb6834", L_MULTI: "#1baf7a"}
SCENARIOS = ["normal", "fast_normal", "shared_ip", "frequent_cancel", "fast", "jitter", "multi_account", "hold_cancel"]
SWAPS = {"예매 간격 2개": ["avg_booking_interval", "booking_interval_variance"], "취소율": ["cancel_rate"],
         "카드·IP": ["card_reuse_count", "ip_booking_density"], "시도 수": ["booking_count"]}


def cross_fit(train_frame, test_frame, train_idx, test_idx, features=BEST_FEATURES, contamination=BEST_CONTAMINATION):
    """fit_and_score 와 같은 모델 설정으로, 학습과 평가에 서로 다른 데이터셋을 쓴다."""
    model = IsolationForest(contamination=contamination, random_state=MODEL_SEED, n_estimators=N_ESTIMATORS)
    model.fit(train_frame.loc[train_idx, features])
    x = test_frame.loc[test_idx, features]
    decision = model.decision_function(x)
    return np.where(model.predict(x) == -1, 1, 0), (decision.max() - decision) / (decision.max() - decision.min() + 1e-9)


def original_method_dataset(seed):
    """기존 생성 코드를 그대로 불러 다른 seed 의 데이터셋을 만든다. 그 코드는 현재 폴더의 data/ 에 쓰므로 임시 폴더에서 실행한다."""
    spec = importlib.util.spec_from_file_location("generate_dummy_data", REPO / "ml_experiment" / "generate_dummy_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as folder:
        os.chdir(folder)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                module.generate_fairticket_dataset(seed=seed)
            return pd.read_csv(Path(folder) / "data" / "fairticket_synthetic_features.csv")
        finally:
            os.chdir(here)


def evaluate_own_split(frame):
    """데이터셋 자체의 레이블로 기존과 같은 분할을 만들어 기준 구성으로 평가한다 (생성 seed 반복용)."""
    train_idx, test_idx = train_test_split(frame.index, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=frame["is_anomaly"])
    pred, _, normalized = fit_and_score(frame, train_idx, test_idx, BEST_FEATURES, BEST_CONTAMINATION)
    return metrics(frame.loc[test_idx, "is_anomaly"].to_numpy(), pred, normalized)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation-seeds", type=int, default=10, help="데이터 생성 seed 반복 수 (0 이면 생략)")
    args = parser.parse_args()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    paths = {L_ORIG: ORIGINAL_PATH, L_V2: V2_PATH,
             L_EVENT: EXP / "data" / "single_session" / "fairticket_event_based_features.csv",
             L_MULTI: EXP / "data" / "multi_session" / "fairticket_event_based_features.csv"}
    hashes = {label: sha256(path) for label, path in paths.items()}
    data = {label: pd.read_csv(path) for label, path in paths.items()}
    original = data[L_ORIG]
    scenario = data[L_V2]["scenario"]  # 모든 데이터셋에 같은 이름표 (원본의 macro 행은 fast / jitter 로 표시)
    alignment = pd.DataFrame([{
        "dataset": label, "rows": len(frame), "user_id_order_identical": bool(frame["user_id"].equals(original["user_id"])),
        "is_anomaly_identical": bool(frame["is_anomaly"].equals(original["is_anomaly"])),
        "columns_identical": list(frame.columns) == list(original.columns),
        "rows_with_null_model_input": int(frame[BEST_FEATURES].isna().any(axis=1).sum()),
    } for label, frame in data.items()])
    if not (alignment["user_id_order_identical"] & alignment["is_anomaly_identical"] & alignment["columns_identical"]).all():
        sys.exit("데이터셋의 행 순서·레이블·컬럼이 서로 다릅니다.")
    alignment.to_csv(OUT / "dataset_alignment.csv", index=False, encoding="utf-8-sig")

    y = original["is_anomaly"]
    train_idx, test_idx = train_test_split(original.index, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)
    y_test = y.loc[test_idx].to_numpy()
    test_scenario = scenario.loc[test_idx].to_numpy()
    pd.DataFrame({"row_index": original.index, "user_id": original["user_id"], "scenario": scenario,
                  "split": np.where(original.index.isin(test_idx), "test", "train")}).to_csv(OUT / "split_indices.csv", index=False)

    # 1) 모든 구성 × 데이터셋
    rows, predictions = [], {}
    for config, features, contamination in CONFIGS:
        for label, frame in data.items():
            pred, decision, normalized = fit_and_score(frame, train_idx, test_idx, features, contamination)
            rows.append({"config": config, "dataset": label, "n_features": len(features), "contamination": contamination,
                         **metrics(y_test, pred, normalized)})
            if config == PRIMARY:
                predictions[label] = pd.DataFrame({
                    "row_index": test_idx, "user_id": original.loc[test_idx, "user_id"].to_numpy(), "scenario": test_scenario,
                    "is_anomaly": y_test, "pred_anomaly": pred, "decision_function": decision, "anomaly_score_normalized": normalized,
                    "has_null_model_input": frame.loc[test_idx, BEST_FEATURES].isna().any(axis=1).to_numpy()})
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "metrics_all_configs.csv", index=False, encoding="utf-8-sig")
    primary = results[results["config"] == PRIMARY].reset_index(drop=True)
    primary.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    primary[["dataset", "TP", "FP", "TN", "FN", "flagged"]].to_csv(OUT / "confusion_matrix.csv", index=False, encoding="utf-8-sig")
    pd.concat([frame.assign(dataset=label) for label, frame in predictions.items()]).to_csv(
        OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # 2) 시나리오별 이상 판정 비율, 결측 행을 뺀 지표
    rows = []
    for label, frame in predictions.items():
        for name in SCENARIOS:
            part = frame[frame["scenario"] == name]
            actual = int(part["is_anomaly"].iloc[0])
            rows.append({"scenario": name, "actual": "이상" if actual else "정상", "dataset": label, "n_test": len(part),
                         "flagged": int(part["pred_anomaly"].sum()), "flag_rate": part["pred_anomaly"].mean(),
                         "meaning": "탐지율" if actual else "오탐률", "decision_median": part["decision_function"].median()})
    by_scenario = pd.DataFrame(rows)
    by_scenario.to_csv(OUT / "scenario_detection.csv", index=False, encoding="utf-8-sig")

    rows = []
    for label, frame in predictions.items():
        nulls = frame[frame["has_null_model_input"]]
        complete = frame[~frame["has_null_model_input"]]
        rows.append({"dataset": label, "test_rows": len(frame), "test_rows_with_null": len(nulls),
                     "train_rows_with_null": int(data[label].loc[train_idx, BEST_FEATURES].isna().any(axis=1).sum()),
                     "null_rows_flagged": int(nulls["pred_anomaly"].sum()), "null_rows_actual_anomaly": int(nulls["is_anomaly"].sum()),
                     **{f"complete_rows_{k}": v for k, v in metrics(complete["is_anomaly"].to_numpy(), complete["pred_anomaly"].to_numpy(),
                                                                    complete["anomaly_score_normalized"].to_numpy()).items()
                        if k in ("precision", "recall", "f1", "fpr", "roc_auc", "FP", "TP")}})
    null_effect = pd.DataFrame(rows)
    null_effect.to_csv(OUT / "null_rows_effect.csv", index=False, encoding="utf-8-sig")

    # 3-a) 모델 seed 만 바꿈
    rows = []
    for seed in SENSITIVITY_SEEDS:
        for label, frame in data.items():
            pred, _, normalized = fit_and_score(frame, train_idx, test_idx, BEST_FEATURES, BEST_CONTAMINATION, seed)
            row = {"model_seed": seed, "dataset": label, **metrics(y_test, pred, normalized)}
            for name in SCENARIOS:
                row[f"flag_rate_{name}"] = pred[test_scenario == name].mean()
            rows.append(row)
    model_seeds = pd.DataFrame(rows)
    model_seeds.to_csv(OUT / "model_seed_runs.csv", index=False, encoding="utf-8-sig")
    seed_cols = ["f1", "precision", "recall", "fpr", "roc_auc"] + [f"flag_rate_{s}" for s in SCENARIOS]
    model_seed_summary = (model_seeds.groupby("dataset", sort=False)[seed_cols].agg(["mean", "std", "min", "max"])
                          .stack(level=0, future_stack=True).reset_index().rename(columns={"level_1": "metric"}))
    model_seed_summary.to_csv(OUT / "model_seed_summary.csv", index=False, encoding="utf-8-sig")

    # 3-b) 데이터 생성 seed 를 바꿈 (모델·분할 seed 는 그대로). 두 방식 모두 seed 마다 데이터셋 전체를 새로 만든다.
    generation = pd.DataFrame()
    if args.generation_seeds > 0:
        config = load_config()
        check = original_method_dataset(42)
        reproduces = bool(check.equals(original))
        rows = []
        for seed in range(args.generation_seeds):
            rows.append({"generation_seed": seed, "method": "Feature 직접 샘플링 (기존 생성 코드)", **evaluate_own_split(original_method_dataset(seed))})
            events, accounts, _ = generate(config, "single_session", seed)
            dataset, _, _ = build_dataset(events, accounts, config)
            rows.append({"generation_seed": seed, "method": "이벤트 기반", **evaluate_own_split(dataset)})
            print(f"생성 seed {seed} 완료", flush=True)
        generation = pd.DataFrame(rows)
        generation.to_csv(OUT / "generation_seed_runs.csv", index=False, encoding="utf-8-sig")
        generation_summary = (generation.groupby("method", sort=False)[["f1", "precision", "recall", "fpr", "roc_auc"]]
                              .agg(["mean", "std", "min", "max"]).stack(level=0, future_stack=True).reset_index()
                              .rename(columns={"level_1": "metric"}))
        generation_summary.to_csv(OUT / "generation_seed_summary.csv", index=False, encoding="utf-8-sig")
    else:
        reproduces = None

    # 4-a) 교차 평가
    rows = []
    for train_label in (L_ORIG, L_V2, L_EVENT):
        for test_label in (L_ORIG, L_V2, L_EVENT):
            pred, normalized = cross_fit(data[train_label], data[test_label], train_idx, test_idx)
            row = {"train": train_label, "test": test_label, **metrics(y_test, pred, normalized)}
            for name in SCENARIOS:
                row[f"flag_rate_{name}"] = pred[test_scenario == name].mean()
            rows.append(row)
    cross = pd.DataFrame(rows)
    cross.to_csv(OUT / "cross_evaluation.csv", index=False, encoding="utf-8-sig")

    # 4-b) 원본에 이벤트 기반 값을 한 묶음씩 넣어 본다 (같은 계정의 값으로 교체. 행 안의 일관성은 깨진다 — 진단용)
    rows = []
    base = primary[primary["dataset"] == L_ORIG].iloc[0]
    for name, columns in SWAPS.items():
        mixed = original.copy()
        mixed[columns] = data[L_EVENT][columns]
        pred, _, normalized = fit_and_score(mixed, train_idx, test_idx, BEST_FEATURES, BEST_CONTAMINATION)
        row = {"swapped_in_from_event": name, **metrics(y_test, pred, normalized)}
        row["f1_change_vs_original"] = row["f1"] - base["f1"]
        for scen in SCENARIOS:
            row[f"flag_rate_{scen}"] = pred[test_scenario == scen].mean()
        rows.append(row)
    swaps = pd.DataFrame(rows)
    swaps.to_csv(OUT / "feature_swap.csv", index=False, encoding="utf-8-sig")

    # ---------------- 그래프 ----------------
    labels = list(data)
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
    shown = [L_ORIG, L_EVENT]
    fig, axes = plt.subplots(len(shown), 1, figsize=(11, 3.2 * len(shown)), sharex=True, sharey=True, constrained_layout=True)
    for ax, label in zip(axes, shown):
        style(ax)
        frame = predictions[label]
        for position, name in enumerate(SCENARIOS):
            part = frame[frame["scenario"] == name]
            ax.scatter(position + rng.uniform(-0.28, 0.28, len(part)), part["decision_function"], s=14, color=COLOR[label],
                       edgecolor=SURFACE, linewidth=0.4)
            ax.hlines(part["decision_function"].median(), position - 0.36, position + 0.36, color=INK, linewidth=1.4)
        ax.axhline(0, color=INK, linewidth=1, linestyle="--")
        ax.set_ylabel("decision_function", fontsize=9, color=MUTED)
        ax.set_title(f"{label}  (점선 아래 = 이상 판정, 가로선 = 중앙값)", loc="left", fontsize=10, color=INK)
    axes[-1].set_xticks(range(len(SCENARIOS)))
    axes[-1].set_xticklabels([f"{s}\n({counts.loc[s, 'actual']})" for s in SCENARIOS], fontsize=9)
    fig.suptitle("이상 점수 분포 (test, 시나리오별)", fontsize=11, color=INK)
    save(fig, OUT / "figures" / "score_distribution_by_scenario.png")

    panels = [("f1", "F1"), ("recall", "Recall"), ("fpr", "오탐률(FPR)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), constrained_layout=True)
    for ax, (key, title) in zip(axes, panels):
        style(ax)
        for position, label in enumerate(labels):
            values = model_seeds.loc[model_seeds["dataset"] == label, key].to_numpy()
            ax.scatter(position + rng.uniform(-0.18, 0.18, len(values)), values, s=18, color=COLOR[label], edgecolor=SURFACE, linewidth=0.5)
            ax.hlines(values.mean(), position - 0.3, position + 0.3, color=INK, linewidth=1.4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l.replace("(", "\n(") for l in labels], fontsize=8)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.suptitle(f"모델 seed {len(SENSITIVITY_SEEDS)}개로 본 흔들림 (같은 데이터, 가로선 = 평균)", fontsize=11, color=INK)
    save(fig, OUT / "figures" / "model_seed_sensitivity.png")

    if len(generation):
        methods = list(generation["method"].unique())
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), constrained_layout=True)
        for ax, (key, title) in zip(axes, panels):
            style(ax)
            for position, (method, color) in enumerate(zip(methods, (COLOR[L_ORIG], COLOR[L_EVENT]))):
                values = generation.loc[generation["method"] == method, key].to_numpy()
                ax.scatter(position + rng.uniform(-0.18, 0.18, len(values)), values, s=22, color=color, edgecolor=SURFACE, linewidth=0.5)
                ax.hlines(values.mean(), position - 0.3, position + 0.3, color=INK, linewidth=1.4)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(["Feature 직접 샘플링", "이벤트 기반"], fontsize=9)
            ax.set_xlim(-0.6, len(methods) - 0.4)
            ax.set_title(title, loc="left", fontsize=10, color=INK)
        fig.suptitle(f"데이터 생성 seed {args.generation_seeds}개로 본 흔들림 (점 = 새로 만든 데이터셋 1개, 가로선 = 평균)", fontsize=11, color=INK)
        save(fig, OUT / "figures" / "generation_seed_sensitivity.png")

    record = {
        "script": "scripts/compare_if.py", "reference_training_script": "ml_experiment/train_and_evaluate_if.py (수정하지 않음)",
        "reused_from": "experiments/issue16-booking-interval/scripts/compare_if_models.py (fit_and_score, metrics, CONFIGS, seed)",
        "datasets": {label: {"path": path.relative_to(REPO).as_posix(), "sha256": hashes[label],
                             "unchanged_after_run": hashes[label] == sha256(path)} for label, path in paths.items()},
        "primary": {"features": BEST_FEATURES, "contamination": BEST_CONTAMINATION}, "preprocessing": "없음 (결측도 그대로)",
        "split": {"test_size": TEST_SIZE, "random_state": SPLIT_SEED, "stratify": "is_anomaly", "train": len(train_idx),
                  "test": len(test_idx), "test_anomalies": int(y_test.sum())},
        "model": {"class": "sklearn.ensemble.IsolationForest", "n_estimators": N_ESTIMATORS, "random_state": MODEL_SEED},
        "model_seeds": SENSITIVITY_SEEDS, "generation_seeds": list(range(args.generation_seeds)),
        "original_generator_seed42_reproduces_committed_dataset": reproduces,
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]},
        "models_saved": False,
    }
    (OUT / "experiment_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 300)
    print(alignment.to_string(index=False))
    print(results.drop(columns=["n_features", "contamination"]).round(3).to_string(index=False))
    print(by_scenario.pivot(index="scenario", columns="dataset", values="flagged").reindex(SCENARIOS)[labels].assign(
        n_test=[counts.loc[s, "n_test"] for s in SCENARIOS]).to_string())
    print(null_effect.round(3).to_string(index=False))
    print(model_seed_summary[model_seed_summary["metric"].isin(["f1", "precision", "recall", "fpr", "roc_auc"])].round(3).to_string(index=False))
    print(model_seeds.groupby("dataset", sort=False)[[f"flag_rate_{s}" for s in SCENARIOS]].mean().round(2).T.to_string())
    if len(generation):
        print(generation_summary.round(3).to_string(index=False))
    print(cross[["train", "test", "precision", "recall", "f1", "fpr", "roc_auc", "TP", "FP", "FN"]].round(3).to_string(index=False))
    print(cross.set_index(["train", "test"])[[f"flag_rate_{s}" for s in SCENARIOS]].round(2).T.to_string())
    print(swaps.round(3).T.to_string())
    print(json.dumps({k: v for k, v in record.items() if k in ("original_generator_seed42_reproduces_committed_dataset",)}, ensure_ascii=False))
    print({k: v["unchanged_after_run"] for k, v in record["datasets"].items()})
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
