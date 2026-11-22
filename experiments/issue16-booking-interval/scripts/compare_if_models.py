"""원본 · 보정 v1 · 보정 v2 더미데이터로 Isolation Forest 를 같은 조건으로 학습·평가해 비교한다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\compare_if_models.py
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\compare_if_models.py --name my-run --versions v1 v2

조건은 `ml_experiment/train_and_evaluate_if.py` 를 그대로 따른다 (그 파일은 읽지도 고치지도 않고, 같은 값을 여기에 적어 둔다).
  - Feature 7개, 전처리 없음(원래 값 그대로)
  - train_test_split(test_size=0.3, random_state=42, stratify=is_anomaly) — 세 데이터셋에 같은 행 번호를 쓴다
  - IsolationForest(contamination=c, random_state=42, n_estimators=100), 학습은 train 전체(이상 포함, 레이블 미사용)
  - 예측: predict == -1 → 이상(1). 점수: decision_function 을 test 안에서 (max − s)/(max − min + 1e-9) 로 0~1 정규화
  - 기준 구성("최적 모델"): seat_selection_latency 를 뺀 Feature 6개, contamination 0.20
  - 그 밖에 기존 스크립트의 실험 1(contamination 0.05/0.10/0.20, Feature 7개)과 실험 2(Feature 제외 4종)도 같이 돌린다

모델 파일은 저장하지 않는다. 서비스용 모델, 기존 학습·서빙 코드, 데이터셋은 건드리지 않는다.
출력: model_comparison/<name>/ (이미 있으면 --force 가 있어야 다시 쓴다)
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = EXPERIMENT_DIR.parents[1]
ORIGINAL_PATH = REPO_DIR / "ml_experiment" / "data" / "fairticket_synthetic_features.csv"
ADJUSTED_NAME = "fairticket_synthetic_features_adjusted.csv"

# ---- ml_experiment/train_and_evaluate_if.py 와 같은 설정 ----
ALL_FEATURES = ["avg_booking_interval", "booking_interval_variance", "seat_selection_latency", "cancel_rate",
                "card_reuse_count", "ip_booking_density", "booking_count"]
TEST_SIZE, SPLIT_SEED, MODEL_SEED, N_ESTIMATORS = 0.3, 42, 42, 100
BEST_FEATURES = [f for f in ALL_FEATURES if f != "seat_selection_latency"]
BEST_CONTAMINATION = 0.20
PRIMARY = "최적 모델 (Feature 6개, contamination 0.20)"
CONFIGS = [
    (PRIMARY, BEST_FEATURES, BEST_CONTAMINATION),
    ("실험1: Feature 7개, contamination 0.05", ALL_FEATURES, 0.05),
    ("실험1: Feature 7개, contamination 0.10", ALL_FEATURES, 0.10),
    ("실험1·2: Feature 7개, contamination 0.20", ALL_FEATURES, 0.20),
    ("실험2: 시간/분산 제외, contamination 0.20",
     [f for f in ALL_FEATURES if f not in ("avg_booking_interval", "booking_interval_variance")], 0.20),
    ("실험2: 메타 식별자 제외, contamination 0.20",
     [f for f in ALL_FEATURES if f not in ("card_reuse_count", "ip_booking_density")], 0.20),
]
SENSITIVITY_SEEDS = list(range(20))  # 보조 확인: 모델 seed 만 바꿔 본다 (분할은 그대로)

COLORS = ["#2a78d6", "#6250d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID, SURFACE = "#1f2328", "#6b7280", "#e5e7eb", "#ffffff"
MACRO_TYPES = ["fast", "jitter"]
SCENARIO_ORDER = ["normal", "fast_normal", "shared_ip", "frequent_cancel", "hold_cancel", "multi_account", "macro 전체",
                  "macro → fast", "macro → jitter"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_and_score(frame, train_idx, test_idx, features, contamination, seed=MODEL_SEED):
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=N_ESTIMATORS)
    model.fit(frame.loc[train_idx, features])
    x_test = frame.loc[test_idx, features]
    pred = np.where(model.predict(x_test) == -1, 1, 0)
    decision = model.decision_function(x_test)
    normalized = (decision.max() - decision) / (decision.max() - decision.min() + 1e-9)
    return pred, decision, normalized


def metrics(y_true, pred, normalized):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "accuracy": (tp + tn) / len(y_true), "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "roc_auc": roc_auc_score(y_true, normalized), "flagged": int(pred.sum())}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def grouped_bars(ax, categories, series, fmt="{:.2f}", ymax=1.0):
    width = 0.8 / len(series)
    x = np.arange(len(categories))
    for i, (label, values, color) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width - 0.03, color=color, label=label)
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.012, fmt.format(value), ha="center", va="bottom",
                        fontsize=7, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, ymax * 1.12)


def table(frame, columns, digits=3):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append("—" if np.isnan(value) else f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="if-original-v1-v2", help="결과 폴더 이름 (model_comparison/<name>)")
    parser.add_argument("--versions", nargs="+", default=["v1", "v2"], help="비교할 dummy_adjusted 버전")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    paths = {"원본": ORIGINAL_PATH}
    for version in args.versions:
        paths[version] = EXPERIMENT_DIR / "dummy_adjusted" / version / ADJUSTED_NAME
    hashes_before = {label: sha256(path) for label, path in paths.items()}
    data = {label: pd.read_csv(path) for label, path in paths.items()}
    labels = list(data)
    original = data["원본"]

    # 세 데이터셋의 행 순서·레이블이 같은지 확인
    alignment = []
    for label, frame in data.items():
        alignment.append({"dataset": label, "rows": len(frame),
                          "user_id_order_identical": bool(frame["user_id"].equals(original["user_id"])),
                          "is_anomaly_identical": bool(frame["is_anomaly"].equals(original["is_anomaly"])),
                          "columns_identical": list(frame.columns) == list(original.columns),
                          "rows_differing_in_model_features": int((frame[ALL_FEATURES] != original[ALL_FEATURES]).any(axis=1).sum()),
                          "features_differing": ", ".join(f for f in ALL_FEATURES if (frame[f] != original[f]).any()) or "-"})
    alignment = pd.DataFrame(alignment)
    if not (alignment["user_id_order_identical"] & alignment["is_anomaly_identical"] & alignment["columns_identical"]).all():
        print(alignment.to_string(index=False))
        sys.exit("데이터셋의 행 순서·레이블·컬럼이 서로 다릅니다. 비교를 중단합니다.")

    out = EXPERIMENT_DIR / "model_comparison" / args.name
    if out.exists():
        if not args.force:
            sys.exit(f"이미 있는 결과입니다: {out}\n다른 --name 을 쓰거나 --force 를 주세요.")
        if not (out / "experiment_config.json").exists():
            sys.exit(f"{out} 에 experiment_config.json 이 없어 지우지 않았습니다.")
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 기존 스크립트와 같은 분할을 한 번만 만들어 세 데이터셋에 같은 행 번호로 쓴다.
    y = original["is_anomaly"]
    train_idx, test_idx = train_test_split(original.index, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)
    y_test = y.loc[test_idx].to_numpy()
    pd.DataFrame({"row_index": original.index, "user_id": original["user_id"],
                  "split": np.where(original.index.isin(test_idx), "test", "train")}).to_csv(out / "split_indices.csv", index=False)

    # 원본의 macro 행이 보정본에서 어떤 유형이 되는지 (같은 행을 나란히 보기 위한 이름표)
    reference = data[args.versions[-1]]["scenario"]
    for version in args.versions:
        if not data[version]["scenario"].equals(reference):
            sys.exit("보정 버전끼리 macro 행의 유형 배정이 다릅니다. 같은 행끼리 비교할 수 없어 중단합니다.")
    group = original["scenario"].where(original["scenario"] != "macro", "macro → " + reference)

    # 1) 모든 구성 × 데이터셋
    rows, predictions = [], {}
    for config, features, contamination in CONFIGS:
        for label, frame in data.items():
            pred, decision, normalized = fit_and_score(frame, train_idx, test_idx, features, contamination)
            rows.append({"config": config, "dataset": label, "n_features": len(features), "contamination": contamination,
                         **metrics(y_test, pred, normalized)})
            if config == PRIMARY:
                predictions[label] = pd.DataFrame({
                    "row_index": test_idx, "user_id": original.loc[test_idx, "user_id"].to_numpy(),
                    "scenario_in_dataset": frame.loc[test_idx, "scenario"].to_numpy(), "group": group.loc[test_idx].to_numpy(),
                    "is_anomaly": y_test, "pred_anomaly": pred, "decision_function": decision, "anomaly_score_normalized": normalized})
                predictions[label].to_csv(out / f"test_predictions_{label}.csv", index=False, encoding="utf-8-sig")
    results = pd.DataFrame(rows)
    results.to_csv(out / "metrics_all_configs.csv", index=False, encoding="utf-8-sig")
    primary = results[results["config"] == PRIMARY].reset_index(drop=True)
    primary.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    primary[["dataset", "TP", "FP", "TN", "FN", "flagged"]].to_csv(out / "confusion_matrix.csv", index=False, encoding="utf-8-sig")

    # 2) 시나리오별 이상 판정 비율 (기준 구성, test)
    rows = []
    for label, frame in predictions.items():
        parts = {name: frame[frame["group"] == name] for name in frame["group"].unique()}
        parts["macro 전체"] = frame[frame["group"].str.startswith("macro")]
        for name in SCENARIO_ORDER:
            if name not in parts:
                continue
            part = parts[name]
            actual = int(part["is_anomaly"].iloc[0])
            rows.append({"group": name, "actual": "이상" if actual else "정상", "dataset": label, "n_test": len(part),
                         "flagged": int(part["pred_anomaly"].sum()), "flag_rate": part["pred_anomaly"].mean(),
                         "meaning": "탐지율" if actual else "오탐률",
                         "decision_median": part["decision_function"].median(),
                         "decision_min": part["decision_function"].min(), "decision_max": part["decision_function"].max()})
    scenario = pd.DataFrame(rows)
    scenario.to_csv(out / "scenario_detection.csv", index=False, encoding="utf-8-sig")

    # 3) 원본 대비 판정이 바뀐 test 행
    rows = []
    base_pred = predictions["원본"].set_index("row_index")
    for label in labels[1:]:
        other = predictions[label].set_index("row_index")
        for name in [n for n in SCENARIO_ORDER if n != "macro 전체"]:
            mask = base_pred["group"] == name
            if not mask.any():
                continue
            before, after = base_pred.loc[mask, "pred_anomaly"], other.loc[mask, "pred_anomaly"]
            rows.append({"dataset": label, "group": name, "actual": "이상" if base_pred.loc[mask, "is_anomaly"].iloc[0] else "정상",
                         "n_test": int(mask.sum()), "flagged_to_not": int(((before == 1) & (after == 0)).sum()),
                         "not_to_flagged": int(((before == 0) & (after == 1)).sum())})
    flips = pd.DataFrame(rows)
    flips.to_csv(out / "prediction_changes_vs_original.csv", index=False, encoding="utf-8-sig")

    # 4) 이상 점수 분포 요약 (decision_function: 0 미만이면 이상 판정)
    rows = []
    for label, frame in predictions.items():
        for name, part in (("정상 전체", frame[frame["is_anomaly"] == 0]), ("이상 전체", frame[frame["is_anomaly"] == 1])):
            d = part["decision_function"]
            rows.append({"dataset": label, "group": name, "n_test": len(part), "mean": d.mean(), "median": d.median(),
                         "std": d.std(), "min": d.min(), "q25": d.quantile(0.25), "q75": d.quantile(0.75), "max": d.max(),
                         "share_below_0": float((d < 0).mean())})
    score_stats = pd.DataFrame(rows)
    score_stats.to_csv(out / "score_distribution.csv", index=False, encoding="utf-8-sig")

    # 5) 보조: 모델 seed 만 바꿔 본 결과 (분할·그 밖의 설정은 그대로)
    rows = []
    for seed in SENSITIVITY_SEEDS:
        for label, frame in data.items():
            pred, decision, normalized = fit_and_score(frame, train_idx, test_idx, BEST_FEATURES, BEST_CONTAMINATION, seed)
            row = {"model_seed": seed, "dataset": label, **metrics(y_test, pred, normalized)}
            test_group = group.loc[test_idx].to_numpy()
            for name in ["hold_cancel", "multi_account"] + [f"macro → {t}" for t in MACRO_TYPES]:
                mask = test_group == name
                row[f"recall_{name}"] = pred[mask].mean() if mask.any() else np.nan
            row["recall_macro 전체"] = pred[np.char.startswith(test_group.astype(str), "macro")].mean()
            rows.append(row)
    seeds = pd.DataFrame(rows)
    seeds.to_csv(out / "seed_sensitivity_runs.csv", index=False, encoding="utf-8-sig")
    seed_cols = ["f1", "precision", "recall", "accuracy", "fpr", "roc_auc", "recall_macro 전체", "recall_macro → fast",
                 "recall_macro → jitter", "recall_hold_cancel", "recall_multi_account"]
    seed_summary = (seeds.groupby("dataset", sort=False)[seed_cols].agg(["mean", "std", "min", "max"]).stack(level=0, future_stack=True)
                    .reset_index().rename(columns={"level_1": "metric"}))
    seed_summary.to_csv(out / "seed_sensitivity_summary.csv", index=False, encoding="utf-8-sig")

    # ---------- 그래프 ----------
    color_of = dict(zip(labels, COLORS))
    fig, ax = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
    style(ax)
    names = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("accuracy", "Accuracy"), ("fpr", "오탐률(FPR)"), ("roc_auc", "ROC-AUC")]
    grouped_bars(ax, [n for _, n in names], [(label, [primary.loc[primary["dataset"] == label, key].squeeze() for key, _ in names],
                                              color_of[label]) for label in labels], fmt="{:.3f}")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=len(labels))
    ax.set_title(f"전체 성능 — {PRIMARY}, test {len(test_idx)}건", loc="left", fontsize=11, color=INK)
    save(fig, out / "figures" / "overall_metrics.png")

    order = [n for n in SCENARIO_ORDER if n in scenario["group"].unique()]
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    style(ax)
    counts = scenario.drop_duplicates("group").set_index("group")
    grouped_bars(ax, [f"{n}\n({counts.loc[n, 'actual']}, n={counts.loc[n, 'n_test']})" for n in order],
                 [(label, [scenario.loc[(scenario["dataset"] == label) & (scenario["group"] == n), "flag_rate"].squeeze() for n in order],
                   color_of[label]) for label in labels])
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=len(labels))
    ax.set_ylabel("이상 판정 비율", fontsize=9, color=MUTED)
    ax.set_title("시나리오별 이상 판정 비율 (정상 = 오탐률, 이상 = 탐지율). macro → fast/jitter 는 같은 행을 세 데이터셋에서 본 것",
                 loc="left", fontsize=10, color=INK)
    save(fig, out / "figures" / "scenario_detection.png")

    rng = np.random.default_rng(0)  # 점을 좌우로 흩는 용도 (값에는 영향 없음)
    point_groups = [n for n in order if n != "macro 전체"]
    fig, axes = plt.subplots(len(labels), 1, figsize=(11, 3.1 * len(labels)), sharex=True, sharey=True, constrained_layout=True)
    for ax, label in zip(axes, labels):
        style(ax)
        frame = predictions[label]
        for position, name in enumerate(point_groups):
            part = frame[frame["group"] == name]
            ax.scatter(position + rng.uniform(-0.28, 0.28, len(part)), part["decision_function"], s=14, color=color_of[label],
                       edgecolor=SURFACE, linewidth=0.4, alpha=0.9)
            ax.hlines(part["decision_function"].median(), position - 0.36, position + 0.36, color=INK, linewidth=1.4)
        ax.axhline(0, color=INK, linewidth=1, linestyle="--")
        ax.set_ylabel("decision_function", fontsize=9, color=MUTED)
        ax.set_title(f"{label}  (점선 아래 = 이상 판정, 가로선 = 중앙값)", loc="left", fontsize=10, color=INK)
    axes[-1].set_xticks(range(len(point_groups)))
    axes[-1].set_xticklabels([f"{n}\n({counts.loc[n, 'actual']})" for n in point_groups], fontsize=9)
    fig.suptitle("이상 점수 분포 (test, 시나리오별). 값이 낮을수록 이상에 가깝다", fontsize=12, color=INK)
    save(fig, out / "figures" / "score_distribution_by_scenario.png")

    bins = np.linspace(min(f["decision_function"].min() for f in predictions.values()),
                       max(f["decision_function"].max() for f in predictions.values()), 41)
    fig, axes = plt.subplots(len(labels), 1, figsize=(8.5, 2.6 * len(labels)), sharex=True, constrained_layout=True)
    for ax, label in zip(axes, labels):
        style(ax)
        frame = predictions[label]
        ax.hist(frame.loc[frame["is_anomaly"] == 0, "decision_function"], bins=bins, color="#9aa3ad", edgecolor=SURFACE, linewidth=0.8,
                label="정상")
        ax.hist(frame.loc[frame["is_anomaly"] == 1, "decision_function"], bins=bins, color=color_of[label], edgecolor=SURFACE,
                linewidth=0.8, alpha=0.85, label="이상")
        ax.axvline(0, color=INK, linewidth=1, linestyle="--")
        ax.set_ylabel("계정 수", fontsize=9, color=MUTED)
        ax.set_title(label, loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    axes[-1].set_xlabel("decision_function (점선 왼쪽 = 이상 판정)", fontsize=9, color=MUTED)
    fig.suptitle("이상 점수 분포: 정상 대 이상 (test)", fontsize=12, color=INK)
    save(fig, out / "figures" / "score_distribution_normal_vs_anomaly.png")

    shown = [("f1", "F1"), ("recall", "Recall"), ("fpr", "오탐률(FPR)"), ("recall_macro 전체", "macro 탐지율"),
             ("recall_hold_cancel", "hold_cancel 탐지율")]
    fig, axes = plt.subplots(1, len(shown), figsize=(3 * len(shown), 4), constrained_layout=True)
    for ax, (key, title) in zip(axes, shown):
        style(ax)
        for position, label in enumerate(labels):
            values = seeds.loc[seeds["dataset"] == label, key].to_numpy()
            ax.scatter(position + rng.uniform(-0.18, 0.18, len(values)), values, s=18, color=color_of[label], edgecolor=SURFACE, linewidth=0.5)
            ax.hlines(values.mean(), position - 0.3, position + 0.3, color=INK, linewidth=1.4)
            fixed = primary.loc[primary["dataset"] == label, key].squeeze() if key in primary else scenario.loc[
                (scenario["dataset"] == label) & (scenario["group"] == key.replace("recall_", "")), "flag_rate"].squeeze()
            ax.scatter([position], [fixed], s=70, marker="D", facecolor="none", edgecolor=INK, linewidth=1.2, zorder=5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.suptitle(f"모델 seed {len(SENSITIVITY_SEEDS)}개로 본 흔들림 (가로선 = 평균, ◇ = 기존 설정 seed {MODEL_SEED})", fontsize=12, color=INK)
    save(fig, out / "figures" / "seed_sensitivity.png")

    # ---------- 설정 기록 ----------
    hashes_after = {label: sha256(path) for label, path in paths.items()}
    config = {
        "name": args.name, "script": "scripts/compare_if_models.py",
        "command": "python scripts/compare_if_models.py --name " + args.name + " --versions " + " ".join(args.versions),
        "reference_training_script": "ml_experiment/train_and_evaluate_if.py (수정하지 않음)",
        "datasets": {label: {"path": path.relative_to(REPO_DIR).as_posix(), "sha256": hashes_before[label],
                             "unchanged_after_run": hashes_before[label] == hashes_after[label]} for label, path in paths.items()},
        "all_features": ALL_FEATURES, "primary_features": BEST_FEATURES, "preprocessing": "없음 (원래 값 그대로)",
        "split": {"test_size": TEST_SIZE, "random_state": SPLIT_SEED, "stratify": "is_anomaly", "train": len(train_idx),
                  "test": len(test_idx), "test_anomaly_ratio": float(y_test.mean())},
        "model": {"class": "sklearn.ensemble.IsolationForest", "n_estimators": N_ESTIMATORS, "random_state": MODEL_SEED,
                  "max_samples": "auto (기본값)", "primary_contamination": BEST_CONTAMINATION},
        "prediction": "predict == -1 → 이상(1)",
        "score": "decision_function; 정규화 점수 = (max − s)/(max − min + 1e-9) (test 안에서)",
        "configs": [{"name": name, "features": features, "contamination": c} for name, features, c in CONFIGS],
        "sensitivity_model_seeds": SENSITIVITY_SEEDS,
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                     "python": sys.version.split()[0]},
        "models_saved": False,
    }
    (out / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    alignment.to_csv(out / "dataset_alignment.csv", index=False, encoding="utf-8-sig")

    # ---------- 보고서 ----------
    metric_cols = ["dataset", "precision", "recall", "f1", "accuracy", "fpr", "roc_auc", "TP", "FP", "TN", "FN", "flagged"]
    wide = scenario.pivot(index="group", columns="dataset", values="flagged").reindex(order)[labels]
    rate = scenario.pivot(index="group", columns="dataset", values="flag_rate").reindex(order)[labels]
    scen_table = pd.DataFrame({"group": order, "actual": [counts.loc[n, "actual"] for n in order],
                               "n_test": [int(counts.loc[n, "n_test"]) for n in order]})
    for label in labels:
        scen_table[label] = [f"{int(wide.loc[n, label])} ({rate.loc[n, label] * 100:.1f}%)" for n in order]
    normal_fp = scenario[(scenario["actual"] == "정상")].pivot(index="group", columns="dataset", values="flagged")[labels]
    base_row = primary.iloc[0]
    delta_lines = []
    for label in labels[1:]:
        row = primary[primary["dataset"] == label].iloc[0]
        delta_lines.append(
            f"- {label}: F1 {base_row['f1']:.3f} → {row['f1']:.3f} ({row['f1'] - base_row['f1']:+.3f}), "
            f"Recall {base_row['recall']:.3f} → {row['recall']:.3f} (TP {int(base_row['TP'])} → {int(row['TP'])}), "
            f"Precision {base_row['precision']:.3f} → {row['precision']:.3f}, "
            f"오탐률 {base_row['fpr']:.3f} → {row['fpr']:.3f} (FP {int(base_row['FP'])} → {int(row['FP'])}), "
            f"Accuracy {base_row['accuracy']:.3f} → {row['accuracy']:.3f}.")
    seed_wide = seed_summary[seed_summary["metric"].isin(["f1", "recall", "fpr", "recall_macro 전체", "recall_macro → fast",
                                                          "recall_macro → jitter", "recall_hold_cancel", "recall_multi_account"])]
    medians = []
    for label, frame in data.items():
        for name, part in frame.loc[train_idx].assign(group=group.loc[train_idx]).groupby("group"):
            if name.startswith("macro") or name in ("hold_cancel", "multi_account"):
                medians.append({"dataset": label, "group": name, "n_train": len(part),
                                "avg_booking_interval_median": part["avg_booking_interval"].median(),
                                "booking_interval_variance_median": part["booking_interval_variance"].median()})
    medians = pd.DataFrame(medians).sort_values(["group", "dataset"])
    medians.to_csv(out / "train_feature_medians.csv", index=False, encoding="utf-8-sig")

    # 분석 문단에 쓰는 값
    seed_std = seeds.groupby("dataset", sort=False)[["f1", "recall", "fpr"]].std()
    seed_mean = seeds.groupby("dataset", sort=False)[seed_cols].mean()
    ranges = original.loc[train_idx, ["avg_booking_interval", "booking_interval_variance"]].agg(["min", "max"])
    no_time = results[results["config"].str.contains("시간/분산 제외")].iloc[0]
    analysis = [
        "## 분석",
        "",
        "### 성능 변화의 크기",
        "",
        *[f"- {label}: 기존 설정(seed {MODEL_SEED})에서 F1 차이는 {primary.loc[primary['dataset'] == label, 'f1'].squeeze() - base_row['f1']:+.3f}, "
          f"TP 차이는 {int(primary.loc[primary['dataset'] == label, 'TP'].squeeze() - base_row['TP']):+d}건, "
          f"FP 차이는 {int(primary.loc[primary['dataset'] == label, 'FP'].squeeze() - base_row['FP']):+d}건이다. "
          f"모델 seed {len(SENSITIVITY_SEEDS)}개의 평균 F1 은 원본 {seed_mean.loc['원본', 'f1']:.3f} 대 {label} {seed_mean.loc[label, 'f1']:.3f} 이다."
          for label in labels[1:]],
        f"- 같은 데이터셋에서 모델 seed 만 바꿔도 F1 이 표준편차 {seed_std.loc['원본', 'f1']:.3f} 만큼 움직인다 (원본 기준). "
        "데이터셋 사이의 차이는 이보다 작으므로, **이 실험으로는 보정이 성능을 높였다고도 낮췄다고도 말할 수 없다.**",
        f"- test 는 {len(test_idx)}건이고 이상은 {int(y_test.sum())}건이다. 1건이 바뀌면 Recall 이 {1 / y_test.sum():.3f} 움직인다.",
        "",
        "### 가능한 해석 (이 실험에서 확인한 것과 추정을 구분)",
        "",
        f"- **확인**: 두 Feature 를 아예 빼도 F1 이 {no_time['f1']:.3f} 이다 (`시간/분산 제외` 구성). 이 모델에서 매크로를 포함한 이상 계정의 판정은 "
        "예매 간격보다 다른 Feature(취소율, 카드·IP 수, 예매 횟수)에 더 기대고 있다.",
        f"- **확인**: 전처리가 없어서 Feature 는 원래 단위 그대로 쓰인다. train 에서 `avg_booking_interval` 은 {ranges.loc['min', 'avg_booking_interval']:.1f}~"
        f"{ranges.loc['max', 'avg_booking_interval']:.0f}초, `booking_interval_variance` 는 {ranges.loc['min', 'booking_interval_variance']:.2f}~"
        f"{ranges.loc['max', 'booking_interval_variance']:.0f}초² 에 걸쳐 있다. 매크로의 값이 0.1~1.3초에서 2.5~5.2초로 옮겨져도 이 범위의 아래쪽 끝에 그대로 있다.",
        "- **추정**: Isolation Forest 는 Feature 범위 안에서 무작위로 나누는 위치를 고르므로, 범위의 아래쪽 끝에서 몇 초 옮겨진 것은 "
        "나뉘는 깊이를 거의 바꾸지 않는다. 그래서 매크로 행의 판정이 거의 그대로였을 것이다.",
        "- **추정**: 보정 뒤 매크로는 `hold_cancel`·`multi_account` 와 같은 2~10초 구간에 놓인다. 이 구간의 계정이 늘어 서로를 덜 고립시킬 수 있지만, "
        "이번 결과에서는 그 영향이 seed 흔들림보다 크게 나타나지 않았다.",
        "- **확인**: 정상 사용자의 오탐 총수는 세 데이터셋에서 같고, 시나리오 안에서 1~2건씩 자리가 바뀌었다. contamination 이 고정이라 "
        "train 점수의 하위 20% 를 이상으로 자르는데, 매크로 행의 점수가 조금 달라지면 경계 근처의 다른 행이 넘어오거나 빠진다.",
        "- `macro → fast` 와 `macro → jitter` 의 탐지율이 원본에서도 다른 것은, 원본에서는 두 묶음이 같은 분포에서 나온 행이므로 "
        f"행 수가 적어서(test {int(counts.loc['macro → fast', 'n_test'])}건, {int(counts.loc['macro → jitter', 'n_test'])}건) 생긴 차이다.",
        "",
        "### 이 결과가 말해 주지 않는 것",
        "",
        "- 평가 데이터도 같은 합성 데이터다. 매크로의 예매 간격만 실측에 맞췄고, 나머지 Feature 와 정상 사용자·다른 이상 시나리오는 임의로 만든 값 그대로다.",
        "- 따라서 여기의 수치는 실제 서비스에서의 탐지율·오탐률이 아니다. 실제 매크로 계정의 취소율·예매 횟수 등이 더미와 다르면 결과는 달라진다.",
        "",
    ]

    lines = [
        f"# Isolation Forest 비교: {' · '.join(labels)}",
        "",
        "자동 생성 (`scripts/compare_if_models.py`). 합성 데이터 안에서의 비교이며, 실제 환경의 탐지 성능을 뜻하지 않는다.",
        "",
        "## 실험 조건 (`ml_experiment/train_and_evaluate_if.py` 와 같음)",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| Feature | 기준 구성은 6개: {', '.join(BEST_FEATURES)} (`seat_selection_latency` 제외) |",
        "| 전처리 | 없음. 스케일링·변환 없이 원래 값 그대로 |",
        f"| 분할 | `train_test_split(test_size={TEST_SIZE}, random_state={SPLIT_SEED}, stratify=is_anomaly)` → train {len(train_idx)} / test {len(test_idx)} (test 이상 비율 {y_test.mean() * 100:.1f}%) |",
        f"| 모델 | `IsolationForest(n_estimators={N_ESTIMATORS}, random_state={MODEL_SEED}, contamination={BEST_CONTAMINATION})`, 나머지 기본값 |",
        "| 학습 | train 전체로 학습 (이상 계정 포함, 레이블은 쓰지 않음) |",
        "| 예측 | `predict == -1` → 이상(1) |",
        "| 점수 | `decision_function` (0 미만이면 이상 판정). ROC-AUC 는 test 안에서 0~1 로 정규화한 점수로 계산 |",
        f"| 버전 | scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__} |",
        "",
        "세 데이터셋에 같은 행 번호(`split_indices.csv`)를 썼다. 모델 파일은 저장하지 않았다.",
        "",
        "### 데이터셋",
        "",
        "| 데이터셋 | 경로 | sha256 |",
        "|---|---|---|",
        *[f"| {label} | `{path.relative_to(REPO_DIR).as_posix()}` | `{hashes_before[label][:16]}…` |" for label, path in paths.items()],
        "",
        table(alignment, list(alignment.columns)),
        "",
        "## 전체 성능 (기준 구성, test)",
        "",
        "![전체 성능](figures/overall_metrics.png)",
        "",
        table(primary, metric_cols),
        "",
        *delta_lines,
        "",
        "## 시나리오별 이상 판정 (기준 구성, test)",
        "",
        "![시나리오별](figures/scenario_detection.png)",
        "",
        table(scen_table, list(scen_table.columns)),
        "",
        "- 값은 `이상 판정 수 (비율)`. 정상 시나리오에서는 오탐률, 이상 시나리오에서는 탐지율이다.",
        "- `macro → fast`, `macro → jitter` 는 보정본에서 fast / jitter 가 된 **같은 행**이다. 원본 열은 그 행들이 원본 값(0.1~1.3초)일 때의 결과다.",
        "",
        "### 원본 대비 판정이 바뀐 test 행",
        "",
        table(flips, list(flips.columns)),
        "",
        "## 이상 점수 분포",
        "",
        "![시나리오별 점수](figures/score_distribution_by_scenario.png)",
        "",
        "![정상 대 이상](figures/score_distribution_normal_vs_anomaly.png)",
        "",
        table(score_stats, list(score_stats.columns)),
        "",
        table(scenario[scenario["group"].isin(["hold_cancel", "multi_account", "macro 전체", "macro → fast", "macro → jitter"])],
              ["group", "dataset", "n_test", "flagged", "flag_rate", "decision_median", "decision_min", "decision_max"]),
        "",
        "## 기존 스크립트의 다른 구성",
        "",
        table(results[results["config"] != PRIMARY], ["config", "dataset", "precision", "recall", "f1", "accuracy", "fpr", "roc_auc", "TP", "FP", "TN", "FN"]),
        "",
        "`시간/분산 제외` 구성은 바뀐 두 Feature 를 쓰지 않으므로 세 데이터셋의 결과가 같아야 한다 (같으면 비교 절차가 올바르다는 확인이 된다).",
        "",
        "## 보조: 모델 seed 에 따른 흔들림",
        "",
        "![seed](figures/seed_sensitivity.png)",
        "",
        f"기존 설정은 seed {MODEL_SEED} 하나다. 차이가 seed 우연인지 보려고 모델 seed 만 {SENSITIVITY_SEEDS[0]}~{SENSITIVITY_SEEDS[-1]} 로 바꿔 다시 학습했다 (분할과 나머지 설정은 그대로).",
        "",
        table(seed_wide, ["dataset", "metric", "mean", "std", "min", "max"]),
        "",
        "## 해석에 쓴 값: train 의 이상 시나리오 중앙값",
        "",
        table(medians, list(medians.columns)),
        "",
        *analysis,
        "## 재현",
        "",
        "```powershell",
        "cd ai\\experiments\\issue16-booking-interval",
        "$py = \"..\\issue14-real-validation\\.venv\\Scripts\\python\"",
        f"& $py scripts\\compare_if_models.py --name {args.name} --versions {' '.join(args.versions)} --force",
        "```",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(alignment.to_string(index=False))
    print(results.round(4).to_string(index=False))
    print(scen_table.to_string(index=False))
    print(flips.to_string(index=False))
    print(seed_wide.round(3).to_string(index=False))
    print(medians.round(3).to_string(index=False))
    print(normal_fp.to_string())
    print({k: v["unchanged_after_run"] for k, v in config["datasets"].items()})
    print(f"보고서: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
