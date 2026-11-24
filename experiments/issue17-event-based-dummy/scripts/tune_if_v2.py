"""이벤트 기반 v2 데이터셋(C)에서 Isolation Forest 의 설정을 검증 데이터로 골라 성능이 얼마나 나아지는지 확인한다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\tune_if_v2.py

바꾸지 않는 것: Feature 6개(BEST_FEATURES), v2 데이터셋·이벤트 생성 규칙, 트리 수(100)·max_samples(기본값). 새 Feature 는 만들지 않는다.
바꿔 보는 것 (모두 기존 서빙 코드에 이미 있는 선택지다)
  - INSUFFICIENT_DATA 기준: min_booking_count 1 (현재 번들) / 2 (docs/predict_feature_spec.md, train_model.py)
  - 결측 처리: 그대로(NaN) / 중앙값 / 0  (preprocessing.build_preprocessor 의 imputation)
  - log1p 변환 여부 (preprocessing.FEATURE_SPECS 의 log 후보 3개: 간격 평균·분산, 시도 수)
  - contamination (= 학습 점수의 분위수로 정하는 판정 임계값), 그 밖의 임계값 방식

분할
  - test: 기존 실험과 같은 300건 (train_test_split(test_size=0.3, random_state=42, stratify=is_anomaly))
  - 나머지 700건을 train 490 / validation 210 으로 나눈다 (random_state=42, stratify=scenario)
  - 설정은 train 으로 학습해 validation F1(모델 seed 10개 평균)으로만 고른다. test 는 고른 뒤에 한 번 평가한다.
  - 분포 중첩·오탐/미탐 원인 분석도 test 를 뺀 700건(또는 validation)에서만 한다.

INSUFFICIENT_DATA 로 분리된 계정은 점수를 받지 않으므로 "이상 아님"으로 세고 전체 300건 기준으로 지표를 계산한다
(ROC-AUC 는 분리된 계정에 가장 낮은 이상 점수를 준 값과, 점수를 받은 계정만으로 계산한 값을 같이 적는다).

출력: output/v2/model_tuning/
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
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP.parent / "issue16-booking-interval" / "scripts"))
sys.path.insert(0, str(REPO))
from compare_if import SCENARIOS  # noqa: E402
from compare_if_models import (BEST_CONTAMINATION, BEST_FEATURES, GRID, INK, MODEL_SEED, MUTED, N_ESTIMATORS, SPLIT_SEED,  # noqa: E402
                               SURFACE, TEST_SIZE, fit_and_score, grouped_bars, metrics, save, style)
from generate_events import ORIGINAL_PATH, sha256  # noqa: E402
from preprocessing import FEATURE_SPECS, SelectiveLog1p, build_preprocessor  # noqa: E402

OUT = EXP / "output" / "v2" / "model_tuning"
C_PATH = EXP / "data" / "v2" / "fairticket_event_based_features.csv"
F = BEST_FEATURES
VAL_SIZE = 0.3
SELECT_SEEDS = list(range(10))   # 설정 선택에 쓰는 모델 seed (validation F1 평균)
MODEL_SEEDS = list(range(20))    # 안정성 확인
SPLIT_SEEDS = list(range(20))
NESTED_SELECT_SEEDS = list(range(5))
CONTAMINATIONS = [round(c, 3) for c in np.arange(0.05, 0.4001, 0.025)]
POLICIES = [(min_bc, missing, log) for min_bc in (1, 2) for missing in ("raw", "median", "zero") for log in (False, True)]
BASELINE = (1, "raw", False)
NORMAL_SCENARIOS = SCENARIOS[:4]
C_A, C_BASE, C_SEL = "#2a78d6", "#1baf7a", "#6250d6"
C_NORMAL, C_ANOM = "#2a78d6", "#eb6834"
LOG_AXIS = ("avg_booking_interval", "booking_interval_variance", "booking_count")


def policy_name(policy, contamination=None):
    min_bc, missing, log = policy
    text = (f"시도 {min_bc}건 미만 분리" if min_bc > 1 else "분리 없음") + " · 결측 " + {"raw": "그대로", "median": "중앙값", "zero": "0"}[missing]
    text += " · log1p" if log else ""
    return text + (f" · c={contamination:g}" if contamination is not None else "")


def transformers(policy):
    """결측 대체(preprocessing.build_preprocessor) → log1p(preprocessing.SelectiveLog1p) 순서. 스케일링은 트리 분할에 영향이 없어 쓰지 않는다."""
    _, missing, log = policy
    imputer = None if missing == "raw" else build_preprocessor(F, imputation=missing, scaling="none", log_transform=False)
    log1p = SelectiveLog1p(indices=tuple(i for i, col in enumerate(F) if FEATURE_SPECS[col]["log"])) if log else None
    return imputer, log1p


class Scorer:
    """정책대로 학습하고 이상 점수(-score_samples, 클수록 이상)를 낸다. 분리된 계정의 점수는 NaN."""

    def __init__(self, frame, fit_idx, policy, seed=MODEL_SEED):
        self.policy, self.frame = policy, frame
        fit_idx = pd.Index(fit_idx)
        fit_idx = fit_idx[(frame.loc[fit_idx, "booking_count"] >= policy[0]).to_numpy()]
        self.imputer, self.log1p = transformers(policy)
        x = frame.loc[fit_idx, F].to_numpy(dtype=float)
        if self.imputer is not None:
            self.imputer.fit(x)
        x = self.prepare(x)
        self.fill = np.nanmedian(frame.loc[fit_idx, F].to_numpy(dtype=float), axis=0)  # 원인 분석에서 Feature 를 중립값으로 바꿀 때만 쓴다
        self.model = IsolationForest(contamination="auto", random_state=seed, n_estimators=N_ESTIMATORS).fit(x)
        self.fit_scores = -self.model.score_samples(x)

    def prepare(self, x):
        x = np.asarray(x, dtype=float)
        x = self.imputer.transform(x) if self.imputer is not None else x
        return self.log1p.transform(x) if self.log1p is not None else x

    def raw(self, x):
        return -self.model.score_samples(self.prepare(x))

    def score(self, idx):
        out = np.full(len(idx), np.nan)
        ok = (self.frame.loc[idx, "booking_count"] >= self.policy[0]).to_numpy()
        if ok.any():
            out[ok] = self.raw(self.frame.loc[pd.Index(idx)[ok], F].to_numpy(dtype=float))
        return out

    def threshold(self, contamination):
        """sklearn 의 contamination 과 같다: 학습 점수의 상위 contamination 분위수."""
        return np.percentile(self.fit_scores, 100 * (1 - contamination))


def evaluate(y, scores, threshold):
    """분리된 계정(점수 NaN)은 이상 아님으로 세고 전체 기준으로 계산한다."""
    scored = ~np.isnan(scores)
    pred = np.where(scored & (scores > threshold), 1, 0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    full = np.where(scored, scores, np.nanmin(scores) - 1.0)
    row = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
           "fpr": fp / (fp + tn) if fp + tn else 0.0, "roc_auc": roc_auc_score(y, full),
           "roc_auc_scored_only": roc_auc_score(y[scored], scores[scored]), "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
           "flagged": int(pred.sum()), "insufficient": int((~scored).sum()), "insufficient_anomaly": int(y[~scored].sum())}
    return row, pred


def best_threshold(y, scores):
    """validation 레이블로 F1 이 가장 높은 점수 임계값을 찾는다."""
    best = (-1.0, np.inf)
    for value in np.unique(scores[~np.isnan(scores)]):
        f1 = evaluate(y, scores, value - 1e-12)[0]["f1"]
        if f1 > best[0]:
            best = (f1, value - 1e-12)
    return best[1]


def flag_rates(pred, scenario):
    return {f"flag_rate_{s}": pred[scenario == s].mean() if (scenario == s).any() else np.nan for s in SCENARIOS}


def mean_std(frame, by, columns):
    out = frame.groupby(by, sort=False)[columns].agg(["mean", "std", "min", "max"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def select(grid):
    """validation F1 평균이 가장 높은 설정. 동률이면 기존 설정에 가까운 쪽(변환 없음, 분리 없음, contamination 0.20 에 가까움)."""
    ranked = grid.assign(distance=(grid["contamination"] - BEST_CONTAMINATION).abs()).sort_values(
        ["f1", "log", "min_booking_count", "distance"], ascending=[False, True, True, True])
    top = ranked.iloc[0]
    return (int(top["min_booking_count"]), top["missing"], bool(top["log"])), float(top["contamination"])


def grid_on_validation(frame, y, scenario, train_idx, val_idx, seeds):
    y_val, rows = y.loc[val_idx].to_numpy(), []
    for policy in POLICIES:
        for seed in seeds:
            scorer = Scorer(frame, train_idx, policy, seed)
            scores = scorer.score(val_idx)
            for c in CONTAMINATIONS:
                row, _ = evaluate(y_val, scores, scorer.threshold(c))
                rows.append({"min_booking_count": policy[0], "missing": policy[1], "log": policy[2], "contamination": c, "seed": seed, **row})
    runs = pd.DataFrame(rows)
    keys = ["min_booking_count", "missing", "log", "contamination"]
    summary = runs.groupby(keys, sort=False)[["f1", "precision", "recall", "fpr", "roc_auc"]].mean().reset_index()
    summary["f1_std"] = runs.groupby(keys, sort=False)["f1"].std().to_numpy()
    return summary


def overlap_coefficient(a, b, log):
    a, b = (np.log1p(a), np.log1p(b)) if log else (a, b)
    edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=30)
    pa, pb = np.histogram(a, edges)[0] / len(a), np.histogram(b, edges)[0] / len(b)
    return float(np.minimum(pa, pb).sum())


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
    dev_idx, test_idx = train_test_split(data.index, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y)
    train_idx, val_idx = train_test_split(dev_idx, test_size=VAL_SIZE, random_state=SPLIT_SEED, stratify=scenario.loc[dev_idx])
    y_val, y_test = y.loc[val_idx].to_numpy(), y.loc[test_idx].to_numpy()
    s_val, s_test = scenario.loc[val_idx].to_numpy(), scenario.loc[test_idx].to_numpy()

    def csv(frame, name):
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    # ---- 0) 기존 결과 재현 (기존 함수 그대로) + 이 스크립트의 점수 계산이 그것과 같은지 확인 ----
    reference = {}
    for label, frame in (("A", original), ("C", data)):
        pred, _, normalized = fit_and_score(frame, dev_idx, test_idx, F, BEST_CONTAMINATION)
        reference[label] = {**metrics(y_test, pred, normalized), **flag_rates(pred, s_test)}
    scorer = Scorer(data, dev_idx, BASELINE)
    own, own_pred = evaluate(y_test, scorer.score(test_idx), scorer.threshold(BEST_CONTAMINATION))
    assert (own["TP"], own["FP"], own["FN"]) == (reference["C"]["TP"], reference["C"]["FP"], reference["C"]["FN"]), "기존 결과와 다릅니다"
    print(f"기존 결과 재현: A F1 {reference['A']['f1']:.3f}, C F1 {reference['C']['f1']:.3f} (이 스크립트의 계산과 일치)")

    # ---- 1) 정상/이상 Feature 분포 중첩 (test 를 뺀 700건) ----
    dev = data.loc[dev_idx]
    normal = dev[dev["is_anomaly"] == 0]
    rows = []
    for feature in F:
        base = normal[feature].dropna().to_numpy()
        low, high = np.percentile(base, [5, 95])
        for group in ["이상 전체"] + SCENARIOS[4:]:
            part = dev[dev["is_anomaly"] == 1] if group == "이상 전체" else dev[dev["scenario"] == group]
            values = part[feature].dropna().to_numpy()
            auc = roc_auc_score(np.r_[np.zeros(len(base)), np.ones(len(values))], np.r_[base, values])
            rows.append({"feature": feature, "group": group, "n_anomaly": len(part), "anomaly_null": int(part[feature].isna().sum()),
                         "normal_null": int(normal[feature].isna().sum()), "normal_median": np.median(base), "normal_p05": low, "normal_p95": high,
                         "anomaly_median": np.median(values), "anomaly_min": values.min(), "anomaly_max": values.max(),
                         "direction": "이상이 더 큼" if auc >= 0.5 else "이상이 더 작음", "separability_auc": max(auc, 1 - auc),
                         "ks": ks_2samp(base, values).statistic, "overlap_coefficient": overlap_coefficient(base, values, feature in LOG_AXIS),
                         "anomaly_inside_normal_p05_p95": float(((values >= low) & (values <= high)).mean()),
                         # Isolation Forest 는 범위의 바깥쪽에 있는 값을 먼저 고립시킨다: 이상 계정이 전체 분포의 어디에 있는지
                         "anomaly_median_percentile_in_all": float((dev[feature].dropna().to_numpy() <= np.median(values)).mean()),
                         "normal_share_beyond_anomaly_median": float((base >= np.median(values)).mean() if auc >= 0.5 else (base <= np.median(values)).mean())})
    overlap = pd.DataFrame(rows)
    csv(overlap, "feature_overlap.csv")

    bounds = {f: np.percentile(normal[f].dropna(), [5, 95]) for f in F}
    inside = pd.DataFrame({f: dev[f].isna() | ((dev[f] >= bounds[f][0]) & (dev[f] <= bounds[f][1])) for f in F})
    joint = (pd.DataFrame({"scenario": dev["scenario"], "n_features_outside": (~inside).sum(axis=1)})
             .groupby("scenario")["n_features_outside"].agg(n="size", all_inside=lambda v: (v == 0).mean(), one_outside=lambda v: (v == 1).mean(),
                                                           two_or_more_outside=lambda v: (v >= 2).mean()).reindex(SCENARIOS).reset_index())
    csv(joint, "feature_overlap_joint.csv")

    # ---- 3a) 결측 현황 (전체 1,000건, 레이블과 무관한 집계) ----
    null_rows = []
    for name in SCENARIOS:
        part = data[data["scenario"] == name]
        interval, cancel = part["avg_booking_interval"].isna(), part["cancel_rate"].isna()
        null_rows.append({"scenario": name, "n": len(part), "booking_count_lt_2": int((part["booking_count"] < 2).sum()),
                          "interval_null": int(interval.sum()), "cancel_rate_null": int(cancel.sum()),
                          "cancel_null_only (시도 2건 이상)": int((cancel & ~interval).sum()), "any_null": int((interval | cancel).sum()),
                          "booking_count_eq_2 (분산 0)": int((part["booking_count"] == 2).sum())})
    nulls = pd.DataFrame(null_rows)
    csv(nulls, "missing_counts.csv")

    # ---- 3b) 결측 처리별 validation 지표 (contamination 0.20 고정, 변환 없음) / 4) 전체 격자 ----
    grid = grid_on_validation(data, y, scenario, train_idx, val_idx, SELECT_SEEDS)
    csv(grid, "validation_grid.csv")
    rows = []
    for policy in [p for p in POLICIES if not p[2]]:
        for seed in SELECT_SEEDS:
            scorer = Scorer(data, train_idx, policy, seed)
            scores = scorer.score(val_idx)
            row, pred = evaluate(y_val, scores, scorer.threshold(BEST_CONTAMINATION))
            part = data.loc[val_idx]
            interval, cancel = part["avg_booking_interval"].isna().to_numpy(), part["cancel_rate"].isna().to_numpy()
            rows.append({"policy": policy_name(policy), "seed": seed, **{k: row[k] for k in ("f1", "precision", "recall", "fpr", "roc_auc", "insufficient")},
                         "val_interval_null": int(interval.sum()), "interval_null_flagged": int(pred[interval].sum()),
                         "val_cancel_null_only": int((cancel & ~interval).sum()), "cancel_null_only_flagged": int(pred[cancel & ~interval].sum()),
                         "cancel_null_only_anomaly": int(y_val[cancel & ~interval].sum()),
                         "normal_flagged_without_null": int(pred[(y_val == 0) & ~interval & ~cancel].sum()), **flag_rates(pred, s_val)})
    missing_policy = pd.DataFrame(rows).groupby("policy", sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
    csv(missing_policy, "missing_policy_validation.csv")

    selected, selected_c = select(grid)
    plain, plain_c = select(grid[~grid["log"]])
    by_policy = (grid.sort_values("f1", ascending=False).groupby(["min_booking_count", "missing", "log"], sort=False).head(1)
                 .rename(columns={"contamination": "best_contamination"}).reset_index(drop=True))
    at_default = grid[grid["contamination"] == BEST_CONTAMINATION].set_index(["min_booking_count", "missing", "log"])["f1"]
    by_policy["f1_at_c0.20"] = [at_default.loc[(r.min_booking_count, r.missing, r.log)] for r in by_policy.itertuples()]
    by_policy.insert(0, "policy", [policy_name((r.min_booking_count, r.missing, r.log)) for r in by_policy.itertuples()])
    csv(by_policy, "validation_best_per_policy.csv")
    print(f"validation 선택: {policy_name(selected, selected_c)} / 변환 없이: {policy_name(plain, plain_c)}")

    # ---- 4b) 임계값 방식 비교 (고른 전처리, train 으로 학습) ----
    rows = []
    for label, policy in (("기존 설정", BASELINE), ("선택 (변환 없음)", plain), ("선택", selected)):
        best_c = float(grid[(grid["min_booking_count"] == policy[0]) & (grid["missing"] == policy[1]) & (grid["log"] == policy[2])]
                       .sort_values("f1", ascending=False)["contamination"].iloc[0])
        for seed in SELECT_SEEDS:
            scorer = Scorer(data, train_idx, policy, seed)
            val_scores, test_scores = scorer.score(val_idx), scorer.score(test_idx)
            thresholds = {"contamination 0.20 (기존)": scorer.threshold(BEST_CONTAMINATION),
                          f"contamination {best_c:g} (validation F1 최고)": scorer.threshold(best_c),
                          "sklearn auto (score_samples < -0.5)": 0.5,
                          "validation F1 이 최고인 점수 임계값 (seed 별)": best_threshold(y_val, val_scores)}
            for method, threshold in thresholds.items():
                v, t = evaluate(y_val, val_scores, threshold)[0], evaluate(y_test, test_scores, threshold)[0]
                rows.append({"setting": label, "policy": policy_name(policy), "threshold_method": method, "seed": seed,
                             "train_flag_share": float((scorer.fit_scores > threshold).mean()),
                             **{f"val_{k}": v[k] for k in ("f1", "precision", "recall")}, **{f"test_{k}": t[k] for k in ("f1", "precision", "recall", "fpr")}})
    thresholds = pd.DataFrame(rows).groupby(["setting", "policy", "threshold_method"], sort=False).mean(numeric_only=True).drop(columns="seed").reset_index()
    csv(thresholds, "threshold_comparison.csv")

    # ---- 2) 오탐·미탐 원인 (validation, train 으로 학습) ----
    def error_analysis(label, policy, contamination):
        rate_rows, detail = [], []
        for seed in SELECT_SEEDS:
            scorer = Scorer(data, train_idx, policy, seed)
            scores = scorer.score(val_idx)
            _, pred = evaluate(y_val, scores, scorer.threshold(contamination))
            rate_rows.append(flag_rates(pred, s_val))
            if seed != SELECT_SEEDS[0]:
                continue
            # Feature 하나를 학습 데이터의 중앙값으로 바꿨을 때 점수가 얼마나 내려가는지 = 그 Feature 가 점수를 올린 정도
            x = data.loc[val_idx, F].to_numpy(dtype=float)
            ok = ~np.isnan(scores)
            contribution = np.full((len(val_idx), len(F)), np.nan)
            for j in range(len(F)):
                swapped = x[ok].copy()
                swapped[:, j] = scorer.fill[j]
                contribution[ok, j] = scores[ok] - scorer.raw(swapped)
            part = data.loc[val_idx].reset_index(names="row_index")
            part["setting"], part["score"], part["margin"] = label, scores, scores - scorer.threshold(contamination)
            part["pred_anomaly"], part["insufficient"] = pred, ~ok
            part["top_feature"] = [F[int(np.nanargmax(row))] if not np.isnan(row).all() else "" for row in contribution]
            for j, feature in enumerate(F):
                part[f"contribution_{feature}"] = contribution[:, j]
            detail.append(part)
        rates = pd.DataFrame(rate_rows).mean()
        detail = pd.concat(detail)
        variance_p90 = data.loc[train_idx, "booking_interval_variance"].quantile(0.9)
        rows = []
        for name in SCENARIOS:
            part = detail[detail["scenario"] == name]
            wrong = part[part["pred_anomaly"] != part["is_anomaly"]]
            row = {"setting": label, "scenario": name, "actual": "정상" if name in NORMAL_SCENARIOS else "이상", "n_val": len(part),
                   "flag_rate_mean_10seeds": rates[f"flag_rate_{name}"], "kind": "오탐" if name in NORMAL_SCENARIOS else "미탐",
                   "errors_seed0": len(wrong), "margin_median_all": part["margin"].median(), "errors_margin_median": wrong["margin"].median(),
                   "errors_insufficient": int(wrong["insufficient"].sum()),
                   "errors_interval_null": int(wrong["avg_booking_interval"].isna().sum()), "errors_cancel_null": int(wrong["cancel_rate"].isna().sum()),
                   "errors_booking_count_eq_2": int((wrong["booking_count"] == 2).sum()),
                   "errors_variance_ge_train_p90": int((wrong["booking_interval_variance"] >= variance_p90).sum()),
                   "errors_ip_ge_5": int((wrong["ip_booking_density"] >= 5).sum()), "errors_cancel_ge_0.5": int((wrong["cancel_rate"] >= 0.5).sum())}
            for feature in F:
                row[f"top_{feature}"] = int((wrong["top_feature"] == feature).sum())
            rows.append(row)
        return pd.DataFrame(rows), detail

    errors, details = zip(*(error_analysis(label, policy, c) for label, policy, c in
                            (("기존 설정", BASELINE, BEST_CONTAMINATION), ("선택", selected, selected_c))))
    errors = pd.concat(errors)
    csv(errors, "error_analysis_validation.csv")
    csv(pd.concat(details), "error_detail_validation.csv")

    # ---- 최종 test 평가 ----
    settings = [("C 기존 설정", BASELINE, BEST_CONTAMINATION), ("C 선택 (변환 없음)", plain, plain_c), ("C 선택", selected, selected_c)]
    fits = [("train 490건으로 학습", train_idx), ("train+validation 700건으로 재학습", dev_idx)]
    rows, predictions = [], []
    for label, policy, c in settings:
        for fit_label, fit_idx in fits:
            for seed in sorted(set(MODEL_SEEDS) | {MODEL_SEED}):
                scorer = Scorer(data, fit_idx, policy, seed)
                scores = scorer.score(test_idx)
                row, pred = evaluate(y_test, scores, scorer.threshold(c))
                rows.append({"setting": label, "policy": policy_name(policy, c), "fit": fit_label, "seed": seed, **row, **flag_rates(pred, s_test)})
                if seed == MODEL_SEED and fit_idx is dev_idx:
                    predictions.append(pd.DataFrame({"setting": label, "row_index": test_idx, "user_id": data.loc[test_idx, "user_id"].to_numpy(),
                                                     "scenario": s_test, "is_anomaly": y_test, "pred_anomaly": pred, "score": scores,
                                                     "margin": scores - scorer.threshold(c), "insufficient_data": np.isnan(scores)}))
    test_runs = pd.DataFrame(rows)
    csv(test_runs, "test_runs.csv")
    csv(pd.concat(predictions), "test_predictions.csv")
    metric_cols = ["f1", "precision", "recall", "fpr", "roc_auc", "roc_auc_scored_only"]
    rate_cols = [f"flag_rate_{s}" for s in SCENARIOS]
    seed42 = test_runs[test_runs["seed"] == MODEL_SEED].drop(columns="seed").reset_index(drop=True)
    csv(seed42, "test_metrics_seed42.csv")
    model_seed_summary = mean_std(test_runs[test_runs["seed"].isin(MODEL_SEEDS)], ["setting", "policy", "fit"], metric_cols + rate_cols)
    csv(model_seed_summary, "model_seed_summary.csv")

    # ---- 5) 분할 seed: 분할마다 validation 에서 다시 고른다 (중첩) + 위에서 고른 설정을 그대로 적용 ----
    rows, chosen = [], []
    for split_seed in SPLIT_SEEDS:
        dev_s, test_s = train_test_split(data.index, test_size=TEST_SIZE, random_state=split_seed, stratify=y)
        train_s, val_s = train_test_split(dev_s, test_size=VAL_SIZE, random_state=split_seed, stratify=scenario.loc[dev_s])
        nested, nested_c = select(grid_on_validation(data, y, scenario, train_s, val_s, NESTED_SELECT_SEEDS))
        chosen.append({"split_seed": split_seed, "policy": policy_name(nested), "contamination": nested_c})
        yt, st = y.loc[test_s].to_numpy(), scenario.loc[test_s].to_numpy()
        for label, policy, c in (("C 기존 설정", BASELINE, BEST_CONTAMINATION), ("C 선택 (고정: seed 42 분할에서 고른 설정)", selected, selected_c),
                                 ("C 선택 (분할마다 validation 에서 다시 선택)", nested, nested_c)):
            scorer = Scorer(data, dev_s, policy)
            row, pred = evaluate(yt, scorer.score(test_s), scorer.threshold(c))
            rows.append({"setting": label, "split_seed": split_seed, **row, **flag_rates(pred, st)})
        pred, _, normalized = fit_and_score(original, dev_s, test_s, F, BEST_CONTAMINATION)
        rows.append({"setting": "A 기존 Feature 기반", "split_seed": split_seed, **metrics(yt, pred, normalized), **flag_rates(pred, st)})
        print(f"분할 seed {split_seed}: {policy_name(nested, nested_c)}", flush=True)
    split_runs, chosen = pd.DataFrame(rows), pd.DataFrame(chosen)
    csv(split_runs, "split_seed_runs.csv")
    csv(chosen, "split_seed_selected.csv")
    split_summary = mean_std(split_runs, ["setting"], ["f1", "precision", "recall", "fpr", "roc_auc"] + rate_cols)
    csv(split_summary, "split_seed_summary.csv")
    paired = split_runs.pivot(index="split_seed", columns="setting", values="f1")
    wins = {name: int((paired[name] > paired["C 기존 설정"]).sum()) for name in paired.columns if name.startswith("C 선택")}

    # ---------------- 그래프 ----------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, feature in zip(axes.ravel(), F):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for position, name in enumerate(SCENARIOS[::-1]):
            values = dev.loc[dev["scenario"] == name, feature].dropna().to_numpy()
            color = C_NORMAL if name in NORMAL_SCENARIOS else C_ANOM
            ax.scatter(values, position + rng.uniform(-0.28, 0.28, len(values)), s=7, color=color, alpha=0.45, linewidth=0)
            ax.vlines(np.median(values), position - 0.38, position + 0.38, color=INK, linewidth=1.6)
        ax.axvspan(*bounds[feature], color=C_NORMAL, alpha=0.07, linewidth=0)
        ax.set_yticks(range(len(SCENARIOS)))
        ax.set_yticklabels(SCENARIOS[::-1], fontsize=9)
        if feature in LOG_AXIS:
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_xlim(left=0)
        ax.set_title(feature + (" (로그 눈금)" if feature in LOG_AXIS else ""), loc="left", fontsize=10, color=INK)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=C_NORMAL, label="정상 시나리오"),
               plt.Line2D([], [], marker="o", linestyle="", color=C_ANOM, label="이상 시나리오"),
               plt.Line2D([], [], color=INK, linewidth=1.6, label="중앙값"),
               plt.Rectangle((0, 0), 1, 1, color=C_NORMAL, alpha=0.12, label="정상 전체의 5~95% 구간")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Feature 별 분포: 시나리오별 계정 (test 를 뺀 700건)", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "feature_overlap.png")

    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    style(ax)
    lines = [("기존 설정", BASELINE, MUTED), ("선택 (변환 없음)", plain, C_BASE), ("선택", selected, C_SEL)]
    drawn = set()
    for label, policy, color in lines:
        if policy in drawn:
            continue
        drawn.add(policy)
        part = grid[(grid["min_booking_count"] == policy[0]) & (grid["missing"] == policy[1]) & (grid["log"] == policy[2])]
        ax.plot(part["contamination"], part["f1"], color=color, linewidth=2, marker="o", markersize=4, label=f"{label}: {policy_name(policy)}")
        ax.fill_between(part["contamination"], part["f1"] - part["f1_std"], part["f1"] + part["f1_std"], color=color, alpha=0.12, linewidth=0)
    ax.axvline(BEST_CONTAMINATION, color=GRID, linewidth=1.2, linestyle="--")
    ax.set_xlabel("contamination (학습 점수의 상위 비율을 이상으로 판정)", fontsize=9, color=MUTED)
    ax.set_ylabel("validation F1", fontsize=9, color=MUTED)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"contamination 에 따른 validation F1 (모델 seed {len(SELECT_SEEDS)}개 평균 ± 표준편차)", loc="left", fontsize=10, color=INK)
    save(fig, OUT / "figures" / "validation_f1_by_contamination.png")

    refit = seed42[seed42["fit"] == fits[1][0]].set_index("setting")
    bars = [("A 기존 Feature 기반", reference["A"], C_A), ("C 이벤트 v2 · 기존 설정", reference["C"], C_BASE),
            ("C 이벤트 v2 · 선택한 설정", refit.loc["C 선택"], C_SEL)]
    names = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC")]
    fig, ax = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    style(ax)
    grouped_bars(ax, [n for _, n in names], [(label, [row[key] for key, _ in names], color) for label, row, color in bars], fmt="{:.3f}")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    ax.set_title("test 300건의 전체 성능 (모델 seed 42, 합성 데이터 안에서의 값)", loc="left", fontsize=10, color=INK)
    save(fig, OUT / "figures" / "overall_metrics.png")

    counts = pd.Series(s_test).value_counts()
    fig, ax = plt.subplots(figsize=(12.5, 4.8), constrained_layout=True)
    style(ax)
    grouped_bars(ax, [f"{s}\n({'정상' if s in NORMAL_SCENARIOS else '이상'}, n={counts[s]})" for s in SCENARIOS],
                 [(label, [row[f"flag_rate_{s}"] for s in SCENARIOS], color) for label, row, color in bars])
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    ax.set_ylabel("이상 판정 비율", fontsize=9, color=MUTED)
    ax.set_title("시나리오별 이상 판정 비율 (정상 = 오탐률, 이상 = 탐지율), test, 모델 seed 42", loc="left", fontsize=10, color=INK)
    save(fig, OUT / "figures" / "scenario_detection.png")

    model_panel = test_runs[(test_runs["fit"] == fits[1][0]) & test_runs["seed"].isin(MODEL_SEEDS) & (test_runs["setting"] != "C 선택 (변환 없음)")]
    split_panel = split_runs[split_runs["setting"].isin(["C 기존 설정", "C 선택 (분할마다 validation 에서 다시 선택)"])]
    tick = {"C 기존 설정": "기존 설정", "C 선택": "선택한 설정", "C 선택 (분할마다 validation 에서 다시 선택)": "분할마다\n다시 선택"}
    color_of = {"C 기존 설정": C_BASE, "C 선택": C_SEL, "C 선택 (분할마다 validation 에서 다시 선택)": C_SEL}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3), sharey=True, constrained_layout=True)
    for ax, (title, frame) in zip(axes, ((f"모델 seed {len(MODEL_SEEDS)}개 (분할 고정)", model_panel), (f"분할 seed {len(SPLIT_SEEDS)}개", split_panel))):
        style(ax)
        order = list(dict.fromkeys(frame["setting"]))
        for position, name in enumerate(order):
            values = frame.loc[frame["setting"] == name, "f1"].to_numpy()
            ax.scatter(position + rng.uniform(-0.16, 0.16, len(values)), values, s=22, color=color_of[name], edgecolor=SURFACE, linewidth=0.6)
            ax.hlines(values.mean(), position - 0.3, position + 0.3, color=INK, linewidth=1.4)
            ax.text(position + 0.34, values.mean(), f"{values.mean():.3f}", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([tick[name] for name in order], fontsize=9)
        ax.set_xlim(-0.6, len(order) - 0.2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("test F1", fontsize=9, color=MUTED)
    axes[0].set_ylim(0, 1)
    fig.suptitle("seed 를 바꿨을 때의 test F1 — C 이벤트 v2 (점 = 실행 1회, 가로선 = 평균)", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "seed_stability.png")

    record = {
        "script": "scripts/tune_if_v2.py", "dataset": {"path": C_PATH.relative_to(REPO).as_posix(), "sha256": hash_before["C"]},
        "datasets_unchanged_after_run": hash_before == {"A": sha256(ORIGINAL_PATH), "C": sha256(C_PATH)},
        "features": F, "model": {"class": "sklearn.ensemble.IsolationForest", "n_estimators": N_ESTIMATORS, "max_samples": "auto"},
        "split": {"test": len(test_idx), "train": len(train_idx), "validation": len(val_idx), "random_state": SPLIT_SEED,
                  "test_stratify": "is_anomaly (기존과 같은 300건)", "validation_stratify": "scenario"},
        "search_space": {"min_booking_count": [1, 2], "missing": ["raw", "median", "zero"], "log1p": [False, True], "contamination": CONTAMINATIONS},
        "selection": {"criterion": f"validation F1, 모델 seed {len(SELECT_SEEDS)}개 평균", "selected": policy_name(selected, selected_c),
                      "selected_without_transform": policy_name(plain, plain_c)},
        "reference_existing": {k: {m: round(float(v[m]), 4) for m in ("f1", "precision", "recall", "roc_auc")} for k, v in reference.items()},
        "model_seeds": MODEL_SEEDS, "split_seeds": SPLIT_SEEDS, "nested_selection_model_seeds": NESTED_SELECT_SEEDS,
        "split_seed_wins_over_baseline": wins,
        "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__, "python": sys.version.split()[0]},
        "models_saved": False,
    }
    (OUT / "experiment_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    show = ["f1", "precision", "recall", "fpr", "roc_auc"]
    print("\n[1] 분포 중첩\n", overlap.drop(columns=["normal_null", "anomaly_min", "anomaly_max"]).round(3).to_string(index=False))
    print(joint.round(3).to_string(index=False))
    print("\n[3] 결측\n", nulls.to_string(index=False))
    print(missing_policy.round(3).T.to_string())
    print("\n[4] 정책별 validation 최고\n", by_policy.round(3).to_string(index=False))
    print(thresholds.round(3).to_string(index=False))
    print("\n[2] 오탐·미탐 (validation)\n", errors.round(3).T.to_string())
    print("\n[최종] test seed 42\n", seed42[["setting", "policy", "fit"] + show + ["roc_auc_scored_only", "TP", "FP", "TN", "FN", "insufficient"]].round(3).to_string(index=False))
    print(seed42[["setting", "fit"] + rate_cols].round(3).to_string(index=False))
    print("기존 A:", {k: round(float(reference["A"][k]), 3) for k in show + rate_cols})
    print("기존 C:", {k: round(float(reference["C"][k]), 3) for k in show + rate_cols})
    print("\n[5] 모델 seed\n", model_seed_summary[["setting", "fit"] + [f"{m}_{s}" for m in show for s in ("mean", "std")]].round(3).to_string(index=False))
    print(model_seed_summary[["setting", "fit"] + [f"{c}_mean" for c in rate_cols]].round(2).to_string(index=False))
    print("\n분할 seed\n", split_summary[["setting"] + [f"{m}_{s}" for m in show for s in ("mean", "std", "min", "max")]].round(3).to_string(index=False))
    print(split_summary[["setting"] + [f"{c}_mean" for c in rate_cols]].round(2).to_string(index=False))
    print(chosen.value_counts(["policy", "contamination"]).to_string())
    print("기존 설정보다 F1 이 높은 분할 수:", wins, "/", len(SPLIT_SEEDS))
    print(record["datasets_unchanged_after_run"], f"저장: {OUT}")


if __name__ == "__main__":
    main()
