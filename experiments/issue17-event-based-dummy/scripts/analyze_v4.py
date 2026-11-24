"""v4 의 Feature 분포와 민감도 분석. 모델은 학습하지 않는다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\analyze_v4.py

입력 (읽기만 함)
  data/v4/<D>/fairticket_event_based_features.csv, interval_definitions.csv, accounts.csv   (D = D1, D2, D3. 생성 seed 42)
  data/v4/_runs/*.csv.gz                                                                      (sweep_v4.py: 조건 × 생성 seed 10개)
  measurement/data/20261124-add/raw_events.csv                                                (실측한 정상 사용자 1명, 8계정. 없으면 그 부분을 건너뛴다)

D1·D2·D3 는 정상 사용자 사이의 속도 분포에 대한 가정이다. 결과를 가정별로 따로 낸다. 평균을 내어 합치지 않는다.
실측 계정 8개는 한 사람의 반복 측정이다. 합성 분포의 어디에 놓이는지만 본다 (8계정을 한 묶음의 참고점으로).
독립 표본으로 보고 비율이나 신뢰구간을 계산하지 않는다.

출력: output/v4/distribution/   distribution_by_scenario.csv, single_feature_auc.csv, overlap.csv, null_share.csv, person_position.csv, figures/
      output/v4/sensitivity/    sensitivity_runs.csv, sensitivity_summary.csv, figures/
      measurement/data/_analysis/v4/person_features.csv (git 제외)
"""
import json
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP.parent / "issue14-real-validation" / "scripts"))
from extract_features import extract, load_events  # noqa: E402
from extract_dataset import DATASET_NAME  # noqa: E402
from extract_v4 import BOUNDARIES, INTERVAL, MEASURED_EVENTS, REPRESENTATIVES, VARIANCE, definition_columns, interval_definitions  # noqa: E402
from generate_events_v4 import DISTS, load_v4  # noqa: E402
from sweep_v4 import OUT as SENS, SEEDS, run_path  # noqa: E402

OUT = EXP / "output" / "v4" / "distribution"
PRIVATE = EXP / "measurement" / "data" / "_analysis" / "v4"
F = [INTERVAL, VARIANCE, "cancel_rate", "card_reuse_count", "ip_booking_density", "booking_count"]
SCENARIOS = ["normal", "fast_normal", "shared_ip", "frequent_cancel", "fast", "jitter", "multi_account", "hold_cancel"]
MACRO = ["fast", "jitter", "multi_account"]
LOW_IS_ANOMALOUS = {INTERVAL, VARIANCE}
INK, MUTED, GRID, SURFACE = "#1f2328", "#6b7280", "#e5e7eb", "#ffffff"
C_NORMAL, C_ANOM, C_REAL = "#2a78d6", "#eb6834", "#1f2328"
Q = [0.05, 0.25, 0.5, 0.75, 0.95]


def describe(values):
    values = pd.Series(values, dtype=float)
    present = values.dropna()
    row = {"n": int(len(values)), "missing": int(values.isna().sum()), "mean": present.mean(), "median": present.median(), "std": present.std(ddof=1)}
    row.update({f"p{int(q * 100):02d}": present.quantile(q) for q in Q if q != 0.5})
    return row


def separation(frame, column, anomaly=None):
    """단일 Feature ROC-AUC. 값이 있는 계정만. 간격·분산은 작을수록 이상, 나머지는 클수록 이상 쪽으로 계산한다 (0.5 미만이면 방향이 반대다)."""
    part = frame.dropna(subset=[column])
    if anomaly is not None:
        part = part[(part["is_anomaly"] == 0) | part["scenario"].isin(anomaly)]
    if part["is_anomaly"].nunique() < 2:
        return np.nan
    sign = -1 if column.split("__")[0] in LOW_IS_ANOMALOUS else 1
    return float(roc_auc_score(part["is_anomaly"], sign * part[column]))


def inside(values, reference):
    """values 중 reference 의 5~95% 구간 안에 드는 비율."""
    values, reference = pd.Series(values).dropna(), pd.Series(reference).dropna()
    if len(values) == 0 or len(reference) == 0:
        return np.nan
    low, high = reference.quantile(0.05), reference.quantile(0.95)
    return float(((values >= low) & (values <= high)).mean())


def overlap_coefficient(a, b, bins=30):
    """log1p 눈금의 히스토그램 겹침 (0 = 겹치지 않음, 1 = 같은 분포)."""
    a, b = np.log1p(pd.Series(a).dropna()), np.log1p(pd.Series(b).dropna())
    if len(a) == 0 or len(b) == 0:
        return np.nan
    edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()) + 1e-9, bins + 1)
    return float(np.minimum(np.histogram(a, edges)[0] / len(a), np.histogram(b, edges)[0] / len(b)).sum())


def columns_to_compare():
    """(표시 이름, 컬럼, 종류). 기존 6개 + 간격 2개의 정의별 값."""
    out = [(name, name, "BTS 정의") for name in F]
    for boundary in BOUNDARIES:
        for representative in REPRESENTATIVES:
            a, b = definition_columns(boundary, representative)
            out += [(f"{INTERVAL} [{boundary} · {representative}]", a, "정의 후보"), (f"{VARIANCE} [{boundary} · {representative}]", b, "정의 후보"),
                    (f"{VARIANCE} [{boundary} · {representative} · 간격 5개 이상]", b + "__min5", "정의 후보")]
    return out


def load_main(dist):
    folder = EXP / "data" / "v4" / dist
    frame = pd.read_csv(folder / DATASET_NAME).merge(pd.read_csv(folder / "interval_definitions.csv"), on="user_id", validate="one_to_one")
    accounts = pd.read_csv(folder / "accounts.csv", keep_default_na=False)
    accounts["speed_s"] = pd.to_numeric(accounts["speed_s"], errors="coerce")
    return frame.merge(accounts[["user_id", "speed_s", "n_sessions", "tool_profile", "ip_group_size"]], on="user_id", validate="one_to_one")


def person_features():
    """실측한 1명의 8계정을 합성 데이터와 같은 추출기·같은 정의로 계산한다."""
    if not MEASURED_EVENTS.exists():
        return None
    events = load_events(MEASURED_EVENTS)
    events = events[events["group"] == "participant"].reset_index(drop=True)  # 동작 확인용 시험 계정 제외
    calculated_at = events["occurred_at"].max() + timedelta(seconds=1)
    features, _, _ = extract(events, calculated_at, 30.0)
    definitions, _ = interval_definitions(events, calculated_at, features["user_id"])
    frame = features.merge(definitions, on="user_id", validate="one_to_one")
    names = events.drop_duplicates("user_id").set_index("user_id")["external_user_id"]
    frame.insert(0, "account", frame["user_id"].map(names))
    PRIVATE.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PRIVATE / "person_features.csv", index=False, encoding="utf-8-sig")
    return frame


def run_metrics(frame):
    """민감도 분석에서 조건·seed 마다 보는 값 (모델 없음)."""
    normal, anomaly = frame[frame["is_anomaly"] == 0], frame[frame["is_anomaly"] == 1]
    macro = frame[frame["scenario"].isin(MACRO)]
    row = {"auc_booking_count": separation(frame, "booking_count"), "normal_one_attempt_share": float((normal["booking_count"] == 1).mean()),
           "normal_booking_count_median": float(normal["booking_count"].median())}
    for boundary in BOUNDARIES:
        a, b = definition_columns(boundary, "mean_var")
        row.update({f"auc_interval__{boundary}": separation(frame, a), f"auc_variance__{boundary}": separation(frame, b),
                    f"auc_interval_macro_only__{boundary}": separation(frame, a, MACRO),
                    f"null_normal__{boundary}": float(normal[a].isna().mean()), f"null_anomaly__{boundary}": float(anomaly[a].isna().mean()),
                    f"macro_inside_normal_range__{boundary}": inside(macro[a], normal[a]), f"normal_inside_macro_range__{boundary}": inside(normal[a], macro[a]),
                    f"normal_interval_median__{boundary}": float(normal[a].median()), f"macro_interval_median__{boundary}": float(macro[a].median())})
    return row


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_axisbelow(True)


def main():
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (SENS / "figures").mkdir(parents=True, exist_ok=True)
    _, v4 = load_v4()
    labels = {dist: v4["normal_speed"]["distributions"][dist]["label"] for dist in DISTS}
    csv = lambda frame, path: frame.to_csv(path, index=False, encoding="utf-8-sig")
    main_frames = {dist: load_main(dist) for dist in DISTS}
    compare = columns_to_compare()

    # ---- 1) 시나리오별 분포 ----
    rows = []
    for dist, frame in main_frames.items():
        groups = [(name, frame[frame["scenario"] == name]) for name in SCENARIOS] + [("정상 전체", frame[frame["is_anomaly"] == 0]), ("이상 전체", frame[frame["is_anomaly"] == 1]),
                                                                                     ("매크로 (fast·jitter·multi_account)", frame[frame["scenario"].isin(MACRO)])]
        for label, column, kind in compare + [("speed_s (생성 기록)", "speed_s", "생성 기록")]:
            for name, part in groups:
                rows.append({"assumption": dist, "feature": label, "kind": kind, "group": name, **describe(part[column])})
    distribution = pd.DataFrame(rows)
    csv(distribution, OUT / "distribution_by_scenario.csv")

    # ---- 2) 단일 Feature AUC, 3) 겹침, 결측 ----
    auc_rows, overlap_rows, null_rows = [], [], []
    for dist, frame in main_frames.items():
        normal, anomaly, macro = frame[frame["is_anomaly"] == 0], frame[frame["is_anomaly"] == 1], frame[frame["scenario"].isin(MACRO)]
        for label, column, kind in compare:
            auc_rows.append({"assumption": dist, "feature": label, "kind": kind, "direction": "작을수록 이상" if column.split("__")[0] in LOW_IS_ANOMALOUS else "클수록 이상",
                             "auc_all_anomalies": separation(frame, column), "auc_macro_only": separation(frame, column, MACRO),
                             "auc_hold_cancel_only": separation(frame, column, ["hold_cancel"]), "accounts_with_value": int(frame[column].notna().sum())})
            overlap_rows.append({"assumption": dist, "feature": label, "kind": kind,
                                 "anomaly_inside_normal_p05_p95": inside(anomaly[column], normal[column]), "macro_inside_normal_p05_p95": inside(macro[column], normal[column]),
                                 "normal_inside_anomaly_p05_p95": inside(normal[column], anomaly[column]), "normal_inside_macro_p05_p95": inside(normal[column], macro[column]),
                                 "overlap_coefficient_normal_vs_anomaly": overlap_coefficient(normal[column], anomaly[column]),
                                 "overlap_coefficient_normal_vs_macro": overlap_coefficient(normal[column], macro[column])})
            null_rows.append({"assumption": dist, "feature": label, "kind": kind, "null_normal": float(normal[column].isna().mean()),
                              "null_anomaly": float(anomaly[column].isna().mean()),
                              **{f"null_{s}": float(frame.loc[frame["scenario"] == s, column].isna().mean()) for s in SCENARIOS}})
    auc, overlap, nulls = pd.DataFrame(auc_rows), pd.DataFrame(overlap_rows), pd.DataFrame(null_rows)
    csv(auc, OUT / "single_feature_auc.csv")
    csv(overlap, OUT / "overlap.csv")
    csv(nulls, OUT / "null_share.csv")

    # ---- 4) 실측한 1명의 위치 ----
    person = person_features()
    position = None
    if person is not None:
        rows = []
        watch = [("booking_count", "booking_count")] + [(f"{name} [{boundary}]", definition_columns(boundary, "mean_var")[i])
                                                         for boundary in BOUNDARIES for i, name in enumerate((INTERVAL, VARIANCE))]
        for dist, frame in main_frames.items():
            normal, macro = frame[frame["is_anomaly"] == 0], frame[frame["scenario"].isin(MACRO)]
            for label, column in watch:
                for _, account in person.iterrows():
                    value = account[column]
                    pct = lambda ref: np.nan if pd.isna(value) else float((ref.dropna() <= value).mean() * 100)
                    rows.append({"assumption": dist, "feature": label, "account": account["account"], "value": value,
                                 "percentile_in_synthetic_normal": pct(normal[column]), "percentile_in_synthetic_macro": pct(macro[column])})
        detail = pd.DataFrame(rows)
        detail.to_csv(PRIVATE / "person_position_by_account.csv", index=False, encoding="utf-8-sig")
        # 8계정을 한 묶음으로: 값이 있는 계정 수, 값의 범위, 합성 정상·매크로 분포에서의 백분위 범위
        position = (detail.groupby(["assumption", "feature"], sort=False)
                    .agg(accounts_with_value=("value", "count"), value_min=("value", "min"), value_median=("value", "median"), value_max=("value", "max"),
                         normal_pct_min=("percentile_in_synthetic_normal", "min"), normal_pct_median=("percentile_in_synthetic_normal", "median"),
                         normal_pct_max=("percentile_in_synthetic_normal", "max"), macro_pct_min=("percentile_in_synthetic_macro", "min"),
                         macro_pct_median=("percentile_in_synthetic_macro", "median"), macro_pct_max=("percentile_in_synthetic_macro", "max")).reset_index())
        position.insert(2, "note", "한 사람의 8계정 (독립 표본 아님). 참고점")
        csv(position, OUT / "person_position.csv")

    # ---- 5) 민감도: 조건 × 생성 seed ----
    listed = json.loads((SENS / "conditions.json").read_text(encoding="utf-8"))["one_at_a_time"]
    conditions = [(c["condition"], c["axis"]) for c in listed]
    choice_path = SENS / "combination_choice.json"
    if choice_path.exists():
        choice = json.loads(choice_path.read_text(encoding="utf-8"))
        conditions += [(c["condition"], "조합: " + "+".join(choice["chosen"])) for c in choice["combinations"]]
    rows = []
    for dist in DISTS:
        for name, axis in conditions:
            for seed in SEEDS:
                path = run_path(dist, name, seed)
                if path.exists():
                    rows.append({"assumption": dist, "condition": name, "axis": axis, "seed": seed, **run_metrics(pd.read_csv(path))})
    runs = pd.DataFrame(rows)
    csv(runs, SENS / "sensitivity_runs.csv")
    metrics = [c for c in runs.columns if c not in ("assumption", "condition", "axis", "seed")]
    grouped = runs.groupby(["assumption", "condition", "axis"], sort=False)[metrics]
    summary = grouped.mean().add_suffix("__mean").join(grouped.std(ddof=1).add_suffix("__sd")).join(grouped.size().rename("seeds")).reset_index()
    base = summary[summary["condition"] == "base"].set_index("assumption")
    for metric in metrics:
        summary[f"{metric}__delta"] = summary[f"{metric}__mean"] - summary["assumption"].map(base[f"{metric}__mean"])
    csv(summary, SENS / "sensitivity_summary.csv")

    # ---- 그림 1: 가정별 간격 분포와 실측한 1명 ----
    for boundary, title in (("activity", "활동 세션 안의 평균 간격"), ("d30", "30일 전체의 평균 간격 (현재 BTS 정의)")):
        column = definition_columns(boundary, "mean_var")[0]
        fig, axes = plt.subplots(3, 1, figsize=(8.6, 6.6), sharex=True, constrained_layout=True)
        pooled = pd.concat([frame[column] for frame in main_frames.values()]).dropna()
        edges = np.logspace(np.log10(max(pooled.min(), 1.0)), np.log10(pooled.max()), 46)
        for ax, (dist, frame) in zip(axes, main_frames.items()):
            style(ax)
            for mask, color, name in ((frame["is_anomaly"] == 0, C_NORMAL, "정상"), (frame["scenario"].isin(MACRO), C_ANOM, "매크로")):
                values = frame.loc[mask, column].dropna()
                ax.hist(values, bins=edges, histtype="stepfilled", color=color, alpha=0.18, weights=np.full(len(values), 1 / len(values)))
                ax.hist(values, bins=edges, histtype="step", color=color, linewidth=2, weights=np.full(len(values), 1 / len(values)), label=f"{name} (값이 있는 {len(values)}계정)")
            if person is not None:
                values = person[column].dropna()
                ax.plot(values, np.full(len(values), ax.get_ylim()[1] * 0.93), linestyle="none", marker="|", markersize=13, markeredgewidth=2, color=C_REAL,
                        label=f"실측한 1명의 계정 {len(values)}개 (참고점)")
            ax.set_xscale("log")
            ax.set_title(f"{dist} · {labels[dist]} (가정)   단일 Feature AUC {separation(frame, column):.3f}", loc="left", fontsize=10, color=INK)
            ax.set_ylabel("계정 비율", fontsize=8, color=MUTED)
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        axes[-1].set_xlabel(f"{title} (초, 로그 눈금)", fontsize=9, color=MUTED)
        fig.savefig(OUT / "figures" / f"interval_{boundary}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    # ---- 그림 2: 민감도 (조건별 단일 Feature AUC, 가정별 작은 그림) ----
    one_at_a_time = [name for name, axis in conditions if not axis.startswith("조합")]
    for boundary, title in (("d30", "30일 전체 (현재 BTS 정의)"), ("activity", "활동 세션")):
        metric = f"auc_interval__{boundary}"
        fig, axes = plt.subplots(1, 3, figsize=(10.4, 5.2), sharey=True, sharex=True, constrained_layout=True)
        for ax, dist in zip(axes, DISTS):
            style(ax)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            part = runs[runs["assumption"] == dist]
            reference = part.loc[part["condition"] == "base", metric].mean()
            ax.axvline(reference, color=MUTED, linewidth=1)
            for y, name in enumerate(one_at_a_time[::-1]):
                values = part.loc[part["condition"] == name, metric]
                if len(values) == 0:
                    continue
                ax.plot([values.min(), values.max()], [y, y], color=C_NORMAL, linewidth=2, solid_capstyle="round")
                ax.plot(values.mean(), y, marker="o", markersize=8, color=C_NORMAL, markeredgecolor=SURFACE, markeredgewidth=2)
            ax.set_yticks(range(len(one_at_a_time)))
            ax.set_yticklabels([("기준" if n == "base" else n) for n in one_at_a_time[::-1]], fontsize=8, color=INK)
            ax.set_title(f"{dist} · {labels[dist]}\n기준 {reference:.3f}", loc="left", fontsize=10, color=INK)
            ax.set_xlabel("평균 간격의 단일 Feature AUC", fontsize=8, color=MUTED)
        fig.suptitle(f"한 번에 한 축만 바꿨을 때 · {title} · 점 = 생성 seed 10개의 평균, 선 = 최소~최대, 세로선 = 기준", fontsize=10, color=INK, x=0.01, ha="left")
        fig.savefig(SENS / "figures" / f"sensitivity_auc_{boundary}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)

    # ---- 화면 출력 ----
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    show = auc[auc["feature"].isin(F) | auc["feature"].str.contains("mean_var]")]
    print("[2] 단일 Feature AUC (가정별)\n", show.pivot(index="feature", columns="assumption", values="auc_all_anomalies").round(3).to_string())
    print("\n[2b] 매크로만\n", show.pivot(index="feature", columns="assumption", values="auc_macro_only").round(3).to_string())
    print("\n[3] 겹침 (mean_var)\n", overlap[overlap["feature"].isin(F) | overlap["feature"].str.contains("mean_var]")].round(3).to_string(index=False))
    if position is not None:
        print("\n[4] 실측한 1명의 위치\n", position.drop(columns="note").round(2).to_string(index=False))
    keep = ["assumption", "condition"] + [f"{m}__mean" for m in ("auc_interval__d30", "auc_interval__concert", "auc_interval__activity", "auc_variance__activity", "auc_booking_count",
                                                                 "null_normal__activity", "macro_inside_normal_range__activity", "normal_interval_median__activity")]
    print("\n[5] 민감도 (seed 평균)\n", summary[keep].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
