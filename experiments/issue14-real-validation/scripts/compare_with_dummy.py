"""실측 Feature를 기존 더미데이터 분포와 비교한다 (실험용).

실행 (실험 폴더에서):
    .venv/Scripts/python scripts/compare_with_dummy.py --tag pilot

입력 : data/<tag>/real_features.csv, (있으면) data/<tag>/real_scores.csv, attempt_intervals.csv
       model_build/ml_experiment/data/fairticket_synthetic_features.csv (더미, 학습에 쓴 파일 그대로)
출력 : output/<tag>/summary_stats.csv        Feature × 집단 요약 통계
       output/<tag>/real_positions.csv       실측 계정이 더미 분포의 몇 번째 백분위인지
       output/<tag>/ks_tests.csv             KS 통계량 (연속형 Feature, 표본 수 함께 기록)
       output/<tag>/fig_<feature>.png        더미 분포 + 실측 계정 위치
       output/<tag>/fig_scores.png           그룹별 이상 점수 (real_scores.csv 가 있을 때)

비교 기준
- 더미 정상  : is_anomaly == 0 (normal, fast_normal, shared_ip, frequent_cancel)
- 더미 매크로: scenario == "macro"
- 간격·분산은 두 가지 실측 컬럼을 따로 비교한다.
    30일 기준(BTS 식)  : avg_booking_interval, booking_interval_variance
    세션 기준(실험 가정): session_interval_mean, session_interval_variance
  더미 쪽 컬럼은 같지만 표·그림 이름에 기준을 적어 섞이지 않게 한다.
- card_reuse_count, ip_booking_density 는 데모 환경 왜곡이 있으면 "비교 불가"로 표시한다.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

EXP = Path(__file__).resolve().parent.parent
DUMMY_PATH = EXP / "model_build" / "ml_experiment" / "data" / "fairticket_synthetic_features.csv"

# (표시 이름, 실측 컬럼, 더미 컬럼, 연속형 여부, 로그 축 여부)
COMPARISONS = [
    ("booking_count", "booking_count", "booking_count", False, False),
    ("cancel_rate", "cancel_rate", "cancel_rate", True, False),
    ("avg_booking_interval [30일 기준]", "avg_booking_interval", "avg_booking_interval", True, True),
    ("avg_booking_interval [세션 기준]", "session_interval_mean", "avg_booking_interval", True, True),
    ("booking_interval_variance [30일 기준]", "booking_interval_variance", "booking_interval_variance", True, True),
    ("booking_interval_variance [세션 기준]", "session_interval_variance", "booking_interval_variance", True, True),
    ("card_reuse_count", "card_reuse_count", "card_reuse_count", False, False),
    ("ip_booking_density", "ip_booking_density", "ip_booking_density", False, False),
]
GROUP_LABELS = {"human": "실측 사람", "macro_fast": "실측 매크로 fast", "macro_jitter": "실측 매크로 jitter",
                "macro_human": "실측 매크로 human", "other": "실측 기타"}
GROUP_ORDER = list(GROUP_LABELS)

# 색: 더미 두 집단만 색으로 구분하고(검증된 2색), 실측 그룹은 행 이름으로 구분한다.
C_NORMAL, C_MACRO = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def describe(values):
    s = pd.Series(values, dtype=float)
    valid = s.dropna()
    if valid.empty:
        return {"n": 0, "n_missing": int(s.isna().sum())}
    q = valid.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(len(valid)), "n_missing": int(s.isna().sum()),
            "mean": valid.mean(), "median": q[0.5], "std": valid.std(ddof=1) if len(valid) > 1 else np.nan,
            "iqr": q[0.75] - q[0.25], "min": valid.min(), "max": valid.max(), "p05": q[0.05], "p95": q[0.95]}


def percentile_in(reference, value):
    reference = np.sort(np.asarray(reference, dtype=float))
    return float(np.searchsorted(reference, value, side="right") / len(reference))


def distorted(real, real_col):
    flag = {"card_reuse_count": "card_env_distorted", "ip_booking_density": "ip_env_distorted"}.get(real_col)
    return bool(flag and flag in real.columns and real[flag].astype(str).str.lower().eq("true").any())


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def plot_feature(title, real, real_col, dummy_normal, dummy_macro, continuous, log_axis, note, path):
    groups = [g for g in GROUP_ORDER if g in set(real["group"])]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.4, 4.6 + 0.32 * len(groups)), sharex=True,
        gridspec_kw={"height_ratios": [3, max(1.0, 0.45 * len(groups))], "hspace": 0.08})
    fig.patch.set_facecolor(SURFACE)

    real_values = pd.to_numeric(real[real_col], errors="coerce")
    everything = pd.concat([dummy_normal, dummy_macro, real_values]).dropna()
    use_log = log_axis and (everything > 0).any()
    if use_log:
        floor = everything[everything > 0].min()
        clip = lambda s: s.clip(lower=floor)  # 0은 로그 축에 못 그리므로 최소 양수 위치에 둔다  # noqa: E731
        low, high = floor, everything.max()
        bins = np.logspace(np.log10(low * 0.9), np.log10(high * 1.1), 36)
    else:
        clip = lambda s: s  # noqa: E731
        if continuous:
            bins = np.linspace(everything.min(), everything.max() + 1e-9, 36)
        else:
            bins = np.arange(np.floor(everything.min()) - 0.5, np.ceil(everything.max()) + 1.5, 1)

    for values, color, label in ((dummy_normal, C_NORMAL, f"더미 정상 (n={len(dummy_normal)})"),
                                 (dummy_macro, C_MACRO, f"더미 매크로 (n={len(dummy_macro)})")):
        weights = np.full(len(values), 1 / len(values))
        top.hist(clip(values), bins=bins, weights=weights, color=color, alpha=0.55,
                 edgecolor=SURFACE, linewidth=1.0, label=label)
    top.set_ylabel("집단 내 비율", color=INK_2, fontsize=9)
    top.grid(axis="y", color=GRID, linewidth=0.8)
    top.set_axisbelow(True)
    top.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    style_axis(top)

    for row, group in enumerate(groups):
        values = clip(real_values[real["group"] == group].dropna())
        missing = int(real_values[real["group"] == group].isna().sum())
        jitter = (np.arange(len(values)) - (len(values) - 1) / 2) * 0.12 if len(values) else []
        bottom.scatter(values, np.full(len(values), row) + jitter, s=52, color=INK, edgecolor=SURFACE,
                       linewidth=1.5, zorder=3)
        label = GROUP_LABELS[group] + f"  (n={len(values)}" + (f", 결측 {missing})" if missing else ")")
        bottom.text(-0.01, row, label, transform=bottom.get_yaxis_transform(), ha="right", va="center",
                    fontsize=9, color=INK)
    bottom.set_ylim(len(groups) - 0.5, -0.5)
    bottom.set_yticks([])
    bottom.grid(axis="x", color=GRID, linewidth=0.8)
    bottom.set_axisbelow(True)
    style_axis(bottom)
    if use_log:
        bottom.set_xscale("log")
        bottom.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    bottom.set_xlabel(title + ("  (로그 축)" if use_log else ""), color=INK_2, fontsize=9)

    fig.suptitle(title, x=0.02, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.02, 0.925, note, ha="left", va="top", fontsize=9, color=INK_2)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.84, bottom=0.12)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_scores(scores, dummy_scores, path):
    groups = [g for g in GROUP_ORDER if g in set(scores["group"])]
    rows = [("더미 정상", dummy_scores["normal"], C_NORMAL), ("더미 매크로", dummy_scores["macro"], C_MACRO)]
    fig, ax = plt.subplots(figsize=(8.4, 1.6 + 0.42 * (len(rows) + len(groups))))
    fig.patch.set_facecolor(SURFACE)
    labels = []
    for row, (name, values, color) in enumerate(rows):
        ax.scatter(values, np.full(len(values), row) + np.random.default_rng(0).uniform(-0.18, 0.18, len(values)),
                   s=10, color=color, alpha=0.45, linewidth=0)
        labels.append(f"{name} (n={len(values)})")
    for offset, group in enumerate(groups):
        row = len(rows) + offset
        values = scores.loc[scores["group"] == group, "anomaly_score"].dropna()
        ax.scatter(values, np.full(len(values), row), s=52, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        labels.append(f"{GROUP_LABELS[group]} (n={len(values)})")
    for threshold, name in ((0.40, "BTS 0.40"), (0.50, "번들 0.5"), (0.70, "BTS 0.70"), (0.85, "BTS 0.85")):
        ax.axvline(threshold, color=GRID, linewidth=1.2, zorder=0)
        ax.text(threshold, -0.75, name, ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("이상 점수 (0~1, 높을수록 이상)", color=INK_2, fontsize=9)
    style_axis(ax)
    fig.suptitle("기존 모델 이상 점수: 더미 대 실측", x=0.02, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.3, right=0.97, top=0.8, bottom=0.2)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    data_dir, out_dir = EXP / "data" / args.tag, EXP / "output" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    real = pd.read_csv(data_dir / "real_features.csv")
    dummy = pd.read_csv(DUMMY_PATH)
    normal, macro = dummy[dummy["is_anomaly"] == 0], dummy[dummy["scenario"] == "macro"]

    stats_rows, position_rows, ks_rows = [], [], []
    for title, real_col, dummy_col, continuous, log_axis in COMPARISONS:
        is_distorted = distorted(real, real_col)
        comparable = "비교 불가(환경 왜곡)" if is_distorted else "비교 가능"
        references = [("더미 정상(전체)", normal[dummy_col]), ("더미 매크로", macro[dummy_col])]
        references += [(f"더미 {name}", part[dummy_col]) for name, part in dummy.groupby("scenario") if name != "macro"]
        for name, values in references:
            stats_rows.append({"feature": title, "population": name, "comparable": comparable, **describe(values)})
        real_values = pd.to_numeric(real[real_col], errors="coerce")
        for group in [g for g in GROUP_ORDER if g in set(real["group"])]:
            values = real_values[real["group"] == group]
            stats_rows.append({"feature": title, "population": GROUP_LABELS[group], "comparable": comparable,
                               **describe(values)})
            valid = values.dropna()
            if continuous and len(valid) >= 2:
                for ref_name, ref in (("더미 정상(전체)", normal[dummy_col]), ("더미 매크로", macro[dummy_col])):
                    result = ks_2samp(valid, ref)
                    ks_rows.append({"feature": title, "real_group": GROUP_LABELS[group], "reference": ref_name,
                                    "n_real": len(valid), "n_reference": len(ref),
                                    "ks_statistic": result.statistic, "p_value": result.pvalue,
                                    "comparable": comparable,
                                    "note": "계정 수가 적어 검정력이 낮다. 참고용" if len(valid) < 20 else ""})
        for _, row in real.iterrows():
            value = pd.to_numeric(pd.Series([row[real_col]]), errors="coerce").iloc[0]
            position_rows.append({
                "feature": title, "external_user_id": row["external_user_id"], "group": row["group"],
                "value": value, "comparable": comparable,
                "percentile_in_dummy_normal": np.nan if pd.isna(value) else percentile_in(normal[dummy_col], value),
                "percentile_in_dummy_macro": np.nan if pd.isna(value) else percentile_in(macro[dummy_col], value),
                "below_dummy_normal_min": bool(value < normal[dummy_col].min()) if pd.notna(value) else pd.NA,
                "above_dummy_normal_max": bool(value > normal[dummy_col].max()) if pd.notna(value) else pd.NA,
            })
        note = "더미: 생성 분포에서 직접 뽑은 값 · 실측: 계정별 값(점 하나가 계정 하나)"
        if is_distorted:
            note += "\n주의: 데모 환경 왜곡(결제 식별자 3개 / 같은 IP)으로 직접 비교 불가"
        if "세션 기준" in title:
            note += "\n세션 기준은 실험용 가정이며 모델 입력이 아니다"
        safe = real_col if "세션" not in title else real_col + "_session"
        plot_feature(title, real, real_col, normal[dummy_col], macro[dummy_col], continuous, log_axis, note,
                     out_dir / f"fig_{safe}.png")

    pd.DataFrame(stats_rows).round(4).to_csv(out_dir / "summary_stats.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(position_rows).round(4).to_csv(out_dir / "real_positions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ks_rows).round(4).to_csv(out_dir / "ks_tests.csv", index=False, encoding="utf-8-sig")

    scores_path = data_dir / "real_scores.csv"
    if scores_path.exists():
        import sys
        sys.path.insert(0, str(EXP.parent.parent))  # AI 레포 루트
        from inference import load_bundle, raw_anomaly_score
        bundle = load_bundle(EXP / "model_build" / "models" / "behavior_model.joblib")
        cols = bundle["feature_columns"]
        to_score = lambda part: bundle["normalizer"].transform(raw_anomaly_score(bundle["model"], part[cols]))  # noqa: E731
        plot_scores(pd.read_csv(scores_path), {"normal": to_score(normal), "macro": to_score(macro)},
                    out_dir / "fig_scores.png")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    stats = pd.DataFrame(stats_rows)
    keep = stats["population"].isin(["더미 정상(전체)", "더미 매크로"] + list(GROUP_LABELS.values()))
    print(stats[keep][["feature", "population", "comparable", "n", "n_missing", "median", "min", "max"]]
          .round(3).to_string(index=False))
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
