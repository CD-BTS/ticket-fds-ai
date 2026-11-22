"""원본 · v1 · v2 더미의 매크로 예매 간격 Feature 가 실측과 얼마나 가까운지, 비교 단위를 맞춰 계산한다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\validation_stats.py

이미 있는 파일을 읽기만 한다 (새 수집·데이터셋 생성·모델 학습 없음).
    ml_experiment/data/fairticket_synthetic_features.csv           원본 더미
    dummy_adjusted/v1, v2/fairticket_synthetic_features_adjusted.csv  보정 더미
    dummy_adjusted/v2/real_account_features.csv                    실측 계정별 Feature (두 실행)
    dummy_adjusted/v2/leave_one_account_out.csv                    실측 계정 하나를 빼고 만든 검증 결과
    runs/20261123-remeasure/dummy_comparison/summary_by_type.csv    기존 실측 대 새 실측 (개별 간격)

비교 단위: 더미의 한 행은 계정 1개의 Feature 이므로, 실측도 **계정별 Feature**(계정 평균, 계정 모분산)로 맞춘다.
개별 예매 간격끼리의 비교는 실측 실행 사이에서만 한다 (더미에는 개별 간격이 없다).
출력: final_validation/ (다시 실행하면 이 폴더만 새로 쓴다)
"""
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = EXPERIMENT_DIR.parents[1]
MEAN_COL, VAR_COL = "avg_booking_interval", "booking_interval_variance"
TYPES = ["fast", "jitter"]
COLORS = {"원본": "#2a78d6", "v1": "#6250d6", "v2": "#eb6834"}
INK, MUTED, GRID, SURFACE = "#1f2328", "#6b7280", "#e5e7eb", "#ffffff"


def main():
    out = EXPERIMENT_DIR / "final_validation"
    if out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    original = pd.read_csv(REPO_DIR / "ml_experiment" / "data" / "fairticket_synthetic_features.csv")
    adjusted = {v: pd.read_csv(EXPERIMENT_DIR / "dummy_adjusted" / v / "fairticket_synthetic_features_adjusted.csv") for v in ("v1", "v2")}
    real = pd.read_csv(EXPERIMENT_DIR / "dummy_adjusted" / "v2" / "real_account_features.csv", encoding="utf-8-sig")
    macro = original[original["scenario"] == "macro"]
    # 원본에는 유형이 없으므로 macro 87행 전체를 각 유형의 실측과 비교한다.
    dummy = {"원본": {name: macro for name in TYPES},
             "v1": {name: adjusted["v1"][adjusted["v1"]["scenario"] == name] for name in TYPES},
             "v2": {name: adjusted["v2"][adjusted["v2"]["scenario"] == name] for name in TYPES}}
    reference_sets = {
        "실측 계정 전체 (간격 5개 이상)": real[real["used_for_generation"]],
        "기존 실측 계정": real[real["used_for_generation"] & (real["run"] == "기존 실측")],
        "새 실측 계정": real[real["used_for_generation"] & (real["run"] == "새 실측")],
    }
    independence = {  # 그 더미를 만들 때 이 실측 계정들을 썼는가
        ("원본", "실측 계정 전체 (간격 5개 이상)"): "독립", ("원본", "기존 실측 계정"): "독립", ("원본", "새 실측 계정"): "독립",
        ("v1", "실측 계정 전체 (간격 5개 이상)"): "일부 사용", ("v1", "기존 실측 계정"): "생성에 사용", ("v1", "새 실측 계정"): "독립",
        ("v2", "실측 계정 전체 (간격 5개 이상)"): "생성에 사용", ("v2", "기존 실측 계정"): "생성에 사용", ("v2", "새 실측 계정"): "생성에 사용",
    }

    rows = []
    for name in TYPES:
        for ref_label, ref in reference_sets.items():
            r = ref[ref["mode"] == name]
            for label in dummy:
                d = dummy[label][name]
                rows.append({
                    "type": name, "dummy": label, "reference": ref_label, "relation": independence[(label, ref_label)],
                    "n_dummy_accounts": len(d), "n_real_accounts": len(r),
                    "dummy_mean_of_account_means": d[MEAN_COL].mean(), "real_mean_of_account_means": r[MEAN_COL].mean(),
                    "mean_diff": d[MEAN_COL].mean() - r[MEAN_COL].mean(),
                    "dummy_median_of_account_means": d[MEAN_COL].median(), "real_median_of_account_means": r[MEAN_COL].median(),
                    "median_diff": d[MEAN_COL].median() - r[MEAN_COL].median(),
                    "dummy_std_of_account_means": d[MEAN_COL].std(ddof=1),
                    "real_std_of_account_means": r[MEAN_COL].std(ddof=1) if len(r) > 1 else np.nan,
                    "dummy_account_mean_min": d[MEAN_COL].min(), "dummy_account_mean_max": d[MEAN_COL].max(),
                    "real_account_mean_min": r[MEAN_COL].min(), "real_account_mean_max": r[MEAN_COL].max(),
                    "wasserstein_account_mean_s": wasserstein_distance(d[MEAN_COL], r[MEAN_COL]),
                    "dummy_mean_of_account_variances": d[VAR_COL].mean(), "real_mean_of_account_variances": r[VAR_COL].mean(),
                    "variance_mean_diff": d[VAR_COL].mean() - r[VAR_COL].mean(),
                    "dummy_median_of_account_variances": d[VAR_COL].median(), "real_median_of_account_variances": r[VAR_COL].median(),
                    "variance_median_diff": d[VAR_COL].median() - r[VAR_COL].median(),
                    "dummy_account_variance_min": d[VAR_COL].min(), "dummy_account_variance_max": d[VAR_COL].max(),
                    "real_account_variance_min": r[VAR_COL].min(), "real_account_variance_max": r[VAR_COL].max(),
                    "wasserstein_account_variance_s2": wasserstein_distance(d[VAR_COL], r[VAR_COL]),
                    "real_accounts_mean_within_dummy_range": int(r[MEAN_COL].between(d[MEAN_COL].min(), d[MEAN_COL].max()).sum()),
                    "real_accounts_variance_within_dummy_range": int(r[VAR_COL].between(d[VAR_COL].min(), d[VAR_COL].max()).sum()),
                })
    stats = pd.DataFrame(rows)
    stats.to_csv(out / "account_level_comparison.csv", index=False, encoding="utf-8-sig")
    real.to_csv(out / "real_accounts.csv", index=False, encoding="utf-8-sig")

    # 그래프: 계정 단위 분포 거리
    def style(ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    main_ref = "실측 계정 전체 (간격 5개 이상)"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, (col, title, unit) in zip(axes, (("wasserstein_account_mean_s", "계정 평균 예매 간격", "초"),
                                              ("wasserstein_account_variance_s2", "계정 예매 간격 분산", "초²"))):
        style(ax)
        width = 0.26
        x = np.arange(len(TYPES))
        for i, label in enumerate(dummy):
            part = stats[(stats["reference"] == main_ref) & (stats["dummy"] == label)].set_index("type").loc[TYPES]
            bars = ax.bar(x + (i - 1) * width, part[col], width=width - 0.03, color=COLORS[label], label=f"더미 {label}")
            for bar, value in zip(bars, part[col]):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n(실측 계정 3개)" for t in TYPES], fontsize=9)
        ax.set_ylabel(f"Wasserstein 거리 ({unit})", fontsize=9, color=MUTED)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("더미 계정과 실측 계정의 분포 거리 (계정 단위, 작을수록 가깝다)", fontsize=12, color=INK)
    fig.savefig(out / "figures" / "wasserstein_account_level.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    show = ["type", "dummy", "reference", "relation", "n_real_accounts", "mean_diff", "median_diff", "dummy_std_of_account_means",
            "real_std_of_account_means", "wasserstein_account_mean_s", "variance_mean_diff", "variance_median_diff",
            "wasserstein_account_variance_s2", "real_accounts_mean_within_dummy_range", "real_accounts_variance_within_dummy_range"]
    print(stats[show].round(3).to_string(index=False))
    loo = pd.read_csv(EXPERIMENT_DIR / "dummy_adjusted" / "v2" / "leave_one_account_out.csv", encoding="utf-8-sig")
    print(loo.round(3).to_string(index=False))
    runs = pd.read_csv(EXPERIMENT_DIR / "runs" / "20261123-remeasure" / "dummy_comparison" / "summary_by_type.csv", encoding="utf-8-sig")
    print(runs[["mode", "n_base", "n_new", "mean_base", "mean_new", "mean_diff", "median_base", "median_new", "median_diff",
                "std_base", "std_new", "wasserstein_s", "ks_statistic"]].round(3).to_string(index=False))
    print(f"출력: {out}")


if __name__ == "__main__":
    main()
