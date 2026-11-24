"""기존 Feature 직접 샘플링 데이터셋과 이벤트 기반 데이터셋의 분포를 비교한다 (읽기만 함, 모델 없음).

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\compare_distributions.py

비교 대상: 원본(ml_experiment/data), #16 v2(매크로 간격만 실측으로 바꾼 것), 이벤트 기반(data/single_session), 참고로 다회 세션 변형.
세 데이터셋은 같은 1,000계정·같은 행 순서다. 시나리오 이름은 #16 v2 의 것(fast / jitter 포함)을 모든 데이터셋에 쓴다.
원본에서 fast 와 jitter 는 같은 분포(macro)에서 나온 행이다.

출력: output/distribution_comparison/
  feature_stats.csv, distance_vs_event.csv, correlations.csv, within_scenario_correlation.csv,
  consistency_checks.csv, macro_interval_vs_measured.csv, macro_individual_intervals.csv, figures/*.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ISSUE16 = EXP.parent / "issue16-booking-interval"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ISSUE16 / "scripts"))
from compare_if_models import GRID, INK, MUTED, SURFACE, save, style  # noqa: E402  (#16 의 그래프 스타일)
from generate_events import ORIGINAL_PATH, V2_PATH  # noqa: E402

OUT = EXP / "output" / "distribution_comparison"
FEATURES = ["avg_booking_interval", "booking_interval_variance", "cancel_rate", "card_reuse_count", "ip_booking_density",
            "booking_count", "seat_selection_latency"]
UNITS = {"avg_booking_interval": "초", "booking_interval_variance": "초²", "cancel_rate": "비율", "card_reuse_count": "계정 수",
         "ip_booking_density": "계정 수", "booking_count": "회", "seat_selection_latency": "초"}
LOG_FEATURES = {"avg_booking_interval", "booking_interval_variance", "seat_selection_latency"}
SCENARIOS = ["normal", "fast_normal", "shared_ip", "frequent_cancel", "fast", "jitter", "multi_account", "hold_cancel"]
L_ORIG, L_V2, L_EVENT, L_MULTI = "원본", "#16 v2", "이벤트 기반", "이벤트 기반(다회 세션)"
COLOR = {L_ORIG: "#2a78d6", L_V2: "#6250d6", L_EVENT: "#eb6834", L_MULTI: "#1baf7a"}
PLAIN = FuncFormatter(lambda value, _: f"{value:g}")  # 로그 축 눈금을 10^n 대신 숫자로 (한글 글꼴에 지수용 빼기 기호가 없다)
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_orange", ["#2a78d6", "#eceef1", "#eb6834"])


def load():
    data = {L_ORIG: pd.read_csv(ORIGINAL_PATH), L_V2: pd.read_csv(V2_PATH),
            L_EVENT: pd.read_csv(EXP / "data" / "single_session" / "fairticket_event_based_features.csv"),
            L_MULTI: pd.read_csv(EXP / "data" / "multi_session" / "fairticket_event_based_features.csv")}
    scenario = data[L_V2]["scenario"]
    for frame in data.values():
        assert frame["user_id"].equals(data[L_ORIG]["user_id"]) and frame["is_anomaly"].equals(data[L_ORIG]["is_anomaly"])
        frame["scenario"] = scenario
    return data


def groups(frame):
    yield "전체", "전체", frame
    yield "레이블", "정상", frame[frame["is_anomaly"] == 0]
    yield "레이블", "이상", frame[frame["is_anomaly"] == 1]
    for name in SCENARIOS:
        yield "시나리오", name, frame[frame["scenario"] == name]


def feature_stats(data):
    rows = []
    for label, frame in data.items():
        for kind, name, part in groups(frame):
            for feature in FEATURES:
                v = part[feature].dropna()
                rows.append({"dataset": label, "group_type": kind, "group": name, "feature": feature, "unit": UNITS[feature],
                             "n": len(part), "n_null": int(part[feature].isna().sum()), "mean": v.mean(), "median": v.median(),
                             "std": v.std(ddof=1), "min": v.min(), "q25": v.quantile(0.25), "q75": v.quantile(0.75), "max": v.max()})
    return pd.DataFrame(rows)


def distances(data):
    rows = []
    event = data[L_EVENT]
    for other in (L_ORIG, L_V2):
        for (kind, name, a), (_, _, b) in zip(groups(data[other]), groups(event)):
            for feature in FEATURES:
                x, y = a[feature].dropna(), b[feature].dropna()
                rows.append({"compared": f"{other} 대 {L_EVENT}", "group_type": kind, "group": name, "feature": feature,
                             "unit": UNITS[feature], "n_existing": len(x), "n_event": len(y),
                             "median_existing": x.median(), "median_event": y.median(), "mean_existing": x.mean(), "mean_event": y.mean(),
                             "std_existing": x.std(ddof=1), "std_event": y.std(ddof=1),
                             "wasserstein": sps.wasserstein_distance(x, y), "ks_statistic": sps.ks_2samp(x, y).statistic})
    return pd.DataFrame(rows)


def correlations(data):
    """Spearman 순위상관. 값이 모두 같은 Feature(예: 정상 사용자의 카드 수 1)는 상관을 계산할 수 없어 빈 칸이다."""
    rows = []
    for label, frame in data.items():
        for kind, name, part in groups(frame):
            usable = [f for f in FEATURES if part[f].nunique(dropna=True) > 1]
            matrix = part[usable].corr(method="spearman")
            for i, a in enumerate(FEATURES):
                for b in FEATURES[i + 1:]:
                    value = matrix.loc[a, b] if a in usable and b in usable else np.nan
                    rows.append({"dataset": label, "group_type": kind, "group": name, "feature_a": a, "feature_b": b, "spearman": value})
    return pd.DataFrame(rows)


def consistency_checks(data):
    """행동에서 나온 값이라면 지켜져야 하는 관계. 한 행 안의 값만으로 판정할 수 있는 것만 넣었다."""
    rows = []
    for label, frame in data.items():
        n_intervals = frame["booking_count"] - 1
        mean, var, rate = frame["avg_booking_interval"], frame["booking_interval_variance"], frame["cancel_rate"]

        def representable(value, count):
            if pd.isna(value):
                return True
            # 취소율 = 취소된 완료 예매 / 완료 예매, 완료 예매는 시도 수 이하. 소수 4자리 반올림을 감안한다
            return any(abs(round(m / k, 4) - value) < 1e-9 for k in range(1, int(count) + 1) for m in range(k + 1))

        rules = [
            ("시도 1건 이하인데 예매 간격 값이 있음", "간격은 연속한 두 시도 사이의 값이라 시도가 2건 이상이어야 한다 (BTS 는 null)",
             (frame["booking_count"] < 2) & mean.notna(), frame["booking_count"] < 2),
            ("시도 2건인데 간격 분산이 0 이 아님", "간격이 1개이면 모분산은 항상 0 이다",
             (frame["booking_count"] == 2) & (var > 0), frame["booking_count"] == 2),
            ("간격 분산이 수학적 상한을 넘음", "음이 아닌 간격 n 개의 평균이 m 이면 모분산은 m²·(n−1) 이하다",
             (n_intervals >= 1) & (var > mean ** 2 * (n_intervals - 1) * (1 + 1e-6) + 1e-3), n_intervals >= 1),
            ("취소율이 (취소 건수 / 완료 예매 수)로 나올 수 없는 값", "완료 예매 수는 시도 수 이하의 정수다",
             ~pd.Series([representable(v, c) for v, c in zip(rate, frame["booking_count"])], index=frame.index), rate.notna()),
            ("좌석 선택 시간이 평균 예매 간격보다 김", "한 번에 한 시도씩 진행하면 다음 시도는 좌석 선택이 끝난 뒤에 시작한다 (seat_selection_latency 정의는 실험 가정)",
             frame["seat_selection_latency"] > mean, frame["seat_selection_latency"].notna() & mean.notna()),
        ]
        for name, why, violated, applicable in rules:
            rows.append({"dataset": label, "check": name, "reason": why, "applicable_rows": int(applicable.sum()),
                         "violating_rows": int(violated.sum()),
                         "violating_share_of_applicable": float(violated.sum() / applicable.sum()) if applicable.sum() else np.nan,
                         "violating_by_scenario": json.dumps(frame.loc[violated, "scenario"].value_counts().to_dict(), ensure_ascii=False)})
        rows.append({"dataset": label, "check": "카드·IP 공유 수의 상대 계정이 실제로 있음",
                     "reason": "공유 수 k 는 같은 값을 쓴 계정이 k 개 있다는 뜻이다",
                     "applicable_rows": int(len(frame)), "violating_rows": 0 if label.startswith(L_EVENT) else np.nan,
                     "violating_share_of_applicable": 0.0 if label.startswith(L_EVENT) else np.nan,
                     "violating_by_scenario": "이벤트 기반: 생성 기록의 공유 묶음과 전 계정 일치 (output/feature_extraction/*/validation.json)"
                     if label.startswith(L_EVENT) else "확인 불가: 계정별 숫자만 있고 누구와 공유하는지에 대한 정보가 없다"})
    return pd.DataFrame(rows)


def macro_vs_measured(data):
    """Fast/Jitter 의 예매 간격 Feature 를 #16 실측 계정(유형당 3개)과 비교한다."""
    real = pd.read_csv(ISSUE16 / "dummy_adjusted" / "v2" / "real_account_features.csv", encoding="utf-8-sig")
    real = real[real["used_for_generation"]]
    rows = []
    for mode in ("fast", "jitter"):
        measured = real[real["mode"] == mode]
        for feature in ("avg_booking_interval", "booking_interval_variance"):
            r = measured[feature].to_numpy()
            rows.append({"mode": mode, "feature": feature, "data": "실측 계정", "n": len(r), "mean": r.mean(), "median": np.median(r),
                         "std": r.std(ddof=1), "min": r.min(), "max": r.max()})
            for label in (L_ORIG, L_V2, L_EVENT):
                v = data[label].loc[data[label]["scenario"] == mode, feature].dropna().to_numpy()
                rows.append({"mode": mode, "feature": feature, "data": label, "n": len(v), "mean": v.mean(), "median": np.median(v),
                             "std": v.std(ddof=1), "min": v.min(), "max": v.max(), "mean_minus_real": v.mean() - r.mean(),
                             "wasserstein_to_real": sps.wasserstein_distance(v, r),
                             "real_accounts_inside_range": f"{int(((r >= v.min()) & (r <= v.max())).sum())} / {len(r)}",
                             "independent_of_real": label == L_ORIG})
    return pd.DataFrame(rows), real


def macro_individual_intervals():
    """이벤트 기반 데이터에는 개별 예매 간격이 있다 (기존 더미에는 없다). 실측 개별 간격과 비교한다."""
    attempts = pd.read_csv(EXP / "data" / "single_session" / "attempts.csv")
    accounts = pd.read_csv(EXP / "data" / "single_session" / "accounts.csv", keep_default_na=False)
    attempts = attempts.merge(accounts[["user_id", "scenario"]], on="user_id")
    attempts["start"] = pd.to_datetime(attempts["start_utc"])
    attempts = attempts.sort_values(["user_id", "start"])
    attempts["gap"] = attempts.groupby("user_id")["start"].diff().dt.total_seconds()
    pool = pd.read_csv(ISSUE16 / "dummy_adjusted" / "v2" / "real_interval_pool.csv", encoding="utf-8-sig")
    donors = pd.read_csv(EXP / "reference" / "macro_attempt_cycles.csv")["account"].unique()
    pool = pool[pool["account"].isin(donors)]
    rows = []
    for mode in ("fast", "jitter"):
        g = attempts.loc[attempts["scenario"] == mode, "gap"].dropna().to_numpy()
        r = pool.loc[pool["mode"] == mode, "gap_seconds"].to_numpy()
        for name, v in (("실측 개별 간격", r), ("이벤트 기반 개별 간격", g)):
            rows.append({"mode": mode, "data": name, "n": len(v), "mean": v.mean(), "median": np.median(v), "std": v.std(ddof=1),
                         "min": v.min(), "max": v.max()})
        rows[-1].update(wasserstein_to_real=sps.wasserstein_distance(g, r),
                        all_values_are_measured_values=bool(np.isin(np.round(g, 3), np.round(r, 3)).all()))
    return pd.DataFrame(rows)


# ---------------- 그래프 ----------------
def fig_feature(data, feature, path):
    fig, ax = plt.subplots(figsize=(11.5, 4.4), constrained_layout=True)
    style(ax)
    width = 0.36
    for offset, label in ((-0.2, L_ORIG), (0.2, L_EVENT)):
        values = [data[label].loc[data[label]["scenario"] == s, feature].dropna().to_numpy() for s in SCENARIOS]
        if feature in LOG_FEATURES:
            values = [v[v > 0] for v in values]
        box = ax.boxplot(values, positions=np.arange(len(SCENARIOS)) + offset, widths=width, patch_artist=True, whis=(5, 95),
                         showfliers=False, medianprops={"color": INK, "linewidth": 1.4})
        for patch in box["boxes"]:
            patch.set(facecolor=COLOR[label], edgecolor=SURFACE, linewidth=1)
        for part in box["whiskers"] + box["caps"]:
            part.set(color=COLOR[label], linewidth=1.2)
        ax.plot([], [], color=COLOR[label], linewidth=7, label=label)
    if feature in LOG_FEATURES:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(PLAIN)
    ax.set_xticks(range(len(SCENARIOS)))
    counts = data[L_EVENT]["scenario"].value_counts()
    ax.set_xticklabels([f"{s}\n(n={counts[s]})" for s in SCENARIOS], fontsize=9)
    ax.set_ylabel(f"{feature} ({UNITS[feature]})" + (", 로그 축" if feature in LOG_FEATURES else ""), fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
    note = "  ·  원본의 fast / jitter 는 같은 분포(macro)에서 나온 행" if feature != "seat_selection_latency" else "  ·  BTS 미산출 컬럼, 정의는 실험 가정"
    ax.set_title(f"{feature}: 시나리오별 분포 (상자 = 25~75%, 수염 = 5~95%, 가로선 = 중앙값){note}", loc="left", fontsize=10, color=INK)
    save(fig, path)


def fig_correlation(corr, path):
    panels = [("전체", "전체 1,000계정"), ("normal", "normal 시나리오 안")]
    short = ["간격 평균", "간격 분산", "취소율", "카드", "IP", "시도 수", "좌석 시간"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.6), constrained_layout=True)
    for row, (group, title) in enumerate(panels):
        for col, label in enumerate((L_ORIG, L_EVENT)):
            ax = axes[row][col]
            part = corr[(corr["dataset"] == label) & (corr["group"] == group)]
            matrix = np.full((len(FEATURES), len(FEATURES)), np.nan)
            for r in part.itertuples():
                i, j = FEATURES.index(r.feature_a), FEATURES.index(r.feature_b)
                matrix[i, j] = matrix[j, i] = r.spearman
            image = ax.imshow(np.ma.masked_invalid(matrix), cmap=DIVERGING, vmin=-1, vmax=1)
            ax.set_facecolor("#f6f7f9")
            for i in range(len(FEATURES)):
                for j in range(len(FEATURES)):
                    text = "" if i == j else ("–" if np.isnan(matrix[i, j]) else f"{matrix[i, j]:.2f}")
                    ax.text(j, i, text, ha="center", va="center", fontsize=8, color=INK)
            ax.set_xticks(range(len(FEATURES)))
            ax.set_yticks(range(len(FEATURES)))
            ax.set_xticklabels(short, fontsize=8, rotation=30, ha="right", color=MUTED)
            ax.set_yticklabels(short, fontsize=8, color=MUTED)
            for side in ax.spines.values():
                side.set_visible(False)
            ax.tick_params(length=0)
            ax.set_title(f"{label} — {title}", loc="left", fontsize=10, color=INK)
    fig.colorbar(image, ax=axes, shrink=0.5, label="Spearman 순위상관 (– = 값이 모두 같아 계산 불가)")
    save(fig, path)


def fig_scatter(data, x, y, path, title, logx=False, logy=False, jitter=0.0):
    rng = np.random.default_rng(0)  # 점 겹침을 푸는 용도 (값에는 영향 없음)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, label in zip(axes, (L_ORIG, L_EVENT)):
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        frame = data[label]
        for flag, name, color in ((0, "정상", "#9aa3ad"), (1, "이상", COLOR[label])):
            part = frame[frame["is_anomaly"] == flag]
            ax.scatter(part[x] + rng.uniform(-jitter, jitter, len(part)), part[y], s=14, color=color, edgecolor=SURFACE, linewidth=0.4,
                       label=name, alpha=0.9)
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(PLAIN)
        if logy:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(PLAIN)
        ax.set_xlabel(f"{x} ({UNITS[x]})", fontsize=9, color=MUTED)
        ax.set_title(label, loc="left", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1, 1), ncol=2, borderaxespad=0.2)
    axes[0].set_ylabel(f"{y} ({UNITS[y]})", fontsize=9, color=MUTED)
    fig.suptitle(title, fontsize=11, color=INK)
    save(fig, path)


def fig_macro(data, real, path):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, feature in zip(axes, ("avg_booking_interval", "booking_interval_variance")):
        style(ax)
        ticks = []
        for m, mode in enumerate(("fast", "jitter")):
            for k, label in enumerate((L_ORIG, L_V2, L_EVENT)):
                position = m * 4 + k
                values = data[label].loc[data[label]["scenario"] == mode, feature].dropna()
                ax.scatter(position + rng.uniform(-0.25, 0.25, len(values)), values, s=14, color=COLOR[label], edgecolor=SURFACE, linewidth=0.4)
                ticks.append((position, f"{label}\n{mode}"))
            measured = real.loc[real["mode"] == mode, feature]
            ax.hlines(measured, m * 4 - 0.45, m * 4 + 2.45, color=INK, linewidth=1.1, linestyle="--")
        ax.plot([], [], color=INK, linewidth=1.1, linestyle="--", label="실측 계정 (유형당 3개)")
        ax.set_xticks([p for p, _ in ticks])
        ax.set_xticklabels([t for _, t in ticks], fontsize=8)
        ax.set_ylabel(f"{feature} ({UNITS[feature]})", fontsize=9, color=MUTED)
        ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1, 1), borderaxespad=0.2)
        ax.set_title(feature, loc="left", fontsize=10, color=INK)
    fig.suptitle("Fast/Jitter 매크로의 예매 간격 Feature 와 #16 실측 계정 (점 = 더미 계정 1개)", fontsize=11, color=INK)
    save(fig, path)


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    data = load()
    main_sets = {k: data[k] for k in (L_ORIG, L_V2, L_EVENT)}

    stats = feature_stats(data)
    stats.to_csv(OUT / "feature_stats.csv", index=False, encoding="utf-8-sig")
    dist = distances(data)
    dist.to_csv(OUT / "distance_vs_event.csv", index=False, encoding="utf-8-sig")
    corr = correlations(main_sets)
    corr.to_csv(OUT / "correlations.csv", index=False, encoding="utf-8-sig")
    within = corr[corr["group_type"] == "시나리오"].pivot_table(index=["group", "feature_a", "feature_b"], columns="dataset",
                                                             values="spearman", dropna=False).reset_index()
    within.to_csv(OUT / "within_scenario_correlation.csv", index=False, encoding="utf-8-sig")
    checks = consistency_checks(data)
    checks.to_csv(OUT / "consistency_checks.csv", index=False, encoding="utf-8-sig")
    macro, real = macro_vs_measured(data)
    macro.to_csv(OUT / "macro_interval_vs_measured.csv", index=False, encoding="utf-8-sig")
    individual = macro_individual_intervals()
    individual.to_csv(OUT / "macro_individual_intervals.csv", index=False, encoding="utf-8-sig")

    for feature in FEATURES:
        fig_feature(data, feature, OUT / "figures" / f"feature_{feature}.png")
    fig_correlation(corr, OUT / "figures" / "correlation_heatmaps.png")
    fig_scatter(data, "avg_booking_interval", "booking_interval_variance", OUT / "figures" / "mean_vs_variance.png",
                "예매 간격의 평균과 분산 (계정 1개 = 점 1개, 두 축 모두 로그)", logx=True, logy=True)
    fig_scatter(data, "booking_count", "cancel_rate", OUT / "figures" / "cancel_rate_vs_booking_count.png",
                "시도 수와 취소율 (계정 1개 = 점 1개, 가로축 로그)", logx=True)
    fig_scatter(data, "seat_selection_latency", "avg_booking_interval", OUT / "figures" / "seat_latency_vs_interval.png",
                "좌석 선택 시간과 평균 예매 간격 (두 축 모두 로그)", logx=True, logy=True)
    fig_macro(data, real, OUT / "figures" / "macro_interval_vs_measured.png")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 400)
    pd.set_option("display.max_colwidth", 90)
    show = stats[(stats["dataset"] != L_V2) & (stats["group_type"] != "시나리오")]
    print(show.drop(columns=["unit", "group_type"]).round(3).to_string(index=False))
    med = stats[stats["group_type"] == "시나리오"].pivot_table(index=["feature", "group"], columns="dataset", values="median")
    print(med.round(3).to_string())
    print(dist[(dist["compared"].str.startswith(L_ORIG)) & (dist["group_type"] != "시나리오")].drop(columns=["compared", "unit"]).round(3).to_string(index=False))
    print(checks.drop(columns=["reason"]).to_string(index=False))
    key = [("avg_booking_interval", "booking_interval_variance"), ("avg_booking_interval", "seat_selection_latency"),
           ("booking_interval_variance", "booking_count"), ("card_reuse_count", "ip_booking_density"), ("cancel_rate", "booking_count")]
    print(within[[(a, b) in key for a, b in zip(within["feature_a"], within["feature_b"])]].round(2).to_string(index=False))
    print(corr[corr["group"].isin(["전체"])].pivot_table(index=["feature_a", "feature_b"], columns="dataset", values="spearman").round(2).to_string())
    print(macro.round(3).to_string(index=False))
    print(individual.round(3).to_string(index=False))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
