"""이벤트 기반 v2 와 v3 의 Feature 분포, 정상/이상의 중첩, 실측과의 유사성, 수학적 일관성을 비교한다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\compare_v3.py

- 데이터셋과 이벤트는 읽기만 한다. 모델은 쓰지 않는다 (모델 평가는 scripts/evaluate_v3.py).
- recent_booking_count 는 모델의 6개 Feature 가 아니라 프론트 전용 Feature 다. v2 보고서와 같은 방식으로 이벤트에서 계산해 참고로 본다
  (좌석 선점 요청이 가장 몰린 1분 / 10분 구간의 횟수). 모델의 6개 중 시도 횟수에 해당하는 것은 booking_count 다.
- 실측 매크로와의 비교는 '같은 오픈 안의 연속 시도 간격'으로 한다. v3 는 계정이 여러 오픈에 참여하므로 30일 평균 간격에는
  오픈 사이의 며칠짜리 간격이 들어가고, 실측(한 번의 연속 실행)과 직접 비교할 수 없다.

출력: output/v3/distribution_comparison/
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP.parent / "issue16-booking-interval" / "scripts"))
sys.path.insert(0, str(EXP.parent / "issue14-real-validation" / "scripts"))
from compare_distributions import SCENARIOS, consistency_checks  # noqa: E402
from compare_if_models import GRID, INK, MUTED, SURFACE, save, style  # noqa: E402
from compare_v2 import aux_features, real_accounts  # noqa: E402
from extract_features import load_events  # noqa: E402
from generate_events import REFERENCE_PATH, V2_PATH, load_config, sha256  # noqa: E402
from generate_events_v3 import CONFIG_PATH, generate  # noqa: E402

OUT = EXP / "output" / "v3" / "distribution_comparison"
L2, L3 = "이벤트 기반 v2", "이벤트 기반 v3"
DIRS = {L2: EXP / "data" / "v2", L3: EXP / "data" / "v3"}
MODEL_FEATURES = ["avg_booking_interval", "booking_interval_variance", "cancel_rate", "card_reuse_count", "ip_booking_density", "booking_count"]
AUX = ["recent_booking_count_1m", "recent_booking_count_10m"]
NORMAL = SCENARIOS[:4]
LOG = {"avg_booking_interval", "booking_interval_variance", "booking_count", "recent_booking_count_1m", "recent_booking_count_10m"}
C_NORMAL, C_ANOM, C_V2, C_V3 = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
MIN_GAPS = 5  # 계정 평균을 실측(계정당 간격 19개)과 비교할 때 필요한 최소 간격 수


def within_open_gaps(attempts):
    """같은 계정이 같은 오픈에서 이어서 한 시도들의 시작 간격(초)."""
    table = attempts.assign(start=pd.to_datetime(attempts["start_utc"])).sort_values(["user_id", "start"])
    gap = table.groupby(["user_id", "concert"])["start"].diff().dt.total_seconds()
    return table.assign(gap=gap).dropna(subset=["gap"])[["user_id", "concert", "gap"]]


def load():
    data = {}
    for label, folder in DIRS.items():
        frame = pd.read_csv(folder / "fairticket_event_based_features.csv")
        frame = frame.merge(aux_features(load_events(folder / "events.csv"))[["user_id"] + AUX], on="user_id", how="left", validate="one_to_one")
        accounts = pd.read_csv(folder / "accounts.csv", keep_default_na=False)
        attempts = pd.read_csv(folder / "attempts.csv")
        gaps = within_open_gaps(attempts)
        per_account = gaps.groupby("user_id")["gap"].agg(open_gap_mean="mean", open_gap_var=lambda v: v.var(ddof=0), open_gap_n="size")
        frame = frame.merge(per_account, left_on="user_id", right_index=True, how="left")
        frame["n_sessions"] = accounts["n_sessions"].to_numpy()
        frame["tool_profile"] = accounts["tool_profile"].to_numpy() if "tool_profile" in accounts else ""
        data[label] = {"frame": frame, "accounts": accounts, "attempts": attempts, "gaps": gaps}
    return data


def quantile_table(data):
    rows = []
    for label, pack in data.items():
        frame = pack["frame"]
        parts = [("레이블", "정상", frame[frame["is_anomaly"] == 0]), ("레이블", "이상", frame[frame["is_anomaly"] == 1])]
        parts += [("시나리오", name, frame[frame["scenario"] == name]) for name in SCENARIOS]
        for kind, name, part in parts:
            for feature in MODEL_FEATURES + AUX:
                values = part[feature].dropna()
                q = values.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).to_numpy()
                rows.append({"dataset": label, "group_type": kind, "group": name, "feature": feature, "n": len(part), "null": int(part[feature].isna().sum()),
                             "mean": values.mean(), "std": values.std(), **dict(zip(["min", "p05", "p25", "median", "p75", "p95", "max"], q))})
    return pd.DataFrame(rows)


def overlap_coefficient(a, b, log):
    a, b = (np.log1p(a), np.log1p(b)) if log else (a, b)
    edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=30)
    return float(np.minimum(np.histogram(a, edges)[0] / len(a), np.histogram(b, edges)[0] / len(b)).sum())


def best_single_threshold(y, x, larger_is_anomaly):
    """Feature 하나에 임계값 하나를 둔 규칙. 결측 계정은 이상 아님으로 센다. 전체 계정 기준의 값이다."""
    score = np.where(np.isnan(x), -np.inf, x if larger_is_anomaly else -x)
    order = np.argsort(-score, kind="stable")
    s, t = score[order], y[order]
    tp, fp = np.cumsum(t), np.cumsum(1 - t)
    last = np.r_[s[1:] != s[:-1], True] & np.isfinite(s)  # 같은 값은 함께 판정된다
    tp, fp, cut = tp[last], fp[last], s[last]
    precision, recall, fpr = tp / (tp + fp), tp / t.sum(), fp / (1 - t).sum()
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1))
    ok = fpr <= 0.05
    return {"best_f1": float(f1[best]), "best_f1_precision": float(precision[best]), "best_f1_recall": float(recall[best]), "best_f1_fpr": float(fpr[best]),
            "best_f1_threshold": float(cut[best] if larger_is_anomaly else -cut[best]),
            "recall_at_fpr_le_5pct": float(recall[ok].max()) if ok.any() else 0.0}


def separability(data):
    rows = []
    for label, pack in data.items():
        frame = pack["frame"]
        normal = frame[frame["is_anomaly"] == 0]
        for feature in MODEL_FEATURES + AUX:
            base = normal[feature].dropna().to_numpy(dtype=float)
            for group in ["이상 전체"] + SCENARIOS[4:]:
                part = frame[frame["is_anomaly"] == 1] if group == "이상 전체" else frame[frame["scenario"] == group]
                values = part[feature].dropna().to_numpy(dtype=float)
                auc = roc_auc_score(np.r_[np.zeros(len(base)), np.ones(len(values))], np.r_[base, values])
                row = {"dataset": label, "feature": feature, "group": group, "n_normal": len(normal), "n_anomaly": len(part),
                       "normal_null": int(normal[feature].isna().sum()), "anomaly_null": int(part[feature].isna().sum()),
                       "direction": "이상이 더 큼" if auc >= 0.5 else "이상이 더 작음", "auc_single_feature": max(auc, 1 - auc),
                       "ks": sps.ks_2samp(base, values).statistic, "overlap_coefficient": overlap_coefficient(base, values, feature in LOG),
                       "normal_median": np.median(base), "anomaly_median": np.median(values),
                       "anomaly_inside_normal_range": float(((values >= base.min()) & (values <= base.max())).mean()),
                       "anomaly_inside_normal_p05_p95": float(((values >= np.quantile(base, 0.05)) & (values <= np.quantile(base, 0.95))).mean()),
                       "normal_inside_anomaly_range": float(((base >= values.min()) & (base <= values.max())).mean())}
                if group == "이상 전체":
                    row.update(best_single_threshold(frame["is_anomaly"].to_numpy(), frame[feature].to_numpy(dtype=float), auc >= 0.5))
                rows.append(row)
    return pd.DataFrame(rows)


def generation_facts(data):
    """어떤 생성 규칙이 두 집단을 나누는지: 계정이 받은 시도 수·오픈 참여 수와 같은 오픈 안의 간격."""
    rows = []
    for label, pack in data.items():
        frame, accounts, attempts = pack["frame"], pack["accounts"], pack["attempts"]
        for name in SCENARIOS:
            part, info = frame[frame["scenario"] == name], accounts[accounts["scenario"] == name]
            tries = attempts[attempts["user_id"].isin(info["user_id"])]
            gaps = part["open_gap_mean"].dropna()
            rows.append({"dataset": label, "scenario": name, "accounts": len(part),
                         "attempts_min": int(info["n_attempts"].min()), "attempts_median": float(info["n_attempts"].median()), "attempts_max": int(info["n_attempts"].max()),
                         "accounts_with_1_attempt": int((info["n_attempts"] == 1).sum()), "accounts_with_10_or_more_attempts": int((info["n_attempts"] >= 10).sum()),
                         "accounts_in_2_or_more_opens": int((info["n_sessions"] >= 2).sum()),
                         "within_open_gap_account_mean_min": gaps.min(), "within_open_gap_account_mean_median": gaps.median(), "within_open_gap_account_mean_max": gaps.max(),
                         "avg_booking_interval_median_single_open": part.loc[part["n_sessions"] == 1, "avg_booking_interval"].median(),
                         "avg_booking_interval_median_multi_open": part.loc[part["n_sessions"] >= 2, "avg_booking_interval"].median(),
                         "attempt_complete_share": float((tries["outcome"] == "complete").mean()),
                         "attempts_with_hold_rejection_share": float((tries["hold_rejections"] > 0).mean()) if "hold_rejections" in tries else 0.0})
    return pd.DataFrame(rows)


def macro_vs_measured(data, real):
    reference = pd.read_csv(REFERENCE_PATH)
    rows = []
    for mode in ("fast", "jitter"):
        measured = real[real["group"] == f"실측 {mode}"]
        real_gaps = reference.loc[reference["mode"] == mode, "gap_to_next_ms"].dropna().to_numpy(dtype=float) / 1000
        for feature, column in (("오픈 안 간격의 계정 평균 (초)", "open_gap_mean"), ("오픈 안 간격의 계정 분산 (초²)", "open_gap_var")):
            r = measured["avg_booking_interval" if column == "open_gap_mean" else "booking_interval_variance"].to_numpy()
            rows.append({"mode": mode, "quantity": feature, "data": "실측 계정", "n": len(r), "mean": r.mean(), "std": r.std(ddof=1), "min": r.min(), "max": r.max()})
            for label, pack in data.items():
                frame = pack["frame"]
                v = frame.loc[(frame["scenario"] == mode) & (frame["open_gap_n"] >= MIN_GAPS), column].to_numpy()
                rows.append({"mode": mode, "quantity": feature, "data": label, "n": len(v), "mean": v.mean(), "std": v.std(ddof=1), "min": v.min(), "max": v.max(),
                             "mean_minus_real": v.mean() - r.mean(), "std_ratio_to_real": v.std(ddof=1) / r.std(ddof=1),
                             "wasserstein_to_real": sps.wasserstein_distance(v, r), "real_accounts_inside_range": f"{int(((r >= v.min()) & (r <= v.max())).sum())} / {len(r)}",
                             "accounts_excluded_fewer_gaps": int(((frame["scenario"] == mode) & ~(frame["open_gap_n"] >= MIN_GAPS)).sum())})
        rows.append({"mode": mode, "quantity": "개별 간격 (초)", "data": "실측 간격", "n": len(real_gaps), "mean": real_gaps.mean(), "std": real_gaps.std(ddof=1),
                     "min": real_gaps.min(), "max": real_gaps.max()})
        for label, pack in data.items():
            users = pack["frame"].loc[pack["frame"]["scenario"] == mode, "user_id"]
            v = pack["gaps"].loc[pack["gaps"]["user_id"].isin(users), "gap"].to_numpy()
            rows.append({"mode": mode, "quantity": "개별 간격 (초)", "data": label, "n": len(v), "mean": v.mean(), "std": v.std(ddof=1), "min": v.min(), "max": v.max(),
                         "mean_minus_real": v.mean() - real_gaps.mean(), "std_ratio_to_real": v.std(ddof=1) / real_gaps.std(ddof=1),
                         "wasserstein_to_real": sps.wasserstein_distance(v, real_gaps)})
    return pd.DataFrame(rows)


def leave_one_account_out(real):
    """실측 계정 하나를 숨기고 나머지로 v3 방식의 매크로 계정을 만든 뒤, 숨긴 계정의 평균 간격이 합성 계정(오픈 안 간격 5개 이상)의 범위에 드는지 본다."""
    reference = pd.read_csv(REFERENCE_PATH)
    assignment = pd.read_csv(V2_PATH)[["user_id", "scenario", "is_anomaly"]]
    config = load_config(CONFIG_PATH)
    rows = []
    for mode in ("fast", "jitter"):
        subset = assignment[assignment["scenario"] == mode].reset_index(drop=True)
        for hidden in real.loc[real["group"] == f"실측 {mode}"].itertuples():
            _, _, attempts = generate(config, config["seed"], subset, reference[reference["account"] != hidden.account])
            gaps = within_open_gaps(attempts).groupby("user_id")["gap"].agg(["mean", "size"])
            v = gaps.loc[gaps["size"] >= MIN_GAPS, "mean"].to_numpy()
            value = hidden.avg_booking_interval
            rows.append({"mode": mode, "hidden_account": hidden.account, "method": "v3 방식", "synthetic_accounts": len(v), "hidden_mean": value,
                         "synthetic_mean_min": v.min(), "synthetic_mean_q05": np.quantile(v, 0.05), "synthetic_mean_q95": np.quantile(v, 0.95),
                         "synthetic_mean_max": v.max(), "mean_inside_range": bool(v.min() <= value <= v.max()),
                         "mean_inside_90pct": bool(np.quantile(v, 0.05) <= value <= np.quantile(v, 0.95)),
                         "mean_distance_outside_range": max(v.min() - value, value - v.max(), 0.0)})
    return pd.DataFrame(rows)


def weak_measured(data, real):
    """근거가 약한 실측(#14): 사람 2계정과 human 프로필 매크로 1계정이 합성 분포의 어디에 있는지."""
    rows = []
    human = pd.read_csv(EXP / "reference" / "macro_human_attempt_cycles.csv")
    cycle = human[["queue_ms", "pre_seat_ms", "seat_ms", "hold_to_booking_ms", "completed_ms"]].sum(axis=1) / 1000
    for label, pack in data.items():
        frame = pack["frame"]
        normal = frame.loc[(frame["is_anomaly"] == 0) & frame["open_gap_n"].notna(), "open_gap_mean"].to_numpy()
        for person in real[real["group"].str.contains("사람")].itertuples():
            rows.append({"dataset": label, "measured": f"사람 {person.account} 의 평균 간격", "value": person.avg_booking_interval,
                         "compared_with": "정상 계정의 오픈 안 평균 간격", "synthetic_min": normal.min(), "synthetic_median": np.median(normal),
                         "synthetic_max": normal.max(), "percentile_of_measured": float((normal <= person.avg_booking_interval).mean())})
        ring = frame.loc[(frame["tool_profile"] == "human") & frame["open_gap_n"].notna(), "open_gap_mean"].to_numpy()
        if len(ring):
            rows.append({"dataset": label, "measured": "human 프로필 매크로의 시도 1건 소요 시간 (다음 시도까지의 대기 제외)", "value": float(cycle.median()),
                         "compared_with": "human 프로필 링의 오픈 안 평균 간격 (대기 포함)", "synthetic_min": ring.min(), "synthetic_median": np.median(ring),
                         "synthetic_max": ring.max(), "percentile_of_measured": np.nan})
    return pd.DataFrame(rows)


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 400)
    before = {label: sha256(folder / "fairticket_event_based_features.csv") for label, folder in DIRS.items()}
    data = load()
    real = real_accounts()

    def csv(frame, name):
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    stats, sep, facts = quantile_table(data), separability(data), generation_facts(data)
    measured, loao, weak = macro_vs_measured(data, real), leave_one_account_out(real), weak_measured(data, real)
    checks = consistency_checks({label: pack["frame"] for label, pack in data.items()})
    previous = pd.read_csv(EXP / "output" / "v2" / "distribution_comparison" / "leave_one_account_out.csv")
    previous = previous.loc[previous["method"].str.startswith("v2"), ["mode", "hidden_account", "hidden_mean", "synthetic_mean_min", "synthetic_mean_max",
                                                                     "mean_inside_range", "mean_inside_90pct"]].assign(method="v2 방식 (REPORT_V2 의 결과)")
    loao = pd.concat([previous, loao], ignore_index=True)
    for frame, name in ((stats, "feature_stats.csv"), (sep, "separability.csv"), (facts, "generation_facts.csv"), (measured, "macro_vs_measured.csv"),
                        (loao, "leave_one_account_out.csv"), (weak, "weak_measured_position.csv"), (checks, "consistency_checks.csv"), (real, "real_accounts.csv")):
        csv(frame, name)
    for label, pack in data.items():
        csv(pack["frame"], f"features_with_aux_{'v2' if label == L2 else 'v3'}.csv")

    # v3 에서 30일 평균 간격이 오픈 참여 횟수에 따라 어떻게 갈리는지
    f3 = data[L3]["frame"]
    by_open = (f3.assign(label=f3["is_anomaly"].map({0: "정상", 1: "이상"}), opens=np.where(f3["n_sessions"] >= 2, "오픈 2회 이상", "오픈 1회"))
               .groupby(["label", "opens"]).agg(accounts=("user_id", "size"), interval_null=("avg_booking_interval", lambda v: int(v.isna().sum())),
                                                interval_p05=("avg_booking_interval", lambda v: v.quantile(0.05)), interval_median=("avg_booking_interval", "median"),
                                                interval_p95=("avg_booking_interval", lambda v: v.quantile(0.95)), attempts_median=("booking_count", "median")).reset_index())
    csv(by_open, "v3_interval_by_open_participation.csv")

    # ---------------- 그래프 ----------------
    rng = np.random.default_rng(0)
    shown = ["avg_booking_interval", "booking_count", "recent_booking_count_1m"]
    fig, axes = plt.subplots(len(shown), 2, figsize=(13, 10.5), constrained_layout=True)
    for r, feature in enumerate(shown):
        low = min(pack["frame"][feature].min() for pack in data.values())
        high = max(pack["frame"][feature].max() for pack in data.values())
        for c, (label, pack) in enumerate(data.items()):
            ax, frame = axes[r, c], pack["frame"]
            style(ax)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=GRID, linewidth=0.8)
            for position, name in enumerate(SCENARIOS[::-1]):
                values = frame.loc[frame["scenario"] == name, feature].dropna().to_numpy()
                ax.scatter(values, position + rng.uniform(-0.3, 0.3, len(values)), s=6, alpha=0.4, linewidth=0, color=C_NORMAL if name in NORMAL else C_ANOM)
                ax.vlines(np.median(values), position - 0.4, position + 0.4, color=INK, linewidth=1.6)
            ax.set_yticks(range(len(SCENARIOS)))
            ax.set_yticklabels(SCENARIOS[::-1] if c == 0 else [], fontsize=9)
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_xlim(0, high * 1.3)
            ax.set_title(f"{feature} — {label}", loc="left", fontsize=10, color=INK)
    fig.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color=C_NORMAL, label="정상 시나리오"),
                        plt.Line2D([], [], marker="o", linestyle="", color=C_ANOM, label="이상 시나리오"), plt.Line2D([], [], color=INK, linewidth=1.6, label="중앙값")],
               loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("시나리오별 분포: v2 (왼쪽) 와 v3 (오른쪽), 로그 눈금. recent_booking_count 는 모델 입력이 아닌 참고 지표", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "distribution_v2_v3.png")

    overall = sep[sep["group"] == "이상 전체"]
    features = MODEL_FEATURES + AUX
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for ax, (column, title) in zip(axes, (("auc_single_feature", "Feature 하나의 AUC (0.5 = 구분 불가, 1.0 = 완전 분리)"), ("best_f1", "Feature 하나 + 임계값 하나의 최고 F1"))):
        style(ax)
        x = np.arange(len(features))
        for offset, label, color in ((-0.2, L2, C_V2), (0.2, L3, C_V3)):
            values = overall[overall["dataset"] == label].set_index("feature").loc[features, column].to_numpy()
            ax.bar(x + offset, values, width=0.37, color=color, label=label)
            for xi, value in zip(x + offset, values):
                ax.text(xi, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("_", "\n") for f in features], fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    save(fig, OUT / "figures" / "single_feature_separability.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, mode in zip(axes, ("fast", "jitter")):
        style(ax)
        for position, (label, color) in enumerate(((L2, C_V2), (L3, C_V3))):
            frame = data[label]["frame"]
            values = frame.loc[(frame["scenario"] == mode) & (frame["open_gap_n"] >= MIN_GAPS), "open_gap_mean"].to_numpy()
            ax.scatter(position + rng.uniform(-0.2, 0.2, len(values)), values, s=18, color=color, edgecolor=SURFACE, linewidth=0.5)
        points = real.loc[real["group"] == f"실측 {mode}", "avg_booking_interval"].to_numpy()
        ax.scatter(np.full(len(points), 2), points, s=60, color=INK, marker="D")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([L2, L3, "실측 3계정"], fontsize=9)
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(f"{mode}: 같은 오픈 안의 평균 간격 (초, 간격 {MIN_GAPS}개 이상인 계정)", loc="left", fontsize=10, color=INK)
    save(fig, OUT / "figures" / "macro_vs_measured.png")

    record = {"datasets": {label: {"path": (folder / "fairticket_event_based_features.csv").relative_to(EXP).as_posix(), "sha256": before[label]}
                           for label, folder in DIRS.items()},
              "datasets_unchanged_after_run": before == {label: sha256(folder / "fairticket_event_based_features.csv") for label, folder in DIRS.items()},
              "min_gaps_for_account_mean": MIN_GAPS}
    (OUT / "comparison_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    show = stats[(stats["feature"].isin(["avg_booking_interval", "booking_count", "recent_booking_count_1m", "recent_booking_count_10m"]))]
    print(show[["dataset", "group", "feature", "n", "null", "min", "p05", "median", "p95", "max"]].round(2).to_string(index=False))
    print(stats[stats["group_type"] == "레이블"][["dataset", "group", "feature", "null", "min", "p05", "median", "p95", "max"]].round(3).to_string(index=False))
    print(overall[["dataset", "feature", "direction", "auc_single_feature", "ks", "overlap_coefficient", "anomaly_inside_normal_range", "anomaly_inside_normal_p05_p95",
                   "normal_inside_anomaly_range", "best_f1", "best_f1_precision", "best_f1_recall", "best_f1_fpr", "best_f1_threshold", "recall_at_fpr_le_5pct"]].round(3).to_string(index=False))
    print(sep[sep["group"] != "이상 전체"].pivot_table(index=["feature", "group"], columns="dataset", values="auc_single_feature").round(3).to_string())
    print(facts.round(2).T.to_string())
    print(by_open.round(2).to_string(index=False))
    print(measured.round(3).to_string(index=False))
    print(loao.round(3).to_string(index=False))
    print(weak.round(2).to_string(index=False))
    print(checks[["dataset", "check", "applicable_rows", "violating_rows"]].to_string(index=False))
    print(record["datasets_unchanged_after_run"], f"저장: {OUT}")


if __name__ == "__main__":
    main()
