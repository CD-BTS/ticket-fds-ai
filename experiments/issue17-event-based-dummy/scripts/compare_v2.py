"""A(기존 직접 샘플링) · B(이벤트 기반 v1) · C(이벤트 기반 v2)의 Feature 분포를 비교한다 (읽기만 함, 모델 없음).

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\compare_v2.py

출력: output/v2/distribution_comparison/
  feature_stats.csv              정상/이상·시나리오별 평균·중앙값·표준편차 등 (보조 지표 포함)
  change_b_to_c.csv              v1 → v2 에서 시나리오별로 달라진 정도 (중앙값, KS, Wasserstein)
  correlations.csv, within_scenario_correlation.csv
  consistency_checks.csv         행동에서 나올 수 없는 값·수학적 제약 위반
  criteria_rates.csv             기존 평가 기준(기준 후보·Credit 규칙) 충족 비율, 실측 계정 포함
  macro_interval_vs_measured.csv 실측 Fast/Jitter 계정과의 거리
  leave_one_account_out.csv      실측 계정 하나를 숨기고 v1 방식·v2 방식으로 만든 결과
  aux_features_<v1|v2>.csv       이벤트에서 계산한 보조 지표 (step_stay_time, recent_booking_count)
  figures/*.png

평가 기준의 출처: demo-ticketing/frontend/src/risk/features.ts (기준 후보), credit.ts (Credit 규칙의 확정 초기값).
기준 후보에는 방향(미만/이상)이 적혀 있지 않은 것이 있다. 여기서는 Credit 규칙의 방향을 따라
간격·분산·좌석 시간은 '미만', 취소율·카드·IP·시도 수는 '이상'으로 계산했다 (이 방향은 이 스크립트의 해석이다).
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

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ISSUE14 = EXP.parent / "issue14-real-validation"
ISSUE16 = EXP.parent / "issue16-booking-interval"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ISSUE16 / "scripts"))
sys.path.insert(0, str(ISSUE14 / "scripts"))
from compare_distributions import DIVERGING, FEATURES, PLAIN, SCENARIOS, UNITS, consistency_checks, correlations, feature_stats  # noqa: E402
from compare_if_models import GRID, INK, MUTED, SURFACE, save, style  # noqa: E402
from extract_dataset import build_dataset  # noqa: E402
from extract_features import load_events  # noqa: E402
from generate_events import ORIGINAL_PATH, REFERENCE_PATH, V2_PATH, generate, load_config  # noqa: E402

OUT = EXP / "output" / "v2" / "distribution_comparison"
L_A, L_B, L_C = "A 원본", "이벤트 기반 v1 (B)", "이벤트 기반 v2 (C)"
PATHS = {L_A: ORIGINAL_PATH, L_B: EXP / "data" / "single_session" / "fairticket_event_based_features.csv",
         L_C: EXP / "data" / "v2" / "fairticket_event_based_features.csv"}
EVENT_DIRS = {L_B: EXP / "data" / "single_session", L_C: EXP / "data" / "v2"}
COLOR = {L_A: "#2a78d6", L_B: "#eb6834", L_C: "#1baf7a"}
AUX = ["step_stay_seat_select", "step_stay_payment", "recent_booking_count_1m", "recent_booking_count_10m"]
AUX_UNITS = {"step_stay_seat_select": "초", "step_stay_payment": "초", "recent_booking_count_1m": "회", "recent_booking_count_10m": "회"}
LOG = {"avg_booking_interval", "booking_interval_variance", "seat_selection_latency", "step_stay_seat_select", "step_stay_payment"}


# ---------------- 보조 지표 (프론트 전용 Feature 를 이벤트에서 다시 계산) ----------------
def aux_features(events):
    """step_stay_time, recent_booking_count 를 계정별로 계산한다. 데이터셋 컬럼이 아니라 검증용이다.

    - step_stay_time.seat_select : SEAT_SELECTION 진입 ~ 그 시도의 다음 단계 이벤트. 계정의 중앙값
    - step_stay_time.payment     : PAYMENT 진입 ~ 그 시도의 다음 단계 이벤트. 계정의 중앙값 (PAYMENT 이벤트가 없는 시도는 제외)
    - recent_booking_count       : 좌석 선점 요청(SEAT_HOLD 이벤트)이 가장 몰린 1분 / 10분 구간의 횟수
    프론트는 '가장 최근 방문'과 '지금 기준 최근 1분'을 쓴다. 계정 1행으로 요약하려고 중앙값과 최대 구간을 쓴 것은 이 실험의 가정이다.
    """
    stage = events[events["event_type"] == "BOOKING_ATTEMPT"].copy()
    stage["attempt"] = stage["meta"].map(lambda m: m["attemptId"])
    stage["kind"] = stage["meta"].map(lambda m: (m["stage"], m["status"]))
    stage["seq"] = stage["meta"].map(lambda m: m["sequence"])
    stage = stage.sort_values(["attempt", "seq"])
    stage["next_t"] = stage.groupby("attempt")["occurred_at"].shift(-1)
    stage["stay"] = (stage["next_t"] - stage["occurred_at"]).dt.total_seconds()
    out = pd.DataFrame(index=pd.Index(events["user_id"].unique(), name="user_id"))
    out["step_stay_seat_select"] = stage[stage["kind"] == ("SEAT_SELECTION", "ENTERED")].groupby("user_id")["stay"].median()
    out["step_stay_payment"] = stage[stage["kind"] == ("PAYMENT", "ENTERED")].groupby("user_id")["stay"].median()
    holds = stage[stage["meta"].map(lambda m: m["stage"] == "SEAT_HOLD" and m["status"] in ("SUCCEEDED", "REJECTED"))]
    for column, window in (("recent_booking_count_1m", 60), ("recent_booking_count_10m", 600)):
        peak = {}
        for user, part in holds.groupby("user_id"):
            t = np.sort(part["occurred_at"].to_numpy().astype("datetime64[ms]").astype("int64")) / 1000
            peak[user] = int((np.arange(len(t)) - np.searchsorted(t, t - window, side="right") + 1).max())
        out[column] = pd.Series(peak)
        out[column] = out[column].fillna(0).astype(int)
    return out.reset_index()


def load():
    scenario = pd.read_csv(V2_PATH)["scenario"]
    data = {}
    for label, path in PATHS.items():
        frame = pd.read_csv(path)
        frame["scenario"] = scenario
        data[label] = frame
    for label, folder in EVENT_DIRS.items():
        aux = aux_features(load_events(folder / "events.csv"))
        aux.to_csv(OUT / f"aux_features_{'v1' if label == L_B else 'v2'}.csv", index=False, encoding="utf-8-sig")
        data[label] = data[label].merge(aux, on="user_id", how="left", validate="one_to_one")
    for column in AUX:
        data[L_A][column] = np.nan  # 기존 더미에는 이벤트가 없어 계산할 수 없다
    return data


def groups(frame):
    yield "전체", "전체", frame
    yield "레이블", "정상", frame[frame["is_anomaly"] == 0]
    yield "레이블", "이상", frame[frame["is_anomaly"] == 1]
    for name in SCENARIOS:
        yield "시나리오", name, frame[frame["scenario"] == name]


def stats_table(data):
    rows = []
    for label, frame in data.items():
        for kind, name, part in groups(frame):
            for feature in FEATURES + AUX:
                v = part[feature].dropna()
                rows.append({"dataset": label, "group_type": kind, "group": name, "feature": feature,
                             "unit": {**UNITS, **AUX_UNITS}[feature], "n": len(part), "n_null": int(part[feature].isna().sum()),
                             "mean": v.mean(), "median": v.median(), "std": v.std(ddof=1), "min": v.min(), "q25": v.quantile(0.25),
                             "q75": v.quantile(0.75), "max": v.max()})
    return pd.DataFrame(rows)


def change_table(data):
    rows = []
    for first, second in ((L_A, L_B), (L_A, L_C), (L_B, L_C)):
        for (kind, name, a), (_, _, b) in zip(groups(data[first]), groups(data[second])):
            for feature in FEATURES + AUX:
                x, y = a[feature].dropna(), b[feature].dropna()
                if len(x) == 0 or len(y) == 0:
                    continue
                rows.append({"compared": f"{first} → {second}", "group_type": kind, "group": name, "feature": feature,
                             "n_first": len(x), "n_second": len(y), "median_first": x.median(), "median_second": y.median(),
                             "mean_first": x.mean(), "mean_second": y.mean(), "std_first": x.std(ddof=1), "std_second": y.std(ddof=1),
                             "wasserstein": sps.wasserstein_distance(x, y), "ks_statistic": sps.ks_2samp(x, y).statistic})
    return pd.DataFrame(rows)


# ---------------- 기존 평가 기준 ----------------
def criteria(frame, normal_reference):
    """계정마다 기준 충족 여부(True/False/결측)를 돌려준다. normal_reference 는 '정상 분포 하위 n%' 를 정할 정상 계정들."""
    out = {}
    for limit in (3, 5, 10):
        out[f"avg_booking_interval < {limit}초 (후보)"] = frame["avg_booking_interval"] < limit
    for limit in (0.5, 1.0, 2.0):
        out[f"booking_interval_variance < {limit}초² (후보)"] = frame["booking_interval_variance"] < limit
    # Credit 규칙: 분산 1.0초² 미만, 요청 5회 이상일 때만 판단 (BTS 에서는 시도 5건 이상으로 옮겨 적용)
    out["[Credit] 분산 < 1.0초² · 시도 5건 이상"] = (frame["booking_interval_variance"] < 1.0).where(frame["booking_count"] >= 5)
    for limit in (0.5, 1.0, 3.0):
        out[f"seat_selection_latency < {limit}초 (후보)"] = frame["seat_selection_latency"] < limit
    out["[Credit] 좌석 선택 < 1.0초"] = frame["seat_selection_latency"] < 1.0
    for limit in (0.3, 0.5, 0.7):
        out[f"cancel_rate ≥ {int(limit * 100)}% (후보)"] = frame["cancel_rate"] >= limit
    out["[Credit] card_reuse_count ≥ 4"] = frame["card_reuse_count"] >= 4
    out["[Credit 약한 감점] card_reuse_count ≥ 2"] = frame["card_reuse_count"] >= 2
    out["[Credit] ip_booking_density ≥ 5"] = frame["ip_booking_density"] >= 5
    if frame["recent_booking_count_1m"].notna().any():
        for limit in (5, 10, 20):
            out[f"recent_booking_count 1분 ≥ {limit}회 (후보)"] = frame["recent_booking_count_1m"] >= limit
        for column, name in (("step_stay_seat_select", "좌석"), ("step_stay_payment", "결제")):
            for pct in (5, 10):
                threshold = normal_reference[column].quantile(pct / 100)
                out[f"step_stay_time {name} < 정상 하위 {pct}% (후보)"] = frame[column] < threshold
    table = pd.DataFrame(out)
    sources = {"avg_booking_interval": "avg_booking_interval", "booking_interval_variance": "booking_interval_variance", "분산": "booking_interval_variance",
               "seat_selection_latency": "seat_selection_latency", "좌석 선택": "seat_selection_latency", "cancel_rate": "cancel_rate",
               "좌석 <": "step_stay_seat_select", "결제 <": "step_stay_payment"}
    for column in table.columns:  # 값이 없으면 판단 불가(결측)로 둔다
        for key, source in sources.items():
            if key in column:
                table[column] = table[column].where(frame[source].notna())
    return table


def criteria_rates(data, real):
    rows = []
    for label, frame in data.items():
        flags = criteria(frame, frame[frame["is_anomaly"] == 0])
        for kind, name, part in groups(frame):
            for column in flags.columns:
                v = flags.loc[part.index, column].dropna()
                rows.append({"dataset": label, "group_type": kind, "group": name, "criterion": column, "evaluable": len(v),
                             "not_evaluable": len(part) - len(v), "met": int(v.sum()), "met_rate": v.mean() if len(v) else np.nan})
    if real is not None:  # 실측 계정: 값이 있는 기준만
        flags = criteria(real.assign(recent_booking_count_1m=np.nan), real)
        for name, part in real.groupby("group", sort=False):
            for column in flags.columns:
                v = flags.loc[part.index, column].dropna()
                if len(v):
                    rows.append({"dataset": "실측", "group_type": "실측 계정", "group": name, "criterion": column, "evaluable": len(v),
                                 "not_evaluable": len(part) - len(v), "met": int(v.sum()), "met_rate": v.mean()})
    return pd.DataFrame(rows)


def real_accounts():
    """실측 계정의 Feature. 매크로는 #16 참조 파일(6계정), 사람은 #14 의 실측(2계정, 파일이 있을 때만).
    카드·IP 는 실측 환경에서 왜곡돼 있고(모든 계정이 같은 카드·IP) 취소는 하지 않았으므로 비워 둔다."""
    reference = pd.read_csv(REFERENCE_PATH)
    rows = []
    for (mode, account), part in reference.groupby(["mode", "account"]):
        gaps = part["gap_to_next_ms"].dropna().to_numpy(dtype=float) / 1000
        rows.append({"group": f"실측 {mode}", "account": account, "booking_count": len(part), "avg_booking_interval": gaps.mean(),
                     "booking_interval_variance": gaps.var(ddof=0), "seat_selection_latency": part["seat_ms"].mean() / 1000,
                     "step_stay_seat_select": part["seat_ms"].median() / 1000,
                     "step_stay_payment": (part["hold_to_booking_ms"] - part["hold_to_payment_ms"]).median() / 1000})
    human_path = ISSUE14 / "data" / "count" / "events.csv"
    if human_path.exists():
        events = load_events(human_path)
        events = events[events["group"] == "human"]
        stage = events[events["event_type"] == "BOOKING_ATTEMPT"].assign(a=lambda d: d["meta"].map(lambda m: m["attemptId"]),
                                                                        s=lambda d: d["meta"].map(lambda m: m["stage"]))
        wide = stage.pivot_table(index=["external_user_id", "a"], columns="s", values="occurred_at", aggfunc="min").reset_index()
        for account, part in wide.groupby("external_user_id"):
            starts = part["STARTED"].sort_values()
            gaps = starts.diff().dt.total_seconds().dropna().to_numpy()
            seat = (part["SEAT_HOLD"] - part["SEAT_SELECTION"]).dt.total_seconds()
            rows.append({"group": "실측 사람 (#14)", "account": account, "booking_count": len(part), "avg_booking_interval": gaps.mean(),
                         "booking_interval_variance": gaps.var(ddof=0), "seat_selection_latency": seat.mean(),
                         "step_stay_seat_select": seat.median(), "step_stay_payment": np.nan})
    real = pd.DataFrame(rows)
    for column in ["cancel_rate", "card_reuse_count", "ip_booking_density"]:
        real[column] = np.nan
    return real


# ---------------- 실측 Fast/Jitter 와의 비교, Leave-One-Account-Out ----------------
def macro_vs_measured(data, real):
    extra = {"#16 v2": pd.read_csv(V2_PATH)}
    rows = []
    for mode in ("fast", "jitter"):
        measured = real[real["group"] == f"실측 {mode}"]
        for feature in ("avg_booking_interval", "booking_interval_variance", "seat_selection_latency"):
            r = measured[feature].to_numpy()
            rows.append({"mode": mode, "feature": feature, "data": "실측 계정", "n": len(r), "mean": r.mean(), "median": np.median(r),
                         "std": r.std(ddof=1), "min": r.min(), "max": r.max()})
            for label, frame in {L_A: data[L_A], **extra, L_B: data[L_B], L_C: data[L_C]}.items():
                if label == "#16 v2" and feature == "seat_selection_latency":
                    continue
                v = frame.loc[frame["scenario"] == mode, feature].dropna().to_numpy()
                rows.append({"mode": mode, "feature": feature, "data": label, "n": len(v), "mean": v.mean(), "median": np.median(v),
                             "std": v.std(ddof=1), "min": v.min(), "max": v.max(), "mean_minus_real": v.mean() - r.mean(),
                             "std_ratio_to_real": v.std(ddof=1) / r.std(ddof=1), "wasserstein_to_real": sps.wasserstein_distance(v, r),
                             "real_accounts_inside_range": f"{int(((r >= v.min()) & (r <= v.max())).sum())} / {len(r)}",
                             "distinct_clusters_note": "실측 계정 값 주변 3개 덩어리" if label in ("#16 v2", L_B) else "",
                             "independent_of_real": label == L_A})
    return pd.DataFrame(rows)


def leave_one_account_out(real):
    """실측 계정 하나를 숨기고 나머지로 같은 방식·같은 seed 로 매크로 계정을 만든 뒤, 숨긴 계정이 그 분포 안에 드는지 본다."""
    reference = pd.read_csv(REFERENCE_PATH)
    assignment = pd.read_csv(V2_PATH)[["user_id", "scenario", "is_anomaly"]]
    configs = {"v1 방식 (실측 계정 값 그대로)": load_config(), "v2 방식 (계정 수준 효과)": load_config(EXP / "config" / "scenarios_v2.json")}
    rows = []
    for mode in ("fast", "jitter"):
        subset = assignment[assignment["scenario"] == mode].reset_index(drop=True)
        for hidden in real.loc[real["group"] == f"실측 {mode}"].itertuples():
            for method, config in configs.items():
                events, accounts, _ = generate(config, "single_session", config["seed"], subset, reference[reference["account"] != hidden.account])
                dataset, _, _ = build_dataset(events, accounts, config)
                row = {"mode": mode, "hidden_account": hidden.account, "method": method, "synthetic_accounts": len(dataset)}
                for feature, short in (("avg_booking_interval", "mean"), ("booking_interval_variance", "variance")):
                    v, value = dataset[feature].to_numpy(), getattr(hidden, feature)
                    row.update({f"hidden_{short}": value, f"synthetic_{short}_min": v.min(), f"synthetic_{short}_q05": np.quantile(v, 0.05),
                                f"synthetic_{short}_q95": np.quantile(v, 0.95), f"synthetic_{short}_max": v.max(),
                                f"{short}_inside_range": bool(v.min() <= value <= v.max()),
                                f"{short}_inside_90pct": bool(np.quantile(v, 0.05) <= value <= np.quantile(v, 0.95)),
                                f"{short}_distance_outside_range": max(v.min() - value, value - v.max(), 0.0),
                                f"{short}_wasserstein_to_hidden": float(np.abs(v - value).mean())})
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------- 그래프 ----------------
def fig_feature(data, feature, path, labels):
    fig, ax = plt.subplots(figsize=(12.5, 4.4), constrained_layout=True)
    style(ax)
    width = 0.8 / len(labels)
    for k, label in enumerate(labels):
        offset = (k - (len(labels) - 1) / 2) * width
        values = [data[label].loc[data[label]["scenario"] == s, feature].dropna().to_numpy() for s in SCENARIOS]
        if feature in LOG:
            values = [v[v > 0] for v in values]
        box = ax.boxplot(values, positions=np.arange(len(SCENARIOS)) + offset, widths=width - 0.04, patch_artist=True, whis=(5, 95),
                         showfliers=False, medianprops={"color": INK, "linewidth": 1.4})
        for patch in box["boxes"]:
            patch.set(facecolor=COLOR[label], edgecolor=SURFACE, linewidth=1)
        for part in box["whiskers"] + box["caps"]:
            part.set(color=COLOR[label], linewidth=1.2)
        ax.plot([], [], color=COLOR[label], linewidth=7, label=label)
    if feature in LOG:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(PLAIN)
    ax.set_xticks(range(len(SCENARIOS)))
    counts = data[L_C]["scenario"].value_counts()
    ax.set_xticklabels([f"{s}\n(n={counts[s]})" for s in SCENARIOS], fontsize=9)
    unit = {**UNITS, **AUX_UNITS}[feature]
    ax.set_ylabel(f"{feature} ({unit})" + (", 로그 축" if feature in LOG else ""), fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1, 1), ncol=len(labels), borderaxespad=0.2)
    ax.set_title(f"{feature}: 시나리오별 분포 (상자 25~75%, 수염 5~95%, 가로선 중앙값)", loc="left", fontsize=10, color=INK)
    save(fig, path)


def fig_correlation(corr, path):
    short = ["간격 평균", "간격 분산", "취소율", "카드", "IP", "시도 수", "좌석 시간"]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6), constrained_layout=True)
    for row, (group, title) in enumerate((("전체", "전체 1,000계정"), ("hold_cancel", "hold_cancel 시나리오 안"))):
        for col, label in enumerate((L_A, L_B, L_C)):
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
                    ax.text(j, i, "" if i == j else ("–" if np.isnan(matrix[i, j]) else f"{matrix[i, j]:.2f}"), ha="center", va="center",
                            fontsize=8, color=INK)
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


def fig_macro(data, real, path):
    rng = np.random.default_rng(0)  # 점 겹침을 푸는 용도
    sets = {L_A: data[L_A], "#16 v2": pd.read_csv(V2_PATH), L_B: data[L_B], L_C: data[L_C]}
    colors = {**COLOR, "#16 v2": "#6250d6"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for ax, feature in zip(axes, ("avg_booking_interval", "booking_interval_variance")):
        style(ax)
        ticks = []
        for m, mode in enumerate(("fast", "jitter")):
            for k, (label, frame) in enumerate(sets.items()):
                position = m * 5 + k
                values = frame.loc[frame["scenario"] == mode, feature].dropna()
                ax.scatter(position + rng.uniform(-0.27, 0.27, len(values)), values, s=13, color=colors[label], edgecolor=SURFACE, linewidth=0.4)
                ticks.append((position, f"{label.split(' (')[0].replace('이벤트 기반 ', '')}\n{mode}"))
            measured = real.loc[real["group"] == f"실측 {mode}", feature]
            ax.hlines(measured, m * 5 - 0.45, m * 5 + 3.45, color=INK, linewidth=1.1, linestyle="--")
        ax.plot([], [], color=INK, linewidth=1.1, linestyle="--", label="실측 계정 (유형당 3개)")
        ax.set_xticks([p for p, _ in ticks])
        ax.set_xticklabels([t for _, t in ticks], fontsize=8)
        ax.set_ylabel(f"{feature} ({UNITS[feature]})", fontsize=9, color=MUTED)
        ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1, 1), borderaxespad=0.2)
        ax.set_title(feature, loc="left", fontsize=10, color=INK)
    fig.suptitle("Fast/Jitter 매크로의 예매 간격 Feature 와 #16 실측 계정 (점 = 합성 계정 1개)", fontsize=11, color=INK)
    save(fig, path)


def fig_criteria(rates, path):
    shown = ["[Credit] 좌석 선택 < 1.0초", "[Credit] 분산 < 1.0초² · 시도 5건 이상", "[Credit] card_reuse_count ≥ 4", "[Credit] ip_booking_density ≥ 5",
             "avg_booking_interval < 5초 (후보)", "cancel_rate ≥ 50% (후보)"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4), constrained_layout=True, sharey=True)
    x = np.arange(len(SCENARIOS))
    for ax, criterion in zip(axes.ravel(), shown):
        style(ax)
        for k, label in enumerate((L_A, L_B, L_C)):
            part = rates[(rates["dataset"] == label) & (rates["criterion"] == criterion) & (rates["group_type"] == "시나리오")].set_index("group")
            ax.bar(x + (k - 1) * 0.27, part.reindex(SCENARIOS)["met_rate"], width=0.24, color=COLOR[label], label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in SCENARIOS], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(criterion, loc="left", fontsize=10, color=INK)
    axes[0][0].legend(frameon=False, fontsize=9, loc="upper left")
    axes[0][0].set_ylabel("기준 충족 비율", fontsize=9, color=MUTED)
    axes[1][0].set_ylabel("기준 충족 비율", fontsize=9, color=MUTED)
    fig.suptitle("기존 평가 기준 충족 비율 (시나리오별, 판단 가능한 계정 기준). 앞의 4개 시나리오가 정상, 뒤의 4개가 이상", fontsize=11, color=INK)
    save(fig, path)


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    data = load()
    real = real_accounts()
    real.to_csv(OUT / "real_accounts.csv", index=False, encoding="utf-8-sig")

    stats = stats_table(data)
    stats.to_csv(OUT / "feature_stats.csv", index=False, encoding="utf-8-sig")
    change = change_table(data)
    change.to_csv(OUT / "change_b_to_c.csv", index=False, encoding="utf-8-sig")
    corr = correlations(data)
    corr.to_csv(OUT / "correlations.csv", index=False, encoding="utf-8-sig")
    within = corr[corr["group_type"] == "시나리오"].pivot_table(index=["group", "feature_a", "feature_b"], columns="dataset",
                                                             values="spearman", dropna=False).reset_index()
    within.to_csv(OUT / "within_scenario_correlation.csv", index=False, encoding="utf-8-sig")
    checks = consistency_checks(data)
    checks.to_csv(OUT / "consistency_checks.csv", index=False, encoding="utf-8-sig")
    rates = criteria_rates(data, real)
    rates.to_csv(OUT / "criteria_rates.csv", index=False, encoding="utf-8-sig")
    macro = macro_vs_measured(data, real)
    macro.to_csv(OUT / "macro_interval_vs_measured.csv", index=False, encoding="utf-8-sig")
    loo = leave_one_account_out(real)
    loo.to_csv(OUT / "leave_one_account_out.csv", index=False, encoding="utf-8-sig")

    for feature in FEATURES:
        fig_feature(data, feature, OUT / "figures" / f"feature_{feature}.png", [L_A, L_B, L_C])
    for feature in AUX:
        fig_feature(data, feature, OUT / "figures" / f"aux_{feature}.png", [L_B, L_C])
    fig_correlation(corr, OUT / "figures" / "correlation_heatmaps.png")
    fig_macro(data, real, OUT / "figures" / "macro_interval_vs_measured.png")
    fig_criteria(rates, OUT / "figures" / "criteria_rates.png")

    pd.set_option("display.width", 280)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 70)
    label_stats = stats[stats["group_type"] == "레이블"].pivot_table(index=["feature", "group"], columns="dataset", values=["mean", "median", "std"])
    print(label_stats.round(3).to_string())
    print(stats[stats["group_type"] == "시나리오"].pivot_table(index=["feature", "group"], columns="dataset", values="median").round(3).to_string())
    bc = change[(change["compared"] == f"{L_B} → {L_C}") & (change["group_type"] == "시나리오")]
    print(bc[bc["ks_statistic"] >= 0.15][["group", "feature", "median_first", "median_second", "std_first", "std_second", "wasserstein", "ks_statistic"]].round(3).to_string(index=False))
    print(checks.drop(columns=["reason", "violating_by_scenario"]).to_string(index=False))
    label_rates = rates[rates["group_type"].isin(["레이블", "실측 계정"])].pivot_table(index="criterion", columns=["dataset", "group"], values="met_rate", sort=False)
    print(label_rates.round(3).to_string())
    print(rates[(rates["group_type"] == "시나리오") & (rates["dataset"] == L_C)].pivot_table(index="criterion", columns="group", values="met_rate", sort=False)[SCENARIOS].round(2).to_string())
    print(rates[(rates["group_type"] == "시나리오") & (rates["dataset"] == L_B)].pivot_table(index="criterion", columns="group", values="met_rate", sort=False)[SCENARIOS].round(2).to_string())
    print(macro.drop(columns=["distinct_clusters_note"]).round(3).to_string(index=False))
    print(loo[["mode", "hidden_account", "method", "hidden_mean", "synthetic_mean_min", "synthetic_mean_max", "mean_inside_range", "mean_inside_90pct",
               "mean_distance_outside_range", "hidden_variance", "synthetic_variance_min", "synthetic_variance_max", "variance_inside_range"]].round(3).to_string(index=False))
    print(loo.groupby("method")[["mean_inside_range", "mean_inside_90pct", "variance_inside_range", "variance_inside_90pct"]].sum().to_string())
    key = [("avg_booking_interval", "booking_interval_variance"), ("avg_booking_interval", "seat_selection_latency"),
           ("cancel_rate", "booking_count"), ("card_reuse_count", "ip_booking_density"), ("booking_interval_variance", "booking_count")]
    print(within[[(a, b) in key for a, b in zip(within["feature_a"], within["feature_b"])]].round(2).to_string(index=False))
    print(real.round(3).to_string(index=False))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
