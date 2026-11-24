"""합성 데이터(v3)의 예매 간격 생성 규칙을 실측 이벤트(#14, #16)와 비교한다. 읽기 전용 분석이다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\analyze_interval_validation.py

- 데이터셋을 만들지 않고, 모델을 학습하지 않고, 기존 파일을 고치지 않는다. 실측 원본과 data/v3, data/v3_session 을 읽기만 한다.
- 실측 원본(#14 data/, #16 runs/)은 git 에 올리지 않는 파일이다. 없는 PC 에서는 이 스크립트를 돌릴 수 없고, 저장된 결과(output/interval_validation/)를 본다.

간격의 정의 (모든 표에서 같다)
  시도의 시작 시각 = attemptId 의 가장 이른 BOOKING_ATTEMPT 이벤트의 occurredAt (BTS FeatureAggregationService 와 같다)
  간격 = 같은 계정의 연속한 두 시도의 시작 시각 차이
  세션 정의 두 가지를 같이 계산한다
    concert+30분 : 앞 시도와 공연이 같고 간격이 30분 이하 (REPORT_V3_SESSION.md 에서 쓴 정의)
    30분만       : 간격이 30분 이하 (공연은 보지 않는다)

출력: output/interval_validation/
"""
import json
import sys
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
sys.path.insert(0, str(EXP.parent / "issue16-booking-interval" / "scripts"))
from compare_if_models import GRID, INK, MUTED, save, style  # noqa: E402
from generate_events import load_config, sha256  # noqa: E402

OUT = EXP / "output" / "interval_validation"
PRIVATE = EXP / "data" / "interval_validation"  # 실측 계정·시도 단위의 표 (data/ 는 git 제외)
ISSUE14, ISSUE16 = EXP.parent / "issue14-real-validation", EXP.parent / "issue16-booking-interval"
RELAUNCH, CONTINUOUS = "시도마다 프로세스 재실행", "연속 실행"
SOURCES = [  # (이름, 파일, 매크로의 실행 방식)
    ("#14 count", ISSUE14 / "data" / "count" / "events.csv", RELAUNCH),
    ("#16 20261122-main", ISSUE16 / "runs" / "20261122-main" / "raw_events.csv", RELAUNCH),
    ("#16 20261122-continuous", ISSUE16 / "runs" / "20261122-continuous" / "raw_events.csv", CONTINUOUS),
    ("#16 20261123-remeasure", ISSUE16 / "runs" / "20261123-remeasure" / "raw_events.csv", CONTINUOUS),
]
KIND = {"human": "사람", "macro_fast": "매크로 fast", "macro_jitter": "매크로 jitter", "macro_human": "매크로 human 프로필", "other": "기타 (시험·수동 계정)"}
SESSION_GAP_S = 1800
SCENARIOS = ["normal", "fast_normal", "shared_ip", "frequent_cancel", "fast", "jitter", "multi_account", "hold_cancel"]
NORMAL = SCENARIOS[:4]
C_REAL, C_NORMAL, C_ANOM = "#1f2328", "#2a78d6", "#eb6834"
Q = [0.05, 0.25, 0.5, 0.75, 0.95]


def describe(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"n": 0}
    q = np.quantile(v, Q)
    return {"n": len(v), "mean": v.mean(), "median": q[2], "std": v.std(ddof=1) if len(v) > 1 else np.nan, "min": v.min(), "p05": q[0], "p25": q[1],
            "p75": q[3], "p95": q[4], "max": v.max()}


def read_real(name, path, mode):
    """실측 이벤트 → 시도 1행 (시작, 공연, 단계 시각, 결과)과 원본 이벤트."""
    events = pd.read_csv(path, dtype=str, keep_default_na=False)
    meta = events["metadata"].map(json.loads)
    events["t"] = pd.to_datetime(events["occurred_at_utc"])
    events["attempt"] = meta.map(lambda m: m.get("attemptId"))
    events["concert"] = meta.map(lambda m: m.get("concertId"))
    events["stage"] = meta.map(lambda m: m.get("stage"))
    events["status"] = meta.map(lambda m: m.get("status"))
    events["recorded"] = pd.to_datetime(meta.map(lambda m: m.get("recordedAt")), utc=True).dt.tz_localize(None)
    events["client"] = pd.to_datetime(meta.map(lambda m: m.get("clientOccurredAt")), utc=True).dt.tz_localize(None)
    events.loc[events["event_type"] == "BOOKING", "stage"] = "BOOKING"
    keyed = events[events["attempt"].notna()]
    wide = keyed.pivot_table(index=["group", "external_user_id", "attempt"], columns="stage", values="t", aggfunc="min").reset_index()
    start = keyed[keyed["event_type"] == "BOOKING_ATTEMPT"].groupby("attempt").agg(start=("t", "min"), concert=("concert", "first"))
    wide = wide.merge(start, left_on="attempt", right_index=True)
    for column in ("SEAT_SELECTION", "SEAT_HOLD", "BOOKING", "COMPLETED"):
        if column not in wide:
            wide[column] = pd.NaT
    wide = wide.sort_values(["external_user_id", "start"]).reset_index(drop=True)
    by = wide.groupby("external_user_id")
    wide["gap"] = by["start"].diff().dt.total_seconds()
    wide["same_concert"] = wide["concert"] == by["concert"].shift()
    wide["in_session_gap_only"] = wide["gap"] <= SESSION_GAP_S
    wide["in_session_concert"] = wide["in_session_gap_only"] & wide["same_concert"]
    wide["completed"] = wide["COMPLETED"].notna()
    wide["start_to_booking"] = (wide["BOOKING"] - wide["start"]).dt.total_seconds()
    wide["seat"] = (wide["SEAT_HOLD"] - wide["SEAT_SELECTION"]).dt.total_seconds()
    wide["hold_to_booking"] = (wide["BOOKING"] - wide["SEAT_HOLD"]).dt.total_seconds()
    wide["previous_done_to_start"] = (wide["start"] - by["COMPLETED"].shift()).dt.total_seconds()
    wide.insert(0, "source", name)
    wide.insert(1, "macro_run_mode", mode)
    wide["kind"] = wide["group"].map(KIND)
    return wide, events


def synthetic_attempts():
    attempts = pd.read_csv(EXP / "data" / "v3" / "attempts.csv")
    accounts = pd.read_csv(EXP / "data" / "v3" / "accounts.csv", keep_default_na=False)
    table = attempts.merge(accounts[["user_id", "scenario", "is_anomaly", "tool_profile", "n_sessions"]], on="user_id")
    table["start"] = pd.to_datetime(table["start_utc"])
    table["booking"] = pd.to_datetime(table["booking_utc"])
    table = table.sort_values(["user_id", "start"]).reset_index(drop=True)
    by = table.groupby("user_id")
    table["gap"] = by["start"].diff().dt.total_seconds()
    table["in_session"] = (table["concert"] == by["concert"].shift()) & (table["gap"] <= SESSION_GAP_S)
    table["start_to_booking"] = (table["booking"] - table["start"]).dt.total_seconds()
    table["previous_booking_to_start"] = (table["start"] - by["booking"].shift()).dt.total_seconds().where(table["in_session"])
    return table, accounts


def main():
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 400)
    protected = [EXP / "data" / "v3" / "events.csv", EXP / "data" / "v3" / "fairticket_event_based_features.csv",
                 EXP / "data" / "v3_session" / "session_features_all.csv", EXP / "config" / "scenarios_v3.json"] + [path for _, path, _ in SOURCES]
    before = {str(path.relative_to(EXP.parent)): sha256(path) for path in protected}

    def csv(frame, name, folder=OUT):
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_csv(folder / name, index=False, encoding="utf-8-sig")

    packs = [read_real(*source) for source in SOURCES]
    real = pd.concat([pack[0] for pack in packs], ignore_index=True)
    events = pd.concat([pack[1].assign(source=name) for pack, (name, _, _) in zip(packs, SOURCES)], ignore_index=True)

    # ---- 1) 실측: 계정별 시도 수와 표본 수 ----
    rows = []
    for (source, mode, kind, account), part in real.groupby(["source", "macro_run_mode", "kind", "external_user_id"], sort=False):
        concert, gap_only = part.loc[part["in_session_concert"], "gap"], part.loc[part["in_session_gap_only"], "gap"]
        rows.append({"source": source, "kind": kind, "account": account, "macro_run_mode": mode if kind.startswith("매크로") else "",
                     "attempts": len(part), "completed": int(part["completed"].sum()), "concerts": part["concert"].nunique(),
                     "gaps_concert_30min": len(concert), "gaps_30min_only": len(gap_only), "gaps_across_concerts": int((part["in_session_gap_only"] & ~part["same_concert"]).sum()),
                     **{f"concert30_{k}": v for k, v in describe(concert).items() if k != "n"}, **{f"gap30_{k}": v for k, v in describe(gap_only).items() if k != "n"}})
    accounts_table = pd.DataFrame(rows)
    csv(accounts_table, "measured_accounts.csv", PRIVATE)
    csv(real[["source", "macro_run_mode", "kind", "external_user_id", "attempt", "concert", "start", "completed", "gap", "same_concert", "in_session_concert",
              "in_session_gap_only", "start_to_booking", "seat", "hold_to_booking", "previous_done_to_start"]], "measured_attempts.csv", PRIVATE)

    # ---- 2) 실측: 종류 × 실행 방식 × 세션 정의별 간격 분포 ----
    rows = []
    for (kind, mode), part in real.groupby(["kind", "macro_run_mode"], sort=False):
        label_mode = mode if kind.startswith("매크로") else "(해당 없음: 사람이 직접 / 시험 계정)"
        for definition, mask in (("concert+30분", part["in_session_concert"]), ("30분만", part["in_session_gap_only"])):
            rows.append({"kind": kind, "macro_run_mode": label_mode, "session_definition": definition, "accounts": part["external_user_id"].nunique(),
                         "attempts": len(part), **describe(part.loc[mask, "gap"])})
    measured = pd.DataFrame(rows)
    csv(measured, "measured_gap_summary.csv")

    # 실측: 시도 1건의 구성 (완료된 시도). 시작→예매, 좌석 선택, 선점→예매, 앞 시도 완료→다음 시작
    rows = []
    for (kind, mode), part in real[real["completed"]].groupby(["kind", "macro_run_mode"], sort=False):
        for column, title in (("start_to_booking", "시도 시작 → 예매 완료"), ("seat", "좌석 선택 진입 → 선점"), ("hold_to_booking", "선점 → 예매 완료"),
                              ("previous_done_to_start", "앞 시도 완료 → 다음 시도 시작 (30분 이하)")):
            values = part[column].where(part["in_session_gap_only"]) if column == "previous_done_to_start" else part[column]
            rows.append({"kind": kind, "macro_run_mode": mode if kind.startswith("매크로") else "", "quantity": title, **describe(values)})
    stage_times = pd.DataFrame(rows)
    csv(stage_times, "measured_stage_times.csv")

    # ---- 3) 이벤트 시각: BTS 가 쓰는 occurredAt 이 무엇을 재는가 ----
    stage_events = events[events["event_type"] == "BOOKING_ATTEMPT"]
    started = stage_events[stage_events["stage"] == "STARTED"]
    lag = (started["t"] - started["client"]).dt.total_seconds()
    timestamps = pd.DataFrame([
        {"check": "occurredAt − metadata.recordedAt (모든 BOOKING_ATTEMPT)", **describe((stage_events["t"] - stage_events["recorded"]).dt.total_seconds())},
        {"check": "STARTED 의 occurredAt − metadata.clientOccurredAt (예매하기 클릭 시각)", **describe(lag)},
        {"check": "위 값이 1초를 넘는 시도 수", "n": int((lag > 1).sum()), "max": lag.max()},
    ])
    csv(timestamps, "timestamp_check.csv")

    # ---- 4) v3 합성: 세션 안 간격 ----
    synthetic, synthetic_accounts = synthetic_attempts()
    inside = synthetic[synthetic["in_session"]]
    rows = []
    for name in SCENARIOS:
        part, info = inside[inside["scenario"] == name], synthetic_accounts[synthetic_accounts["scenario"] == name]
        means = part.groupby("user_id")["gap"].mean()
        row = {"scenario": name, "label": "정상" if name in NORMAL else "이상", "accounts": len(info), "accounts_with_session_gap": int(means.size),
               **{f"gap_{k}": v for k, v in describe(part["gap"]).items()}, **{f"account_mean_{k}": v for k, v in describe(means).items() if k != "n"}}
        for limit in (5, 7, 10, 15, 20):
            row[f"gaps_below_{limit}s"] = float((part["gap"] < limit).mean())
            row[f"account_means_below_{limit}s"] = float((means < limit).mean())
        rows.append(row)
    synthetic_summary = pd.DataFrame(rows)
    csv(synthetic_summary, "synthetic_gap_summary.csv")
    rows = []
    for name in SCENARIOS:
        part = synthetic[(synthetic["scenario"] == name) & (synthetic["outcome"] == "complete")]
        rows.append({"scenario": name, "quantity": "시도 시작 → 예매 완료", **describe(part["start_to_booking"])})
        rows.append({"scenario": name, "quantity": "좌석 선택 진입 → 선점", **describe(part["seat_ms"] / 1000)})
        rows.append({"scenario": name, "quantity": "앞 시도 예매 완료 → 다음 시도 시작", **describe(synthetic.loc[synthetic["scenario"] == name, "previous_booking_to_start"])})
    synthetic_stage = pd.DataFrame(rows)
    csv(synthetic_stage, "synthetic_stage_times.csv")
    by_profile = (inside[inside["tool_profile"] != ""].groupby(["scenario", "tool_profile"])["gap"].apply(lambda v: pd.Series(describe(v))).unstack().reset_index())
    csv(by_profile, "synthetic_gap_by_tool_profile.csv")

    # ---- 5) 생성 설정: 간격을 정하는 파라미터와 근거 ----
    config = load_config(EXP / "config" / "scenarios_v3.json")
    floors = config["stage_floor_s"]
    rows = []
    for name, spec in config["scenarios"].items():
        if "stages_median_s" in spec:
            stages = {k: v for k, v in spec["stages_median_s"].items() if k != "basis"}
            cycle = sum(stages[k] for k in ("queue", "pre_seat", "seat", "hold_to_payment", "payment", "completed"))
            rows.append({"scenario": name, "timing_source": spec["timing_source"], "used_in_v3": spec["timing_source"] != "tool",
                         "stage_medians_sum_s": cycle, "idle_median_s": stages["idle"], "interval_at_medians_s": cycle + stages["idle"],
                         "seat_median_s": stages["seat"], "payment_median_s": stages["payment"],
                         "account_pace_log_sd": spec["account_pace_sigma"]["value"], "pace_basis": spec["account_pace_sigma"]["basis"],
                         "within_attempt_log_sd": spec["within_sigma"]["value"], "within_basis": spec["within_sigma"]["basis"],
                         "floor_sum_s": sum(floors[k] for k in ("queue", "pre_seat", "seat", "hold_to_payment", "payment", "completed")),
                         "stage_basis": stages and spec["stages_median_s"]["basis"]})
        else:
            rows.append({"scenario": name, "timing_source": spec["timing_source"], "used_in_v3": True,
                         "stage_basis": "실측 시도를 통째로 복원추출 (reference/macro_attempt_cycles.csv), 계정 수준은 실측 계정 평균 6개에서 추정"})
    parameters = pd.DataFrame(rows)
    csv(parameters, "generation_parameters.csv")

    # ---- 6) 실측 값이 합성 분포의 어디에 있는가 ----
    normal_gaps = inside.loc[inside["is_anomaly"] == 0, "gap"].to_numpy()
    normal_means = inside[inside["is_anomaly"] == 0].groupby("user_id")["gap"].mean().to_numpy()
    anomaly_means = inside[inside["is_anomaly"] == 1].groupby("user_id")["gap"].mean().to_numpy()
    rows = []
    for row in accounts_table.itertuples():
        value = row.gap30_mean
        if pd.isna(value):
            continue
        rows.append({"source": row.source, "kind": row.kind, "account": row.account, "macro_run_mode": row.macro_run_mode, "gaps_30min_only": row.gaps_30min_only,
                     "gaps_concert_30min": row.gaps_concert_30min, "mean_gap_30min_only": value,
                     "share_of_v3_normal_account_means_below": float((normal_means < value).mean()),
                     "share_of_v3_normal_gaps_below": float((normal_gaps < value).mean()),
                     "share_of_v3_anomaly_account_means_above": float((anomaly_means > value).mean())})
    position = pd.DataFrame(rows)
    csv(position, "measured_position_in_v3.csv", PRIVATE)

    # ---- 7) 30일 분산이 실제로 나타내는 것 ----
    features = pd.read_csv(EXP / "data" / "v3_session" / "session_features_all.csv")
    multi = (features["n_sessions"] >= 2).astype(int)
    old, new = features["booking_interval_variance__existing_30d"], features["booking_interval_variance__pooled"]
    has = old.notna()
    variance = pd.DataFrame([
        {"quantity": "30일 분산으로 '오픈 2회 이상 참여'를 맞히는 AUC (값이 있는 계정)", "value": roc_auc_score(multi[has], old[has]), "n": int(has.sum())},
        {"quantity": "세션 분산으로 '오픈 2회 이상 참여'를 맞히는 AUC", "value": roc_auc_score(multi[new.notna()], new[new.notna()]), "n": int(new.notna().sum())},
        {"quantity": "30일 분산과 세션 수의 Spearman 상관", "value": old.corr(features["n_sessions"], method="spearman"), "n": int(has.sum())},
        {"quantity": "30일 분산으로 정상/이상을 나누는 AUC: 오픈 1회 계정", "value": max(a := roc_auc_score(features.loc[has & (multi == 0), "is_anomaly"], old[has & (multi == 0)]), 1 - a),
         "n": int((has & (multi == 0)).sum())},
        {"quantity": "30일 분산으로 정상/이상을 나누는 AUC: 오픈 2회 이상 계정", "value": max(a := roc_auc_score(features.loc[has & (multi == 1), "is_anomaly"], old[has & (multi == 1)]), 1 - a),
         "n": int((has & (multi == 1)).sum())},
        {"quantity": "30일 분산의 중앙값: 오픈 1회 계정", "value": old[has & (multi == 0)].median(), "n": int((has & (multi == 0)).sum())},
        {"quantity": "30일 분산의 중앙값: 오픈 2회 이상 계정", "value": old[has & (multi == 1)].median(), "n": int((has & (multi == 1)).sum())},
        {"quantity": "30일 분산은 있지만 세션 분산은 결측인 계정 (오픈마다 1건씩 시도)", "value": int((has & new.isna()).sum()), "n": len(features)},
        {"quantity": "  그중 이상 계정", "value": int((has & new.isna() & (features["is_anomaly"] == 1)).sum()), "n": int((has & new.isna()).sum())},
    ])
    csv(variance, "variance_30d_meaning.csv")

    # ---------------- 그래프 ----------------
    rng = np.random.default_rng(0)
    groups = []
    person = real[real["kind"] == "사람"]
    groups.append(("실측 사람 2계정\n(다른 공연 사이, 4개)", person.loc[person["in_session_gap_only"], "gap"], C_REAL))
    for kind, short in (("매크로 fast", "fast"), ("매크로 jitter", "jitter"), ("매크로 human 프로필", "human 프로필")):
        for mode, tag in ((CONTINUOUS, "연속"), (RELAUNCH, "재실행")):
            part = real[(real["kind"] == kind) & (real["macro_run_mode"] == mode)]
            groups.append((f"실측 매크로 {short}\n({tag}, {int(part['in_session_concert'].sum())}개)", part.loc[part["in_session_concert"], "gap"], C_REAL))
    for name in SCENARIOS:
        part = inside[inside["scenario"] == name]
        groups.append((f"v3 {name}\n({len(part):,}개)", part["gap"], C_NORMAL if name in NORMAL else C_ANOM))
    fig, ax = plt.subplots(figsize=(12, 8.6), constrained_layout=True)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for position_, (label, values, color) in enumerate(groups[::-1]):
        values = values.dropna().to_numpy()
        if len(values) > 400:
            values = rng.choice(values, 400, replace=False)
        ax.scatter(values, position_ + rng.uniform(-0.28, 0.28, len(values)), s=9, alpha=0.5, linewidth=0, color=color)
        if len(values):
            ax.vlines(np.median(values), position_ - 0.4, position_ + 0.4, color=INK, linewidth=1.5)
    ax.axhline(len(SCENARIOS) - 0.5, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups[::-1]], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1500)
    ax.set_xlabel("연속한 두 시도의 시작 간격 (초, 로그 눈금)", fontsize=9, color=MUTED)
    ax.set_title("실측(위)과 v3 합성(아래)의 세션 안 시도 간격. 세로선 = 중앙값. 사람 실측은 같은 공연 안의 간격이 없어 다른 공연 사이의 간격이다",
                 loc="left", fontsize=10, color=INK)
    save(fig, OUT / "figures" / "gaps_measured_vs_v3.png")

    panels = [("start_to_booking", "시도 시작 → 예매 완료 (초)"), ("idle", "앞 시도 완료 → 다음 시도 시작 (초)")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for ax, (key, title) in zip(axes, panels):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        rows_ = []
        done = real[real["completed"]]
        for account in ("human-01", "human-02"):
            part = done[(done["external_user_id"] == account) & (done["source"] == "#14 count")]
            rows_.append((f"실측 사람 {account}", part["start_to_booking"] if key == "start_to_booking" else part["previous_done_to_start"], C_REAL))
        part = done[(done["kind"] == "매크로 human 프로필") & (done["macro_run_mode"] == CONTINUOUS)]
        rows_.append(("실측 매크로 human 프로필 (연속)", part["start_to_booking"] if key == "start_to_booking" else part["previous_done_to_start"], C_REAL))
        for name in ("normal", "fast_normal", "shared_ip", "frequent_cancel"):
            part = synthetic[synthetic["scenario"] == name]
            rows_.append((f"v3 {name}", part.loc[part["outcome"] == "complete", "start_to_booking"] if key == "start_to_booking" else part["previous_booking_to_start"], C_NORMAL))
        for position_, (label, values, color) in enumerate(rows_[::-1]):
            values = values.dropna().to_numpy()
            if len(values) > 400:
                values = rng.choice(values, 400, replace=False)
            ax.scatter(values, position_ + rng.uniform(-0.28, 0.28, len(values)), s=9, alpha=0.5, linewidth=0, color=color)
            ax.vlines(np.median(values), position_ - 0.4, position_ + 0.4, color=INK, linewidth=1.5)
        ax.set_yticks(range(len(rows_)))
        ax.set_yticklabels([r[0] for r in rows_[::-1]] if ax is axes[0] else [], fontsize=8)
        ax.set_xscale("log")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.suptitle("시도 1건의 구성: 실측 사람 2계정(각 3건)과 v3 정상 시나리오. 세로선 = 중앙값", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, OUT / "figures" / "attempt_composition.png")

    record = {"script": "scripts/analyze_interval_validation.py", "read_only": True,
              "sources": [{"name": name, "path": str(path.relative_to(EXP.parent).as_posix()), "macro_run_mode": mode} for name, path, mode in SOURCES],
              "session_gap_seconds": SESSION_GAP_S, "inputs_sha256": before,
              "inputs_unchanged_after_run": before == {str(path.relative_to(EXP.parent)): sha256(path) for path in protected},
              "datasets_generated": False, "models_trained": False}
    (OUT / "analysis_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cols = ["n", "mean", "median", "std", "min", "p05", "p25", "p75", "p95", "max"]
    print("[1] 실측 계정\n", accounts_table[["source", "kind", "account", "macro_run_mode", "attempts", "completed", "concerts", "gaps_concert_30min", "gaps_30min_only",
                                         "gaps_across_concerts", "gap30_mean", "gap30_median", "gap30_min", "gap30_max"]].round(2).to_string(index=False))
    print("\n[2] 실측 간격 분포\n", measured.round(2).to_string(index=False))
    print(stage_times.round(2).to_string(index=False))
    print("\n[3] 이벤트 시각\n", timestamps.round(3).to_string(index=False))
    print("\n[4] v3 합성\n", synthetic_summary.round(2).T.to_string())
    print(synthetic_stage.round(2).to_string(index=False))
    print(by_profile.round(2).to_string(index=False))
    print("\n[5] 생성 파라미터\n", parameters.drop(columns="stage_basis").round(3).to_string(index=False))
    print("\n[6] 실측 값의 위치\n", position.round(3).to_string(index=False))
    print("\n[7] 30일 분산\n", variance.round(3).to_string(index=False))
    print(record["inputs_unchanged_after_run"], f"저장: {OUT}")


if __name__ == "__main__":
    main()
