"""v3 이벤트에서 예매 간격 Feature 2개를 '같은 세션 안의 시도 간격'으로 다시 계산한다. 나머지 4개 Feature 는 그대로 둔다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\extract_session_features.py

- data/v3 (이벤트, 기존 추출 결과)는 읽기만 한다. 결과는 data/v3_session/ 에 따로 저장한다.
- 기존 추출기(#14 extract_features.py = BTS FeatureAggregationService 재현)는 수정하지 않는다.

세션의 정의 (demo-ticketing 의 이벤트 구조에서 가져왔다)
  1. 시도의 시작 시각 = 그 attemptId 의 가장 이른 BOOKING_ATTEMPT 이벤트 시각 (BTS 와 같다)
  2. 시도의 공연 = 그 이벤트들의 metadata.concertId. BOOKING_ATTEMPT 의 필수 필드이고 attemptId 는 사용자·공연·서비스에 고정된다
     (demo-ticketing/docs/booking-attempts.md)
  3. 한 계정의 시도를 시작 시각 순으로 놓고, 바로 앞 시도와 (서비스, 공연)이 같고 시작 간격이 SESSION_GAP_MINUTES 이하이면 같은 세션이다.
     30분은 대기열에서 한 번 받은 순번이 유지되는 최대 시간이다 (demo.queue.wait-ttl = PT30M). 그보다 오래 비우면 대기열에 다시 들어가야 한다.
  공연이 바뀌면 간격이 짧아도 다른 세션이고, 같은 공연이라도 30분 넘게 비우면 다른 세션이다.

세션 안의 값
  간격 = 같은 세션에서 연속한 두 시도의 시작 시각 차이 (초). 세션 평균 = 그 간격들의 평균, 세션 분산 = 모분산 (BTS 와 같은 식. 간격이 1개면 0).
  시도가 1건인 세션에는 간격이 없다. 그 세션은 평균·분산 계산에서 빠진다 (0 이나 다른 값으로 채우지 않는다).

계정 수준 집계 (4가지를 모두 계산해 비교한다)
  pooled : 모든 세션의 간격을 한데 모아 평균과 모분산
  median : 세션 평균의 중앙값, 세션 분산의 중앙값
  min    : 세션 평균이 가장 작은 세션의 평균과 그 세션의 분산
  latest : 간격이 있는 세션 중 가장 최근 세션의 평균과 분산
  간격이 있는 세션이 하나도 없는 계정(모든 세션이 시도 1건)은 두 Feature 가 결측이다.

출력: data/v3_session/
  sessions.csv                         세션 1행 (계정, 공연, 시도 수, 시작·끝, 세션 평균·분산)
  session_features_all.csv             계정 1행, 네 집계의 값과 기존 값
  fairticket_event_based_features_<집계>.csv   기존 데이터셋과 같은 컬럼·순서·레이블. 간격 2개만 바뀐다
  validation.json                      검증 결과
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP.parent / "issue14-real-validation" / "scripts"))
from extract_features import load_events  # noqa: E402
from generate_events import load_config, sha256  # noqa: E402

SOURCE = EXP / "data" / "v3"
OUT = EXP / "data" / "v3_session"
DATASET_NAME = "fairticket_event_based_features.csv"
SESSION_GAP_MINUTES = 30.0  # demo.queue.wait-ttl
AGGREGATIONS = ["pooled", "median", "min", "latest"]
INTERVAL, VARIANCE = "avg_booking_interval", "booking_interval_variance"
DECIMALS = {INTERVAL: 2, VARIANCE: 4}  # 기존 데이터셋과 같은 자릿수
WINDOW_DAYS = 30


def attempt_table(events, calculated_at):
    """시도 1행: 계정, 서비스, attemptId, 공연, 시작 시각. 30일 창과 시작 시각의 정의는 BTS 와 같다."""
    window = events[(events["occurred_at"] >= calculated_at - pd.Timedelta(days=WINDOW_DAYS)) & (events["occurred_at"] <= calculated_at)]
    stage = window[window["event_type"].str.upper() == "BOOKING_ATTEMPT"].sort_values(["occurred_at", "event_id"])
    stage = stage.assign(attempt_id=stage["meta"].map(lambda m: m.get("attemptId")), concert=stage["meta"].map(lambda m: m.get("concertId")))
    stage = stage[stage["attempt_id"].notna() & (stage["service_id"] != "")]
    table = (stage.groupby(["user_id", "service_id", "attempt_id"], sort=False)
             .agg(start=("occurred_at", "min"), concert=("concert", "first"), concerts=("concert", "nunique")).reset_index())
    if (table["concerts"] != 1).any() or table["concert"].isna().any():
        sys.exit("concertId 가 없거나 한 시도 안에서 바뀌는 시도가 있습니다. 세션을 나눌 수 없습니다.")
    return table.drop(columns="concerts").sort_values(["user_id", "start", "attempt_id"]).reset_index(drop=True)


def assign_sessions(table, gap_minutes=SESSION_GAP_MINUTES, use_concert=True):
    previous = table.groupby("user_id")
    gap = previous["start"].diff().dt.total_seconds()
    new = gap.isna() | (gap > gap_minutes * 60)
    if use_concert:
        new |= (table["concert"] != previous["concert"].shift()) | (table["service_id"] != previous["service_id"].shift())
    out = table.assign(gap=gap.where(~new), session_no=new.groupby(table["user_id"]).cumsum())
    return out


def session_table(attempts):
    """세션 1행. 간격은 모분산(ddof=0), 음수는 0 으로 (BTS 와 같다)."""
    gaps = attempts["gap"].clip(lower=0)
    grouped = attempts.assign(gap=gaps).groupby(["user_id", "session_no"], sort=False)
    sessions = grouped.agg(service_id=("service_id", "first"), concert=("concert", "first"), n_attempts=("attempt_id", "size"),
                           start=("start", "min"), end=("start", "max"), session_mean=("gap", "mean"),
                           session_variance=("gap", lambda v: v.var(ddof=0) if v.notna().any() else np.nan)).reset_index()
    return sessions


def aggregate(sessions, attempts, users):
    """계정 수준의 값. 간격이 있는 세션만 쓴다."""
    valid = sessions[sessions["n_attempts"] >= 2].sort_values(["user_id", "start"])
    out = pd.DataFrame(index=pd.Index(users, name="user_id"))
    out["n_sessions"] = sessions.groupby("user_id").size().reindex(out.index).fillna(0).astype(int)
    out["n_sessions_with_interval"] = valid.groupby("user_id").size().reindex(out.index).fillna(0).astype(int)
    out["n_single_attempt_sessions"] = out["n_sessions"] - out["n_sessions_with_interval"]
    gaps = attempts.dropna(subset=["gap"]).assign(gap=lambda d: d["gap"].clip(lower=0)).groupby("user_id")["gap"]
    out[f"{INTERVAL}__pooled"] = gaps.mean().reindex(out.index)
    out[f"{VARIANCE}__pooled"] = gaps.agg(lambda v: v.var(ddof=0)).reindex(out.index)
    by_user = valid.groupby("user_id")
    out[f"{INTERVAL}__median"] = by_user["session_mean"].median().reindex(out.index)
    out[f"{VARIANCE}__median"] = by_user["session_variance"].median().reindex(out.index)
    fastest = valid.loc[by_user["session_mean"].idxmin()].set_index("user_id")
    out[f"{INTERVAL}__min"] = fastest["session_mean"].reindex(out.index)
    out[f"{VARIANCE}__min"] = fastest["session_variance"].reindex(out.index)
    latest = by_user.tail(1).set_index("user_id")
    out[f"{INTERVAL}__latest"] = latest["session_mean"].reindex(out.index)
    out[f"{VARIANCE}__latest"] = latest["session_variance"].reindex(out.index)
    # 가장 최근 세션이 시도 1건이라 그 앞의 세션을 쓴 계정
    last_any = sessions.sort_values(["user_id", "start"]).groupby("user_id").tail(1).set_index("user_id")
    out["latest_session_had_one_attempt"] = ((last_any["n_attempts"] < 2).reindex(out.index).fillna(False)) & (out["n_sessions_with_interval"] > 0)
    return out.reset_index()


def build(events, base_dataset, calculated_at):
    """이벤트와 기존 데이터셋 → (집계별 데이터셋, 계정별 전체 표, 세션 표, 시도 표)."""
    attempts = assign_sessions(attempt_table(events, calculated_at))
    sessions = session_table(attempts)
    features = aggregate(sessions, attempts, base_dataset["user_id"])
    datasets = {}
    for name in AGGREGATIONS:
        frame = base_dataset.copy()
        for column in (INTERVAL, VARIANCE):
            frame[column] = features[f"{column}__{name}"].astype(float).round(DECIMALS[column]).to_numpy()
        datasets[name] = frame
    return datasets, features, sessions, attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(EXP / "config" / "scenarios_v3.json")
    calculated_at = pd.Timestamp(config["calculated_at_utc"])
    protected = [SOURCE / "events.csv", SOURCE / DATASET_NAME, SOURCE / "features_full.csv"]
    before = {path.name: sha256(path) for path in protected}

    events = load_events(SOURCE / "events.csv")
    base = pd.read_csv(SOURCE / DATASET_NAME)
    datasets, features, sessions, attempts = build(events, base, calculated_at)

    # ---- 검증 ----
    checks = {}
    # (1) 세션이 하나뿐인 계정은 30일 전체 간격과 세션 간격이 같아야 한다 → 기존 추출 결과와 같은 값
    #     (반올림 전의 값으로 비교한다. 반올림한 값은 계산 순서에 따른 부동소수 오차로 마지막 자리가 달라질 수 있다)
    one = (features["n_sessions"] == 1).to_numpy()
    full = pd.read_csv(SOURCE / "features_full.csv").set_index("user_id").reindex(features["user_id"])
    for name in AGGREGATIONS:
        for column in (INTERVAL, VARIANCE):
            a, b = features.loc[one, f"{column}__{name}"].to_numpy(dtype=float), full[column].to_numpy(dtype=float)[one]
            checks[f"single_session_accounts_equal_existing__{name}__{column}"] = bool(np.isclose(a, b, rtol=1e-9, atol=1e-6, equal_nan=True).all())
    # (2) 생성기가 기록한 시도(공연, 시작 시각)에서 같은 규칙으로 다시 계산한 값과 같아야 한다
    truth = pd.read_csv(SOURCE / "attempts.csv").assign(start=lambda d: pd.to_datetime(d["start_utc"])).sort_values(["user_id", "start"])
    gap = truth.groupby("user_id")["start"].diff().dt.total_seconds()
    same = (truth["concert"] == truth.groupby("user_id")["concert"].shift()) & (gap <= SESSION_GAP_MINUTES * 60)
    again = truth.assign(gap=gap.where(same)).dropna(subset=["gap"]).groupby("user_id")["gap"].agg(mean="mean", var=lambda v: v.var(ddof=0))
    again = again.reindex(features["user_id"])
    checks["pooled_equals_recompute_from_generation_record_mean"] = bool(np.isclose(features[f"{INTERVAL}__pooled"], again["mean"], rtol=1e-9, atol=1e-6, equal_nan=True).all())
    checks["pooled_equals_recompute_from_generation_record_variance"] = bool(np.isclose(features[f"{VARIANCE}__pooled"], again["var"], rtol=1e-9, atol=1e-6, equal_nan=True).all())
    # (3) 시도 수와 세션의 시도 수 합이 같고, 기존 시도 수와 같다
    checks["attempts_sum_over_sessions_equals_booking_count"] = bool(
        (sessions.groupby("user_id")["n_attempts"].sum().reindex(base["user_id"]).to_numpy() == base["booking_count"].to_numpy()).all())
    # (4) 나머지 컬럼은 기존 데이터셋과 같다
    others = [c for c in base.columns if c not in (INTERVAL, VARIANCE)]
    checks["other_columns_identical_to_existing"] = all(frame[others].equals(base[others]) for frame in datasets.values())
    # (5) 결측 규칙: 간격이 있는 세션이 없으면 결측, 있으면 값이 있다. 시도 2건인 세션 하나뿐이면 분산 0
    none = (features["n_sessions_with_interval"] == 0).to_numpy()
    checks["null_iff_no_session_with_interval"] = all(bool(frame.loc[none, [INTERVAL, VARIANCE]].isna().all().all()
                                                           and frame.loc[~none, [INTERVAL, VARIANCE]].notna().all().all()) for frame in datasets.values())
    # (6) 세션 평균의 최솟값 ≤ 중앙값, 분산은 음수가 아니고 수학적 상한(m²·(n−1)) 이하
    checks["min_not_above_median"] = bool((features[f"{INTERVAL}__min"].fillna(0) <= features[f"{INTERVAL}__median"].fillna(0) + 1e-9).all())
    with_gap = sessions[sessions["n_attempts"] >= 2]
    bound = with_gap["session_mean"] ** 2 * (with_gap["n_attempts"] - 2) * (1 + 1e-6) + 1e-3
    checks["session_variance_within_mathematical_bound"] = bool(((with_gap["session_variance"] >= 0) & (with_gap["session_variance"] <= bound)).all())
    # 세션 경계를 달리 잡았을 때 (이 데이터에서는 공연마다 오픈이 일주일씩 떨어져 있어 같아야 한다)
    table = attempt_table(events, calculated_at)
    variants = {"concert_and_30min (채택)": assign_sessions(table), "concert_only": assign_sessions(table, gap_minutes=1e9),
                "gap_30min_only": assign_sessions(table, use_concert=False), "gap_10min_only": assign_sessions(table, gap_minutes=10, use_concert=False),
                "gap_60min_only": assign_sessions(table, gap_minutes=60, use_concert=False)}
    boundary = {name: {"sessions": int(frame.groupby("user_id")["session_no"].nunique().sum()),
                       "same_assignment_as_adopted": bool((frame["session_no"].to_numpy() == variants["concert_and_30min (채택)"]["session_no"].to_numpy()).all())}
                for name, frame in variants.items()}
    checks["all_passed"] = all(checks.values())

    sessions.to_csv(OUT / "sessions.csv", index=False, encoding="utf-8", lineterminator="\n")
    merged = base[["user_id", "scenario", "is_anomaly", "booking_count", INTERVAL, VARIANCE]].rename(
        columns={INTERVAL: f"{INTERVAL}__existing_30d", VARIANCE: f"{VARIANCE}__existing_30d"}).merge(features, on="user_id", validate="one_to_one")
    merged.to_csv(OUT / "session_features_all.csv", index=False, encoding="utf-8", lineterminator="\n")
    for name, frame in datasets.items():
        frame.to_csv(OUT / f"fairticket_event_based_features_{name}.csv", index=False, encoding="utf-8", lineterminator="\n")

    label = merged["is_anomaly"].map({0: "정상", 1: "이상"})
    record = {
        "source_events": (SOURCE / "events.csv").relative_to(EXP).as_posix(), "source_dataset": (SOURCE / DATASET_NAME).relative_to(EXP).as_posix(),
        "session_rule": {"same": ["service_id", "metadata.concertId"], "max_gap_minutes": SESSION_GAP_MINUTES,
                         "basis": "concertId 는 BOOKING_ATTEMPT 의 필수 필드 (demo-ticketing/docs/booking-attempts.md), 30분은 demo.queue.wait-ttl"},
        "attempt_start": "attemptId 의 가장 이른 BOOKING_ATTEMPT 이벤트 시각 (BTS FeatureAggregationService 와 같다)",
        "variance": "모분산 (BTS 와 같다). 간격 1개면 0", "single_attempt_session": "간격이 없으므로 집계에서 제외. 값을 채우지 않는다",
        "aggregations": AGGREGATIONS, "rounding": DECIMALS,
        "counts": {"accounts": int(len(merged)), "attempts": int(len(attempts)), "sessions": int(len(sessions)),
                   "sessions_with_one_attempt": int((sessions["n_attempts"] < 2).sum()),
                   "accounts_by_n_sessions": {int(k): int(v) for k, v in merged["n_sessions"].value_counts().sort_index().items()},
                   "accounts_null_existing": merged[f"{INTERVAL}__existing_30d"].isna().groupby(label).sum().astype(int).to_dict(),
                   "accounts_null_session_based": merged[f"{INTERVAL}__pooled"].isna().groupby(label).sum().astype(int).to_dict(),
                   "accounts_with_value_only_in_existing": int((merged[f"{INTERVAL}__existing_30d"].notna() & merged[f"{INTERVAL}__pooled"].isna()).sum()),
                   "accounts_latest_session_had_one_attempt": int(merged["latest_session_had_one_attempt"].sum())},
        "boundary_variants": boundary, "checks": checks,
        "files_sha256": {path.name: sha256(path) for path in sorted(OUT.glob("*.csv"))},
        "source_unchanged": before == {path.name: sha256(path) for path in protected},
    }
    (OUT / "validation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "files_sha256"}, ensure_ascii=False, indent=2))
    if not (checks["all_passed"] and record["source_unchanged"]):
        sys.exit("검증 실패")


if __name__ == "__main__":
    main()
