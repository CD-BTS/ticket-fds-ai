"""v4 이벤트를 기존 BTS Feature 추출기(#14 extract_features.py, 수정 없음)에 넣어 데이터셋을 만들고 검증한다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\extract_v4.py

입력: data/v4/<D>/events.csv, accounts.csv, attempts.csv  (D = D1, D2, D3)
출력: data/v4/<D>/fairticket_event_based_features.csv   기존 더미와 같은 컬럼·순서·레이블 (BTS 정의 그대로)
      data/v4/<D>/features_full.csv                      추출기의 원래 출력 + 보조 컬럼
      data/v4/<D>/interval_definitions.csv               간격 Feature 2개를 여러 정의로 계산한 값 (계정 1행)
      output/v4/validation/<D>/validation.json, null_report.csv

Feature 종류는 6개 그대로다. interval_definitions.csv 는 avg_booking_interval, booking_interval_variance 두 개의 '계산 방식'만 바꾼 값이다.
후보는 REPORT_MEASUREMENT_ADD.md 6.2절에 평가 전에 적어 둔 것이다.
  세션 경계   d30      = 30일 창의 모든 간격 (현재 BTS)
              concert  = 같은 서비스·공연이고 시작 간격 30분 이하인 간격만 (공연별 세션)
              activity = 시작 간격 30분 이하인 간격만 (활동 세션)
  대표값      mean_var      = 평균, 모분산 (BTS 와 같은 식)
              median_robust = 중앙값, (사분위 범위 / 1.349)² (정규분포면 분산과 같아지는 강건한 값)
              end_start     = 간격을 '앞 시도의 마지막 이벤트 → 이번 시작'으로 바꾼 평균, 모분산.
                              앞 시도가 종료 이벤트로 끝나지 않았거나 그 이벤트가 이번 시작보다 늦으면 시작 → 시작 간격을 쓴다
  최소 간격 수 __min5      = 간격이 5개 미만이면 분산을 결측으로
30분, 사분위 범위, 5개라는 값은 실측으로 검증된 것이 아니다 (가정).
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP.parent / "issue14-real-validation" / "scripts"))
from extract_features import MODEL_FEATURES, extract, load_events  # noqa: E402  (#14 의 추출기, 수정 없음)
from extract_dataset import DATASET_NAME, build_dataset, null_report, validate  # noqa: E402
from extract_session_features import assign_sessions, attempt_table  # noqa: E402  (v3 세션 실험의 계산. 대조용)
from generate_events import sha256  # noqa: E402
from generate_events_v4 import DISTS, load_v4  # noqa: E402

INTERVAL, VARIANCE = "avg_booking_interval", "booking_interval_variance"
BOUNDARIES = {"d30": "30일 전체 (현재 BTS)", "concert": "공연별 세션", "activity": "활동 세션"}
REPRESENTATIVES = {"mean_var": "평균·모분산", "median_robust": "중앙값·강건한 흔들림", "end_start": "종료 → 시작 간격의 평균·모분산"}
SESSION_GAP_S = 1800.0
MIN_GAPS = 5
WINDOW_DAYS = 30
TERMINAL_STATUS = {"ABANDONED", "EXPIRED", "REJECTED"}
MEASURED_EVENTS = EXP / "measurement" / "data" / "20261124-add" / "raw_events.csv"
NOT_GENERATED_KEYS = {"performanceStartsAt", "performanceEndsAt"}


def definition_columns(boundary, representative, min5=False):
    return f"{INTERVAL}__{boundary}__{representative}", f"{VARIANCE}__{boundary}__{representative}" + ("__min5" if min5 else "")


def attempts_with_end(events, calculated_at):
    """시도 1행: 계정, 공연, 시작(가장 이른 이벤트), 끝(마지막 이벤트가 종료 이벤트일 때 그 시각)."""
    calculated_at = pd.Timestamp(calculated_at)
    window = events[(events["occurred_at"] >= calculated_at - pd.Timedelta(days=WINDOW_DAYS)) & (events["occurred_at"] <= calculated_at)]
    stage = window[window["event_type"].str.upper() == "BOOKING_ATTEMPT"].sort_values(["occurred_at", "event_id"])
    stage = stage.assign(attempt_id=stage["meta"].map(lambda m: m.get("attemptId")), concert=stage["meta"].map(lambda m: m.get("concertId")),
                         stage=stage["meta"].map(lambda m: m.get("stage")), status=stage["meta"].map(lambda m: m.get("status")))
    stage = stage[stage["attempt_id"].notna() & (stage["service_id"] != "")]
    table = (stage.groupby(["user_id", "service_id", "attempt_id"], sort=False)
             .agg(start=("occurred_at", "min"), last=("occurred_at", "max"), concert=("concert", "first"), last_stage=("stage", "last"),
                  last_status=("status", "last")).reset_index())
    terminal = table["last_status"].isin(TERMINAL_STATUS) | (table["last_stage"] == "COMPLETED")
    table["end"] = table["last"].where(terminal)
    return table.sort_values(["user_id", "start", "attempt_id"]).reset_index(drop=True)


def interval_definitions(events, calculated_at, users):
    """계정 1행. 세션 경계 3 × 대표값 3 (+ 분산의 최소 간격 수)."""
    table = attempts_with_end(events, calculated_at)
    by = table.groupby("user_id")
    gap = by["start"].diff().dt.total_seconds()
    after_end = (table["start"] - by["end"].shift()).dt.total_seconds()
    table["gap"] = gap.clip(lower=0)
    table["gap_end"] = after_end.where(after_end >= 0, table["gap"])  # 종료 이벤트가 없거나 다음 시작보다 늦으면 시작 → 시작
    same = (table["concert"] == by["concert"].shift()) & (table["service_id"] == by["service_id"].shift())
    masks = {"d30": gap.notna(), "activity": gap.notna() & (gap <= SESSION_GAP_S), "concert": gap.notna() & (gap <= SESSION_GAP_S) & same}
    out = pd.DataFrame(index=pd.Index(users, name="user_id"))
    out["n_attempts"] = by.size().reindex(out.index).fillna(0).astype(int)
    for boundary, mask in masks.items():
        part = table[mask].groupby("user_id")
        count = part.size().reindex(out.index).fillna(0).astype(int)
        out[f"n_gaps__{boundary}"] = count
        quartile = part["gap"].quantile([0.25, 0.75]).unstack()
        values = {"mean_var": (part["gap"].mean(), part["gap"].var(ddof=0)),
                  "median_robust": (part["gap"].median(), ((quartile[0.75] - quartile[0.25]) / 1.349) ** 2 if len(quartile) else pd.Series(dtype=float)),
                  "end_start": (part["gap_end"].mean(), part["gap_end"].var(ddof=0))}
        for representative, (center, spread) in values.items():
            a, b = definition_columns(boundary, representative)
            out[a], out[b] = center.reindex(out.index), spread.reindex(out.index)
            out[b + "__min5"] = out[b].where(count >= MIN_GAPS)
    return out.reset_index(), table


def with_definition(dataset, definitions, boundary, representative, min5=False):
    """기존 형식의 데이터셋에서 간격 Feature 2개만 그 정의의 값으로 바꾼 것."""
    a, b = definition_columns(boundary, representative, min5)
    frame = dataset.copy()
    frame[INTERVAL], frame[VARIANCE] = definitions[a].to_numpy(dtype=float), definitions[b].to_numpy(dtype=float)
    return frame


def build_all(events, accounts, base):
    dataset, full, summary = build_dataset(events, accounts, base)
    definitions, table = interval_definitions(events, base["calculated_at_utc"], dataset["user_id"])
    return dataset, full, definitions, table, summary


def event_consistency(events, attempts, base):
    """이벤트 순서·시각과 성공·실패·취소의 일관성."""
    checks = {}
    end = datetime.fromisoformat(base["calculated_at_utc"])
    checks["all_events_inside_30d_window"] = bool(((events["occurred_at"] >= end - timedelta(days=WINDOW_DAYS)) & (events["occurred_at"] <= end)).all())
    checks["event_id_follows_time_order"] = bool(events.sort_values("event_id")["occurred_at"].is_monotonic_increasing)
    checks["external_event_id_unique"] = bool(events["external_event_id"].is_unique)
    stage = events[events["event_type"] == "BOOKING_ATTEMPT"].copy()
    for key in ("attemptId", "stage", "status", "sequence", "bookingId", "transitionId"):
        stage[key] = stage["meta"].map(lambda m, k=key: m.get(k))
    stage = stage.sort_values(["attemptId", "sequence"])
    grouped = stage.groupby("attemptId", sort=False)
    checks["sequence_starts_at_1_and_is_contiguous"] = bool((grouped["sequence"].apply(lambda s: list(s) == list(range(1, len(s) + 1)))).all())
    checks["time_does_not_go_back_within_attempt"] = bool(grouped["occurred_at"].apply(lambda s: s.is_monotonic_increasing).all())
    checks["first_event_is_STARTED"] = bool((grouped["stage"].first() == "STARTED").all())
    checks["one_account_per_attempt"] = bool((grouped["user_id"].nunique() == 1).all())
    checks["transition_id_unique"] = bool(stage["transitionId"].is_unique)
    last = grouped.agg(stage=("stage", "last"), status=("status", "last"), end=("occurred_at", "max"), start=("occurred_at", "min"), user_id=("user_id", "first"))
    last["kind"] = np.where(last["stage"] == "COMPLETED", "completed", last["stage"] + "/" + last["status"])
    allowed = {"completed", "SEAT_SELECTION/ABANDONED", "PAYMENT/ABANDONED", "SEAT_HOLD/EXPIRED", "SEAT_HOLD/REJECTED"}
    checks["every_attempt_ends_with_a_terminal_event"] = bool(last["kind"].isin(allowed).all())
    terminal = stage["status"].isin(TERMINAL_STATUS - {"REJECTED"}) | (stage["stage"] == "COMPLETED")
    checks["at_most_one_terminal_event_per_attempt"] = bool((terminal.groupby(stage["attemptId"]).sum() <= 1).all())

    bookings = events[events["event_type"] == "BOOKING"].assign(attemptId=lambda d: d["meta"].map(lambda m: m.get("attemptId")))
    completed = stage[stage["stage"] == "COMPLETED"]
    checks["booking_id_unique"] = bool(bookings["booking_id"].is_unique)
    checks["completed_attempts_equal_bookings"] = bool(len(completed) == len(bookings) and set(completed["bookingId"]) == set(bookings["booking_id"])
                                                       and set(completed["attemptId"]) == set(bookings["attemptId"]))
    checks["failed_attempts_have_no_booking"] = bool(not set(last.index[last["kind"] != "completed"]) & set(bookings["attemptId"]))
    order = completed.set_index("attemptId")["occurred_at"]
    hold = stage[(stage["stage"] == "SEAT_HOLD") & (stage["status"] == "SUCCEEDED")].set_index("attemptId")["occurred_at"]
    booked = bookings.set_index("attemptId")["occurred_at"]
    checks["booking_is_between_hold_and_completed"] = bool((hold.reindex(booked.index) < booked).all() and (booked <= order.reindex(booked.index)).all())
    cancels = events[events["event_type"] == "CANCEL"]
    owner, booked_at = bookings.set_index("booking_id")["user_id"], bookings.set_index("booking_id")["occurred_at"]
    checks["cancel_refers_to_own_booking"] = bool(cancels["booking_id"].isin(owner.index).all() and (owner.reindex(cancels["booking_id"]).to_numpy() == cancels["user_id"].to_numpy()).all())
    checks["cancel_is_after_booking"] = bool((cancels["occurred_at"].to_numpy() > booked_at.reindex(cancels["booking_id"]).to_numpy()).all())
    checks["at_most_one_cancel_per_booking"] = bool(cancels["booking_id"].is_unique)
    # 한 계정의 시도는 겹치지 않는다. 선점만 해 두고 넘어가는 hold_cancel 의 만료 이벤트(5분 뒤)만 예외다
    truth = attempts.set_index("attempt_id")["outcome"]
    ordered = last.assign(outcome=truth.reindex(last.index)).sort_values(["user_id", "start"])
    following = ordered.groupby("user_id")["start"].shift(-1)
    sequential = ordered[(ordered["outcome"] != "hold_left") & following.notna()]
    checks["next_attempt_starts_after_previous_ends"] = bool((following[sequential.index] > sequential["end"]).all())
    checks["generated_outcome_matches_last_event"] = bool((ordered["kind"].map({"completed": "complete", "SEAT_SELECTION/ABANDONED": "abandon_seat", "PAYMENT/ABANDONED": "abandon_payment",
                                                                              "SEAT_HOLD/REJECTED": "hold_rejected"}).fillna("expired")
                                                           == ordered["outcome"].replace({"hold_left": "expired", "hold_expired_wait": "expired"})).all())
    return checks, last["kind"].value_counts().to_dict()


def signature(events):
    """(이벤트 종류, 단계, 상태, 출처, 사유) → metadata 키 집합, 그리고 시도별 단계 순서."""
    meta = events["meta"]
    kind = [(t, m.get("stage"), m.get("status"), m.get("source"), m.get("reasonCode")) for t, m in zip(events["event_type"], meta)]
    keys = {}
    for k, m in zip(kind, meta):
        keys.setdefault(k, set()).add(frozenset(m))
    stage = events[events["event_type"] == "BOOKING_ATTEMPT"].assign(a=lambda d: d["meta"].map(lambda m: m["attemptId"]), q=lambda d: d["meta"].map(lambda m: m["sequence"]),
                                                                     s=lambda d: d["meta"].map(lambda m: f"{m['stage']}/{m['status']}"))
    sequences = stage.sort_values(["a", "q"]).groupby("a")["s"].apply(" > ".join)
    return keys, sequences.value_counts().to_dict()


def structure_match(events):
    """실측 이벤트(2026-11-24, 정상 사용자 1명)에 나온 이벤트 종류·필드·단계 순서가 합성 이벤트에도 같은 모양으로 있는지."""
    if not MEASURED_EVENTS.exists():
        return {"skipped": "실측 원본이 이 PC 에 없다 (git 제외 파일)"}
    measured = load_events(MEASURED_EVENTS)
    real_keys, real_sequences = signature(measured)
    synthetic_keys, synthetic_sequences = signature(events)
    rows, same_fields = [], True
    for kind, key_sets in sorted(real_keys.items(), key=str):
        real = set().union(*key_sets)
        synthetic = set().union(*synthetic_keys[kind]) if kind in synthetic_keys else None
        missing = None if synthetic is None else sorted(real - synthetic - NOT_GENERATED_KEYS)
        extra = None if synthetic is None else sorted(synthetic - real)
        same_fields &= synthetic is not None and not missing and not extra
        rows.append({"event": "/".join(str(v) for v in kind if v is not None), "in_synthetic": synthetic is not None,
                     "fields_missing_in_synthetic": missing, "fields_only_in_synthetic": extra,
                     "fields_not_generated_on_purpose": None if synthetic is None else sorted(real & NOT_GENERATED_KEYS - synthetic)})
    return {"measured_event_kinds": len(real_keys), "all_measured_kinds_exist_with_same_fields": bool(same_fields),
            "all_measured_stage_sequences_exist": bool(set(real_sequences) <= set(synthetic_sequences)),
            "measured_stage_sequences": real_sequences,
            "synthetic_only_stage_sequences (실측에 없던 경우: 선점 거절, 결제 화면 없는 실측 매크로 시도 등)": {k: v for k, v in synthetic_sequences.items() if k not in real_sequences},
            "synthetic_only_event_kinds": sorted("/".join(str(v) for v in k if v is not None) for k in set(synthetic_keys) - set(real_keys)), "by_event": rows}


def main():
    base, v4 = load_v4()
    protected = [EXP / "data" / "v3" / "events.csv", EXP / "data" / "v3" / DATASET_NAME, EXP / "data" / "v2" / DATASET_NAME]
    before = {str(p.relative_to(EXP)): sha256(p) for p in protected}
    failed = False
    for dist in DISTS:
        data_dir, out_dir = EXP / "data" / "v4" / dist, EXP / "output" / "v4" / "validation" / dist
        out_dir.mkdir(parents=True, exist_ok=True)
        events = load_events(data_dir / "events.csv")  # 저장된 CSV 를 #14 의 읽기 함수로 읽는다
        accounts = pd.read_csv(data_dir / "accounts.csv", keep_default_na=False)
        attempts = pd.read_csv(data_dir / "attempts.csv")
        dataset, full, definitions, table, summary = build_all(events, accounts, base)

        checks = validate(full, events, accounts, attempts, base)  # 독립 재계산, 생성 기록과의 대조, 공유 관계
        order_checks, endings = event_consistency(events, attempts, base)
        by_user = full.set_index("user_id").reindex(definitions["user_id"])
        a, b = definition_columns("d30", "mean_var")
        same = {"d30_mean_equals_bts_extractor": bool(np.isclose(definitions[a], by_user[INTERVAL], rtol=1e-9, atol=1e-6, equal_nan=True).all()),
                "d30_variance_equals_bts_extractor": bool(np.isclose(definitions[b], by_user[VARIANCE], rtol=1e-9, atol=1e-6, equal_nan=True).all())}
        # 공연별 세션·활동 세션의 평균: v3 세션 실험의 계산(extract_session_features)과 대조
        old = attempt_table(events, pd.Timestamp(base["calculated_at_utc"]))
        for boundary, use_concert in (("concert", True), ("activity", False)):
            again = assign_sessions(old, use_concert=use_concert).dropna(subset=["gap"]).groupby("user_id")["gap"].agg(mean="mean", var=lambda v: v.var(ddof=0))
            again = again.reindex(definitions["user_id"])
            x, y = definition_columns(boundary, "mean_var")
            same[f"{boundary}_mean_var_equals_v3_session_code"] = bool(np.isclose(definitions[x], again["mean"], rtol=1e-9, atol=1e-6, equal_nan=True).all()
                                                                       and np.isclose(definitions[y], again["var"], rtol=1e-9, atol=1e-6, equal_nan=True).all())
        structure = structure_match(events)
        ok = checks["all_passed"] and all(order_checks.values()) and all(same.values()) and (
            "skipped" in structure or (structure["all_measured_kinds_exist_with_same_fields"] and structure["all_measured_stage_sequences_exist"]))
        failed |= not ok

        dataset.to_csv(data_dir / DATASET_NAME, index=False, encoding="utf-8", lineterminator="\n")
        full.to_csv(data_dir / "features_full.csv", index=False, encoding="utf-8", lineterminator="\n")
        definitions.to_csv(data_dir / "interval_definitions.csv", index=False, encoding="utf-8", lineterminator="\n")
        nulls = null_report(full)
        nulls.to_csv(out_dir / "null_report.csv", index=False, encoding="utf-8-sig")
        label = dataset["is_anomaly"].map({0: "정상", 1: "이상"}).to_numpy()
        definition_nulls = {f"{boundary}": {name: float(definitions.loc[label == name, definition_columns(boundary, "mean_var")[0]].isna().mean()) for name in ("정상", "이상")}
                            for boundary in BOUNDARIES}
        record = {"speed_assumption": dist, "extractor": "experiments/issue14-real-validation/scripts/extract_features.py (import, 수정 없음)",
                  "bts_reference": summary["bts_reference"], "calculated_at_utc": summary["calculated_at_utc"], "events": int(len(events)),
                  "all_passed": bool(ok), "feature_checks": checks, "event_consistency": order_checks, "attempt_endings": endings,
                  "interval_definition_checks": same, "structure_match_with_measured": structure,
                  "model_input_null_accounts": int(full[MODEL_FEATURES].isna().any(axis=1).sum()),
                  "interval_null_share_by_boundary": definition_nulls,
                  "dataset_sha256": sha256(data_dir / DATASET_NAME), "events_sha256": sha256(data_dir / "events.csv")}
        (out_dir / "validation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n[{dist}] all_passed={ok}")
        print(" feature checks:", {k: v for k, v in checks.items() if v is not True})
        print(" event consistency failed:", [k for k, v in order_checks.items() if not v], "| endings:", endings)
        print(" definition checks:", same)
        print(" structure:", {k: v for k, v in structure.items() if k not in ("by_event", "measured_stage_sequences")})
        print(nulls.to_string(index=False))
        print(" 간격 결측 비율:", definition_nulls)
    unchanged = before == {str(p.relative_to(EXP)): sha256(p) for p in protected}
    print("\n기존 v2·v3 데이터 변경 없음:", unchanged)
    if failed or not unchanged:
        sys.exit("검증 실패")


if __name__ == "__main__":
    main()
