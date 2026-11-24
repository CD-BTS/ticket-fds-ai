"""생성한 이벤트에서 기존 추출 로직(#14 extract_features.py = BTS FeatureAggregationService 재현)으로 Feature 를 뽑아 데이터셋을 만든다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\extract_dataset.py
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\extract_dataset.py --profile multi_session

입력: data/<profile>/events.csv, accounts.csv, attempts.csv
출력: data/<profile>/fairticket_event_based_features.csv   기존 더미와 같은 컬럼·순서·레이블
      data/<profile>/features_full.csv                      추출기의 원래 출력(반올림 없음) + 보조 컬럼
      output/feature_extraction/<profile>/validation.json, null_report.csv, extraction_summary.json

- Feature 6개의 계산식은 바꾸지 않는다. 추출 함수를 import 해서 그대로 부른다.
- 계산할 수 없는 값은 채우지 않고 빈 칸으로 둔다 (시도 2건 미만 → 간격·분산, 완료 예매 0건 → 취소율).
- `seat_selection_latency` 는 BTS 가 계산하지 않는 컬럼이다. 기존 더미와 컬럼을 맞추기 위해
  '시도별 (SEAT_HOLD 성공 시각 − SEAT_SELECTION 진입 시각)의 계정 평균'으로 넣었고, 이 정의는 이 실험의 가정이다.
  기준 모델(Feature 6개)은 이 컬럼을 쓰지 않는다.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ISSUE14_SCRIPTS = EXP.parent / "issue14-real-validation" / "scripts"
sys.path.insert(0, str(ISSUE14_SCRIPTS))
sys.path.insert(0, str(HERE))
from extract_features import MODEL_FEATURES, extract, load_events  # noqa: E402  (#14 의 추출기, 수정 없음)
from generate_events import ORIGINAL_PATH, V2_PATH, load_config, sha256  # noqa: E402

DATASET_NAME = "fairticket_event_based_features.csv"
# 기존 생성 코드(generate_dummy_data.py)와 같은 자릿수. 하한(max(0.1, ..) 등)은 적용하지 않는다: 계산값을 바꾸기 때문이다.
DECIMALS = {"avg_booking_interval": 2, "booking_interval_variance": 4, "seat_selection_latency": 2, "cancel_rate": 4}
SESSION_GAP_MINUTES = 30.0  # #14 추출기의 세션 보조 컬럼용 기본값 (모델 입력과 무관)


def seat_latency(events):
    """계정별 좌석 선택 소요 시간 평균(초). 선점에 성공한 시도만 쓴다. 없으면 결측."""
    attempts = events[events["event_type"] == "BOOKING_ATTEMPT"]
    stage = attempts["meta"].map(lambda m: (m.get("stage"), m.get("status")))
    frame = pd.DataFrame({"user_id": attempts["user_id"], "attempt": attempts["meta"].map(lambda m: m.get("attemptId")),
                          "t": attempts["occurred_at"], "kind": stage})
    entered = frame[frame["kind"] == ("SEAT_SELECTION", "ENTERED")].groupby(["user_id", "attempt"])["t"].min()
    held = frame[frame["kind"] == ("SEAT_HOLD", "SUCCEEDED")].groupby(["user_id", "attempt"])["t"].min()
    seconds = (held - entered).dropna().dt.total_seconds()
    return seconds.groupby(level="user_id").mean()


def build_dataset(events, accounts, config):
    """이벤트 → (기존 형식 데이터셋, 추출기 원래 출력, 추출 요약)."""
    calculated_at = datetime.fromisoformat(config["calculated_at_utc"])
    features, _, summary = extract(events, calculated_at, SESSION_GAP_MINUTES)
    full = accounts[["user_id", "scenario", "is_anomaly"]].merge(features, on="user_id", how="left", validate="one_to_one")
    full["seat_selection_latency"] = full["user_id"].map(seat_latency(events))
    columns = list(pd.read_csv(ORIGINAL_PATH, nrows=0).columns)
    dataset = full[columns].copy()
    for column, digits in DECIMALS.items():
        dataset[column] = dataset[column].astype(float).round(digits)
    for column in ["card_reuse_count", "ip_booking_density", "booking_count"]:
        dataset[column] = dataset[column].astype("int64")
    return dataset, full, summary


def recompute_independently(events, config):
    """추출기와 다른 방식(표 연산)으로 6개 Feature 를 다시 계산한다. 추출기가 이 데이터에서 제대로 돌았는지 대조하는 용도."""
    end = datetime.fromisoformat(config["calculated_at_utc"])
    window = events[(events["occurred_at"] >= end - timedelta(days=config["window_days"])) & (events["occurred_at"] <= end)]
    attempts = window[window["event_type"] == "BOOKING_ATTEMPT"].assign(a=lambda d: d["meta"].map(lambda m: m["attemptId"]))
    starts = attempts.groupby(["user_id", "a"])["occurred_at"].min().reset_index().sort_values(["user_id", "occurred_at"])
    starts["gap"] = starts.groupby("user_id")["occurred_at"].diff().dt.total_seconds()
    out = pd.DataFrame({"booking_count": starts.groupby("user_id").size()})
    out["avg_booking_interval"] = starts.groupby("user_id")["gap"].mean()
    out["booking_interval_variance"] = starts.groupby("user_id")["gap"].var(ddof=0)
    bookings = window[window["event_type"] == "BOOKING"].drop_duplicates(["user_id", "booking_id"])
    cancelled = set(window.loc[window["event_type"] == "CANCEL", "booking_id"])
    out["cancel_rate"] = bookings.assign(c=bookings["booking_id"].isin(cancelled)).groupby("user_id")["c"].mean()
    for column, key in [("card_reuse_count", "paymentIdentifier"), ("ip_booking_density", "ipAddress")]:
        value = bookings["meta"].map(lambda m: m[key])
        users_per_value = bookings.groupby(value)["user_id"].nunique()
        out[column] = value.map(users_per_value).groupby(bookings["user_id"]).max()
        out[column] = out[column].fillna(1).astype(int)
    return out


def validate(full, events, accounts, attempts, config):
    """추출 결과가 (1) 독립 재계산, (2) 생성기가 기록한 사실(시도·예매·취소·공유 관계)과 맞는지 확인한다."""
    by_user = full.set_index("user_id")
    again = recompute_independently(events, config).reindex(by_user.index)
    checks = {}
    for column in MODEL_FEATURES:
        a, b = by_user[column].astype(float), again[column].astype(float)
        close = np.isclose(a, b, rtol=1e-9, atol=1e-6, equal_nan=True)  # 분산이 1e10 대까지 커지므로 상대 오차로 본다
        checks[f"independent_recompute_{column}"] = bool(close.all() and (a.isna() == b.isna()).all())

    truth = attempts.groupby("user_id").agg(n=("attempt_id", "size"), booked=("booking_id", lambda s: (s.fillna("") != "").sum()),
                                           cancelled=("cancel_utc", lambda s: s.notna().sum())).reindex(by_user.index)
    checks["booking_count_equals_generated_attempts"] = bool((by_user["booking_count"] == truth["n"]).all())
    checks["completed_bookings_equal_generated"] = bool((by_user["n_completed_bookings"] == truth["booked"]).all())
    checks["cancelled_bookings_equal_generated"] = bool((by_user["n_cancelled_bookings"] == truth["cancelled"]).all())
    # 공유 수 = 같은 IP(카드)를 쓰는 계정 중 완료 예매가 있는 계정 수. 본인에게 완료 예매가 없으면 기본값 1
    info = accounts.set_index("user_id").reindex(by_user.index)
    has_booking = truth["booked"] > 0
    for column, key in [("ip_booking_density", "ip_group_id"), ("card_reuse_count", "payment_identifier")]:
        expected = has_booking.groupby(info[key]).transform("sum").where(has_booking, 1).clip(lower=1)
        checks[f"{column}_equals_sharing_groups"] = bool((by_user[column] == expected).all())
    checks["every_account_has_a_feature_row"] = bool(len(full) == len(accounts) and full["booking_count"].notna().all())
    checks["all_events_inside_window"] = bool(len(events) == sum(
        (events["occurred_at"] >= datetime.fromisoformat(config["calculated_at_utc"]) - timedelta(days=config["window_days"]))
        & (events["occurred_at"] <= datetime.fromisoformat(config["calculated_at_utc"]))))
    checks["all_passed"] = all(checks.values())
    return checks


def null_report(full):
    rows = []
    for scenario, part in full.groupby("scenario", sort=False):
        rows.append({
            "scenario": scenario, "accounts": len(part),
            "interval_null (시도 1건)": int(part["avg_booking_interval"].isna().sum()),
            "variance_zero_by_formula (시도 2건)": int(part["variance_degenerate"].sum()),
            "cancel_rate_null (완료 예매 0건)": int(part["cancel_rate"].isna().sum()),
            "card_ip_default_1 (완료 예매 0건)": int((part["n_completed_bookings"] == 0).sum()),
            "seat_latency_null (선점 성공 0건)": int(part["seat_selection_latency"].isna().sum()),
            "any_model_input_null": int(part[MODEL_FEATURES].isna().any(axis=1).sum()),
        })
    table = pd.DataFrame(rows)
    total = table.drop(columns="scenario").sum().to_dict()
    return pd.concat([table, pd.DataFrame([{"scenario": "전체", **total}])], ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="single_session", help="data/<profile> 폴더 이름 (v2 는 --profile v2 --config config/scenarios_v2.json)")
    parser.add_argument("--config", default=None, help="생성에 쓴 설정 파일. 기본 config/scenarios.json")
    args = parser.parse_args()
    data_dir = EXP / "data" / args.profile
    out_dir = EXP / "output" / "feature_extraction" / args.profile
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config) if args.config else load_config()
    before = {"original": sha256(ORIGINAL_PATH), "issue16_v2": sha256(V2_PATH)}

    events = load_events(data_dir / "events.csv")  # #14 의 읽기 함수 그대로
    accounts = pd.read_csv(data_dir / "accounts.csv", keep_default_na=False)
    attempts = pd.read_csv(data_dir / "attempts.csv")
    dataset, full, summary = build_dataset(events, accounts, config)
    checks = validate(full, events, accounts, attempts, config)

    # BTS 계산식과 추출기의 일치는 #14 의 테스트(FeatureAggregationServiceTest.java 를 옮긴 것)로 확인한다
    parity = subprocess.run([sys.executable, str(ISSUE14_SCRIPTS / "test_extract_features.py")], capture_output=True, text=True, encoding="utf-8")
    checks["issue14_bts_parity_tests_passed"] = parity.returncode == 0
    checks["issue14_bts_parity_tests_output_tail"] = parity.stdout.strip().splitlines()[-1] if parity.stdout.strip() else parity.stderr[-300:]

    original = pd.read_csv(ORIGINAL_PATH)
    v2 = pd.read_csv(V2_PATH)
    form = {
        "columns_identical_to_original": list(dataset.columns) == list(original.columns),
        "dtypes_identical_to_original": bool((dataset.dtypes == original.dtypes).all()),
        "rows": int(len(dataset)), "user_id_order_identical": bool(dataset["user_id"].equals(original["user_id"])),
        "is_anomaly_identical": bool(dataset["is_anomaly"].equals(original["is_anomaly"])),
        "scenario_identical_to_issue16_v2": bool(dataset["scenario"].equals(v2["scenario"])),
        "anomaly_ratio": float(dataset["is_anomaly"].mean()),
    }
    dataset.to_csv(data_dir / DATASET_NAME, index=False, encoding="utf-8", lineterminator="\n")
    full.to_csv(data_dir / "features_full.csv", index=False, encoding="utf-8", lineterminator="\n")
    nulls = null_report(full)
    nulls.to_csv(out_dir / "null_report.csv", index=False, encoding="utf-8-sig")
    (out_dir / "extraction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    record = {
        "profile": args.profile, "extractor": "experiments/issue14-real-validation/scripts/extract_features.py (import, 수정 없음)",
        "bts_reference": summary["bts_reference"], "calculated_at_utc": summary["calculated_at_utc"],
        "events_total": summary["events_total"], "events_in_window": summary["events_in_window"],
        "checks": checks, "format": form, "rounding": DECIMALS,
        "dataset_sha256": sha256(data_dir / DATASET_NAME), "events_sha256": sha256(data_dir / "events.csv"),
        "existing_datasets_unchanged": before == {"original": sha256(ORIGINAL_PATH), "issue16_v2": sha256(V2_PATH)},
    }
    (out_dir / "validation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(nulls.to_string(index=False))
    if not (checks["all_passed"] and checks["issue14_bts_parity_tests_passed"] and all(v for v in form.values() if isinstance(v, bool))):
        sys.exit("검증 실패: 위 checks 를 확인하세요.")


if __name__ == "__main__":
    main()
