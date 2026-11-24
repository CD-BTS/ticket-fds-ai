"""이벤트 기반 v2 의 재현성·일관성 테스트, 그리고 v1 이 그대로인지 확인.

    ..\\issue14-real-validation\\.venv\\Scripts\\python tests\\test_v2_dataset.py

구조 검사(단계 순서, 예매·취소의 논리, 한 번에 한 시도, 공유 관계)는 v1 테스트의 함수를 v2 데이터에 그대로 적용한다.
"""
import hashlib
import io
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(HERE))
import test_event_dataset as v1_tests  # noqa: E402
from extract_dataset import DATASET_NAME, build_dataset  # noqa: E402
from generate_events import ORIGINAL_PATH, generate, load_config, sha256, to_csv_frame  # noqa: E402

V2_CONFIG = EXP / "config" / "scenarios_v2.json"
DATA = EXP / "data" / "v2"
# v1 산출물의 해시 (v2 작업 전의 값). 바뀌면 v1 이 수정된 것이다
V1_SHA = {
    "config/scenarios.json": "0f874591b7799629ea3d528de6973234c70caed6a06c38de44d3dc1b493383b5",
    "data/single_session/events.csv": "da37da3e218226efa482ff69135dc08ce606620a2b2e2b7851fa718794b3ae5c",
    "data/single_session/fairticket_event_based_features.csv": "f430317a01c8e5b28fb4043443232364ffa5c5bec3c06bba13a3112b69e209b2",
}
MEASURED = ["fast", "jitter"]


@lru_cache(maxsize=None)
def generated(seed=42):
    return generate(load_config(V2_CONFIG), "single_session", seed)


def csv_bytes(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def test_v1_files_are_untouched():
    for name, digest in V1_SHA.items():
        assert sha256(EXP / name) == digest, name
    v1_tests.test_existing_datasets_are_untouched()


def test_v2_config_changes_only_what_it_declares():
    v1, v2 = load_config(), load_config(V2_CONFIG)
    assert v2["version"] == "v2" and len(v2["changes_from_v1"]) == 4
    for key in ("seed", "service_id", "calculated_at_utc", "window_days", "hold_duration_s", "concert_opens", "profiles", "cancel_delay"):
        assert v1[key] == v2[key], key
    for name, spec in v1["scenarios"].items():
        other = json.loads(json.dumps(v2["scenarios"][name]))
        other.pop("account_level", None)                                   # C1
        if name == "hold_cancel":                                          # C3
            other["outcomes"]["complete"] = spec["outcomes"]["complete"]
            other["outcomes"]["note"] = spec["outcomes"]["note"]
        if name == "fast_normal":                                          # C4
            other["stages_median_s"] = spec["stages_median_s"]
            other["account_pace_sigma"] = spec["account_pace_sigma"]
        assert other == spec, name


def test_v2_same_seed_reproduces_saved_files_byte_for_byte():
    events, accounts, attempts = generated()
    record = json.loads((DATA / "generation.json").read_text(encoding="utf-8"))
    assert record["config_version"] == "v2"
    for name, frame in (("events.csv", to_csv_frame(events)), ("accounts.csv", accounts), ("attempts.csv", attempts)):
        assert hashlib.sha256(csv_bytes(frame)).hexdigest() == record["files_sha256"][name] == sha256(DATA / name), name


def test_v2_features_come_from_events_and_keep_the_format():
    events, accounts, _ = generated()
    dataset, _, _ = build_dataset(events, accounts, load_config(V2_CONFIG))
    saved = pd.read_csv(DATA / DATASET_NAME)
    pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(csv_bytes(dataset))), saved)
    original = pd.read_csv(ORIGINAL_PATH)
    v1 = pd.read_csv(EXP / "data" / "single_session" / DATASET_NAME)
    assert list(saved.columns) == list(original.columns) and (saved.dtypes == original.dtypes).all()
    assert saved["user_id"].equals(original["user_id"]) and saved["is_anomaly"].equals(original["is_anomaly"])
    assert saved["scenario"].equals(v1["scenario"])
    assert saved.loc[saved["booking_count"] < 2, "avg_booking_interval"].isna().all()   # 결측은 채우지 않는다
    assert saved.loc[saved["booking_count"] >= 2, "avg_booking_interval"].notna().all()


def test_v2_keeps_the_structural_consistency_of_v1():
    """v1 테스트의 구조 검사를 v2 데이터로 돌린다."""
    original = v1_tests.generated
    v1_tests.generated = generated
    try:
        v1_tests.test_accounts_keep_existing_scenarios_and_labels()
        v1_tests.test_event_ids_follow_time_and_ids_are_unique()
        v1_tests.test_each_attempt_follows_the_stage_order()
        v1_tests.test_bookings_and_cancels_are_logically_consistent()
        v1_tests.test_one_account_does_one_attempt_at_a_time()
    finally:
        v1_tests.generated = original
    # 공유 관계: 묶음 크기는 설정 범위 안, 카드를 같이 쓰는 계정은 IP 도 같이 쓴다 (v1 과 같은 규칙, 같은 seed 라 묶음도 같다)
    _, accounts, _ = generated()
    _, v1_accounts, _ = v1_tests.generated()
    for column in ("ip_group_id", "ip_address", "payment_identifier", "donor_account"):
        assert accounts[column].equals(v1_accounts[column]), column


def test_assumed_stage_times_respect_measured_floors():
    events, accounts, attempts = generated()
    floors = load_config(V2_CONFIG)["stage_floor_s"]
    scenario = accounts.set_index("user_id")["scenario"]
    assumed = attempts[~attempts["user_id"].map(scenario).isin(MEASURED)]
    assert assumed["seat_ms"].dropna().min() >= floors["seat"] * 1000
    # 결제 화면 진입 ~ 예매 완료 이벤트: 가정 시나리오에서는 결제 화면 지연보다 짧지 않다
    stage = events[events["event_type"] == "BOOKING_ATTEMPT"]
    paid_at = stage[stage["meta"].map(lambda m: (m["stage"], m["status"]) == ("PAYMENT", "ENTERED"))]
    paid_at = pd.Series(paid_at["occurred_at"].to_numpy(), index=paid_at["meta"].map(lambda m: m["attemptId"]))
    bookings = events[(events["event_type"] == "BOOKING") & ~events["group"].isin(MEASURED)]
    waited = bookings["occurred_at"].to_numpy() - paid_at.loc[bookings["meta"].map(lambda m: m["attemptId"])].to_numpy()
    assert waited.min() >= np.timedelta64(int(floors["payment"] * 1000), "ms")


def test_macro_accounts_are_scaled_copies_of_measured_attempts():
    _, accounts, attempts = generated()
    reference = pd.read_csv(EXP / "reference" / "macro_attempt_cycles.csv")
    macro = accounts[accounts["scenario"].isin(MEASURED)]
    assert macro["time_scale"].between(0.6, 1.6).all() and macro["account_level_s"].notna().all()
    assert accounts.loc[~accounts["scenario"].isin(MEASURED), "time_scale"].isna().all()
    table = attempts.merge(macro[["user_id", "donor_account", "time_scale"]], on="user_id")
    table = table.assign(start=pd.to_datetime(table["start_utc"])).sort_values(["user_id", "start"])
    table["gap_ms"] = (table.groupby("user_id")["start"].diff().dt.total_seconds() * 1000).round()
    assert (table["outcome"] == "complete").all()
    for (donor, scale), part in table.groupby(["donor_account", "time_scale"]):
        measured = reference[reference["account"] == donor]
        allowed_gaps = {round(g * scale) for g in measured["gap_to_next_ms"].dropna()}
        allowed_seats = {max(1, round(s * scale)) for s in measured["seat_ms"]}
        assert set(part["gap_ms"].dropna()) <= allowed_gaps, donor
        assert set(part["seat_ms"]) <= allowed_seats, donor
    # 계정 수준은 유형 안에서 연속적으로 퍼진다 (v1 처럼 실측 계정 수만큼의 덩어리가 아니다)
    assert macro.groupby("scenario")["account_level_s"].nunique().min() >= 40


def test_hold_cancel_mostly_leaves_holds_unpaid():
    _, accounts, attempts = generated()
    hold = attempts[attempts["user_id"].isin(accounts.loc[accounts["scenario"] == "hold_cancel", "user_id"])]
    share = hold["outcome"].value_counts(normalize=True)
    assert set(share.index) == {"complete", "hold_left"} and share["hold_left"] > 0.7
    assert accounts.loc[accounts["scenario"] == "hold_cancel", "complete_prob"].between(0.05, 0.3).all()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}개 통과")
