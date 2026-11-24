"""이벤트 기반 v3 의 재현성·일관성 테스트, 그리고 v1·v2 가 그대로인지 확인.

    ..\\issue14-real-validation\\.venv\\Scripts\\python tests\\test_v3_dataset.py

구조 검사(단계 순서, 예매·취소의 논리, 한 번에 한 시도)는 v1 테스트의 함수를 v3 데이터에 그대로 적용한다.
v3 에서 새로 생긴 결과(hold_rejected: 선점이 계속 거절돼 그만둠)만 마지막 이벤트 표에 더한다.
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
import test_v2_dataset as v2_tests  # noqa: E402
from extract_dataset import DATASET_NAME, build_dataset  # noqa: E402
from generate_events import ORIGINAL_PATH, load_config, sha256, to_csv_frame  # noqa: E402
from generate_events_v3 import CONFIG_PATH, RULES, generate  # noqa: E402

DATA = EXP / "data" / "v3"
V2_SHA = {  # v3 작업 전의 값 (REPORT_V2.md, data/v2/generation.json)
    "config/scenarios_v2.json": "fe8cb362a08497fa125b2fe320067ebc8861133d1a68bb5df28820b1e076d416",
    "data/v2/events.csv": "c61bf72208822195586907b4d5aaf0058c3490140d78ef234042e60dd108b95e",
    "data/v2/fairticket_event_based_features.csv": "b39aa53337d2992ab8226e8cada80ea76a6276195a4c0d16811bb87d52d4b180",
    "reference/macro_attempt_cycles.csv": "6b45af0fe503e79864cb4374647587c95b37cbb6432a2c91e906f8a7dbfc2dce",
}
NORMAL = ["normal", "fast_normal", "shared_ip", "frequent_cancel"]
FEATURE_NAMES = ["avg_booking_interval", "booking_interval_variance", "cancel_rate", "card_reuse_count", "ip_booking_density", "booking_count",
                 "seat_selection_latency"]


@lru_cache(maxsize=None)
def generated(seed=42):
    return generate(load_config(CONFIG_PATH), seed)


def csv_bytes(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def stage_events(events):
    stage = events[events["event_type"] == "BOOKING_ATTEMPT"]
    return stage.assign(attempt=stage["meta"].map(lambda m: m["attemptId"]), kind=stage["meta"].map(lambda m: (m["stage"], m["status"])))


def test_v1_and_v2_files_are_untouched():
    v2_tests.test_v1_files_are_untouched()
    for name, digest in V2_SHA.items():
        assert sha256(EXP / name) == digest, name


def test_v3_config_changes_only_what_it_declares():
    v2, v3 = load_config(EXP / "config" / "scenarios_v2.json"), load_config(CONFIG_PATH)
    assert v3["version"] == "v3" and [c["id"] for c in v3["changes_from_v2"]] == list(RULES)
    for key in ("seed", "service_id", "calculated_at_utc", "window_days", "hold_duration_s", "concert_opens", "profiles", "cancel_delay", "stage_floor_s"):
        assert v2[key] == v3[key], key
    assert set(v3) - set(v2) == {"participation", "seat_contention", "payment_error", "human_profile", "changes_from_v2"}
    for name, spec in v2["scenarios"].items():
        other = json.loads(json.dumps(v3["scenarios"][name]))
        if "attempts_v2" in other:                                        # S2, S3: v2 의 값은 attempts_v2 에 그대로 남아 있다
            other["attempts"] = other.pop("attempts_v2")
        if other.pop("tool_profile", None) is not None:                   # S4
            other["timing_source"] = spec["timing_source"]
        assert other == spec, name


def test_v3_same_seed_reproduces_saved_files_byte_for_byte():
    events, accounts, attempts = generated()
    record = json.loads((DATA / "generation.json").read_text(encoding="utf-8"))
    assert record["config_version"] == "v3" and record["existing_datasets_unchanged"]
    for name, frame in (("events.csv", to_csv_frame(events)), ("accounts.csv", accounts), ("attempts.csv", attempts)):
        assert hashlib.sha256(csv_bytes(frame)).hexdigest() == record["files_sha256"][name] == sha256(DATA / name), name
    assert csv_bytes(to_csv_frame(generate(load_config(CONFIG_PATH), 43)[0])) != csv_bytes(to_csv_frame(events))


def test_v3_features_come_from_events_and_keep_the_format():
    events, accounts, _ = generated()
    dataset, _, _ = build_dataset(events, accounts, load_config(CONFIG_PATH))
    saved = pd.read_csv(DATA / DATASET_NAME)
    pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(csv_bytes(dataset))), saved)
    original = pd.read_csv(ORIGINAL_PATH)
    v2 = pd.read_csv(EXP / "data" / "v2" / DATASET_NAME)
    assert list(saved.columns) == list(original.columns) and (saved.dtypes == original.dtypes).all()
    assert saved["user_id"].equals(original["user_id"]) and saved["is_anomaly"].equals(original["is_anomaly"])
    assert saved["scenario"].equals(v2["scenario"])                       # 같은 계정, 같은 시나리오, 같은 레이블
    assert saved.loc[saved["booking_count"] < 2, "avg_booking_interval"].isna().all()   # 결측은 채우지 않는다
    assert saved.loc[saved["booking_count"] >= 2, "avg_booking_interval"].notna().all()
    assert (saved.loc[saved["booking_count"] == 2, "booking_interval_variance"] == 0).all()


def test_v3_keeps_the_structural_consistency_of_v1():
    original, ends = v1_tests.generated, dict(v1_tests.SEQUENTIAL_OUTCOMES_END)
    v1_tests.generated = generated
    v1_tests.SEQUENTIAL_OUTCOMES_END["hold_rejected"] = ("SEAT_HOLD", "REJECTED")
    try:
        v1_tests.test_accounts_keep_existing_scenarios_and_labels()
        v1_tests.test_event_ids_follow_time_and_ids_are_unique()
        v1_tests.test_each_attempt_follows_the_stage_order()
        v1_tests.test_bookings_and_cancels_are_logically_consistent()
        v1_tests.test_one_account_does_one_attempt_at_a_time()
        v1_tests.test_sharing_relations_are_mutual()
    finally:
        v1_tests.generated = original
        v1_tests.SEQUENTIAL_OUTCOMES_END.clear()
        v1_tests.SEQUENTIAL_OUTCOMES_END.update(ends)
    # 공유 관계는 v2 와 같다 (같은 규칙, 같은 seed). 실측 계정 배정은 fast·jitter 시나리오에서 같다
    _, accounts, _ = generated()
    _, v2_accounts, _ = v2_tests.generated()
    for column in ("ip_group_id", "ip_address", "payment_identifier"):
        assert accounts[column].equals(v2_accounts[column]), column
    measured = accounts["scenario"].isin(["fast", "jitter"])
    assert accounts.loc[measured, "donor_account"].equals(v2_accounts.loc[measured, "donor_account"])


def test_generator_does_not_take_feature_values_as_input():
    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    code = (EXP / "scripts" / "generate_events_v3.py").read_text(encoding="utf-8").split('"""', 2)[2]
    for name in FEATURE_NAMES:
        assert name not in config_text and name not in code, name
    assert "import extract" not in code and "from extract" not in code


def test_rejected_holds_stay_inside_the_attempt():
    """선점 거절은 같은 attemptId 안의 SEAT_HOLD/REJECTED 이벤트다. holdId 가 없고, 시도 수를 늘리지 않는다."""
    events, accounts, attempts = generated()
    stage = stage_events(events)
    rejected = stage[stage["kind"] == ("SEAT_HOLD", "REJECTED")]
    assert len(rejected) == attempts["hold_rejections"].sum() > 0
    assert rejected["meta"].map(lambda m: m.get("reasonCode") == "SEAT_ALREADY_TAKEN" and "holdId" not in m).all()
    assert rejected.groupby("attempt").size().equals(attempts.set_index("attempt_id")["hold_rejections"].loc[lambda s: s > 0].sort_index().rename(None))
    assert stage.groupby("user_id")["attempt"].nunique().reindex(accounts["user_id"]).to_numpy().tolist() == accounts["n_attempts"].tolist()
    # 매크로는 5번까지 요청하고, 5번 모두 거절되면 그 오픈에서의 실행을 멈춘다
    tool_users = set(accounts.loc[accounts["tool_profile"] != "", "user_id"])
    tool = attempts[attempts["user_id"].isin(tool_users)]
    assert tool["hold_rejections"].max() <= 5
    assert (tool.loc[tool["outcome"] == "hold_rejected", "hold_rejections"] == 5).all()
    last = tool.assign(start=pd.to_datetime(tool["start_utc"])).sort_values("start").groupby(["user_id", "concert"]).tail(1)
    assert set(tool.loc[tool["outcome"] == "hold_rejected", "attempt_id"]) <= set(last["attempt_id"])
    assert set(stage["kind"].map(lambda k: k[1])) <= {"ENTERED", "SUCCEEDED", "REJECTED", "ABANDONED", "EXPIRED"}  # FAILED 는 데모가 보내지 않는다


def test_payment_errors_only_for_people_and_leave_no_event():
    _, accounts, attempts = generated()
    scenario = accounts.set_index("user_id")["scenario"]
    errors = attempts[attempts["payment_errors"] > 0]
    assert len(errors) > 0 and errors["user_id"].map(scenario).isin(NORMAL).all()
    assert set(errors["outcome"]) <= {"complete", "abandon_payment", "hold_expired_wait"}


def test_measured_macro_timing_is_kept():
    """fast·jitter 시나리오와 fast·jitter 프로필의 링: 거절이 없는 시도의 시간은 실측 시도를 계정 수준에 맞게 같은 비율로 조정한 값이다."""
    _, accounts, attempts = generated()
    reference = pd.read_csv(EXP / "reference" / "macro_attempt_cycles.csv")
    macro = accounts[accounts["tool_profile"].isin(["fast", "jitter"])]
    assert set(accounts.loc[accounts["scenario"].isin(["fast", "jitter"]), "tool_profile"]) == {"fast", "jitter"}
    assert macro["account_level_s"].notna().all() and macro["time_scale"].between(0.6, 1.6).all()
    table = attempts.merge(macro[["user_id", "donor_account", "time_scale"]], on="user_id")
    table = table.assign(start=pd.to_datetime(table["start_utc"])).sort_values(["user_id", "start"])
    same_open = table["concert"] == table.groupby("user_id")["concert"].shift()
    clean_before = (table.groupby("user_id")["hold_rejections"].shift() == 0) & (table.groupby("user_id")["outcome"].shift() == "complete")
    table["gap_ms"] = (table.groupby("user_id")["start"].diff().dt.total_seconds() * 1000).round().where(same_open & clean_before)
    for (donor, scale), part in table.groupby(["donor_account", "time_scale"]):
        measured = reference[reference["account"] == donor]
        gaps = measured["gap_to_next_ms"].dropna().to_numpy(dtype=float) * scale
        seats = np.maximum(1, np.round(measured["seat_ms"].to_numpy(dtype=float) * scale))
        for gap in part["gap_ms"].dropna():
            assert np.abs(gaps - gap).min() <= 4, (donor, gap)                  # 단계별로 반올림해서 몇 ms 차이가 난다
        assert set(part.loc[part["hold_rejections"] == 0, "seat_ms"].dropna()) <= set(seats), donor
    # human 프로필: 단계 시간은 #14 실측 20건에서만 온다
    human = pd.read_csv(EXP / "reference" / "macro_human_attempt_cycles.csv")
    ring = attempts.merge(accounts.loc[accounts["tool_profile"] == "human", ["user_id", "time_scale"]], on="user_id")
    for scale, part in ring.groupby("time_scale"):
        allowed = set(np.maximum(1, np.round(human["seat_ms"].to_numpy(dtype=float) * scale)))
        assert set(part.loc[part["hold_rejections"] == 0, "seat_ms"].dropna()) <= allowed


def test_assumed_stage_times_still_respect_measured_floors():
    _, accounts, attempts = generated()
    floors = load_config(CONFIG_PATH)["stage_floor_s"]
    assumed = attempts[attempts["user_id"].isin(accounts.loc[accounts["tool_profile"] == "", "user_id"])]
    assert assumed["seat_ms"].dropna().min() >= floors["seat"] * 1000


def test_rules_can_be_switched_off_one_by_one():
    """규칙을 모두 끄면 오픈 참여 1회, 거절·결제 오류 없음, 매크로 시도 수는 v2 의 범위로 돌아간다."""
    _, accounts, attempts = generate(load_config(CONFIG_PATH), 42, off=RULES)
    assert (accounts["n_sessions"] == 1).all() and attempts["hold_rejections"].sum() == 0 and attempts["payment_errors"].sum() == 0
    assert accounts.loc[accounts["scenario"].isin(["fast", "jitter"]), "n_attempts"].between(15, 49).all()
    assert accounts.loc[accounts["scenario"] == "multi_account", "n_attempts"].between(10, 24).all()
    assert (accounts.loc[accounts["scenario"] == "multi_account", "tool_profile"] == "").all()


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}개 통과")
