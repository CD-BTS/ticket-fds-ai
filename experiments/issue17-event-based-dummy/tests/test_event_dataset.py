"""이벤트 기반 더미데이터의 재현성·일관성 테스트.

    ..\\issue14-real-validation\\.venv\\Scripts\\python tests\\test_event_dataset.py

pytest 가 있으면 `pytest tests` 로도 돌릴 수 있다 (#14 의 테스트와 같은 방식).
data/single_session 에 저장된 파일이 있어야 한다 (generate_events.py → extract_dataset.py 를 먼저 실행).
"""
import hashlib
import io
import json
import sys
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP / "scripts"))
from extract_dataset import DATASET_NAME, build_dataset  # noqa: E402
from generate_events import ORIGINAL_PATH, V2_PATH, generate, load_config, sha256, to_csv_frame  # noqa: E402

DATA = EXP / "data" / "single_session"
ORIGINAL_SHA = "2e1e27c1ebd434676e8079c7e4a65763be3f4bb2b4a8fe3e788897729fabf416"  # #14·#16 보고서에 기록된 값
V2_SHA = "426a0967dc59c0fca80f9408820f3454d772c027eac71bc4faee095f514af4a7"
SEQUENTIAL_OUTCOMES_END = {"complete": ("COMPLETED", "SUCCEEDED"), "abandon_seat": ("SEAT_SELECTION", "ABANDONED"),
                           "abandon_payment": ("PAYMENT", "ABANDONED"), "hold_expired_wait": ("SEAT_HOLD", "EXPIRED"),
                           "hold_left": ("SEAT_HOLD", "EXPIRED")}


@lru_cache(maxsize=None)
def generated(seed=42):
    return generate(load_config(), "single_session", seed)


def csv_bytes(frame):
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def test_existing_datasets_are_untouched():
    assert sha256(ORIGINAL_PATH) == ORIGINAL_SHA
    assert sha256(V2_PATH) == V2_SHA


def test_same_seed_reproduces_saved_files_byte_for_byte():
    events, accounts, attempts = generated()
    record = json.loads((DATA / "generation.json").read_text(encoding="utf-8"))
    for name, frame in (("events.csv", to_csv_frame(events)), ("accounts.csv", accounts), ("attempts.csv", attempts)):
        digest = hashlib.sha256(csv_bytes(frame)).hexdigest()
        assert digest == record["files_sha256"][name] == sha256(DATA / name), name


def test_second_run_is_identical_and_other_seed_differs():
    generated.cache_clear()
    first = csv_bytes(to_csv_frame(generated()[0]))
    generated.cache_clear()
    assert csv_bytes(to_csv_frame(generated()[0])) == first
    assert csv_bytes(to_csv_frame(generate(load_config(), "single_session", 43)[0])) != first


def test_accounts_keep_existing_scenarios_and_labels():
    _, accounts, _ = generated()
    v2 = pd.read_csv(V2_PATH)
    original = pd.read_csv(ORIGINAL_PATH)
    assert len(accounts) == 1000
    assert accounts["user_id"].equals(v2["user_id"]) and accounts["scenario"].equals(v2["scenario"])
    assert accounts["is_anomaly"].equals(original["is_anomaly"])
    config = load_config()
    for name, spec in config["scenarios"].items():
        assert set(accounts.loc[accounts["scenario"] == name, "is_anomaly"]) == {spec["is_anomaly"]}


def test_event_ids_follow_time_and_ids_are_unique():
    events, _, _ = generated()
    assert events["event_id"].tolist() == list(range(1, len(events) + 1))
    assert events["occurred_at"].is_monotonic_increasing
    assert events["external_event_id"].is_unique
    end = datetime.fromisoformat(load_config()["calculated_at_utc"])
    assert events["occurred_at"].min() >= end - timedelta(days=30) and events["occurred_at"].max() <= end


def test_each_attempt_follows_the_stage_order():
    events, _, attempts = generated()
    outcome = attempts.set_index("attempt_id")["outcome"]
    order = {"STARTED": 0, "QUEUE": 1, "SEAT_SELECTION": 2, "SEAT_HOLD": 3, "PAYMENT": 4, "COMPLETED": 5}
    stage_events = events[events["event_type"] == "BOOKING_ATTEMPT"]
    grouped = stage_events.groupby(stage_events["meta"].map(lambda m: m["attemptId"]), sort=False)
    assert len(grouped) == len(attempts)
    for attempt_id, part in grouped:
        metas = part.sort_values("event_id")["meta"].tolist()
        assert [m["sequence"] for m in metas] == list(range(1, len(metas) + 1)), attempt_id
        assert (metas[0]["stage"], metas[0]["status"]) == ("STARTED", "ENTERED")
        ranks = [order[m["stage"]] for m in metas]
        # 단계는 뒤로 가지 않는다. 예외: 선점 만료(SEAT_HOLD EXPIRED)는 결제 화면 진입 뒤에 올 수 있다
        for previous, current, meta in zip(ranks, ranks[1:], metas[1:]):
            assert current >= previous or (meta["stage"], meta["status"]) == ("SEAT_HOLD", "EXPIRED"), attempt_id
        assert (metas[-1]["stage"], metas[-1]["status"]) == SEQUENTIAL_OUTCOMES_END[outcome[attempt_id]], attempt_id


def test_bookings_and_cancels_are_logically_consistent():
    events, _, attempts = generated()
    bookings = events[events["event_type"] == "BOOKING"].set_index("booking_id")
    cancels = events[events["event_type"] == "CANCEL"].set_index("booking_id")
    completed = events[(events["event_type"] == "BOOKING_ATTEMPT") & events["meta"].map(lambda m: m.get("stage") == "COMPLETED")]
    assert bookings.index.is_unique and cancels.index.is_unique
    assert set(cancels.index) <= set(bookings.index)                         # 취소는 완료된 예매에만
    assert (cancels["occurred_at"] > bookings.loc[cancels.index, "occurred_at"]).all()   # 예매 뒤에 취소
    assert (cancels["user_id"] == bookings.loc[cancels.index, "user_id"]).all()          # 같은 계정이 취소
    assert set(completed["meta"].map(lambda m: m["bookingId"])) == set(bookings.index)   # 완료된 시도 = 예매 이벤트
    assert set(bookings.index) == set(attempts.loc[attempts["outcome"] == "complete", "booking_id"])
    # 예매 이벤트는 그 시도의 선점 이후, 선점 유효 시간(5분) 안에 일어난다
    holds = events[(events["event_type"] == "BOOKING_ATTEMPT") & events["meta"].map(lambda m: (m.get("stage"), m.get("status")) == ("SEAT_HOLD", "SUCCEEDED"))]
    hold_at = pd.Series(holds["occurred_at"].to_numpy(), index=holds["meta"].map(lambda m: m["attemptId"]))
    waited = bookings["occurred_at"].to_numpy() - hold_at.loc[bookings["meta"].map(lambda m: m["attemptId"])].to_numpy()
    assert (waited > np.timedelta64(0)).all() and (waited < np.timedelta64(300, "s")).all()


def test_one_account_does_one_attempt_at_a_time():
    """다음 시도는 이전 시도를 떠난 뒤에 시작한다. 선점만 해 두고 넘어가는 경우(hold_left)는 선점 성공 시각이 기준이다."""
    events, _, attempts = generated()
    stage_events = events[events["event_type"] == "BOOKING_ATTEMPT"].assign(
        attempt=lambda d: d["meta"].map(lambda m: m["attemptId"]), stage=lambda d: d["meta"].map(lambda m: (m["stage"], m["status"])))
    last = stage_events.groupby("attempt")["occurred_at"].max()
    hold = stage_events[stage_events["stage"] == ("SEAT_HOLD", "SUCCEEDED")].groupby("attempt")["occurred_at"].min()
    table = attempts.assign(start=pd.to_datetime(attempts["start_utc"]))
    table["left_at"] = [hold[a] if o == "hold_left" else last[a] for a, o in zip(table["attempt_id"], table["outcome"])]
    table = table.sort_values(["user_id", "start"])
    previous_left = table.groupby("user_id")["left_at"].shift()
    assert (table["start"][previous_left.notna()] > previous_left.dropna()).all()


def test_sharing_relations_are_mutual():
    events, accounts, _ = generated()
    bookings = events[events["event_type"] == "BOOKING"]
    used = pd.DataFrame({"user_id": bookings["user_id"], "ip": bookings["meta"].map(lambda m: m["ipAddress"]),
                         "card": bookings["meta"].map(lambda m: m["paymentIdentifier"])}).drop_duplicates()
    assert used.groupby("user_id").size().max() == 1                         # 계정은 자기 IP·카드만 쓴다
    info = accounts.set_index("user_id")
    assert (used["ip"].to_numpy() == info.loc[used["user_id"], "ip_address"].to_numpy()).all()
    assert (used["card"].to_numpy() == info.loc[used["user_id"], "payment_identifier"].to_numpy()).all()
    # IP 묶음과 카드 묶음은 한 시나리오 안에서만 만들어진다. 카드를 같이 쓰는 계정은 IP 도 같이 쓴다
    assert accounts.groupby("ip_group_id")["scenario"].nunique().max() == 1
    assert accounts.groupby("payment_identifier")["ip_group_id"].nunique().max() == 1
    config = load_config()
    for name, spec in config["scenarios"].items():
        sizes = accounts.loc[accounts["scenario"] == name, "ip_group_size"]
        assert sizes.between(spec["ip_group_size"]["min"], spec["ip_group_size"]["max"]).all(), name


def test_macro_timing_comes_only_from_measured_attempts():
    _, accounts, attempts = generated()
    reference = pd.read_csv(EXP / "reference" / "macro_attempt_cycles.csv")
    table = attempts.merge(accounts[["user_id", "scenario", "donor_account"]], on="user_id")
    table = table[table["scenario"].isin(["fast", "jitter"])].assign(start=lambda d: pd.to_datetime(d["start_utc"]))
    table = table.sort_values(["user_id", "start"])
    table["gap_ms"] = (table.groupby("user_id")["start"].diff().dt.total_seconds() * 1000).round()
    assert (table["outcome"] == "complete").all()
    for donor, part in table.groupby("donor_account"):
        measured = reference[reference["account"] == donor]
        assert set(part["gap_ms"].dropna()) <= set(measured["gap_to_next_ms"].dropna()), donor
        assert set(part["seat_ms"]) <= set(measured["seat_ms"]), donor


def test_features_are_reproduced_from_events_and_match_saved_dataset():
    events, accounts, _ = generated()
    dataset, _, _ = build_dataset(events, accounts, load_config())
    saved = pd.read_csv(DATA / DATASET_NAME)
    pd.testing.assert_frame_equal(pd.read_csv(io.BytesIO(csv_bytes(dataset))), saved)
    original = pd.read_csv(ORIGINAL_PATH)
    assert list(saved.columns) == list(original.columns) and (saved.dtypes == original.dtypes).all()
    # 계산할 수 없는 값은 채우지 않는다
    assert saved.loc[saved["booking_count"] < 2, "avg_booking_interval"].isna().all()
    assert saved.loc[saved["booking_count"] >= 2, "avg_booking_interval"].notna().all()
    assert (saved.loc[saved["booking_count"] == 2, "booking_interval_variance"] == 0).all()


def test_generator_does_not_take_feature_values_as_input():
    """생성 설정과 생성기에는 Feature 컬럼 이름이 등장하지 않는다 (Feature 를 먼저 정하고 이벤트를 맞추지 않았다는 확인)."""
    config_text = (EXP / "config" / "scenarios.json").read_text(encoding="utf-8")
    source = (EXP / "scripts" / "generate_events.py").read_text(encoding="utf-8")
    code = source.split('"""', 2)[2]  # 맨 위 설명문 제외
    for name in ["avg_booking_interval", "booking_interval_variance", "cancel_rate", "card_reuse_count", "ip_booking_density",
                 "booking_count", "seat_selection_latency"]:
        assert name not in config_text and name not in code, name
    assert "import extract" not in code and "from extract" not in code  # 추출기를 불러 쓰지 않는다


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}개 통과")
