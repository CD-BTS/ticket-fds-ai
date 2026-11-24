"""이벤트 기반 v4 의 재현성·일관성 테스트, 그리고 v2·v3 가 그대로인지 확인.

    ..\\issue14-real-validation\\.venv\\Scripts\\python tests\\test_v4_dataset.py
"""
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
from extract_dataset import DATASET_NAME  # noqa: E402
from extract_v4 import MIN_GAPS, build_all, definition_columns, event_consistency  # noqa: E402
from generate_events import sha256, to_csv_frame  # noqa: E402
from generate_events_v4 import DISTS, generate, load_v4, make_params  # noqa: E402

DATA = EXP / "data" / "v4"
NORMAL = ["normal", "fast_normal", "shared_ip", "frequent_cancel"]


@lru_cache(maxsize=None)
def generated(dist, key="{}"):
    return generate(dist, json.loads(key), 42)


def alt(dist, **overrides):
    return generated(dist, json.dumps(overrides, sort_keys=True))


def test_saved_files_are_reproduced_byte_for_byte():
    for dist in DISTS:
        events, accounts, attempts, _ = generated(dist)
        for frame, name in [(to_csv_frame(events), "events.csv"), (accounts, "accounts.csv"), (attempts, "attempts.csv")]:
            buffer = io.StringIO()
            frame.to_csv(buffer, index=False, lineterminator="\n")
            assert buffer.getvalue().encode("utf-8") == (DATA / dist / name).read_bytes(), (dist, name)


def test_existing_v2_v3_files_match_their_generation_records():
    for version in ("v2", "v3"):
        record = json.loads((EXP / "data" / version / "generation.json").read_text(encoding="utf-8"))
        for name, digest in record["files_sha256"].items():
            assert sha256(EXP / "data" / version / name) == digest, (version, name)
    for dist in DISTS:
        assert json.loads((DATA / dist / "generation.json").read_text(encoding="utf-8"))["existing_datasets_unchanged"]


def test_accounts_labels_and_sharing_are_the_same_as_v3():
    v3 = pd.read_csv(EXP / "data" / "v3" / "accounts.csv", keep_default_na=False)
    for dist in DISTS:
        accounts = generated(dist)[1]
        for column in ["user_id", "scenario", "is_anomaly", "ip_group_id", "payment_identifier", "tool_profile"]:
            assert (accounts[column].astype(str).to_numpy() == v3[column].astype(str).to_numpy()).all(), (dist, column)


def test_speed_assumption_changes_only_normal_accounts():
    """D1·D2·D3 는 정상 계정의 속도만 다르다. 매크로와 hold_cancel 의 이벤트는 세 데이터셋에서 같다."""
    frames = [to_csv_frame(generated(dist)[0]) for dist in DISTS]
    anomalous = [f[~f["group"].isin(NORMAL)].drop(columns="event_id").reset_index(drop=True) for f in frames]
    assert anomalous[0].equals(anomalous[1]) and anomalous[0].equals(anomalous[2])
    normal = [f[f["group"].isin(NORMAL)] for f in frames]
    assert not normal[0]["occurred_at_utc"].reset_index(drop=True).equals(normal[2]["occurred_at_utc"].reset_index(drop=True))


def test_speed_levels_follow_the_stated_assumptions():
    for dist in DISTS:
        accounts = generated(dist)[1]
        normal = accounts[accounts["is_anomaly"] == 0]
        speed = normal["speed_s"]
        assert speed.min() >= 0.1 - 1e-12, "빠른 끝(실측한 1명)보다 빠른 정상 계정은 만들지 않는다"
        assert (normal.loc[normal["scenario"] == "fast_normal", "speed_s"] <= 10 ** -0.5 + 1e-12).all()
        assert accounts.loc[accounts["is_anomaly"] == 1, "speed_s"].isna().all()
        if dist != "D1":
            assert speed.max() <= 1 + 1e-12
    medians = [generated(dist)[1].query("scenario == 'normal'")["speed_s"].median() for dist in DISTS]
    assert medians[0] > medians[1] > medians[2]  # 느린 쪽 중심 > 로그 균등 > 빠른 쪽 중심


def test_normal_sessions_stop_at_the_goal_or_after_a_failure():
    for dist in DISTS:
        _, accounts, attempts, _ = generated(dist)
        users = accounts.loc[accounts["is_anomaly"] == 0, "user_id"]
        part = attempts[attempts["user_id"].isin(users)]
        by = part.groupby(["user_id", "session_no"])
        last = by["outcome"].last()
        completed = by["outcome"].apply(lambda s: (s == "complete").sum())
        # 목표를 채우고 끝난 세션의 마지막 시도는 완료, 아니면 실패한 뒤 그만둔 세션이다
        assert (completed[last == "complete"] >= 1).all()
        done = completed.groupby(level="user_id").sum()
        failed_end = (last != "complete").groupby(level="user_id").sum()
        goal = accounts.set_index("user_id")["goal_bookings"].reindex(done.index)
        assert (done <= goal).all()
        assert (done[failed_end == 0] == goal[failed_end == 0]).all()  # 실패로 끝난 세션이 없으면 목표를 모두 채웠다


def test_event_order_and_booking_logic_hold_in_alternative_conditions():
    base, _ = load_v4()
    for overrides in ({"queue_wait_s": 300}, {"macro_delay": "loguniform"}, {"skill": True, "switch_prob": 0.5}, {"rules_off": ["S1", "S2"], "retry_wait": "measured_end"}):
        events, accounts, attempts, _ = alt("D2", **overrides)
        checks, _ = event_consistency(events, attempts, base)
        assert all(checks.values()), (overrides, [k for k, v in checks.items() if not v])


def test_queue_wait_is_added_to_people_and_macros_alike():
    _, accounts, attempts, _ = alt("D2", queue_wait_s=60)
    merged = attempts.merge(accounts[["user_id", "is_anomaly"]], on="user_id")
    in_rush = merged[merged["concert"] == merged["open_concert"]]
    medians = in_rush.groupby("is_anomaly")["queue_wait_ms"].median()
    assert (in_rush["queue_wait_ms"] > 0).all() and abs(medians[0] / medians[1] - 1) < 0.15
    assert (merged.loc[merged["concert"] != merged["open_concert"], "queue_wait_ms"] == 0).all()
    assert generated("D2")[2]["queue_wait_ms"].eq(0).all()


def test_human_profile_macro_uses_the_continuous_measurement():
    _, accounts, attempts, _ = generated("D1")
    human = accounts[(accounts["tool_profile"] == "human") & (accounts["n_attempts"] >= 6)]
    starts = attempts[attempts["user_id"].isin(human["user_id"])].assign(t=lambda d: pd.to_datetime(d["start_utc"]))
    gap = starts.groupby(["user_id", "session_no"])["t"].diff().dt.total_seconds().dropna()
    assert 9.0 < gap.median() < 16.5, gap.median()  # 실측 10.1~15.1초 (v3 는 17.3초)


def test_continuous_macro_delay_stays_between_the_measured_profiles():
    _, accounts, _, _ = alt("D1", macro_delay="loguniform")
    drawn = accounts[accounts["pause_ms"].notna()]
    assert set(drawn["scenario"]) == {"jitter", "multi_account"} and drawn["pause_ms"].between(10, 1750).all()
    assert drawn["target_level_s"].between(2.8, 12.9).all()
    assert drawn.groupby("ip_group_id")["pause_ms"].nunique().loc[lambda s: s.index.str.startswith("multi_account")].eq(1).all()  # 링마다 설정 하나
    assert generated("D1")[1]["pause_ms"].isna().all()


def test_interval_definitions_follow_their_rules():
    base, _ = load_v4()
    events, accounts, _, _ = generated("D2")
    dataset, full, definitions, table, _ = build_all(events, accounts, base)
    a, b = definition_columns("d30", "mean_var")
    assert np.allclose(definitions[a], full["avg_booking_interval"], equal_nan=True) and np.allclose(definitions[b], full["booking_interval_variance"], equal_nan=True, rtol=1e-9)
    assert (definitions["n_gaps__concert"] <= definitions["n_gaps__activity"]).all() and (definitions["n_gaps__activity"] <= definitions["n_gaps__d30"]).all()
    for boundary in ("d30", "concert", "activity"):
        mean, variance = definition_columns(boundary, "mean_var")
        none = definitions[f"n_gaps__{boundary}"] == 0
        assert definitions.loc[none, [mean, variance]].isna().all().all() and definitions.loc[~none, [mean, variance]].notna().all().all()
        assert definitions.loc[definitions[f"n_gaps__{boundary}"] < MIN_GAPS, variance + "__min5"].isna().all()
        end_mean = definition_columns(boundary, "end_start")[0]
        assert (definitions[end_mean].dropna() <= definitions.loc[definitions[end_mean].notna(), mean] + 1e-6).all()  # 종료 → 시작 간격은 시작 → 시작 간격보다 길 수 없다
    assert (definitions.loc[definitions["n_gaps__activity"] > 0, definition_columns("activity", "mean_var")[0]] <= 1800).all()


def test_no_parameter_outside_the_fixed_config_is_accepted():
    _, v4 = load_v4()
    try:
        make_params(v4, "D1", {"seat_median": 3})
    except AssertionError:
        return
    raise AssertionError("설정에 없는 파라미터가 받아들여졌다")


def test_saved_dataset_has_the_original_format():
    original = pd.read_csv(EXP.parents[1] / "ml_experiment" / "data" / "fairticket_synthetic_features.csv")
    for dist in DISTS:
        dataset = pd.read_csv(DATA / dist / DATASET_NAME)
        assert list(dataset.columns) == list(original.columns) and len(dataset) == 1000 and dataset["is_anomaly"].equals(original["is_anomaly"])


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}개 통과")
