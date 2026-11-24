"""이벤트 기반 더미데이터 v4 의 이벤트를 생성한다. Feature 는 여기서 정하지 않는다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\generate_events_v4.py            # data/v4/D1, D2, D3
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\generate_events_v4.py --force

설계: REPORT_MEASUREMENT_ADD.md 5절·8절. 값과 근거: config/scenarios_v4.json (v3 설정 위에 덮어쓴다).
v1~v3 의 생성기는 수정하지 않고 import 해서 쓴다. 계정 목록, 공유 관계, 이벤트 형식, fast·jitter 매크로, hold_cancel, 선점 거절, 취소는 v3 와 같다.

v3 → v4
  R1 정상 사용자의 속도: 계정마다 속도 수준 s. 빠른 끝(s = 0.1)은 실측한 1명, 느린 끝(s = 1)은 v3 normal.
     사용자 사이의 분포는 D1(느린 쪽 중심) / D2(로그 균등) / D3(빠른 쪽 중심) 세 가정. 어느 것도 실제 분포가 아니다.
  R3 정상 세션: 목표 예매 수를 정하고 실패하면 다시 시도한다. 다음 시도까지의 대기는 앞 시도의 결과에 따라 다르다. 공연을 바꿀 수 있다.
  R5 human 프로필 매크로: #16 연속 실행 실측 시도를 복원추출한다. (선택) 지연 설정을 연속 범위에서 뽑는다.
  R6 대기열 대기: STARTED → QUEUE 통과 사이의 별도 변수. 사람과 매크로에 같은 분포.
  (선택) 숙련도: 처음 몇 건이 느리다.
이벤트에 recordedAt, transitionId, clientOccurredAt 을 넣어 실측 이벤트의 필드와 맞춘다.

params (config 의 baseline 에 덮어쓰는 값. 민감도 분석에서 한 번에 하나씩 바꾼다)
  speed_dist, within_sigma, goal_lambda, retry_wait, switch_prob, macro_delay, queue_wait_s, rules_off, skill

출력: data/v4/<D>/events.csv, accounts.csv, attempts.csv, generation.json
"""
import argparse
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_events import (EXP, ORIGINAL_PATH, REFERENCE_PATH, REPO, STAGE_KEYS, V2_PATH, Account, build_accounts, draw, load_assignment,  # noqa: E402
                             load_config, macro_levels, outcome_table, rng_for, sha256, to_csv_frame)
from generate_events_v3 import CONFIG_PATH as V3_CONFIG_PATH, CYCLE, assign_tools, draw_attempts, run_attempt  # noqa: E402

CONFIG_PATH = EXP / "config" / "scenarios_v4.json"
DISTS = ("D1", "D2", "D3")
PARAM_KEYS = ("speed_dist", "within_sigma", "goal_lambda", "retry_wait", "switch_prob", "macro_delay", "queue_wait_s", "rules_off", "skill")
SPEED_STAGES = ("queue", "pre_seat", "seat", "hold_to_payment", "payment")
IDLE_KIND = {"complete": "complete", "abandon_seat": "abandon", "abandon_payment": "abandon", "hold_expired_wait": "expired", "hold_rejected": "hold_rejected"}
FAST_S = 0.1


def load_v4():
    """(v3 설정, v4 설정). v3 설정 파일은 읽기만 한다."""
    v4 = load_config(CONFIG_PATH)
    return load_config(EXP / "config" / v4["extends"]), v4


def make_params(v4, dist, overrides=None):
    params = {key: v4["baseline"][key] for key in PARAM_KEYS if key != "speed_dist"}
    params["speed_dist"] = dist
    params.update(overrides or {})
    assert set(params) == set(PARAM_KEYS) and params["speed_dist"] in DISTS, params
    assert params["retry_wait"] in ("by_speed", "measured_end", "same_as_completion") and params["macro_delay"] in ("three_points", "loguniform", "slow_skewed")
    assert set(params["rules_off"]) <= {"S1", "S2"}
    return params


def draw_speed(rng, dist, fast_only):
    """정상 계정의 속도 수준 s (1 = 느린 끝, 0.1 = 빠른 끝). fast_only 면 같은 분포에서 s ≤ 10^-0.5 인 값만 받는다."""
    while True:
        if dist == "D1":
            s = max(FAST_S, math.exp(0.5 * rng.standard_normal()))
        elif dist == "D2":
            s = 10 ** rng.uniform(-1.0, 0.0)
        else:
            s = min(1.0, FAST_S * math.exp(abs(0.5 * rng.standard_normal())))
        if not fast_only or s <= 10 ** -0.5:
            return s


def assign_delay(accounts, base, v4, params, seed, levels):
    """매크로의 지연 설정. three_points 면 v3 의 배정 그대로다. 아니면 jitter 계정과 multi_account 링이 pause 를 연속 범위에서 뽑는다."""
    accounts["pause_ms"] = np.nan
    accounts["target_level_s"] = np.nan
    if params["macro_delay"] == "three_points":
        return accounts
    points = v4["macro"]["pause_ms_profile_mean"]
    profiles = ["fast", "jitter", "human"]
    pauses = [points[name] for name in profiles]
    gaps = [math.exp(levels["log_mean"][name]) for name in profiles]
    donors = {name: sorted(a for a, m in levels["account_mode"].items() if m == name) for name in profiles}
    low, high = math.log(pauses[0]), math.log(pauses[-1])
    for s_index, name in enumerate(base["scenarios"]):
        if name not in ("jitter", "multi_account"):
            continue
        rng = rng_for(seed, 7, s_index)
        members = accounts.index[accounts["scenario"] == name].to_numpy()
        units = [[i] for i in members] if name == "jitter" else [accounts.index[accounts["ip_group_id"] == g].to_numpy()
                                                              for g in dict.fromkeys(accounts.loc[members, "ip_group_id"])]
        for unit in units:
            position = rng.uniform() if params["macro_delay"] == "loguniform" else rng.beta(2, 1)
            pause = math.exp(low + (high - low) * position)
            profile = "fast" if pause <= 20 else min(("jitter", "human"), key=lambda p: abs(math.log(pause) - math.log(points[p])))
            accounts.loc[unit, "pause_ms"] = pause
            accounts.loc[unit, "target_level_s"] = float(np.interp(pause, pauses, gaps))
            accounts.loc[unit, "tool_profile"] = profile
            for k, index in enumerate(rng.permutation(unit)):
                accounts.loc[index, "donor_account"] = donors[profile][k % len(donors[profile])]
    return accounts


def finish_cancels(account, rng, spec, base, cancel_propensity):
    """완료된 예매의 취소 (v3 와 같다)."""
    delay, minimum = spec["cancel_delay"], base["cancel_delay"]["min_s"]
    for attempt in account.attempts:
        attempt.update(cancel_ms=None, cancel_after_cutoff=False)
        if attempt["booking_ms"] is None or rng.random() >= cancel_propensity:
            continue
        wait_s = max(minimum, delay["median_s"] * float(np.exp(delay["sigma"] * rng.standard_normal())))
        cancel_ms = attempt["booking_ms"] + round(wait_s * 1000)
        if cancel_ms > 0:
            attempt["cancel_after_cutoff"] = True
            continue
        attempt["cancel_ms"] = cancel_ms
        account.emit(cancel_ms, "CANCEL", {"concertId": attempt["concert"], "ticketIds": [attempt["booking_id"].replace("bk-", "tk-")]},
                     booking_id=attempt["booking_id"])


def queue_wait_ms(rng_q, v4, params):
    if params["queue_wait_s"] <= 0:
        return 0
    return max(1, round(1000 * params["queue_wait_s"] * math.exp(v4["queue_wait"]["sigma"] * rng_q.standard_normal())))


def simulate_normal(index, row, spec, base, v4, params, seed, fast_end, opens_ms):
    """정상 계정. 속도 수준 → 세션마다 목표 예매 수 → 실패하면 재시도."""
    rng, rng_speed, rng_skill, rng_q = (rng_for(seed, k, index) for k in (2, 4, 5, 6))
    account = Account(index, row, base["service_id"])
    hold_ms = int(base["hold_duration_s"]["value"] * 1000)
    speed = draw_speed(rng_speed, params["speed_dist"], row.scenario == "fast_normal")
    u = 1 + math.log10(speed)
    slow = v4["normal_speed"]["slow_end_stage_median_s"]
    median = {key: fast_end["stage_median_s"][key] * (slow[key] / fast_end["stage_median_s"][key]) ** u for key in SPEED_STAGES}
    median["completed"] = v4["normal_speed"]["completed_s"]["value"]
    idle_fast, idle_slow = v4["idle_s"]["fast_end"], v4["idle_s"]["slow_end"]

    def idle_median(kind):
        if params["retry_wait"] == "measured_end":
            return idle_fast[kind]
        if params["retry_wait"] == "same_as_completion":
            kind = "complete"
        return idle_fast[kind] * (idle_slow[kind] / idle_fast[kind]) ** u

    session = v4["normal_session"]
    names, probs, complete_prob = outcome_table(rng, spec)
    cancel_propensity = draw(rng, spec["cancel_propensity"])
    sigma_goal = 0.8
    goal_rate = params["goal_lambda"] * session["goal"]["scenario_weight"][row.scenario] * math.exp(sigma_goal * rng.standard_normal() - sigma_goal ** 2 / 2)
    participation = None
    if "S1" in params["rules_off"]:
        n_sessions = 1
    else:
        participation = float(rng.beta(*base["participation"]["q_beta"]))
        n_sessions = 1 + int(rng.binomial(len(opens_ms) - 1, participation))
    chosen = np.sort(rng.choice(len(opens_ms), size=n_sessions, replace=False))
    slow_first, slow_factor = 0, 1.0
    if params["skill"]:
        slow_first = int(rng_skill.integers(v4["skill"]["k"][0], v4["skill"]["k"][1] + 1))
        slow_factor = float(rng_skill.uniform(*v4["skill"]["multiplier"]))
    floors = base["stage_floor_s"]
    contention, error = base["seat_contention"], base["payment_error"]
    context = {"tool": False, "macro_max_tries": contention["macro_max_tries"]["value"], "patience": contention["human_patience"],
               "repick": contention["repick_fraction"], "payment_error": error if row.scenario in error["scenarios"] else None}
    sigma = params["within_sigma"]

    number = goals = 0
    for session_no, open_index in enumerate(chosen, start=1):
        open_concert, open_ms = opens_ms[open_index]
        concert = open_concert
        goal = 1 + int(rng.poisson(goal_rate))
        goals += goal
        arrival = spec["arrival"]
        start = open_ms + max(1, round(rng.exponential(arrival["mean_s"]) * 1000))
        done = tried = 0
        while done < goal and tried < session["max_attempts_per_session"]["value"]:
            number += 1
            tried += 1
            outcome = str(rng.choice(names, p=probs))
            noise = np.exp(sigma * rng.standard_normal(len(STAGE_KEYS)))  # queue, pre_seat, seat, hold_to_payment, payment, completed, idle
            timing = {key: max(1, round(1000 * (floors.get(key, 0.0) + max(median[key] - floors.get(key, 0.0), 0.001) * z)))
                      for key, z in zip(STAGE_KEYS[:6], noise[:6])}
            if number <= slow_first:
                timing["seat"], timing["payment"] = round(timing["seat"] * slow_factor), round(timing["payment"] * slow_factor)
            in_rush = concert == open_concert
            wait = queue_wait_ms(rng_q, v4, params) if in_rush else 0
            timing["queue"] += wait
            timing["hold_to_booking"] = timing["hold_to_payment"] + timing["payment"]
            context["hazard"] = contention["hazard_per_s"][concert] if in_rush else session["post_rush_hazard_per_s"]["value"]
            free_at = run_attempt(account, rng, start, concert, number, timing, outcome, hold_ms, context)
            last = account.attempts[-1]
            last.update(session_no=session_no, open_concert=open_concert, queue_wait_ms=wait)
            if last["outcome"] == "complete":
                done += 1
            elif rng.random() >= session["retry_prob"]["value"]:
                break
            start = free_at + max(1, round(1000 * idle_median(IDLE_KIND[last["outcome"]]) * noise[6]))
            if params["switch_prob"] > 0 and rng.random() < params["switch_prob"]:
                others = [name for name, opened in opens_ms if opened <= start and name != concert]
                if others:
                    concert = str(rng.choice(others))
    finish_cancels(account, rng, spec, base, cancel_propensity)
    traits = {"n_attempts": len(account.attempts), "n_sessions": n_sessions, "concerts": "|".join(opens_ms[i][0] for i in chosen),
              "speed_s": speed, "goal_bookings": goals, "goal_rate": goal_rate, "complete_prob": complete_prob, "cancel_propensity": cancel_propensity,
              "participation_q": participation, "skill_slow_first": slow_first, "skill_factor": slow_factor}
    return account, traits


def simulate_other(index, row, spec, base, v4, params, seed, reference, opens_ms, levels):
    """매크로(실측 시도 복원추출)와 hold_cancel. v3 의 규칙 그대로이고, human 프로필의 참조와 대기열 대기만 다르다."""
    rng, rng_q = rng_for(seed, 2, index), rng_for(seed, 6, index)
    account = Account(index, row, base["service_id"])
    hold_ms = int(base["hold_duration_s"]["value"] * 1000)
    tool = row.tool_profile
    attempts_spec = spec["attempts"]
    n_attempts = draw_attempts(rng, spec["attempts_v2"] if (attempts_spec["dist"] == "loguniform_int" and "S2" in params["rules_off"]) else attempts_spec)
    names, probs, complete_prob = outcome_table(rng, spec)
    cancel_propensity = draw(rng, spec["cancel_propensity"])
    pace = 1.0 if tool else float(np.exp(spec["account_pace_sigma"]["value"] * rng.standard_normal()))
    participation = None
    if "S1" in params["rules_off"]:
        n_sessions = 1
    else:
        participation = float(rng.beta(*base["participation"]["q_beta"]))
        n_sessions = min(n_attempts, 1 + int(rng.binomial(len(opens_ms) - 1, participation)))
    chosen = np.sort(rng.choice(len(opens_ms), size=n_sessions, replace=False))
    per_session = np.ones(n_sessions, dtype=int) + rng.multinomial(n_attempts - n_sessions, np.ones(n_sessions) / n_sessions)

    time_scale, account_level = 1.0, None
    if tool:
        donor = reference[reference["account"] == row.donor_account].reset_index(drop=True)
        with_gap = donor.index[donor["gap_to_next_ms"].notna()].to_numpy()
        center = math.log(row.target_level_s) if not pd.isna(row.target_level_s) else levels["log_mean"][tool]
        account_level = float(np.exp(center + levels["log_sd"] * rng.standard_normal()))
        time_scale = account_level / levels["account_mean_s"][row.donor_account]
    floors = base["stage_floor_s"]
    contention = base["seat_contention"]
    context = {"tool": bool(tool), "macro_max_tries": contention["macro_max_tries"]["value"], "patience": contention["human_patience"],
               "repick": contention["repick_fraction"], "payment_error": None}
    scale = lambda value: max(1, round(value * time_scale))

    number = 0
    for session_no, (open_index, count) in enumerate(zip(chosen, per_session), start=1):
        concert, open_ms = opens_ms[open_index]
        context["hazard"] = contention["hazard_per_s"][concert]
        arrival = spec["arrival"]
        wait_s = rng.exponential(arrival["mean_s"]) if arrival["dist"] == "exponential" else rng.uniform(arrival["low_s"], arrival["high_s"])
        start = open_ms + max(1, round(wait_s * 1000))
        for _ in range(count):
            number += 1
            outcome = str(rng.choice(names, p=probs))
            if tool:
                pick = donor.loc[int(rng.choice(with_gap))]  # 실측 시도 1건: 단계 시간과 다음 시도까지의 대기를 함께 가져온다
                h2p = None if pd.isna(pick["hold_to_payment_ms"]) else scale(int(pick["hold_to_payment_ms"]))
                timing = {key: scale(int(pick[f"{key}_ms"])) for key in CYCLE}
                timing["hold_to_payment"] = h2p
                timing["payment"] = timing["hold_to_booking"] - (h2p or 0)
                timing["idle"] = scale(int(pick["gap_to_next_ms"]) - sum(int(pick[f"{key}_ms"]) for key in CYCLE))
            else:
                noise = np.exp(spec["within_sigma"]["value"] * rng.standard_normal(len(STAGE_KEYS)))
                timing = {key: max(1, round(1000 * (floors.get(key, 0.0) + max(spec["stages_median_s"][key] - floors.get(key, 0.0), 0.001) * pace * z)))
                          for key, z in zip(STAGE_KEYS, noise)}
                timing["hold_to_booking"] = timing["hold_to_payment"] + timing["payment"]
            wait = queue_wait_ms(rng_q, v4, params)
            timing["queue"] += wait
            free_at = run_attempt(account, rng, start, concert, number, timing, outcome, hold_ms, context)
            account.attempts[-1].update(session_no=session_no, open_concert=concert, queue_wait_ms=wait)
            start = free_at + timing["idle"]
            if tool and account.attempts[-1]["outcome"] == "hold_rejected":
                break
    finish_cancels(account, rng, spec, base, cancel_propensity)
    traits = {"n_attempts": len(account.attempts), "n_attempts_planned": n_attempts, "n_sessions": n_sessions,
              "concerts": "|".join(opens_ms[i][0] for i in chosen), "pace_multiplier": pace, "complete_prob": complete_prob,
              "cancel_propensity": cancel_propensity, "participation_q": participation, "account_level_s": account_level,
              "time_scale": time_scale if tool else None}
    return account, traits


def generate(dist="D1", overrides=None, seed=None, configs=None):
    """이벤트·계정·시도 표와 쓰인 params 를 돌려준다. 파일은 쓰지 않는다."""
    base, v4 = configs or load_v4()
    params = make_params(v4, dist, overrides)
    seed = base["seed"] if seed is None else seed
    end = datetime.fromisoformat(base["calculated_at_utc"])
    opens_ms = [(item["concertId"], round((datetime.fromisoformat(item["open_utc"]) - end).total_seconds() * 1000)) for item in base["concert_opens"]["items"]]
    fast_end = load_config(EXP / v4["normal_speed"]["fast_end"]["file"])
    human = pd.read_csv(EXP / v4["macro"]["human_profile_reference"]["file"])
    accounts, reference = build_accounts(load_assignment(), base, seed)
    accounts = assign_tools(accounts, base, seed, reference, human, off=set())  # 링마다 프로필 하나 (v3 S4)
    reference = pd.concat([reference, human], ignore_index=True)
    levels = macro_levels(reference)  # human 은 계정이 1개라 계정 간 표준편차에는 영향이 없다
    levels["account_mode"] = reference.drop_duplicates("account").set_index("account")["mode"].to_dict()
    accounts = assign_delay(accounts, base, v4, params, seed, levels)

    raw_events, attempt_rows, trait_rows = [], [], []
    for index, row in enumerate(accounts.itertuples(index=False)):
        spec = base["scenarios"][row.scenario]
        if row.is_anomaly == 0:
            account, traits = simulate_normal(index, row, spec, base, v4, params, seed, fast_end, opens_ms)
        else:
            account, traits = simulate_other(index, row, spec, base, v4, params, seed, reference, opens_ms, levels)
        raw_events.extend(account.events)
        attempt_rows.extend(account.attempts)
        trait_rows.append(traits)
    accounts = pd.concat([accounts, pd.DataFrame(trait_rows)], axis=1)

    raw_events.sort(key=lambda e: e[:3])
    stamp = lambda ms: end + timedelta(milliseconds=int(ms))
    users, scenarios = accounts["user_id"].to_numpy(), accounts["scenario"].to_numpy()
    rows = []
    for number, (t, i, n, event_type, booking_id, ticket_id, metadata) in enumerate(raw_events, start=1):
        at = stamp(t)
        if event_type == "BOOKING_ATTEMPT":  # 실측 이벤트에 있는 필드 (값은 occurredAt 과 같다)
            iso = at.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            metadata = {**metadata, "recordedAt": iso, "transitionId": f"tr-{i:04d}-{n:04d}"}
            if metadata["stage"] == "STARTED":
                metadata["clientOccurredAt"] = iso
        rows.append({"event_id": number, "user_id": users[i], "external_user_id": users[i], "group": scenarios[i], "service_id": base["service_id"],
                     "external_event_id": f"evt-{i:04d}-{n:04d}", "event_type": event_type, "booking_id": booking_id, "ticket_id": ticket_id,
                     "occurred_at": at, "meta": metadata})
    events = pd.DataFrame(rows)

    attempts = pd.DataFrame(attempt_rows)
    for column in ["start_ms", "booking_ms", "cancel_ms"]:
        attempts[column.replace("_ms", "_utc")] = attempts[column].map(lambda ms: None if pd.isna(ms) else stamp(ms).isoformat(timespec="milliseconds"))
    attempts = attempts[["user_id", "attempt_id", "concert", "open_concert", "session_no", "start_utc", "outcome", "seat_ms", "queue_wait_ms", "booking_id",
                         "booking_utc", "cancel_utc", "cancel_after_cutoff", "hold_rejections", "payment_errors"]]
    return events, accounts, attempts, params


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="기본값은 v3 config 의 seed (42)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    base, v4 = load_v4()
    seed = base["seed"] if args.seed is None else args.seed
    root = EXP / "data" / "v4"
    protected = {"original": ORIGINAL_PATH, "issue16_v2": V2_PATH, "event_v2_events": EXP / "data" / "v2" / "events.csv",
                 "event_v3_events": EXP / "data" / "v3" / "events.csv", "event_v3_features": EXP / "data" / "v3" / "fairticket_event_based_features.csv",
                 "v3_config": V3_CONFIG_PATH}
    before = {name: sha256(path) for name, path in protected.items()}
    for dist in DISTS:
        out = root / dist
        if out.exists() and any(out.iterdir()):
            record_path = out / "generation.json"
            if not args.force:
                sys.exit(f"이미 있는 폴더입니다: {out}\n--force 를 주면 다시 씁니다.")
            if not record_path.exists() or json.loads(record_path.read_text(encoding="utf-8")).get("config_version") != "v4":
                sys.exit(f"{out} 은 v4 가 만든 폴더가 아니어서 덮어쓰지 않았습니다.")
        out.mkdir(parents=True, exist_ok=True)
        events, accounts, attempts, params = generate(dist, seed=seed, configs=(base, v4))
        to_csv_frame(events).to_csv(out / "events.csv", index=False, encoding="utf-8", lineterminator="\n")
        accounts.to_csv(out / "accounts.csv", index=False, encoding="utf-8", lineterminator="\n")
        attempts.to_csv(out / "attempts.csv", index=False, encoding="utf-8", lineterminator="\n")
        normal = accounts[accounts["is_anomaly"] == 0]
        record = {
            "speed_assumption": dist, "speed_assumption_label": v4["normal_speed"]["distributions"][dist]["label"],
            "note": "D1·D2·D3 는 정상 사용자 사이의 속도 분포에 대한 가정이다. 실제 분포가 아니다.",
            "params": params, "seed": seed, "rng": "numpy.random.default_rng(SeedSequence([seed, 단계, 번호]))", "numpy": np.__version__, "pandas": pd.__version__,
            "generator": "scripts/generate_events_v4.py", "config": CONFIG_PATH.name, "config_version": v4["version"], "config_sha256": sha256(CONFIG_PATH),
            "base_config": v4["extends"], "base_config_sha256": sha256(V3_CONFIG_PATH), "reference_sha256": sha256(REFERENCE_PATH),
            "human_reference_sha256": sha256(EXP / v4["macro"]["human_profile_reference"]["file"]),
            "fast_end_sha256": sha256(EXP / v4["normal_speed"]["fast_end"]["file"]),
            "account_source": V2_PATH.relative_to(REPO).as_posix(), "calculated_at_utc": base["calculated_at_utc"], "window_days": base["window_days"],
            "accounts": int(len(accounts)), "accounts_by_scenario": accounts["scenario"].value_counts().to_dict(), "anomaly_ratio": float(accounts["is_anomaly"].mean()),
            "normal_speed_s_quantiles": {f"p{int(q * 100):02d}": round(float(normal["speed_s"].quantile(q)), 3) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "normal_accounts_within_2x_of_fast_end": float((normal["speed_s"] <= 0.2).mean()),
            "accounts_by_sessions": {int(k): int(v) for k, v in accounts["n_sessions"].value_counts().sort_index().items()},
            "multi_account_rings_by_tool_profile": accounts[accounts["scenario"] == "multi_account"].groupby("tool_profile")["ip_group_id"].nunique().to_dict(),
            "events": int(len(events)), "events_by_type": events["event_type"].value_counts().to_dict(),
            "attempts": int(len(attempts)), "attempts_by_outcome": attempts["outcome"].value_counts().to_dict(),
            "normal_attempts_on_other_concert": int((attempts["concert"] != attempts["open_concert"]).sum()),
            "hold_rejections": int(attempts["hold_rejections"].sum()), "payment_errors": int(attempts["payment_errors"].sum()),
            "bookings": int((attempts["booking_id"] != "").sum()), "cancels": int(attempts["cancel_utc"].notna().sum()),
            "cancels_after_cutoff_not_emitted": int(attempts["cancel_after_cutoff"].sum()),
            "occurred_at_min": events["occurred_at"].min().isoformat(), "occurred_at_max": events["occurred_at"].max().isoformat(),
            "files_sha256": {name: sha256(out / name) for name in ["events.csv", "accounts.csv", "attempts.csv"]},
        }
        record["existing_datasets_unchanged"] = before == {name: sha256(path) for name, path in protected.items()}
        (out / "generation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: record[k] for k in ("speed_assumption", "normal_speed_s_quantiles", "normal_accounts_within_2x_of_fast_end", "events", "attempts",
                                                 "attempts_by_outcome", "normal_attempts_on_other_concert", "bookings", "cancels", "existing_datasets_unchanged")},
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
