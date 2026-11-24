"""이벤트 기반 더미데이터 v3 의 이벤트를 생성한다. Feature 는 여기서 정하지 않는다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\generate_events_v3.py            # data/v3
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\generate_events_v3.py --force

v1·v2 의 생성기(scripts/generate_events.py)는 수정하지 않고 import 해서 쓴다: 계정 목록, 공유 관계(IP 묶음·카드), 실측 계정 배정,
이벤트 형식, 계정 수준 효과(v2 C1), 단계 하한(v2 C2), hold_cancel 의 미결제 선점(v2 C3)은 v2 와 같다.
같은 seed 면 IP 묶음과 카드 묶음이 v2 와 같다.

v3 에서 바뀌는 규칙 (config/scenarios_v3.json 의 changes_from_v2, 값과 근거는 그 파일에 있다)
  S1 티켓 오픈 참여 횟수가 계정마다 다르다
  S2 매크로의 시도 수가 1건부터다
  S3 정상 사용자의 시도 수가 계정마다 더 넓게 퍼진다
  S4 다계정 링이 매크로 도구의 프로필(fast / jitter / human) 하나를 쓴다
  S5 좌석 선점이 거절될 수 있다 (같은 attemptId 안의 SEAT_HOLD/REJECTED)
  S6 사람의 결제 입력 오류 (이벤트를 남기지 않는다)
`off` 로 규칙을 하나씩 끄면 그 규칙만 v2 의 동작으로 돌아간다 (어떤 가정이 결과를 바꾸는지 확인하는 용도).

출력: data/v3/events.csv, accounts.csv, attempts.csv, generation.json
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

CONFIG_PATH = EXP / "config" / "scenarios_v3.json"
HUMAN_REFERENCE_PATH = EXP / "reference" / "macro_human_attempt_cycles.csv"
RULES = ("S1", "S2", "S3", "S4", "S5", "S6")
CYCLE = ("queue", "pre_seat", "seat", "hold_to_booking", "completed")


def draw_attempts(rng, spec):
    kind = spec["dist"]
    if kind == "poisson_plus1_overdispersed":  # 평균이 lam 인 계정별 강도
        sigma = spec["sigma"]
        return int(rng.poisson(spec["lam"] * math.exp(sigma * rng.standard_normal() - sigma ** 2 / 2))) + 1
    if kind == "loguniform_int":
        return int(round(math.exp(rng.uniform(math.log(spec["low"]), math.log(spec["high"])))))
    return draw(rng, spec)


def assign_tools(accounts, config, seed, reference, human, off):
    """매크로 도구의 프로필과 실측 계정. fast·jitter 시나리오는 v2 의 배정 그대로, 다계정 링은 링마다 프로필 하나."""
    accounts["tool_profile"] = ""
    for s_index, (name, spec) in enumerate(config["scenarios"].items()):
        members = accounts["scenario"] == name
        if spec["timing_source"] == "measured":
            accounts.loc[members, "tool_profile"] = spec["measured_mode"]
        elif spec["timing_source"] == "tool" and "S4" not in off:
            rng = rng_for(seed, 3, s_index)
            choices = spec["tool_profile"]["choices"]
            for group in dict.fromkeys(accounts.loc[members, "ip_group_id"]):
                profile = str(rng.choice(choices))
                rows = accounts.index[accounts["ip_group_id"] == group].to_numpy()
                source = human if profile == "human" else reference[reference["mode"] == profile]
                donors = sorted(source["account"].unique())
                accounts.loc[rows, "tool_profile"] = profile
                for position, index in enumerate(rng.permutation(rows)):
                    accounts.loc[index, "donor_account"] = donors[position % len(donors)]
    return accounts


def run_attempt(account, rng, start_ms, concert, number, timing, outcome, hold_ms, context):
    """시도 1건의 이벤트. v2 의 run_attempt 에 선점 거절(S5)과 결제 입력 오류(S6)를 더한 것이다. 반환: 이 시도를 떠나는 시각."""
    row = account.row
    tag = f"{account.index:04d}-{number:03d}"
    attempt = {"user_id": row.user_id, "attempt_id": f"att-{tag}", "concert": concert, "sequence": 0, "start_ms": start_ms, "outcome": outcome,
               "seat_ms": None, "booking_id": "", "booking_ms": None, "hold_rejections": 0, "payment_errors": 0}
    account.attempts.append(attempt)
    t = start_ms
    account.stage(t, attempt, "STARTED", "ENTERED", "CLIENT_REPORT")
    t += timing["queue"]
    account.stage(t, attempt, "QUEUE", "SUCCEEDED", "SERVER")
    t += timing["pre_seat"]
    account.stage(t, attempt, "SEAT_SELECTION", "ENTERED", "CLIENT_REPORT")
    if outcome == "abandon_seat":
        t += max(1, round(timing["seat"] * rng.uniform(0.5, 2.0)))
        account.stage(t, attempt, "SEAT_SELECTION", "ABANDONED", "CLIENT_REPORT", reasonCode="USER_CANCELLED")
        return t

    # 좌석을 고르고 선점을 요청한다. 고르는 동안 그 좌석이 넘어갔으면 거절되고(같은 attemptId), 다른 좌석을 다시 고른다
    hazard, pick_ms, spent, tries = context["hazard"], timing["seat"], 0, 0
    max_tries = context["macro_max_tries"] if context["tool"] else (draw(rng, context["patience"]) if hazard > 0 else 1)
    while True:
        t += pick_ms
        spent += pick_ms
        tries += 1
        if hazard > 0 and rng.random() < 1 - math.exp(-hazard * pick_ms / 1000):
            account.stage(t, attempt, "SEAT_HOLD", "REJECTED", "SERVER", reasonCode="SEAT_ALREADY_TAKEN")  # 선점이 없으므로 holdId 도 없다
            attempt["hold_rejections"] += 1
            if tries >= max_tries:  # 그만둔다. 거절 뒤에는 이탈 보고가 받아들여지지 않아 마지막 이벤트는 거절이다
                attempt["outcome"] = "hold_rejected"
                return t
            if not context["tool"]:
                pick_ms = max(1, round(timing["seat"] * rng.uniform(context["repick"]["low"], context["repick"]["high"])))
            continue
        break
    attempt["seat_ms"] = spent
    hold_at, hold_id = t, f"hold-{tag}"
    account.stage(t, attempt, "SEAT_HOLD", "SUCCEEDED", "SERVER", holdId=hold_id)
    if outcome == "hold_left":
        account.stage(hold_at + hold_ms, attempt, "SEAT_HOLD", "EXPIRED", "SERVER", holdId=hold_id, reasonCode="HOLD_EXPIRED")
        return t
    if timing["hold_to_payment"] is not None:
        account.stage(t + timing["hold_to_payment"], attempt, "PAYMENT", "ENTERED", "CLIENT_REPORT", holdId=hold_id)

    hold_to_booking = timing["hold_to_booking"]
    error = context["payment_error"]
    if outcome == "complete" and error is not None and rng.random() < error["prob"]:
        attempt["payment_errors"] = 1  # 결제 실패는 이벤트를 남기지 않는다
        if rng.random() < error["retry_prob"]:
            hold_to_booking += timing["payment"]  # 다시 입력
            if hold_to_booking >= hold_ms:
                outcome = "hold_expired_wait"
        else:
            outcome = "abandon_payment"
        attempt["outcome"] = outcome
    if outcome == "abandon_payment":
        t += timing["hold_to_payment"] + max(1, round(timing["payment"] * rng.uniform(0.2, 1.0)))
        account.stage(t, attempt, "PAYMENT", "ABANDONED", "CLIENT_REPORT", holdId=hold_id, reasonCode="USER_CANCELLED")
        return t
    if outcome == "hold_expired_wait":
        t = hold_at + hold_ms
        account.stage(t, attempt, "SEAT_HOLD", "EXPIRED", "SERVER", holdId=hold_id, reasonCode="HOLD_EXPIRED")
        return t
    t = hold_at + hold_to_booking
    booking_id, ticket_id = f"bk-{tag}", f"tk-{tag}"
    attempt.update(booking_id=booking_id, booking_ms=t)
    account.emit(t, "BOOKING", {"attemptId": attempt["attempt_id"], "concertId": concert, "ipAddress": row.ip_address,
                                "ticketIds": [ticket_id], "paymentIdentifier": row.payment_identifier,
                                "bookingDurationSeconds": hold_to_booking / 1000}, booking_id=booking_id)
    t += timing["completed"]
    account.stage(t, attempt, "COMPLETED", "SUCCEEDED", "SERVER", holdId=hold_id, bookingId=booking_id)
    return t


def simulate_account(index, row, spec, config, seed, reference, human, fast_idle, opens_ms, levels, off):
    rng = rng_for(seed, 2, index)
    account = Account(index, row, config["service_id"])
    hold_ms = int(config["hold_duration_s"]["value"] * 1000)
    tool = row.tool_profile

    # --- 계정의 행동 성향 ---
    attempts_spec = spec["attempts"]
    rule = {"loguniform_int": "S2", "poisson_plus1_overdispersed": "S3"}.get(attempts_spec["dist"])
    n_attempts = draw_attempts(rng, spec["attempts_v2"] if rule in off else attempts_spec)
    names, probs, complete_prob = outcome_table(rng, spec)
    cancel_propensity = draw(rng, spec["cancel_propensity"])
    pace = 1.0 if tool else float(np.exp(spec["account_pace_sigma"]["value"] * rng.standard_normal()))
    participation = None
    if "S1" in off:
        n_sessions = 1
    else:
        participation = float(rng.beta(*config["participation"]["q_beta"]))
        n_sessions = min(n_attempts, 1 + int(rng.binomial(len(opens_ms) - 1, participation)))
    chosen = np.sort(rng.choice(len(opens_ms), size=n_sessions, replace=False))
    per_session = np.ones(n_sessions, dtype=int) + rng.multinomial(n_attempts - n_sessions, np.ones(n_sessions) / n_sessions)

    time_scale, account_level = 1.0, None
    if tool == "human":
        donor = human[human["account"] == row.donor_account].reset_index(drop=True)
        time_scale = float(np.exp(levels["log_sd"] * rng.standard_normal()))  # 계정 간 편차는 #16 Fast/Jitter 의 값을 그대로 쓴다
    elif tool:
        donor = reference[reference["account"] == row.donor_account].reset_index(drop=True)
        with_gap = donor.index[donor["gap_to_next_ms"].notna()].to_numpy()
        account_level = float(np.exp(levels["log_mean"][tool] + levels["log_sd"] * rng.standard_normal()))
        time_scale = account_level / levels["account_mean_s"][row.donor_account]
    floors = config.get("stage_floor_s", {})
    contention = config["seat_contention"]
    error = config["payment_error"]
    context = {"tool": bool(tool), "macro_max_tries": contention["macro_max_tries"]["value"], "patience": contention["human_patience"],
               "repick": contention["repick_fraction"],
               "payment_error": error if ("S6" not in off and row.scenario in error["scenarios"]) else None}
    scale = lambda value: max(1, round(value * time_scale))

    # --- 오픈마다 접속해 시도를 이어 간다 ---
    number = 0
    for open_index, count in zip(chosen, per_session):
        concert, open_ms = opens_ms[open_index]
        context["hazard"] = 0.0 if "S5" in off else contention["hazard_per_s"][concert]
        arrival = spec["arrival"]
        wait_s = rng.exponential(arrival["mean_s"]) if arrival["dist"] == "exponential" else rng.uniform(arrival["low_s"], arrival["high_s"])
        start = open_ms + max(1, round(wait_s * 1000))
        for _ in range(count):
            number += 1
            outcome = str(rng.choice(names, p=probs))
            if tool:
                if tool == "human":
                    pick = donor.loc[int(rng.integers(len(donor)))]
                    idle = int(rng.choice(fast_idle)) + int(round(rng.uniform(*config["human_profile"]["pause_ms"], config["human_profile"]["idle_pauses"]).sum()))
                    h2p = None
                else:
                    pick = donor.loc[int(rng.choice(with_gap))]  # 실측 시도 1건: 단계 시간과 다음 시도까지의 대기를 함께 가져온다
                    h2p = None if pd.isna(pick["hold_to_payment_ms"]) else scale(int(pick["hold_to_payment_ms"]))
                    idle = int(pick["gap_to_next_ms"]) - sum(int(pick[f"{key}_ms"]) for key in CYCLE)
                timing = {key: scale(int(pick[f"{key}_ms"])) for key in CYCLE}
                timing["hold_to_payment"] = h2p
                timing["payment"] = timing["hold_to_booking"] - (h2p or 0)
                timing["idle"] = scale(idle)
            else:
                noise = np.exp(spec["within_sigma"]["value"] * rng.standard_normal(len(STAGE_KEYS)))
                timing = {key: max(1, round(1000 * (floors.get(key, 0.0) + max(spec["stages_median_s"][key] - floors.get(key, 0.0), 0.001) * pace * z)))
                          for key, z in zip(STAGE_KEYS, noise)}
                timing["hold_to_booking"] = timing["hold_to_payment"] + timing["payment"]
            free_at = run_attempt(account, rng, start, concert, number, timing, outcome, hold_ms, context)
            start = free_at + timing["idle"]
            if tool and account.attempts[-1]["outcome"] == "hold_rejected":
                break  # 매크로는 선점에 5번 실패하면 그 실행을 멈춘다

    # --- 완료된 예매의 취소 (v2 와 같다) ---
    delay, minimum = spec["cancel_delay"], config["cancel_delay"]["min_s"]
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
    traits = {"n_attempts": len(account.attempts), "n_attempts_planned": n_attempts, "n_sessions": n_sessions,
              "concerts": "|".join(opens_ms[i][0] for i in chosen), "pace_multiplier": pace, "complete_prob": complete_prob,
              "cancel_propensity": cancel_propensity, "participation_q": participation, "account_level_s": account_level,
              "time_scale": time_scale if tool else None}
    return account, traits


def generate(config=None, seed=None, assignment=None, reference=None, off=()):
    """이벤트·계정·시도 표를 만든다. 파일은 쓰지 않는다. off: 끌 규칙 (예: ("S1",))."""
    config = config or load_config(CONFIG_PATH)
    off = set(off)
    assert off <= set(RULES), off
    seed = config["seed"] if seed is None else seed
    assignment = load_assignment() if assignment is None else assignment
    end = datetime.fromisoformat(config["calculated_at_utc"])
    opens_ms = [(item["concertId"], round((datetime.fromisoformat(item["open_utc"]) - end).total_seconds() * 1000))
                for item in config["concert_opens"]["items"]]
    accounts, reference = build_accounts(assignment, config, seed, reference)
    human = pd.read_csv(HUMAN_REFERENCE_PATH)
    accounts = assign_tools(accounts, config, seed, reference, human, off)
    levels = macro_levels(reference)
    fast = reference[(reference["mode"] == "fast") & reference["gap_to_next_ms"].notna()]
    fast_idle = (fast["gap_to_next_ms"] - fast[[f"{key}_ms" for key in CYCLE]].sum(axis=1)).to_numpy(dtype="int64")

    raw_events, attempt_rows, trait_rows = [], [], []
    for index, row in enumerate(accounts.itertuples(index=False)):
        account, traits = simulate_account(index, row, config["scenarios"][row.scenario], config, seed, reference, human, fast_idle, opens_ms, levels, off)
        raw_events.extend(account.events)
        attempt_rows.extend(account.attempts)
        trait_rows.append(traits)
    accounts = pd.concat([accounts, pd.DataFrame(trait_rows)], axis=1)

    raw_events.sort(key=lambda e: e[:3])
    stamp = lambda ms: end + timedelta(milliseconds=int(ms))
    users, scenarios = accounts["user_id"].to_numpy(), accounts["scenario"].to_numpy()
    events = pd.DataFrame([{
        "event_id": number, "user_id": users[i], "external_user_id": users[i], "group": scenarios[i],
        "service_id": config["service_id"], "external_event_id": f"evt-{i:04d}-{n:04d}", "event_type": event_type,
        "booking_id": booking_id, "ticket_id": ticket_id, "occurred_at": stamp(t), "meta": metadata,
    } for number, (t, i, n, event_type, booking_id, ticket_id, metadata) in enumerate(raw_events, start=1)])

    attempts = pd.DataFrame(attempt_rows)
    for column in ["start_ms", "booking_ms", "cancel_ms"]:
        attempts[column.replace("_ms", "_utc")] = attempts[column].map(lambda ms: None if pd.isna(ms) else stamp(ms).isoformat(timespec="milliseconds"))
    attempts = attempts[["user_id", "attempt_id", "concert", "start_utc", "outcome", "seat_ms", "booking_id", "booking_utc",
                         "cancel_utc", "cancel_after_cutoff", "hold_rejections", "payment_errors"]]
    return events, accounts, attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="기본값은 config 의 seed (42)")
    parser.add_argument("--out", default=None, help="출력 폴더 (기본 data/v3)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(CONFIG_PATH)
    seed = config["seed"] if args.seed is None else args.seed
    out = Path(args.out) if args.out else EXP / "data" / "v3"
    if out.exists() and any(out.iterdir()):
        record_path = out / "generation.json"
        if not args.force:
            sys.exit(f"이미 있는 폴더입니다: {out}\n--force 를 주면 다시 씁니다.")
        if not record_path.exists() or json.loads(record_path.read_text(encoding="utf-8")).get("config_version") != "v3":
            sys.exit(f"{out} 은 v3 가 만든 폴더가 아니어서 덮어쓰지 않았습니다.")
    out.mkdir(parents=True, exist_ok=True)
    protected = {"original": ORIGINAL_PATH, "issue16_v2": V2_PATH, "event_v2": EXP / "data" / "v2" / "fairticket_event_based_features.csv",
                 "event_v2_events": EXP / "data" / "v2" / "events.csv"}
    before = {name: sha256(path) for name, path in protected.items()}

    events, accounts, attempts = generate(config, seed)
    to_csv_frame(events).to_csv(out / "events.csv", index=False, encoding="utf-8", lineterminator="\n")
    accounts.to_csv(out / "accounts.csv", index=False, encoding="utf-8", lineterminator="\n")
    attempts.to_csv(out / "attempts.csv", index=False, encoding="utf-8", lineterminator="\n")
    record = {
        "seed": seed, "rng": "numpy.random.default_rng(SeedSequence([seed, 단계, 번호]))", "numpy": np.__version__, "pandas": pd.__version__,
        "generator": "scripts/generate_events_v3.py", "config": CONFIG_PATH.name, "config_version": config["version"],
        "config_sha256": sha256(CONFIG_PATH), "reference_sha256": sha256(REFERENCE_PATH), "human_reference_sha256": sha256(HUMAN_REFERENCE_PATH),
        "account_source": V2_PATH.relative_to(REPO).as_posix(), "calculated_at_utc": config["calculated_at_utc"], "window_days": config["window_days"],
        "accounts": int(len(accounts)), "accounts_by_scenario": accounts["scenario"].value_counts().to_dict(),
        "anomaly_ratio": float(accounts["is_anomaly"].mean()),
        "accounts_by_sessions": {int(k): int(v) for k, v in accounts["n_sessions"].value_counts().sort_index().items()},
        "multi_account_rings_by_tool_profile": accounts[accounts["scenario"] == "multi_account"].groupby("tool_profile")["ip_group_id"].nunique().to_dict(),
        "events": int(len(events)), "events_by_type": events["event_type"].value_counts().to_dict(),
        "attempts": int(len(attempts)), "attempts_by_outcome": attempts["outcome"].value_counts().to_dict(),
        "attempts_with_hold_rejection": int((attempts["hold_rejections"] > 0).sum()), "hold_rejections": int(attempts["hold_rejections"].sum()),
        "payment_errors": int(attempts["payment_errors"].sum()),
        "bookings": int((attempts["booking_id"] != "").sum()), "cancels": int(attempts["cancel_utc"].notna().sum()),
        "cancels_after_cutoff_not_emitted": int(attempts["cancel_after_cutoff"].sum()),
        "occurred_at_min": events["occurred_at"].min().isoformat(), "occurred_at_max": events["occurred_at"].max().isoformat(),
        "files_sha256": {name: sha256(out / name) for name in ["events.csv", "accounts.csv", "attempts.csv"]},
        "existing_datasets_sha256": before,
        "existing_datasets_unchanged": before == {name: sha256(path) for name, path in protected.items()},
    }
    (out / "generation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
