"""계정별 행동 이벤트(behavior_event 형식)를 시간순으로 생성한다. Feature 는 여기서 정하지 않는다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\generate_events.py
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\generate_events.py --profile multi_session

순서
  1. 계정 목록(user_id, scenario, is_anomaly)은 #16 v2 데이터셋에서 그대로 가져온다 (같은 1,000계정, 같은 시나리오 수, 같은 레이블).
  2. 공유 관계를 먼저 만든다: 같은 IP 를 쓰는 계정 묶음(가구, 공용 IP, 운영자, 다계정 링)과 카드.
  3. 계정마다 행동 성향(속도, 완료 확률, 취소 성향, 시도 수)을 뽑고, 티켓 오픈 시각부터 예매 시도를 차례로 진행시킨다.
     시도 1건 = STARTED → QUEUE → SEAT_SELECTION → SEAT_HOLD → PAYMENT → BOOKING → COMPLETED (중간 이탈·선점 만료 포함).
     다음 시도는 이전 시도가 끝난 뒤(또는 선점만 해 두고 넘어간 뒤) 시작한다. 완료된 예매는 계정의 취소 성향에 따라 나중에 CANCEL 된다.
  4. Fast/Jitter 매크로의 시간은 #16 실측 시도를 통째로 복원추출한다 (reference/macro_attempt_cycles.csv).
     그 밖의 값의 근거는 config/scenarios.json 의 basis 에 적혀 있다.

출력: data/<profile>/events.csv (export_events.py 와 같은 컬럼), accounts.csv, attempts.csv, generation.json
이미 있는 폴더는 --force 가 있어야 다시 쓴다. seed 가 같으면 파일이 바이트 단위로 같다.
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent.parent
REPO = EXP.parents[1]
CONFIG_PATH = EXP / "config" / "scenarios.json"
REFERENCE_PATH = EXP / "reference" / "macro_attempt_cycles.csv"
ORIGINAL_PATH = REPO / "ml_experiment" / "data" / "fairticket_synthetic_features.csv"
V2_PATH = REPO / "experiments" / "issue16-booking-interval" / "dummy_adjusted" / "v2" / "fairticket_synthetic_features_adjusted.csv"
EVENT_COLUMNS = ["event_id", "user_id", "external_user_id", "group", "service_id", "external_event_id", "event_type",
                 "booking_id", "ticket_id", "occurred_at_utc", "metadata"]
STAGE_KEYS = ["queue", "pre_seat", "seat", "hold_to_payment", "payment", "completed", "idle"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_assignment():
    """계정·시나리오·레이블. #16 v2 와 같은 행 순서."""
    return pd.read_csv(V2_PATH)[["user_id", "scenario", "is_anomaly"]]


def rng_for(seed, *key):
    return np.random.default_rng(np.random.SeedSequence([seed, *key]))


def draw(rng, spec):
    kind = spec["dist"]
    if kind == "poisson_plus1":
        return int(rng.poisson(spec["lam"])) + 1
    if kind == "randint":
        return int(rng.integers(spec["low"], spec["high"]))
    if kind == "beta":
        return float(rng.beta(spec["a"], spec["b"]))
    if kind == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    raise ValueError(kind)


def partition(rng, members, low, high):
    """계정들을 크기 low~high 의 묶음으로 나눈다. 계정 기준으로 묶음 크기가 고르게 나오도록 크기 s 를 1/s 비중으로 뽑는다."""
    sizes = np.arange(low, high + 1)
    weights = (1 / sizes) / (1 / sizes).sum()
    while True:
        groups, left = [], len(members)
        while left >= low:
            size = int(rng.choice(sizes, p=weights))
            if size > left:
                continue
            groups.append(size)
            left -= size
        if left == 0:
            break
    order, out, start = rng.permutation(members), [], 0
    for size in groups:
        out.append([int(i) for i in order[start:start + size]])
        start += size
    return out


def macro_levels(reference):
    """실측 매크로 계정의 평균 예매 간격(초)에서 유형별 로그 평균과 계정 간 로그 표준편차를 추정한다 (v2 의 account_level 용).

    계정이 유형당 2~3개뿐이라 표준편차는 유형을 합쳐서 구한다 (유형 안 편차의 제곱합 / 자유도).
    """
    with_gap = reference.dropna(subset=["gap_to_next_ms"])
    means = with_gap.groupby(["mode", "account"])["gap_to_next_ms"].mean() / 1000
    logs = np.log(means)
    center = logs.groupby(level="mode").mean()
    deviation = logs - center.reindex(logs.index.get_level_values("mode")).to_numpy()
    freedom = len(logs) - logs.index.get_level_values("mode").nunique()
    spread = float(np.sqrt((deviation ** 2).sum() / freedom)) if freedom > 0 else 0.0
    return {"log_mean": center.to_dict(), "log_sd": spread, "account_mean_s": means.droplevel("mode").to_dict()}


def build_accounts(assignment, config, seed, reference=None):
    """공유 관계(IP 묶음, 카드)와 실측 계정 배정. 이벤트보다 먼저 정해지는 것은 '누가 누구와 무엇을 같이 쓰는가'뿐이다."""
    reference = pd.read_csv(REFERENCE_PATH) if reference is None else reference
    accounts = assignment.copy()
    for column in ["ip_group_id", "ip_address", "payment_identifier", "donor_account"]:
        accounts[column] = ""
    ip_no = card_no = 0
    for s_index, (name, spec) in enumerate(config["scenarios"].items()):
        rng = rng_for(seed, 1, s_index)
        members = accounts.index[accounts["scenario"] == name].to_numpy()
        if len(members) == 0:  # 일부 시나리오만 만들 때 (Leave-One-Account-Out 검증)
            continue
        size = spec["ip_group_size"]
        card = spec["card"]
        for group in partition(rng, members, size["min"], size["max"]):
            ip_no += 1
            accounts.loc[group, "ip_group_id"] = f"{name}-g{ip_no:04d}"
            accounts.loc[group, "ip_address"] = f"100.{64 + ip_no // 65536}.{ip_no // 256 % 256}.{ip_no % 256}"
            if card["rule"] == "own":
                card_groups = [[i] for i in group]
            elif card["rule"] == "pair_within_group":
                paired = len(group) >= 2 and rng.random() < card["pair_prob"]
                card_groups = [group[:2]] + [[i] for i in group[2:]] if paired else [[i] for i in group]
            elif card["rule"] == "shared_in_group":
                n_cards = math.ceil(len(group) / card["max_accounts_per_card"])
                card_groups = [list(map(int, part)) for part in np.array_split(group, n_cards)]
            else:
                raise ValueError(card["rule"])
            for card_group in card_groups:
                card_no += 1
                accounts.loc[card_group, "payment_identifier"] = f"pay-{card_no:05d}"
        if spec["timing_source"] == "measured":
            donors = sorted(reference.loc[reference["mode"] == spec["measured_mode"], "account"].unique())
            for position, index in enumerate(rng.permutation(members)):
                accounts.loc[index, "donor_account"] = donors[position % len(donors)]
    accounts["ip_group_size"] = accounts.groupby("ip_group_id")["user_id"].transform("size")
    accounts["card_group_size"] = accounts.groupby("payment_identifier")["user_id"].transform("size")
    return accounts, reference


def outcome_table(rng, spec):
    """계정의 시도 결과 확률. complete 가 분포이면 계정마다 한 번 뽑고, 나머지 결과는 남은 확률을 비중대로 나눈다."""
    raw = {k: v for k, v in spec["outcomes"].items() if k not in ("basis", "note")}
    complete = draw(rng, raw["complete"]) if isinstance(raw["complete"], dict) else float(raw["complete"])
    others = {k: float(v) for k, v in raw.items() if k != "complete" and float(v) > 0}
    total = sum(others.values())
    names = ["complete"] + list(others)
    probs = [complete] + [(1 - complete) * v / total for v in others.values()]
    return names, np.array(probs) / sum(probs), complete


class Account:
    """한 계정의 이벤트를 모은다. 시간은 계산 시각 기준 정수 밀리초."""

    def __init__(self, index, row, service_id):
        self.index, self.row, self.service_id = index, row, service_id
        self.events, self.attempts = [], []

    def emit(self, t_ms, event_type, metadata, booking_id="", ticket_id=""):
        self.events.append((t_ms, self.index, len(self.events), event_type, booking_id, ticket_id, metadata))

    def stage(self, t_ms, attempt, stage, status, source, **extra):
        attempt["sequence"] += 1
        self.emit(t_ms, "BOOKING_ATTEMPT", {"stage": stage, "source": source, "status": status, "sequence": attempt["sequence"],
                                            "attemptId": attempt["attempt_id"], "concertId": attempt["concert"], **extra})


def run_attempt(account, rng, start_ms, concert, number, timing, outcome, hold_ms):
    """시도 1건의 이벤트를 만든다. 반환: 사용자가 이 시도를 떠나 다음 행동을 할 수 있게 되는 시각."""
    row = account.row
    tag = f"{account.index:04d}-{number:03d}"
    attempt = {"user_id": row.user_id, "attempt_id": f"att-{tag}", "concert": concert, "sequence": 0, "start_ms": start_ms,
               "outcome": outcome, "seat_ms": None, "booking_id": "", "booking_ms": None}
    account.attempts.append(attempt)
    t = start_ms
    account.stage(t, attempt, "STARTED", "ENTERED", "CLIENT_REPORT")
    t += timing["queue"]
    account.stage(t, attempt, "QUEUE", "SUCCEEDED", "SERVER")
    t += timing["pre_seat"]
    account.stage(t, attempt, "SEAT_SELECTION", "ENTERED", "CLIENT_REPORT")
    if outcome == "abandon_seat":  # 좌석을 고르다가 그만둠
        t += max(1, round(timing["seat"] * rng.uniform(0.5, 2.0)))
        account.stage(t, attempt, "SEAT_SELECTION", "ABANDONED", "CLIENT_REPORT", reasonCode="USER_CANCELLED")
        return t
    t += timing["seat"]
    attempt["seat_ms"] = timing["seat"]
    hold_at, hold_id = t, f"hold-{tag}"
    account.stage(t, attempt, "SEAT_HOLD", "SUCCEEDED", "SERVER", holdId=hold_id)
    if outcome == "hold_left":  # 결제하지 않고 선점만 둔 채 다음 좌석으로. 선점은 나중에 서버가 만료시킨다
        account.stage(hold_at + hold_ms, attempt, "SEAT_HOLD", "EXPIRED", "SERVER", holdId=hold_id, reasonCode="HOLD_EXPIRED")
        return t
    if timing["hold_to_payment"] is not None:
        account.stage(t + timing["hold_to_payment"], attempt, "PAYMENT", "ENTERED", "CLIENT_REPORT", holdId=hold_id)
    if outcome == "abandon_payment":  # 결제 화면에서 선점을 풀고 나감
        t += timing["hold_to_payment"] + max(1, round(timing["payment"] * rng.uniform(0.2, 1.0)))
        account.stage(t, attempt, "PAYMENT", "ABANDONED", "CLIENT_REPORT", holdId=hold_id, reasonCode="USER_CANCELLED")
        return t
    if outcome == "hold_expired_wait":  # 결제 화면에 머물다 선점 시간이 끝남
        t = hold_at + hold_ms
        account.stage(t, attempt, "SEAT_HOLD", "EXPIRED", "SERVER", holdId=hold_id, reasonCode="HOLD_EXPIRED")
        return t
    t = hold_at + timing["hold_to_booking"]
    booking_id, ticket_id = f"bk-{tag}", f"tk-{tag}"
    attempt.update(booking_id=booking_id, booking_ms=t)
    account.emit(t, "BOOKING", {"attemptId": attempt["attempt_id"], "concertId": concert, "ipAddress": row.ip_address,
                                "ticketIds": [ticket_id], "paymentIdentifier": row.payment_identifier,
                                "bookingDurationSeconds": timing["hold_to_booking"] / 1000}, booking_id=booking_id)
    t += timing["completed"]
    account.stage(t, attempt, "COMPLETED", "SUCCEEDED", "SERVER", holdId=hold_id, bookingId=booking_id)
    return t


def simulate_account(index, row, spec, config, profile, seed, reference, opens_ms, levels=None):
    rng = rng_for(seed, 2, index)
    account = Account(index, row, config["service_id"])
    hold_ms = int(config["hold_duration_s"]["value"] * 1000)
    measured = spec["timing_source"] == "measured"

    # --- 계정의 행동 성향 ---
    n_attempts = draw(rng, spec["attempts"])
    names, probs, complete_prob = outcome_table(rng, spec)
    cancel_propensity = draw(rng, spec["cancel_propensity"])
    pace = 1.0 if measured else float(np.exp(spec["account_pace_sigma"]["value"] * rng.standard_normal()))
    extra = config["profiles"][profile]["extra_sessions_lambda"]
    n_sessions = min(n_attempts, len(opens_ms), 1 + (int(rng.poisson(extra)) if extra > 0 else 0))
    chosen = np.sort(rng.choice(len(opens_ms), size=n_sessions, replace=False))
    per_session = np.ones(n_sessions, dtype=int) + rng.multinomial(n_attempts - n_sessions, np.ones(n_sessions) / n_sessions)
    time_scale, account_level = 1.0, None
    if measured:
        donor = reference[reference["account"] == row.donor_account].reset_index(drop=True)
        with_gap = donor.index[donor["gap_to_next_ms"].notna()].to_numpy()
        if "account_level" in spec:
            # v2: 계정의 평균 간격 수준을 실측 계정 간 변동에서 뽑고, 배정된 실측 계정의 시도 전체를 그 수준에 맞게 같은 비율로 늘이거나 줄인다
            account_level = float(np.exp(levels["log_mean"][spec["measured_mode"]] + levels["log_sd"] * rng.standard_normal()))
            time_scale = account_level / levels["account_mean_s"][row.donor_account]
    floors = config.get("stage_floor_s", {})

    # --- 오픈마다 접속해 시도를 이어 간다 ---
    number = 0
    for open_index, count in zip(chosen, per_session):
        concert, open_ms = opens_ms[open_index]
        arrival = spec["arrival"]
        wait_s = rng.exponential(arrival["mean_s"]) if arrival["dist"] == "exponential" else rng.uniform(arrival["low_s"], arrival["high_s"])
        start = open_ms + max(1, round(wait_s * 1000))
        for k in range(count):
            number += 1
            outcome = str(rng.choice(names, p=probs))
            if measured:
                pick = donor.loc[int(rng.choice(with_gap))]  # 실측 시도 1건: 단계 시간과 다음 시도까지의 간격을 함께 가져온다
                h2p = None if pd.isna(pick["hold_to_payment_ms"]) else int(pick["hold_to_payment_ms"])
                timing = {"queue": int(pick["queue_ms"]), "pre_seat": int(pick["pre_seat_ms"]), "seat": int(pick["seat_ms"]),
                          "hold_to_payment": h2p, "hold_to_booking": int(pick["hold_to_booking_ms"]), "completed": int(pick["completed_ms"])}
                gap = int(pick["gap_to_next_ms"])
                if time_scale != 1.0:
                    timing = {key: value if value is None else max(1, round(value * time_scale)) for key, value in timing.items()}
                    cycle = sum(timing[key] for key in ("queue", "pre_seat", "seat", "hold_to_booking", "completed"))
                    gap = max(cycle + 1, round(gap * time_scale))
                run_attempt(account, rng, start, concert, number, timing, outcome, hold_ms)
                start += gap
            else:
                sigma = spec["within_sigma"]["value"]
                noise = np.exp(sigma * rng.standard_normal(len(STAGE_KEYS)))
                if floors:  # v2: 단계 시간 = 하한 + (중앙값 − 하한) × 계정 속도 × 시도별 변동
                    timing = {key: max(1, round(1000 * (floors.get(key, 0.0) + max(spec["stages_median_s"][key] - floors.get(key, 0.0), 0.001) * pace * z)))
                              for key, z in zip(STAGE_KEYS, noise)}
                else:
                    timing = {key: max(1, round(spec["stages_median_s"][key] * 1000 * pace * z)) for key, z in zip(STAGE_KEYS, noise)}
                timing["hold_to_booking"] = timing["hold_to_payment"] + timing["payment"]
                free_at = run_attempt(account, rng, start, concert, number, timing, outcome, hold_ms)
                start = free_at + timing["idle"]

    # --- 완료된 예매의 취소 ---
    delay, minimum = spec["cancel_delay"], config["cancel_delay"]["min_s"]
    for attempt in account.attempts:
        attempt.update(cancel_ms=None, cancel_after_cutoff=False)
        if attempt["booking_ms"] is None or rng.random() >= cancel_propensity:
            continue
        wait_s = max(minimum, delay["median_s"] * float(np.exp(delay["sigma"] * rng.standard_normal())))
        cancel_ms = attempt["booking_ms"] + round(wait_s * 1000)
        if cancel_ms > 0:  # 계산 시각 이후: 아직 일어나지 않은 일이므로 이벤트를 만들지 않는다
            attempt["cancel_after_cutoff"] = True
            continue
        attempt["cancel_ms"] = cancel_ms
        account.emit(cancel_ms, "CANCEL", {"concertId": attempt["concert"], "ticketIds": [attempt["booking_id"].replace("bk-", "tk-")]},
                     booking_id=attempt["booking_id"])
    traits = {"n_attempts": n_attempts, "n_sessions": n_sessions, "concerts": "|".join(opens_ms[i][0] for i in chosen),
              "pace_multiplier": pace, "complete_prob": complete_prob, "cancel_propensity": cancel_propensity}
    if account_level is not None:
        traits.update(account_level_s=account_level, time_scale=time_scale)
    return account, traits


def generate(config=None, profile="single_session", seed=None, assignment=None, reference=None):
    """이벤트·계정·시도 표를 만든다. 파일은 쓰지 않는다."""
    config = config or load_config()
    seed = config["seed"] if seed is None else seed
    assignment = load_assignment() if assignment is None else assignment
    end = datetime.fromisoformat(config["calculated_at_utc"])
    opens_ms = [(item["concertId"], round((datetime.fromisoformat(item["open_utc"]) - end).total_seconds() * 1000))
                for item in config["concert_opens"]["items"]]
    accounts, reference = build_accounts(assignment, config, seed, reference)
    levels = macro_levels(reference)

    raw_events, attempt_rows, trait_rows = [], [], []
    for index, row in enumerate(accounts.itertuples(index=False)):
        account, traits = simulate_account(index, row, config["scenarios"][row.scenario], config, profile, seed, reference, opens_ms, levels)
        raw_events.extend(account.events)
        attempt_rows.extend(account.attempts)
        trait_rows.append(traits)
    accounts = pd.concat([accounts, pd.DataFrame(trait_rows)], axis=1)

    # behavior_event 처럼 시간순으로 event_id 를 붙인다 (같은 시각이면 계정 번호, 계정 안의 생성 순서)
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
                         "cancel_utc", "cancel_after_cutoff"]]
    return events, accounts, attempts


def to_csv_frame(events):
    """export_events.py 가 쓰는 CSV 컬럼으로 바꾼다 (extract_features.load_events 가 그대로 읽는다)."""
    out = events.copy()
    out["occurred_at_utc"] = out["occurred_at"].map(lambda t: t.strftime("%Y-%m-%dT%H:%M:%S.%f"))
    out["metadata"] = out["meta"].map(lambda m: json.dumps(m, ensure_ascii=False))
    return out[EVENT_COLUMNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="single_session")
    parser.add_argument("--seed", type=int, default=None, help="기본값은 config 의 seed (42)")
    parser.add_argument("--out", default=None, help="출력 폴더 (기본 data/<profile>)")
    parser.add_argument("--config", default=None, help="생성 설정 파일 (기본 config/scenarios.json = v1). v2 는 --config config/scenarios_v2.json --out data/v2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else CONFIG_PATH
    if config_path.resolve() != CONFIG_PATH.resolve() and not args.out:
        sys.exit("기본 설정이 아닌 설정을 쓸 때는 --out 으로 출력 폴더를 지정하세요 (v1 폴더를 덮어쓰지 않기 위함).")
    config = load_config(config_path)
    if args.profile not in config["profiles"]:
        sys.exit(f"profile 은 {list(config['profiles'])} 중 하나여야 합니다.")
    seed = config["seed"] if args.seed is None else args.seed
    out = Path(args.out) if args.out else EXP / "data" / args.profile
    if out.exists() and any(out.iterdir()):
        if not args.force:
            sys.exit(f"이미 있는 폴더입니다: {out}\n--force 를 주면 다시 씁니다.")
        if not (out / "generation.json").exists():
            sys.exit(f"{out} 에 generation.json 이 없어 덮어쓰지 않았습니다.")
    out.mkdir(parents=True, exist_ok=True)
    before = {"original": sha256(ORIGINAL_PATH), "issue16_v2": sha256(V2_PATH)}

    events, accounts, attempts = generate(config, args.profile, seed)
    to_csv_frame(events).to_csv(out / "events.csv", index=False, encoding="utf-8", lineterminator="\n")
    accounts.to_csv(out / "accounts.csv", index=False, encoding="utf-8", lineterminator="\n")
    attempts.to_csv(out / "attempts.csv", index=False, encoding="utf-8", lineterminator="\n")

    record = {
        "profile": args.profile, "seed": seed, "rng": "numpy.random.default_rng(SeedSequence([seed, 단계, 번호]))",
        "numpy": np.__version__, "pandas": pd.__version__,
        "config": config_path.name, "config_version": config.get("version", "v1"),
        "config_sha256": sha256(config_path), "reference_sha256": sha256(REFERENCE_PATH),
        "account_source": V2_PATH.relative_to(REPO).as_posix(),
        "calculated_at_utc": config["calculated_at_utc"], "window_days": config["window_days"],
        "accounts": int(len(accounts)), "accounts_by_scenario": accounts["scenario"].value_counts().to_dict(),
        "anomaly_ratio": float(accounts["is_anomaly"].mean()),
        "events": int(len(events)), "events_by_type": events["event_type"].value_counts().to_dict(),
        "attempts": int(len(attempts)), "attempts_by_outcome": attempts["outcome"].value_counts().to_dict(),
        "bookings": int((attempts["booking_id"] != "").sum()), "cancels": int(attempts["cancel_utc"].notna().sum()),
        "cancels_after_cutoff_not_emitted": int(attempts["cancel_after_cutoff"].sum()),
        "occurred_at_min": events["occurred_at"].min().isoformat(), "occurred_at_max": events["occurred_at"].max().isoformat(),
        "files_sha256": {name: sha256(out / name) for name in ["events.csv", "accounts.csv", "attempts.csv"]},
        "existing_datasets_sha256": before,
        "existing_datasets_unchanged": before == {"original": sha256(ORIGINAL_PATH), "issue16_v2": sha256(V2_PATH)},
    }
    (out / "generation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
