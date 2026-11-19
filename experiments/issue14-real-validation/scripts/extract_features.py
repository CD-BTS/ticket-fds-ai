"""실측 이벤트에서 기존 6개 Feature를 계산한다 (실험용).

실행 (실험 폴더에서):
    .venv/Scripts/python scripts/extract_features.py --tag pilot --calculated-at 2026-11-20T12:00:00Z

입력 : data/<tag>/events.csv (export_events.py 출력)
출력 : data/<tag>/real_features.csv       계정별 Feature
       data/<tag>/attempt_intervals.csv   계정별 연속 시도 간격 (한 줄에 간격 하나)
       data/<tag>/extraction_summary.json 계산 조건과 환경 왜곡 기록

[A] BTS 재현 컬럼 — ticket-fds-admin feat/#40 FeatureAggregationService.calculate 와 같은 식.
    기존 모델 입력과 비교하는 용도이며, 식을 바꾸지 않는다 (시도 2건일 때 분산 0도 그대로).
      booking_count, avg_booking_interval, booking_interval_variance,
      cancel_rate, card_reuse_count, ip_booking_density
    기준: 계산 시각 이전 30일 [calculated_at - 30일, calculated_at], 양 끝 포함.

[B] 세션 컬럼 — 동일 세션 내 반복 시도 간격. 실제 반복 행동 분석용이며 모델 입력이 아니다.
      session_interval_mean, session_interval_median, session_interval_variance(표본분산),
      session_interval_cv, n_sessions, n_session_intervals
    세션 기준(제안값): 같은 계정의 연속 시도 시작 간격이 --session-gap-minutes 이하이면 같은 세션.
    기존 이벤트에는 세션 식별자가 없으므로 이 기준은 실험용 가정이다.

계산할 수 없는 값은 0으로 채우지 않고 빈 칸(결측)으로 둔다.
"""

import argparse
import ipaddress
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parent.parent
WINDOW_DAYS = 30
MODEL_FEATURES = ["avg_booking_interval", "booking_interval_variance", "cancel_rate",
                  "card_reuse_count", "ip_booking_density", "booking_count"]


def parse_utc(text):
    value = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def metadata_string(metadata, key):
    """Java metadataString 과 동일: 없거나 공백이면 None, 아니면 trim 한 문자열."""
    value = metadata.get(key) if isinstance(metadata, dict) else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_events(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["event_id"] = df["event_id"].astype("int64")
    df["occurred_at"] = df["occurred_at_utc"].map(parse_utc)
    df["meta"] = df["metadata"].map(lambda s: json.loads(s) if s else {})
    return df.sort_values(["occurred_at", "event_id"]).reset_index(drop=True)


def shared_account_counter(window_events, key):
    """BehaviorEventRepository.countDistinctUsersBy* 와 동일:
    같은 서비스에서 기간 내 BOOKING 이벤트의 metadata 값이 같은 고유 계정 수."""
    bookings = window_events[window_events["event_type"] == "BOOKING"]
    table = {}
    for service_id, user_id, meta in zip(bookings["service_id"], bookings["user_id"], bookings["meta"]):
        raw = meta.get(key) if isinstance(meta, dict) else None
        if raw is None:
            continue
        table.setdefault((service_id, str(raw)), set()).add(user_id)  # SQL은 trim 없이 비교
    return lambda service_id, value: len(table.get((service_id, value), ()))


def calculate_bts(events, count_card, count_ip):
    """한 계정의 기간 내 이벤트(시간순)로 BTS와 같은 6개 Feature를 계산한다."""
    distinct_bookings, distinct_cancels, attempt_starts = {}, set(), {}
    for row in events.itertuples(index=False):
        if not row.service_id:
            continue
        if row.event_type.upper() == "BOOKING_ATTEMPT":
            attempt_id = metadata_string(row.meta, "attemptId")
            if attempt_id is not None:
                attempt_starts.setdefault(f"{row.service_id}:{attempt_id}", row.occurred_at)
            continue
        if not row.booking_id:
            continue
        key = f"{row.service_id}:{row.booking_id}"
        if row.event_type.upper() == "BOOKING":
            distinct_bookings.setdefault(key, row)
        elif row.event_type.upper() == "CANCEL":
            distinct_cancels.add(key)

    bookings = list(distinct_bookings.values())
    cancelled = sum(1 for key in distinct_bookings if key in distinct_cancels)
    cancel_rate = cancelled / len(bookings) if bookings else None

    attempt_times = sorted(attempt_starts.values())
    booking_count = len(attempt_times)
    intervals = [max(0, int((b - a) / timedelta(milliseconds=1))) / 1000.0
                 for a, b in zip(attempt_times, attempt_times[1:])]
    avg_interval = variance = None
    if booking_count >= 2:
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)  # 모분산

    def max_shared(key, counter):
        values = []
        for booking in bookings:
            value = metadata_string(booking.meta, key)
            if value is not None and (booking.service_id, value) not in values:
                values.append((booking.service_id, value))
        return max([1] + [counter(service_id, value) for service_id, value in values]), [v for _, v in values]

    card_reuse, cards = max_shared("paymentIdentifier", count_card)
    ip_density, ips = max_shared("ipAddress", count_ip)

    return {
        "booking_count": booking_count, "avg_booking_interval": avg_interval,
        "booking_interval_variance": variance, "cancel_rate": cancel_rate,
        "card_reuse_count": card_reuse, "ip_booking_density": ip_density,
    }, {
        "attempt_times": attempt_times, "intervals": intervals,
        "n_completed_bookings": len(bookings), "n_cancelled_bookings": cancelled,
        "payment_identifiers_used": cards, "ip_addresses_used": ips,
    }


def session_stats(intervals, gap_seconds, n_attempts):
    """세션 내 간격만 골라 요약한다. 세션 = 간격이 gap_seconds 이하로 이어진 시도 묶음."""
    inside = [x for x in intervals if x <= gap_seconds]
    n_sessions = (1 + sum(1 for x in intervals if x > gap_seconds)) if n_attempts else 0
    stats = {"n_sessions": n_sessions, "n_session_intervals": len(inside),
             "session_interval_mean": None, "session_interval_median": None,
             "session_interval_variance": None, "session_interval_cv": None}
    if inside:
        mean = sum(inside) / len(inside)
        stats["session_interval_mean"] = mean
        stats["session_interval_median"] = float(pd.Series(inside).median())
        if len(inside) >= 2:
            sample_variance = sum((x - mean) ** 2 for x in inside) / (len(inside) - 1)  # 표본분산
            stats["session_interval_variance"] = sample_variance
            stats["session_interval_cv"] = math.sqrt(sample_variance) / mean if mean > 0 else None
    return stats


def is_local_ip(text):
    try:
        ip = ipaddress.ip_address(text)
        return ip.is_loopback or ip.is_private
    except ValueError:
        return False


def extract(events, calculated_at, session_gap_minutes):
    window_start = calculated_at - timedelta(days=WINDOW_DAYS)
    in_window = events[(events["occurred_at"] >= window_start) & (events["occurred_at"] <= calculated_at)]
    count_card = shared_account_counter(in_window, "paymentIdentifier")
    count_ip = shared_account_counter(in_window, "ipAddress")

    booking_meta = in_window.loc[in_window["event_type"] == "BOOKING", "meta"]
    all_cards = sorted({str(m["paymentIdentifier"]) for m in booking_meta if m.get("paymentIdentifier") is not None})
    all_ips = sorted({str(m["ipAddress"]) for m in booking_meta if m.get("ipAddress") is not None})
    n_booking_users = in_window.loc[in_window["event_type"] == "BOOKING", "user_id"].nunique()
    # 데모는 결제 식별자를 3개만 허용하고, 프록시를 거치면 모든 계정이 같은 IP로 기록된다.
    card_distorted = n_booking_users > 0 and (n_booking_users > len(all_cards) or len(all_cards) <= 3)
    ip_distorted = n_booking_users > 0 and (
        (len(all_ips) <= 1 and n_booking_users > 1) or any(is_local_ip(ip) for ip in all_ips))

    users = events.drop_duplicates("user_id")[["user_id", "external_user_id", "group"]]
    feature_rows, interval_rows = [], []
    gap_seconds = session_gap_minutes * 60
    for user in users.itertuples(index=False):
        own = in_window[in_window["user_id"] == user.user_id]
        features, detail = calculate_bts(own, count_card, count_ip)
        intervals = detail["intervals"]
        row = {"user_id": user.user_id, "external_user_id": user.external_user_id, "group": user.group,
               "calculated_at_utc": calculated_at.isoformat(), **features,
               "n_intervals_30d": len(intervals),
               **session_stats(intervals, gap_seconds, features["booking_count"]),
               "n_completed_bookings": detail["n_completed_bookings"],
               "n_cancelled_bookings": detail["n_cancelled_bookings"],
               "first_attempt_utc": detail["attempt_times"][0].isoformat() if detail["attempt_times"] else None,
               "last_attempt_utc": detail["attempt_times"][-1].isoformat() if detail["attempt_times"] else None,
               # 시도 2건이면 간격이 1개라 BTS 식의 모분산이 항상 0이 된다 (측정값이 아니라 계산상 결과)
               "variance_degenerate": len(intervals) == 1,
               "has_null_model_input": any(features[c] is None for c in MODEL_FEATURES),
               "card_env_distorted": card_distorted, "ip_env_distorted": ip_distorted,
               "payment_identifiers_used": "|".join(detail["payment_identifiers_used"]),
               "ip_addresses_used": "|".join(detail["ip_addresses_used"])}
        feature_rows.append(row)
        session_index = 1
        for position, gap in enumerate(intervals):
            within = gap <= gap_seconds
            if not within:
                session_index += 1
            interval_rows.append({"user_id": user.user_id, "external_user_id": user.external_user_id,
                                  "group": user.group, "interval_index": position + 1,
                                  "from_attempt_utc": detail["attempt_times"][position].isoformat(),
                                  "to_attempt_utc": detail["attempt_times"][position + 1].isoformat(),
                                  "gap_seconds": gap, "within_session": within,
                                  "session_index_of_to_attempt": session_index})

    summary = {
        "calculated_at_utc": calculated_at.isoformat(), "window_start_utc": window_start.isoformat(),
        "window_days": WINDOW_DAYS, "session_gap_minutes": session_gap_minutes,
        "session_rule": "제안값(실험용 가정): 같은 계정의 연속 시도 시작 간격이 기준 이하이면 같은 세션",
        "events_total": int(len(events)), "events_in_window": int(len(in_window)),
        "users": int(len(users)), "users_with_bookings": int(n_booking_users),
        "distinct_payment_identifiers": all_cards, "distinct_ip_addresses": all_ips,
        "card_env_distorted": bool(card_distorted), "ip_env_distorted": bool(ip_distorted),
        "bts_reference": "ticket-fds-admin feat/#40 @46f8c67 FeatureAggregationService.calculate",
    }
    return pd.DataFrame(feature_rows), pd.DataFrame(interval_rows), summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--calculated-at", required=True,
                        help="Feature 계산 기준 시각(UTC, ISO-8601). 모든 계정에 같은 값을 쓴다.")
    parser.add_argument("--session-gap-minutes", type=float, default=30.0,
                        help="세션 구분 기준(분). 제안값 30, 파일럿 결과로 조정")
    args = parser.parse_args()

    data_dir = EXP / "data" / args.tag
    events = load_events(data_dir / "events.csv")
    features, intervals, summary = extract(events, parse_utc(args.calculated_at), args.session_gap_minutes)

    features.to_csv(data_dir / "real_features.csv", index=False)
    intervals.to_csv(data_dir / "attempt_intervals.csv", index=False)
    (data_dir / "extraction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if len(features):
        print(features[["external_user_id", "group"] + MODEL_FEATURES
                       + ["session_interval_mean", "n_session_intervals", "has_null_model_input"]].round(3).to_string(index=False))
    print(f"저장: {data_dir}")


if __name__ == "__main__":
    main()
