import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


# ==========================================
# 1. 기본 설정 (BTS AI Feature 명세서 2. 분석 기준)
# ==========================================

random.seed(42)

NUM_USERS = 1000

# 분석 기준 시각(assessed_at)과 분석 기간(기준 시각 이전 30일).
# 시간은 UTC로 저장하고, 계산할 때도 UTC로 통일한다.
ASSESSED_AT = datetime(2026, 10, 10, tzinfo=timezone.utc)
WINDOW_DAYS = 30
WINDOW_START = ASSESSED_AT - timedelta(days=WINDOW_DAYS)

# 짧은 예매 간격 기준(초). 명세 초기값은 5초이며 실험 후 조정한다.
SHORT_INTERVAL_SECONDS = 5

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

EVENT_PATH = DATA_DIR / "behavior_event.csv"
TICKET_PATH = DATA_DIR / "ticket.csv"              # 관람 여부 확인용 (외부 서비스/ticket 대용)
LABEL_PATH = DATA_DIR / "synthetic_user_labels.csv"  # 합성 데이터 정답 유형 (모델 학습에는 사용하지 않음)
FEATURE_PATH = DATA_DIR / "user_features.csv"

SUPPORTED_EVENT_TYPES = {"BOOKING", "CANCEL", "CHECK_IN"}

FEATURE_COLUMNS = [
    "booking_count",
    "cancel_count",
    "cancel_rate",
    "avg_booking_interval",
    "short_interval_booking_count",
    "attendance_rate",
    "avg_booking_duration",
]


# ==========================================
# 2. 합성 데이터 생성
# ==========================================

def generate_data():
    # 분석 기간 안에서 예매가 발생하도록 생성 기간을 분석 기간과 일치시킨다.
    start_date = WINDOW_START
    end_date = ASSESSED_AT

    user_types = (
        ["normal"] * 700
        + ["frequent_cancel"] * 200
        + ["rapid_booking"] * 100
    )
    random.shuffle(user_types)

    # 공연별 예매 오픈 시간: 같은 공연이면 모든 사용자에게 동일하다.
    performances = {}
    for i in range(10):
        opening_day = start_date + timedelta(days=random.randint(0, 5))
        performances[f"PERF-{i + 1:03d}"] = opening_day.replace(
            hour=random.randint(9, 20),
            minute=random.randint(0, 59),
            second=0,
            microsecond=0,
        )

    events = []
    tickets = []
    labels = []
    event_id = 1

    def add_event(user_id, event_type, booking_id, ticket_id,
                  performance_id, occurred_at, metadata):
        nonlocal event_id
        events.append({
            "event_id": event_id,
            # 중복 제거용 외부 이벤트 ID (서비스별로 구분)
            "external_event_id": f"EXT-{event_id:07d}",
            "user_id": user_id,
            "event_type": event_type,
            "booking_id": booking_id,
            "ticket_id": ticket_id,
            "performance_id": performance_id,
            "occurred_at": occurred_at.isoformat(),
            "metadata": json.dumps(metadata),
        })
        event_id += 1

    for user_number, user_type in enumerate(user_types, start=1):
        user_id = f"U{user_number:04d}"
        labels.append({"user_id": user_id, "user_type": user_type})

        performance_id = f"PERF-{(user_number - 1) % len(performances) + 1:03d}"
        booking_open_at = performances[performance_id]

        # 첫 예매는 오픈 이후에 발생한다.
        current_time = booking_open_at + timedelta(
            seconds=random.randint(10, 3600)
        )

        num_bookings = (
            random.randint(3, 10) if user_type == "normal"
            else random.randint(10, 30)
        )

        for booking_number in range(num_bookings):
            if booking_number > 0:
                if user_type == "rapid_booking":
                    # 매크로성 연속 예매: 대부분 짧은 간격(5초 이하)
                    interval = random.choices(
                        [
                            random.randint(1, SHORT_INTERVAL_SECONDS),
                            random.randint(SHORT_INTERVAL_SECONDS + 1, 600),
                            random.randint(601, 3600),
                        ],
                        weights=[60, 25, 15],
                    )[0]
                else:
                    interval = random.randint(600, 172800)

                current_time += timedelta(seconds=interval)

            # 평가 시각 이후는 미래 이벤트이므로 생성하지 않는다.
            if current_time >= end_date:
                break

            booking_id = f"B{user_number:04d}-{booking_number + 1:03d}"
            ticket_id = f"T{user_number:04d}-{booking_number + 1:03d}"

            if user_type == "normal":
                duration = random.randint(30, 300)
                weights = [85, 10, 5]
            elif user_type == "frequent_cancel":
                duration = random.randint(10, 180)
                weights = [20, 10, 70]
            else:
                duration = random.randint(2, 60)
                weights = [65, 10, 25]

            completed_at = current_time
            started_at = completed_at - timedelta(seconds=duration)

            # 예매 시작 시간은 오픈 시간보다 빠를 수 없다.
            if started_at < booking_open_at:
                started_at = booking_open_at
                duration = int((completed_at - started_at).total_seconds())

            outcome = random.choices(
                ["ATTENDED", "NO_SHOW", "CANCELLED"], weights=weights
            )[0]

            add_event(
                user_id, "BOOKING", booking_id, ticket_id, performance_id,
                completed_at,
                {
                    "booking_open_at": booking_open_at.isoformat(),
                    "booking_started_at": started_at.isoformat(),
                    "booking_completed_at": completed_at.isoformat(),
                    "booking_duration_seconds": duration,
                },
            )

            # 티켓(공연 종료 시각): 관람 여부 확정 판단에 사용한다.
            show_start = completed_at + timedelta(
                days=random.randint(1, 7),
                hours=random.randint(0, 23),
            )
            show_ended_at = show_start + timedelta(hours=3)

            tickets.append({
                "ticket_id": ticket_id,
                "user_id": user_id,
                "booking_id": booking_id,
                "performance_id": performance_id,
                "show_ended_at": show_ended_at.isoformat(),
            })

            if outcome == "CANCELLED":
                cancel_time = completed_at + timedelta(
                    seconds=random.randint(10, 3600)
                )
                add_event(
                    user_id, "CANCEL", booking_id, ticket_id,
                    performance_id, cancel_time, {},
                )

            elif outcome == "ATTENDED":
                check_in_time = show_start - timedelta(
                    minutes=random.randint(10, 60)
                )
                add_event(
                    user_id, "CHECK_IN", booking_id, ticket_id,
                    performance_id, check_in_time, {},
                )

    events_df = pd.DataFrame(events)
    events_df.to_csv(EVENT_PATH, index=False)
    pd.DataFrame(tickets).to_csv(TICKET_PATH, index=False)
    pd.DataFrame(labels).to_csv(LABEL_PATH, index=False)

    print("이벤트 데이터 생성 완료!")
    print("전체 이벤트 수:", len(events_df))
    print("이벤트 유형별 개수:")
    print(events_df["event_type"].value_counts())


# ==========================================
# 3. 이벤트 로드 및 전처리 (명세 6. 전처리 및 예외 처리)
# ==========================================

def parse_metadata(value):
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def load_events():
    df = pd.read_csv(EVENT_PATH)

    required_columns = [
        "event_id", "external_event_id", "user_id", "event_type",
        "booking_id", "ticket_id", "performance_id", "occurred_at",
        "metadata",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"behavior_event.csv에 필요한 컬럼이 없습니다: {missing}")

    # 지원하지 않는 이벤트 유형은 검증 오류
    unsupported = set(df["event_type"].dropna().unique()) - SUPPORTED_EVENT_TYPES
    if unsupported or df["event_type"].isna().any():
        raise ValueError(f"지원하지 않는 이벤트 유형: {unsupported}")

    # 필수 ID 누락은 검증 오류
    id_requirements = {
        "BOOKING": ["external_event_id", "user_id", "booking_id", "occurred_at"],
        "CANCEL": ["external_event_id", "user_id", "booking_id", "occurred_at"],
        "CHECK_IN": ["external_event_id", "user_id", "ticket_id", "occurred_at"],
    }
    for event_type, columns in id_requirements.items():
        subset = df[df["event_type"] == event_type]
        for column in columns:
            if subset[column].isna().any():
                raise ValueError(
                    f"{event_type} 이벤트에 필수값 {column} 누락이 있습니다."
                )

    # 명세의 serviceId는 공연 식별자인 performance_id에 대응한다.
    df["serviceId"] = df["performance_id"]
    df["bookingId"] = df["booking_id"]

    # external_event_id 기준 중복 제거 (서비스별로 구분)
    df = df.drop_duplicates(subset=["serviceId", "external_event_id"])

    # 시간은 UTC로 통일하고 occurred_at, event_id 오름차순으로 정렬한다.
    df["occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True)
    df = df.sort_values(["occurred_at", "event_id"]).reset_index(drop=True)

    # 평가 시각보다 미래인 이벤트는 분석에서 제외한다.
    df = df[df["occurred_at"] <= pd.Timestamp(ASSESSED_AT)].copy()

    return df


# ==========================================
# 4. 사용자별 Feature 계산 (명세 4. Feature별 계산 조건)
# ==========================================

def compute_features(events_df):
    assessed_at = pd.Timestamp(ASSESSED_AT)
    window_start = pd.Timestamp(WINDOW_START)

    # 분석 기간 내 이벤트: [assessed_at - 30일, assessed_at]
    window_events = events_df[events_df["occurred_at"] >= window_start].copy()

    # 예매가 한 건도 없는 사용자도 결과에 남긴다.
    all_users = sorted(events_df["user_id"].dropna().unique())
    features = pd.DataFrame(index=pd.Index(all_users, name="user_id"))

    # ---- 예매 이벤트 준비 ----
    booking_df = window_events[window_events["event_type"] == "BOOKING"].copy()

    # 동일한 serviceId + bookingId 조합은 한 번만 집계한다.
    booking_df = booking_df.drop_duplicates(
        subset=["user_id", "serviceId", "bookingId"]
    )

    meta = pd.DataFrame(
        booking_df["metadata"].apply(parse_metadata).tolist(),
        index=booking_df.index,
    ).reindex(columns=["booking_started_at", "booking_completed_at"])

    booking_df["booking_started_at"] = pd.to_datetime(
        meta["booking_started_at"], errors="coerce", utc=True
    )
    booking_df["booking_completed_at"] = pd.to_datetime(
        meta["booking_completed_at"], errors="coerce", utc=True
    )

    # occurred_at 오름차순, 동일 시각이면 event_id 기준으로 정렬
    booking_df = booking_df.sort_values(["user_id", "occurred_at", "event_id"])

    # ---- 평가 시각까지 발생한 취소 (예매 건 기준 취소 판별용) ----
    cancel_asof = events_df[events_df["event_type"] == "CANCEL"]
    cancelled_keys = set(
        zip(cancel_asof["user_id"], cancel_asof["serviceId"], cancel_asof["bookingId"])
    )
    booking_df["is_cancelled"] = [
        key in cancelled_keys
        for key in zip(
            booking_df["user_id"], booking_df["serviceId"], booking_df["bookingId"]
        )
    ]

    # 4.1 booking_count: 분석 기간 내 고유 예매 건수, 없으면 0
    booking_count = booking_df.groupby("user_id").size()
    features["booking_count"] = (
        booking_count.reindex(features.index, fill_value=0).astype(int)
    )

    # 4.2 cancel_count: 분석 기간 내 "취소 발생" 고유 건수, 없으면 0
    cancel_in_window = window_events[
        window_events["event_type"] == "CANCEL"
    ].drop_duplicates(subset=["user_id", "serviceId", "bookingId"])

    cancel_count = cancel_in_window.groupby("user_id").size()
    features["cancel_count"] = (
        cancel_count.reindex(features.index, fill_value=0).astype(int)
    )

    # 4.3 cancel_rate: 분석 기간 내 "예매" 중 평가 시각까지 취소된 비율
    # (cancel_count는 취소 발생 시각 기준, cancel_rate는 예매 발생 시각 기준)
    cancelled_bookings = (
        booking_df.groupby("user_id")["is_cancelled"].sum()
        .reindex(features.index, fill_value=0)
    )
    features["cancel_rate"] = (
        cancelled_bookings / features["booking_count"].replace(0, float("nan"))
    )

    # 4.4 avg_booking_interval: 연속된 예매 완료 시각 차이의 평균(초)
    # 예매가 2건 미만이면 null
    booking_df["interval_seconds"] = (
        booking_df.groupby("user_id")["occurred_at"].diff().dt.total_seconds()
    )
    features["avg_booking_interval"] = (
        booking_df.groupby("user_id")["interval_seconds"].mean()
    )

    # 4.5 short_interval_booking_count: 간격이 기준 이하인 횟수, 예매 2건 미만이면 0
    short_flag = booking_df["interval_seconds"] <= SHORT_INTERVAL_SECONDS
    short_count = short_flag.groupby(booking_df["user_id"]).sum()
    features["short_interval_booking_count"] = (
        short_count.reindex(features.index, fill_value=0).astype(int)
    )

    # 4.6 attendance_rate: 관람 완료 티켓 / 관람 여부가 확정된 유효 티켓
    features["attendance_rate"] = compute_attendance_rate(
        booking_df, events_df, assessed_at
    ).reindex(features.index)

    # 4.7 avg_booking_duration: 예매 완료 시각 - 예매 시작 시각의 평균(초)
    valid_duration = booking_df[
        booking_df["booking_started_at"].notna()
        & booking_df["booking_completed_at"].notna()
        & (booking_df["booking_completed_at"] >= booking_df["booking_started_at"])
    ].copy()

    valid_duration["duration_seconds"] = (
        valid_duration["booking_completed_at"] - valid_duration["booking_started_at"]
    ).dt.total_seconds()

    features["avg_booking_duration"] = (
        valid_duration.groupby("user_id")["duration_seconds"].mean()
    )

    return features.reset_index()[["user_id"] + FEATURE_COLUMNS]


def compute_attendance_rate(booking_df, events_df, assessed_at):
    """관람 여부가 확정된 유효 티켓 중 실제 관람이 완료된 비율.

    - 취소된 티켓은 제외
    - 공연이 평가 시각까지 종료되지 않은 티켓(관람 여부 미확정)은 제외
    - 관람 완료 여부는 CHECK_IN 이벤트로 판단
    """
    if not TICKET_PATH.exists():
        # 관람 데이터가 없으면 계산 불가(null)
        return pd.Series(dtype=float, name="attendance_rate")

    tickets = pd.read_csv(TICKET_PATH)
    tickets["show_ended_at"] = pd.to_datetime(
        tickets["show_ended_at"], errors="coerce", utc=True
    )
    tickets = tickets.drop_duplicates(subset=["ticket_id"])

    ticket_df = booking_df[
        ["user_id", "ticket_id", "is_cancelled"]
    ].drop_duplicates(subset=["user_id", "ticket_id"])

    ticket_df = ticket_df.merge(
        tickets[["ticket_id", "show_ended_at"]], on="ticket_id", how="left"
    )

    resolved = ticket_df[
        (~ticket_df["is_cancelled"])
        & ticket_df["show_ended_at"].notna()
        & (ticket_df["show_ended_at"] <= assessed_at)
    ].copy()

    checked_in = set(
        events_df.loc[events_df["event_type"] == "CHECK_IN", "ticket_id"]
    )
    resolved["attended"] = resolved["ticket_id"].isin(checked_in).astype(int)

    return resolved.groupby("user_id")["attended"].mean().rename("attendance_rate")


# ==========================================
# 5. 실행
# ==========================================

if __name__ == "__main__":
    generate_data()

    events = load_events()
    user_features = compute_features(events)

    # 계산 불가 Feature는 null로 표현한다.
    user_features.to_csv(FEATURE_PATH, index=False, na_rep="null")

    print("\nFeature 계산 완료!")
    print("Feature 데이터 사용자 수:", len(user_features))
    print("\nFeature 데이터 샘플:")
    print(user_features.head())

    print("\nFeature 결측치 개수:")
    print(user_features[FEATURE_COLUMNS].isnull().sum())

    print("\n분석 기준 시각(assessed_at):", ASSESSED_AT.isoformat())
    print("분석 기간:", WINDOW_START.isoformat(), "~", ASSESSED_AT.isoformat())
    print("짧은 간격 예매 기준:", SHORT_INTERVAL_SECONDS, "초")