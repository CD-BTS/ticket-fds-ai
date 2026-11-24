"""정상 사용자 실측 이벤트를 분석한다. 읽기 전용이며 데이터셋 생성·모델 학습을 하지 않는다.

    cd ai\\experiments\\issue17-event-based-dummy
    $py = "..\\issue14-real-validation\\.venv\\Scripts\\python"
    & $py measurement\\scripts\\analyze.py --run-id 20261125-u01
    & $py measurement\\scripts\\analyze.py --run-id 20261125-u01 --run-id 20261126-u02 --name all    # 여러 수집을 합쳐서

입력: measurement/data/<run-id>/raw_events.csv (collect.py export 의 출력)

계산 방식
  - 시도의 시작 시각 = attemptId 의 가장 이른 BOOKING_ATTEMPT 이벤트의 occurredAt. BTS FeatureAggregationService 와 같다.
  - 6개 Feature 는 #14 의 추출기(extract_features.py, BTS 계산식 재현)를 import 해서 그대로 계산한다. 추출기는 수정하지 않는다.
    카드·IP 공유 수는 같은 DB 에 있던 모든 계정(보조 계정 포함)을 대상으로 센다 (BTS 와 같다).
  - 세션 기준 간격은 extract_session_features.py (REPORT_V3_SESSION.md 의 정의)의 함수를 그대로 쓴다. 세션 정의 두 가지를 같이 계산한다.
        공연+30분 : 앞 시도와 공연이 같고 시작 간격이 30분 이하
        30분만    : 시작 간격이 30분 이하 (활동 세션. 공연은 보지 않는다)
  - 이벤트에 없는 값은 만들지 않는다. 해당 단계 이벤트가 없으면 그 시각과 소요 시간은 결측이다.
    결제 실패는 이벤트로 남지 않으므로(코드 확인) 이 분석에 나타나지 않는다.

출력
  measurement/data/_analysis/<이름>/   시도 1행(attempts.csv), 간격 1행(gaps.csv), 계정별 Feature(features.csv)   — git 제외
  measurement/output/<이름>/           집계표(CSV)와 그래프
"""
import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MEASUREMENT = HERE.parent
EXP = MEASUREMENT.parent
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP.parent / "issue14-real-validation" / "scripts"))
sys.path.insert(0, str(EXP.parent / "issue16-booking-interval" / "scripts"))
from analyze_interval_validation import CONTINUOUS, SOURCES, describe, read_real, synthetic_attempts  # noqa: E402
from compare_if_models import GRID, INK, MUTED, save, style  # noqa: E402
from extract_features import MODEL_FEATURES, extract, load_events  # noqa: E402  (#14 추출기, 수정 없음)
from extract_session_features import INTERVAL, VARIANCE, aggregate, assign_sessions, attempt_table, session_table  # noqa: E402

SESSION_GAP_S = 1800
EXPERIMENTS = {"A": "A 자연스러운 예매", "B": "B 숙련된 사용자 (같은 공연 반복)", "C": "C 실패 후 재시도"}
OUTCOMES = {"completed": "예매 완료", "abandoned_seat_selection": "좌석 선택에서 이탈", "abandoned_payment": "결제 화면에서 이탈",
            "abandoned_other": "대기열·시작 단계에서 이탈", "hold_expired": "선점 만료", "open": "종료 이벤트 없음 (탭 닫기 등)"}
DURATIONS = [("queue_s", "시작 → 대기열 통과"), ("to_seat_page_s", "대기열 통과 → 좌석 선택 진입"), ("seat_first_request_s", "좌석 선택 진입 → 첫 선점 요청"),
             ("seat_to_hold_s", "좌석 선택 진입 → 선점 성공"), ("hold_to_payment_page_s", "선점 성공 → 결제 화면 진입"),
             ("payment_s", "결제 화면 진입 → 예매 완료"), ("hold_to_booking_s", "선점 성공 → 예매 완료"), ("start_to_booking_s", "시작 → 예매 완료"),
             ("start_to_end_s", "시작 → 종료 이벤트 (완료·이탈·만료)")]
C_NEW, C_REAL, C_NORMAL, C_ANOM = "#6250d6", "#1f2328", "#2a78d6", "#eb6834"


def first_time(frame, stage, statuses):
    part = frame[(frame["stage"] == stage) & frame["status"].isin(statuses)]
    return part.groupby("attempt_id")["occurred_at"].min()


def lookup(series, keys, kind=None):
    """series(색인 = attemptId 등)에서 keys 순서대로 값을 꺼낸다. 해당 이벤트가 한 건도 없어도(빈 series) 결측으로 채운다."""
    values = series.reindex(keys).to_numpy()
    if kind == "time":
        return pd.to_datetime(pd.Series(values, index=keys.index), errors="coerce")
    return pd.Series(values, index=keys.index)


def build_attempts(events):
    """참가자 계정의 시도 1행. 시각은 모두 이벤트에서 온다. 이벤트가 없으면 NaT."""
    stage = events[events["event_type"] == "BOOKING_ATTEMPT"].copy()
    for key, column in (("attemptId", "attempt_id"), ("concertId", "concert"), ("stage", "stage"), ("status", "status"), ("reasonCode", "reason")):
        stage[column] = stage["meta"].map(lambda m, k=key: m.get(k))
    stage = stage[stage["attempt_id"].notna()].sort_values(["occurred_at", "event_id"])
    keys = ["external_user_id", "participant", "experiment", "round", "user_id"]
    table = stage.groupby("attempt_id").agg(start=("occurred_at", "min"), concert=("concert", "first"), n_events=("event_id", "size"),
                                            **{k: (k, "first") for k in keys}).reset_index()
    times = {"t_queue_ok": ("QUEUE", ["SUCCEEDED"]), "t_seat_page": ("SEAT_SELECTION", ["ENTERED"]), "t_first_hold_request": ("SEAT_HOLD", ["SUCCEEDED", "REJECTED"]),
             "t_hold_ok": ("SEAT_HOLD", ["SUCCEEDED"]), "t_payment_page": ("PAYMENT", ["ENTERED"]), "t_completed": ("COMPLETED", ["SUCCEEDED"]),
             "t_expired": ("SEAT_HOLD", ["EXPIRED"])}
    for column, (name, statuses) in times.items():
        table[column] = lookup(first_time(stage, name, statuses), table["attempt_id"], "time")
    abandoned = stage[stage["status"] == "ABANDONED"].groupby("attempt_id")
    table["t_abandoned"] = lookup(abandoned["occurred_at"].min(), table["attempt_id"], "time")
    table["abandoned_stage"] = lookup(abandoned["stage"].first(), table["attempt_id"])
    rejected = stage[(stage["stage"] == "SEAT_HOLD") & (stage["status"] == "REJECTED")].groupby("attempt_id")
    table["hold_rejections"] = lookup(rejected.size(), table["attempt_id"]).fillna(0).astype(int)
    table["reject_reasons"] = lookup(rejected["reason"].agg(lambda v: "|".join(sorted(set(map(str, v))))), table["attempt_id"]).fillna("")
    table["t_last_rejection"] = lookup(rejected["occurred_at"].max(), table["attempt_id"], "time")
    last = stage.groupby("attempt_id").tail(1).set_index("attempt_id")
    table["last_event"] = lookup(last["stage"] + "/" + last["status"], table["attempt_id"])
    table["t_last_event"] = lookup(last["occurred_at"], table["attempt_id"], "time")

    bookings = events[events["event_type"] == "BOOKING"].assign(attempt_id=lambda d: d["meta"].map(lambda m: m.get("attemptId")),
                                                               seats=lambda d: d["meta"].map(lambda m: len(m.get("ticketIds") or []) or np.nan))
    bookings = bookings.drop_duplicates("booking_id").set_index("attempt_id")
    bookings = bookings[bookings.index.notna()]
    table["t_booking"] = lookup(bookings["occurred_at"], table["attempt_id"], "time")
    table["booking_id"] = lookup(bookings["booking_id"], table["attempt_id"]).fillna("")
    table["seats_booked"] = lookup(bookings["seats"], table["attempt_id"])
    cancels = events[events["event_type"] == "CANCEL"].groupby("booking_id")["occurred_at"].min()
    table["t_cancel"] = lookup(cancels, table["booking_id"], "time")

    def outcome(row):
        if pd.notna(row["t_completed"]) or pd.notna(row["t_booking"]):
            return "completed"
        if pd.notna(row["t_expired"]):
            return "hold_expired"
        if pd.notna(row["t_abandoned"]):
            return {"SEAT_SELECTION": "abandoned_seat_selection", "PAYMENT": "abandoned_payment"}.get(row["abandoned_stage"], "abandoned_other")
        return "open"

    table["outcome"] = table.apply(outcome, axis=1)
    table["failure_reason"] = np.select([table["outcome"] == "hold_expired", table["outcome"].str.startswith("abandoned")], ["HOLD_EXPIRED", "USER_CANCELLED"], default="")
    table["cancelled_later"] = table["t_cancel"].notna()
    table["t_end"] = table["t_completed"].fillna(table["t_booking"]).fillna(table["t_expired"]).fillna(table["t_abandoned"])
    seconds = lambda a, b: (table[b] - table[a]).dt.total_seconds()
    table["queue_s"] = seconds("start", "t_queue_ok")
    table["to_seat_page_s"] = seconds("t_queue_ok", "t_seat_page")
    table["seat_first_request_s"] = seconds("t_seat_page", "t_first_hold_request")
    table["seat_to_hold_s"] = seconds("t_seat_page", "t_hold_ok")
    table["rejection_to_hold_s"] = seconds("t_last_rejection", "t_hold_ok")
    table["hold_to_payment_page_s"] = seconds("t_hold_ok", "t_payment_page")
    table["payment_s"] = seconds("t_payment_page", "t_booking")
    table["hold_to_booking_s"] = seconds("t_hold_ok", "t_booking")
    table["start_to_booking_s"] = seconds("start", "t_booking")
    table["start_to_end_s"] = seconds("start", "t_end")
    table["booking_to_cancel_s"] = seconds("t_booking", "t_cancel")

    table = table.sort_values(["external_user_id", "start", "attempt_id"]).reset_index(drop=True)
    by = table.groupby("external_user_id")
    table["attempt_no"] = by.cumcount() + 1
    table["gap_s"] = by["start"].diff().dt.total_seconds()
    table["previous_outcome"] = by["outcome"].shift()
    table["previous_had_rejection"] = by["hold_rejections"].shift().fillna(0) > 0
    table["previous_end_to_start_s"] = (table["start"] - by["t_end"].shift()).dt.total_seconds()
    table["same_concert_as_previous"] = table["concert"] == by["concert"].shift()
    table["in_session_gap_only"] = table["gap_s"] <= SESSION_GAP_S
    table["in_session_concert"] = table["in_session_gap_only"] & table["same_concert_as_previous"]
    return table


def summarize(frame, by, value_columns):
    rows = []
    for key, part in frame.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for column, title in value_columns:
            rows.append({**dict(zip(by, key)), "quantity": title, **describe(part[column])})
    return pd.DataFrame(rows)


def session_features(events, calculated_at, accounts):
    """계정별 세션 기준 간격 Feature (모든 간격의 평균·모분산). extract_session_features 의 함수 그대로."""
    table = attempt_table(events, pd.Timestamp(calculated_at))
    out = accounts[["user_id"]].copy()
    for label, use_concert in (("concert30", True), ("gap30", False)):
        attempts = assign_sessions(table, use_concert=use_concert)
        values = aggregate(session_table(attempts), attempts, accounts["user_id"]).set_index("user_id")
        out[f"n_sessions__{label}"] = values["n_sessions"].to_numpy()
        out[f"session_gaps__{label}"] = attempts.dropna(subset=["gap"]).groupby("user_id").size().reindex(accounts["user_id"]).fillna(0).astype(int).to_numpy()
        out[f"{INTERVAL}__session_{label}"] = values[f"{INTERVAL}__pooled"].to_numpy()
        out[f"{VARIANCE}__session_{label}"] = values[f"{VARIANCE}__pooled"].to_numpy()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", action="append", required=True, help="measurement/data/<run-id>. 여러 번 줄 수 있다")
    parser.add_argument("--name", default=None, help="결과 폴더 이름 (기본: run-id, 여러 개면 필수)")
    parser.add_argument("--calculated-at", default=None, help="Feature 계산 기준 시각(UTC, ISO). 기본: 마지막 이벤트 + 1초")
    parser.add_argument("--earlier-run-id", action="append", default=[],
                        help="이전 수집의 run-id. 그 수집에 있던 계정을 '이전 수집', 나머지를 '추가 수집' 으로 나눠 본다 (읽기만 한다)")
    args = parser.parse_args()
    if len(args.run_id) > 1 and not args.name:
        sys.exit("run-id 가 여러 개면 --name 으로 결과 폴더 이름을 정하세요.")
    name = args.name or args.run_id[0]
    private, public = MEASUREMENT / "data" / "_analysis" / name, MEASUREMENT / "output" / name
    private.mkdir(parents=True, exist_ok=True)
    (public / "figures").mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 300)

    frames, person_source = [], {}
    for run_id in args.run_id:
        path = MEASUREMENT / "data" / run_id / "raw_events.csv"
        if not path.exists():
            sys.exit(f"{path} 가 없습니다. 먼저 collect.py export --run-id {run_id} 를 실행하세요.")
        frame = load_events(path)
        frame["run_id"] = run_id
        # 수집마다 DB 가 새로 만들어지므로 event_id 와 BTS user_id 가 겹칠 수 있다. 수집 이름을 붙여 구분한다
        frame["user_id"] = run_id + ":" + frame["user_id"]
        # 계정 ID 의 참가자 번호(u01, u02 ...)는 계정을 구분하는 표시일 뿐이다. 실제로 누가 했는지는 manifest.csv 의 person 열을 따른다.
        # 한 사람이 여러 번호를 썼으면 같은 person 값을 적는다. 그 계정들은 한 사람의 반복 측정으로 집계된다.
        manifest_path = MEASUREMENT / "data" / run_id / "manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
            if "person" in manifest.columns:
                people = {row["account"]: row["person"] for _, row in manifest.iterrows() if row["person"]}
                frame["participant"] = [people.get(account, old) for account, old in zip(frame["external_user_id"], frame["participant"])]
                person_source[run_id] = f"manifest.csv 의 person 열 ({len(people)}개 계정)"
        person_source.setdefault(run_id, "계정 ID 의 참가자 번호 (manifest.csv 에 person 열이 없다. 서로 다른 사람인지 확인되지 않은 값)")
        frames.append(frame)
    events = pd.concat(frames, ignore_index=True).sort_values(["occurred_at", "event_id"]).reset_index(drop=True)
    mine = events[events["group"] == "participant"]
    if mine.empty:
        sys.exit("참가자 계정(u01-a-01 형식)의 이벤트가 없습니다.")

    def csv(frame, folder, filename):
        frame.to_csv(folder / filename, index=False, encoding="utf-8-sig")

    attempts = build_attempts(mine)
    earlier_accounts = set()
    for run_id in args.earlier_run_id:
        earlier = pd.read_csv(MEASUREMENT / "data" / run_id / "raw_events.csv", dtype=str, keep_default_na=False, usecols=["external_user_id"])
        earlier_accounts |= set(earlier["external_user_id"])
    attempts["wave"] = np.where(attempts["external_user_id"].isin(earlier_accounts), "이전 수집", "추가 수집") if args.earlier_run_id else "(구분 없음)"
    csv(attempts, private, "attempts.csv")
    gaps = attempts.dropna(subset=["gap_s"])
    csv(gaps[["external_user_id", "participant", "experiment", "attempt_no", "concert", "start", "gap_s", "previous_outcome", "previous_had_rejection",
              "previous_end_to_start_s", "same_concert_as_previous", "in_session_concert", "in_session_gap_only"]], private, "gaps.csv")

    # ---- 1) 사용자별·실험 유형별 시도 횟수 ----
    counts = (attempts.groupby(["participant", "experiment", "external_user_id"])
              .agg(attempts=("attempt_id", "size"), concerts=("concert", "nunique"), completed=("outcome", lambda v: int((v == "completed").sum())),
                   abandoned_seat_selection=("outcome", lambda v: int((v == "abandoned_seat_selection").sum())),
                   abandoned_payment=("outcome", lambda v: int((v == "abandoned_payment").sum())),
                   abandoned_other=("outcome", lambda v: int((v == "abandoned_other").sum())),
                   hold_expired=("outcome", lambda v: int((v == "hold_expired").sum())), open_no_terminal_event=("outcome", lambda v: int((v == "open").sum())),
                   attempts_with_hold_rejection=("hold_rejections", lambda v: int((v > 0).sum())), hold_rejections=("hold_rejections", "sum"),
                   cancelled_later=("cancelled_later", "sum"), gaps_same_concert_30min=("in_session_concert", "sum"), gaps_30min_only=("in_session_gap_only", "sum"),
                   first_start_utc=("start", "min"), last_start_utc=("start", "max")).reset_index())
    csv(counts, private, "attempt_counts.csv")  # 계정 단위 표: git 에 올리지 않는다

    # ---- 2) 단계별 소요 시간 ----
    label = attempts.assign(experiment_name=attempts["experiment"].map(EXPERIMENTS))
    stage_by_experiment = summarize(label, ["experiment_name"], DURATIONS + [("rejection_to_hold_s", "마지막 선점 거절 → 선점 성공"), ("booking_to_cancel_s", "예매 완료 → 취소")])
    stage_by_participant = summarize(label, ["participant", "experiment_name"], DURATIONS)
    csv(stage_by_experiment, public, "stage_times_by_experiment.csv")
    csv(stage_by_participant, public, "stage_times_by_participant.csv")

    # ---- 3) 연속 시도 간격: 같은 공연 안 / 활동 세션(시간 기준) ----
    rows = []
    for definition, mask in (("공연+30분 (같은 공연 안)", "in_session_concert"), ("30분만 (활동 세션)", "in_session_gap_only")):
        part = label[label[mask]]
        for by in (["experiment_name"], ["participant", "experiment_name"]):
            for key, group in part.groupby(by, sort=False):
                key = key if isinstance(key, tuple) else (key,)
                rows.append({"session_definition": definition, "participant": "(전체)", **dict(zip(by, key)), **describe(group["gap_s"])})
    gap_summary = pd.DataFrame(rows)
    csv(gap_summary, public, "gap_summary.csv")

    # ---- 3b) 계정별, 수집 차수별, 참가자 단위 ----
    # 한 참가자의 시도 여러 건은 독립 표본이 아니다. 참가자(계정)마다 중앙값을 하나씩 내고, 그 값들을 나란히 본다.
    keys = ["wave", "participant", "experiment", "external_user_id"]
    stage_by_account = summarize(label, keys, DURATIONS)
    csv(stage_by_account, private, "stage_times_by_account.csv")  # 계정 단위 표: git 에 올리지 않는다
    csv(summarize(label, ["wave", "experiment_name"], DURATIONS), public, "stage_times_by_wave.csv")
    rows = []
    for key, part in label.groupby(keys, sort=False):
        done = part[part["outcome"] == "completed"]
        same, active = part.loc[part["in_session_concert"], "gap_s"], part.loc[part["in_session_gap_only"], "gap_s"]
        rest = part.loc[part["in_session_gap_only"] & (part["previous_outcome"] == "completed"), "previous_end_to_start_s"]
        rows.append({**dict(zip(keys, key)), "attempts": len(part), "completed": len(done), "concerts": part["concert"].nunique(),
                     "start_to_booking_median": done["start_to_booking_s"].median(), "start_to_booking_min": done["start_to_booking_s"].min(),
                     "start_to_booking_max": done["start_to_booking_s"].max(), "seat_median": done["seat_to_hold_s"].median(), "payment_median": done["payment_s"].median(),
                     "gaps_same_concert": int(same.notna().sum()), "gap_same_concert_median": same.median(), "gap_same_concert_mean": same.mean(),
                     "gaps_activity": int(active.notna().sum()), "gap_activity_median": active.median(), "gap_activity_mean": active.mean(),
                     "gap_activity_min": active.min(), "gap_activity_max": active.max(), "gap_activity_std": active.std(ddof=1) if active.notna().sum() > 1 else np.nan,
                     "after_completion_to_next_start_n": int(rest.notna().sum()), "after_completion_to_next_start_median": rest.median(),
                     "first_attempt_start_to_booking": done.sort_values("start")["start_to_booking_s"].iloc[0] if len(done) else np.nan,
                     "first_attempt_seat": done.sort_values("start")["seat_to_hold_s"].iloc[0] if len(done) else np.nan,
                     "account_started_utc": part["start"].min(), "account_last_event_utc": part["t_last_event"].max()})
    by_account = pd.DataFrame(rows).sort_values("account_started_utc").reset_index(drop=True)
    by_account["seconds_since_previous_account_ended"] = (by_account["account_started_utc"] - by_account["account_last_event_utc"].shift()).dt.total_seconds()
    csv(by_account, private, "account_summary.csv")  # 계정 단위 표: git 에 올리지 않는다
    # 참가자 단위: 계정별 중앙값의 분포 (표본 수 = 참가자·계정 수)
    rows = []
    for experiment, part in by_account.groupby("experiment", sort=False):
        for column, title in (("start_to_booking_median", "예매 1건 소요 시간의 계정 중앙값"), ("seat_median", "좌석 선택 시간의 계정 중앙값"),
                              ("payment_median", "결제 시간의 계정 중앙값"), ("gap_activity_median", "활동 세션 간격의 계정 중앙값"),
                              ("gap_same_concert_median", "같은 공연 안 간격의 계정 중앙값"), ("after_completion_to_next_start_median", "완료 → 다음 시작의 계정 중앙값")):
            values = part[column].dropna()
            rows.append({"experiment": experiment, "quantity": title, "accounts": len(values), "participants": part.loc[part[column].notna(), "participant"].nunique(),
                         "min": values.min(), "median": values.median(), "max": values.max(), "max_over_min": values.max() / values.min() if len(values) and values.min() > 0 else np.nan,
                         "values": " / ".join(f"{a}: {v:.2f}" for a, v in zip(part.loc[part[column].notna(), "external_user_id"], values))})
    between = pd.DataFrame(rows)
    csv(between, private, "between_account_summary.csv")  # 계정 단위 표: git 에 올리지 않는다
    order_rows = []
    for account, part in label.sort_values("start").groupby("external_user_id", sort=False):
        done = part[part["outcome"] == "completed"].reset_index(drop=True)
        for position, row in done.iterrows():
            order_rows.append({"external_user_id": account, "experiment": row["experiment"], "completed_order": position + 1, "start_to_booking_s": row["start_to_booking_s"],
                               "seat_to_hold_s": row["seat_to_hold_s"], "payment_s": row["payment_s"]})
    by_order = pd.DataFrame(order_rows)
    csv(by_order, private, "completed_attempts_in_order.csv")

    # ---- 4) 앞 시도의 결과별 재시도 간격 ----
    retry = label[label["in_session_gap_only"]].assign(previous=lambda d: d["previous_outcome"].map(OUTCOMES) + np.where(d["previous_had_rejection"], " (선점 거절 있었음)", ""))
    retry_summary = summarize(retry, ["experiment_name", "previous"], [("gap_s", "앞 시도 시작 → 이번 시도 시작"), ("previous_end_to_start_s", "앞 시도 종료 이벤트 → 이번 시도 시작")])
    csv(retry_summary, public, "retry_gap_by_previous_outcome.csv")
    events_by_kind = (mine[mine["event_type"] == "BOOKING_ATTEMPT"].assign(stage=lambda d: d["meta"].map(lambda m: m.get("stage")), status=lambda d: d["meta"].map(lambda m: m.get("status")),
                                                                         reason=lambda d: d["meta"].map(lambda m: m.get("reasonCode") or ""))
                      .groupby(["experiment", "stage", "status", "reason"]).size().reset_index(name="events"))
    other = mine[mine["event_type"] != "BOOKING_ATTEMPT"].groupby(["experiment", "event_type"]).size().reset_index(name="events").rename(columns={"event_type": "stage"})
    recorded = pd.concat([events_by_kind, other.assign(status="", reason="")], ignore_index=True)
    csv(recorded, public, "recorded_event_kinds.csv")

    # ---- 5) 6개 Feature (BTS 계산식) 와 세션 기준 간격 ----
    calculated_at = pd.Timestamp(args.calculated_at).tz_localize(None).to_pydatetime() if args.calculated_at else (events["occurred_at"].max() + timedelta(seconds=1))
    features, _, extraction = extract(events, calculated_at, SESSION_GAP_S / 60)
    features = features[features["group"] == "participant"].reset_index(drop=True)
    features = features.merge(session_features(events, calculated_at, features), on="user_id")
    meta = attempts.drop_duplicates("external_user_id").set_index("external_user_id")[["participant", "experiment"]]
    features.insert(1, "participant", features["external_user_id"].map(meta["participant"]))
    features.insert(2, "experiment", features["external_user_id"].map(meta["experiment"]))
    csv(features.drop(columns="user_id"), private, "features.csv")
    feature_columns = MODEL_FEATURES + [f"{INTERVAL}__session_concert30", f"{VARIANCE}__session_concert30", f"{INTERVAL}__session_gap30", f"{VARIANCE}__session_gap30"]
    feature_summary = summarize(features.assign(experiment_name=features["experiment"].map(EXPERIMENTS)), ["experiment_name"], [(c, c) for c in feature_columns])
    feature_summary["accounts_null"] = [int(features.loc[features["experiment"].map(EXPERIMENTS) == row.experiment_name, row.quantity].isna().sum()) for row in feature_summary.itertuples()]
    csv(feature_summary, public, "feature_summary.csv")
    # 일치 확인: 추출기의 시도 수·간격 평균이 시도 표에서 직접 계산한 값과 같은가
    direct = attempts.groupby("external_user_id").agg(n=("attempt_id", "size"), mean_gap=("gap_s", "mean")).reindex(features["external_user_id"])
    checks = {"booking_count_equals_distinct_attempts": bool((features["booking_count"].to_numpy() == direct["n"].to_numpy()).all()),
              "avg_booking_interval_equals_mean_start_gap": bool(np.allclose(features["avg_booking_interval"].astype(float), direct["mean_gap"].astype(float), atol=1e-6, equal_nan=True)),
              "interval_null_iff_fewer_than_2_attempts": bool((features["avg_booking_interval"].isna() == (features["booking_count"] < 2)).all())}

    # ---- 6) 기존 실측(#14, #16)·v3 합성과의 비교 ----
    rows = []
    for key, part in label.groupby("experiment_name", sort=False):
        rows.append({"source": "이번 실측", "group": key, "quantity": "연속 시도 간격: 같은 공연 안", **describe(part.loc[part["in_session_concert"], "gap_s"])})
        rows.append({"source": "이번 실측", "group": key, "quantity": "연속 시도 간격: 활동 세션(30분)", **describe(part.loc[part["in_session_gap_only"], "gap_s"])})
        done = part[part["outcome"] == "completed"]
        rows.append({"source": "이번 실측", "group": key, "quantity": "시작 → 예매 완료", **describe(done["start_to_booking_s"])})
        rows.append({"source": "이번 실측", "group": key, "quantity": "좌석 선택 진입 → 선점 성공", **describe(done["seat_to_hold_s"])})
    reference_groups = []
    if all(path.exists() for _, path, _ in SOURCES):
        real = pd.concat([read_real(*source)[0] for source in SOURCES], ignore_index=True)
        picks = [("#14 사람 (1명 2계정)", real[real["kind"] == "사람"])]
        picks += [(f"#16 {kind} (연속 실행)", real[(real["kind"] == kind) & (real["macro_run_mode"] == CONTINUOUS)]) for kind in ("매크로 fast", "매크로 jitter", "매크로 human 프로필")]
        for title, part in picks:
            done = part[part["completed"]]
            rows.append({"source": "기존 실측", "group": title, "quantity": "연속 시도 간격: 같은 공연 안", **describe(part.loc[part["in_session_concert"], "gap"])})
            rows.append({"source": "기존 실측", "group": title, "quantity": "연속 시도 간격: 활동 세션(30분)", **describe(part.loc[part["in_session_gap_only"], "gap"])})
            rows.append({"source": "기존 실측", "group": title, "quantity": "시작 → 예매 완료", **describe(done["start_to_booking"])})
            rows.append({"source": "기존 실측", "group": title, "quantity": "좌석 선택 진입 → 선점 성공", **describe(done["seat"])})
            reference_groups.append((title, part.loc[part["in_session_gap_only"], "gap"], C_REAL))
    else:
        print("참고: #14·#16 실측 원본이 이 PC 에 없어 기존 실측과의 비교는 건너뜁니다.")
    if (EXP / "data" / "v3" / "attempts.csv").exists():
        synthetic, _ = synthetic_attempts()
        for scenario in ("normal", "fast_normal", "shared_ip", "frequent_cancel", "fast", "jitter", "multi_account", "hold_cancel"):
            part = synthetic[synthetic["scenario"] == scenario]
            done = part[part["outcome"] == "complete"]
            rows.append({"source": "v3 합성", "group": scenario, "quantity": "연속 시도 간격: 같은 공연 안", **describe(part.loc[part["in_session"], "gap"])})
            rows.append({"source": "v3 합성", "group": scenario, "quantity": "시작 → 예매 완료", **describe(done["start_to_booking"])})
            rows.append({"source": "v3 합성", "group": scenario, "quantity": "좌석 선택 진입 → 선점 성공", **describe(done["seat_ms"] / 1000)})
            reference_groups.append((f"v3 {scenario}", part.loc[part["in_session"], "gap"], C_NORMAL if scenario in ("normal", "fast_normal", "shared_ip", "frequent_cancel") else C_ANOM))
    comparison = pd.DataFrame(rows)
    csv(comparison, public, "comparison_with_existing.csv")

    # 이번 실측의 간격이 기존 자료의 범위와 얼마나 겹치는가 (간격 1개 단위)
    rows = []
    for key, part in label.groupby("experiment_name", sort=False):
        for definition, mask in (("같은 공연 안", "in_session_concert"), ("활동 세션(30분)", "in_session_gap_only")):
            mine_gaps = part.loc[part[mask], "gap_s"].dropna().to_numpy()
            if len(mine_gaps) == 0:
                continue
            for title, values, _ in reference_groups:
                ref = values.dropna().to_numpy()
                if len(ref) == 0:
                    continue
                low, high = np.quantile(ref, [0.05, 0.95])
                rows.append({"experiment": key, "session_definition": definition, "measured_gaps": len(mine_gaps), "measured_min": mine_gaps.min(),
                             "measured_median": float(np.median(mine_gaps)), "measured_max": mine_gaps.max(), "reference": title, "reference_n": len(ref),
                             "reference_min": ref.min(), "reference_p05": low, "reference_median": float(np.median(ref)), "reference_p95": high, "reference_max": ref.max(),
                             "measured_inside_reference_range": float(((mine_gaps >= ref.min()) & (mine_gaps <= ref.max())).mean()),
                             "measured_inside_reference_p05_p95": float(((mine_gaps >= low) & (mine_gaps <= high)).mean()),
                             "measured_below_reference_min": float((mine_gaps < ref.min()).mean()), "measured_above_reference_max": float((mine_gaps > ref.max()).mean()),
                             "reference_share_below_measured_median": float((ref < np.median(mine_gaps)).mean())})
    overlap = pd.DataFrame(rows)
    csv(overlap, public, "gap_overlap_with_existing.csv")

    # ---------------- 그래프 ----------------
    rng = np.random.default_rng(0)
    groups = []
    for key, part in label.groupby("experiment_name", sort=False):
        groups.append((f"이번 실측 {key}\n같은 공연 안 ({int(part['in_session_concert'].sum())}개)", part.loc[part["in_session_concert"], "gap_s"], C_NEW))
        groups.append((f"이번 실측 {key}\n활동 세션 ({int(part['in_session_gap_only'].sum())}개)", part.loc[part["in_session_gap_only"], "gap_s"], C_NEW))
    groups += [(f"{title} ({int(values.notna().sum())}개)", values, color) for title, values, color in reference_groups]
    fig, ax = plt.subplots(figsize=(12, 0.52 * len(groups) + 1.6), constrained_layout=True)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for position, (title, values, color) in enumerate(groups[::-1]):
        values = values.dropna().to_numpy()
        if len(values) > 400:
            values = rng.choice(values, 400, replace=False)
        ax.scatter(values, position + rng.uniform(-0.28, 0.28, len(values)), s=10, alpha=0.55, linewidth=0, color=color)
        if len(values):
            ax.vlines(np.median(values), position - 0.4, position + 0.4, color=INK, linewidth=1.5)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in groups[::-1]], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("연속한 두 시도의 시작 간격 (초, 로그 눈금)", fontsize=9, color=MUTED)
    ax.set_title("연속 시도 간격: 이번 실측과 기존 실측·v3 합성. 세로선 = 중앙값", loc="left", fontsize=10, color=INK)
    save(fig, public / "figures" / "gaps.png")

    shown = [("seat_to_hold_s", "좌석 선택 진입 → 선점 성공"), ("payment_s", "결제 화면 진입 → 예매 완료"), ("start_to_booking_s", "시작 → 예매 완료"),
             ("previous_end_to_start_s", "앞 시도 종료 → 다음 시도 시작")]
    fig, axes = plt.subplots(1, len(shown), figsize=(15, 3.8), constrained_layout=True)
    order = [EXPERIMENTS[k] for k in "ABC" if EXPERIMENTS[k] in set(label["experiment_name"])]
    for ax, (column, title) in zip(axes, shown):
        style(ax)
        for position, key in enumerate(order):
            values = label.loc[label["experiment_name"] == key, column].dropna().to_numpy()
            values = values[values > 0]
            ax.scatter(position + rng.uniform(-0.18, 0.18, len(values)), values, s=14, alpha=0.6, linewidth=0, color=C_NEW)
            if len(values):
                ax.hlines(np.median(values), position - 0.3, position + 0.3, color=INK, linewidth=1.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([k.split(" ")[0] for k in order], fontsize=9)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    axes[0].set_ylabel("초 (로그 눈금)", fontsize=9, color=MUTED)
    fig.suptitle("실험 유형별 단계 소요 시간 (점 = 시도 1건, 가로선 = 중앙값)", fontsize=11, color=INK, x=0.01, ha="left")
    save(fig, public / "figures" / "stage_times.png")

    record = {"runs": args.run_id, "name": name, "calculated_at_utc": calculated_at.isoformat(),
              "participants": sorted(attempts["participant"].unique()), "participant_source": person_source,
              "independent_people": int(attempts["participant"].nunique()),
              "note_on_independence": "한 사람의 여러 계정·여러 시도는 독립 표본이 아니다. 계정을 독립 사용자로 보고 신뢰구간을 계산하거나 학습·평가 데이터로 나눠 쓰지 않는다.",
              "accounts": int(attempts["external_user_id"].nunique()), "attempts": int(len(attempts)),
              "attempts_by_experiment": attempts["experiment"].value_counts().sort_index().to_dict(),
              "feature_extractor": "issue14-real-validation/scripts/extract_features.py (BTS FeatureAggregationService 재현, import, 수정 없음)",
              "session_code": "issue17 scripts/extract_session_features.py 의 attempt_table · assign_sessions · session_table · aggregate",
              "consistency_checks": checks, "card_env_distorted": extraction["card_env_distorted"], "ip_env_distorted": extraction["ip_env_distorted"],
              "not_measurable": ["결제 실패 (이벤트 없음)", "세션·브라우저 식별자 (이벤트에 없음)", "탭 닫기·뒤로 가기 (이벤트 없음. 종료 이벤트 없는 시도로만 보인다)",
                                 "좌석을 클릭한 시각 (서버에는 선점 요청 시각만 남는다)"],
              "datasets_generated": False, "models_trained": False}
    (public / "analysis_summary.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cols = ["n", "mean", "median", "std", "min", "p05", "p25", "p75", "p95", "max"]
    print("[1] 계정별 시도 수\n", counts.drop(columns=["first_start_utc", "last_start_utc"]).to_string(index=False))
    print("\n[2] 단계별 소요 시간 (초)\n", stage_by_experiment[["experiment_name", "quantity"] + cols].round(2).to_string(index=False))
    print("\n[3] 연속 시도 간격 (초)\n", gap_summary[gap_summary["participant"] == "(전체)"][["session_definition", "experiment_name"] + cols].round(2).to_string(index=False))
    print("\n[4] 앞 시도의 결과별 재시도 간격 (초)\n", retry_summary[["experiment_name", "previous", "quantity"] + cols].round(2).to_string(index=False))
    print("\n기록된 이벤트 종류\n", recorded.to_string(index=False))
    print("\n[5] 계정별 Feature\n", features[["external_user_id", "experiment"] + feature_columns + ["n_completed_bookings", "n_cancelled_bookings"]].round(3).to_string(index=False))
    print("일치 확인:", checks, "| 카드·IP 왜곡:", extraction["card_env_distorted"], extraction["ip_env_distorted"])
    print("\n[6] 기존 자료와의 비교 (초)\n", comparison[["source", "group", "quantity"] + [c for c in cols if c in comparison]].round(2).to_string(index=False))
    print("\n[3b] 계정별 요약 (시작 순서)\n", by_account.drop(columns=["account_started_utc", "account_last_event_utc"]).round(2).to_string(index=False))
    print(between.round(2).to_string(index=False))
    if len(overlap):
        print("\n간격의 겹침 (이번 실측의 간격 중 기존 자료의 범위 안에 드는 비율)\n", overlap[
            ["experiment", "session_definition", "measured_gaps", "measured_median", "reference", "reference_min", "reference_max", "measured_inside_reference_range",
             "measured_inside_reference_p05_p95", "measured_below_reference_min", "reference_share_below_measured_median"]].round(2).to_string(index=False))
    print(f"\n저장: {public}  (계정·시도 단위 표는 {private}, git 제외)")


if __name__ == "__main__":
    main()
