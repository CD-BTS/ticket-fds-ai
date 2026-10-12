from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from inference import (
    assess,
    build_bundle,
    load_bundle,
    raw_anomaly_score,
    save_bundle,
)
from preprocessing import (
    CAMEL_TO_SNAKE,
    FEATURE_SPECS,
    PayloadValidationError,
    ScoreNormalizer,
    build_preprocessor,
)


# ========================================
# 1. 경로 및 설정 (BTS AI Feature 명세서)
# ========================================

BASE_DIR = Path(__file__).resolve().parent

FEATURE_PATH = BASE_DIR / "data" / "user_features.csv"
LABEL_PATH = BASE_DIR / "data" / "synthetic_user_labels.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "behavior_model.joblib"
RESULT_PATH = BASE_DIR / "data" / "user_predictions.csv"
COMPARISON_PATH = BASE_DIR / "data" / "feature_set_comparison.csv"

MODEL_VERSION = "iforest-v1"

# 전체 사용자 중 이상치로 분류할 비율의 예상값
CONTAMINATION = 0.05

# 신규 사용자 최소 데이터 기준 (예매 건수). 실험 후 조정한다.
MIN_BOOKING_COUNT = 2

# 1차 MVP: 상위 5개 Feature
MVP_FEATURES = [
    "booking_count",
    "cancel_count",
    "cancel_rate",
    "avg_booking_interval",
    "short_interval_booking_count",
]

# 추후 구현(추가 데이터 수집 필요): attendance_rate, avg_booking_duration
FULL_FEATURES = MVP_FEATURES + [
    "attendance_rate",
    "avg_booking_duration",
]

# 명세 3. "Feature 전부 vs 일부만" 실험 대상
FEATURE_SETS = {
    "mvp": MVP_FEATURES,
    "full": FULL_FEATURES,
}

# 최종 저장할 Feature 구성
SELECTED_FEATURE_SET = "mvp"

# 전처리 설정 (experiment_preprocessing.py 실험으로 결정)
PREPROCESS = {
    "imputation": "median",
    "scaling": "standard",
    "log_transform": False,
}

# 이상도 점수 정규화 방식: "percentile" | "minmax" | "raw"
SCORE_METHOD = "percentile"


# ========================================
# 2. 데이터 불러오기 및 검증
# ========================================

def load_features():
    features_df = pd.read_csv(FEATURE_PATH)

    required_columns = ["user_id"] + FEATURE_SETS["full"]
    missing_columns = [
        col for col in required_columns if col not in features_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"user_features.csv에 필요한 컬럼이 없습니다: {missing_columns}"
        )

    if features_df.empty:
        raise ValueError("학습할 사용자 데이터가 없습니다.")

    if features_df["user_id"].duplicated().any():
        raise ValueError("user_features.csv에 중복 user_id가 있습니다.")

    # 숫자 형태로 변환할 수 없는 값(null 포함)은 결측치로 처리
    for column in FEATURE_SETS["full"]:
        features_df[column] = pd.to_numeric(features_df[column], errors="coerce")

    return features_df


# ========================================
# 3. 모델 정의 및 학습
# ========================================

def build_model(feature_columns, preprocess=None, random_state=42):
    preprocess = {**PREPROCESS, **(preprocess or {})}

    # 전처리(결측 대체 → 변환 → 스케일링) → 이상치 탐지
    return Pipeline([
        ("preprocess", build_preprocessor(feature_columns, **preprocess)),
        (
            "isolation_forest",
            IsolationForest(
                n_estimators=200,
                contamination=CONTAMINATION,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    ])


def train_and_score(
    features_df,
    feature_columns,
    preprocess=None,
    score_method=None,
    random_state=42,
):
    """최소 데이터 기준을 만족한 사용자만 학습/평가하고, 나머지는 별도 상태로 둔다."""
    eligible = (features_df["booking_count"] >= MIN_BOOKING_COUNT).to_numpy()

    if eligible.sum() == 0:
        raise ValueError("최소 데이터 기준을 만족하는 사용자가 없습니다.")

    X = features_df.loc[eligible, feature_columns]

    all_missing = X.columns[X.isna().all()].tolist()
    if all_missing:
        raise ValueError(f"모든 값이 비어 있는 특징이 있습니다: {all_missing}")

    model = build_model(feature_columns, preprocess, random_state)
    model.fit(X)  # 정답 라벨 없이 특징 데이터만으로 학습

    labels = model.predict(X)  # 1: 정상, -1: 이상치
    raw_scores = raw_anomaly_score(model, X)

    normalizer = ScoreNormalizer(score_method or SCORE_METHOD).fit(raw_scores)

    results = features_df[["user_id"]].copy()
    results["status"] = np.where(eligible, "OK", "INSUFFICIENT_DATA")
    results["raw_score"] = np.nan
    results["anomaly_score"] = np.nan
    results["is_anomaly"] = False

    results.loc[eligible, "raw_score"] = raw_scores
    results.loc[eligible, "anomaly_score"] = normalizer.transform(raw_scores)
    results.loc[eligible, "is_anomaly"] = labels == -1

    return model, normalizer, results


# ========================================
# 4. 합성 데이터 기준 비교 (모델 학습에는 사용하지 않음)
# ========================================

def load_labels():
    if not LABEL_PATH.exists():
        return None
    return pd.read_csv(LABEL_PATH)


def evaluate(results, labels_df):
    """생성된 사용자 유형과 비교한다. 정답 라벨 기반 평가는 합성 데이터에서만 가능하다."""
    if labels_df is None:
        return {}

    merged = results.merge(labels_df, on="user_id", how="left")
    merged = merged[merged["status"] == "OK"]

    summary = {}

    # 연속 예매가 매우 빠른 유형(rapid_booking)을 얼마나 상위로 올리는지
    summary["auc_rapid_vs_rest"] = roc_auc_score(
        merged["user_type"] == "rapid_booking", merged["anomaly_score"]
    )

    # 비정상 유형(rapid_booking + frequent_cancel) vs 정상
    abnormal = merged["user_type"].isin(["rapid_booking", "frequent_cancel"])
    summary["auc_abnormal_vs_normal"] = roc_auc_score(
        abnormal, merged["anomaly_score"]
    )

    for user_type, group in merged.groupby("user_type"):
        summary[f"detect_rate_{user_type}"] = group["is_anomaly"].mean()
        summary[f"mean_score_{user_type}"] = group["anomaly_score"].mean()

    return summary


# ========================================
# 5. 번들 생성 / 입력 형식 데모
#    (저장·로드·예측 로직은 inference.py에 있고, FastAPI 서버와 공용이다)
# ========================================

def make_bundle(model, normalizer, feature_columns):
    """학습 결과를 서빙용 번들로 묶는다. 이상치 경계(정규화 점수)도 함께 계산해 둔다."""
    threshold_raw = -model.named_steps["isolation_forest"].offset_
    anomaly_threshold = float(normalizer.transform([threshold_raw])[0])

    return build_bundle(
        model=model,
        normalizer=normalizer,
        feature_columns=feature_columns,
        preprocess=PREPROCESS,
        score_method=SCORE_METHOD,
        anomaly_threshold=anomaly_threshold,
        model_version=MODEL_VERSION,
        min_booking_count=MIN_BOOKING_COUNT,
    )


def to_payload(row, calculated_at):
    """학습 결과의 한 행을 Spring Boot 입력 형식으로 바꾼다 (데모용)."""
    features = {}
    for camel, snake in CAMEL_TO_SNAKE.items():
        value = row[snake]
        if pd.isna(value):
            features[camel] = None
        elif FEATURE_SPECS[snake]["integer"]:
            features[camel] = int(value)
        else:
            features[camel] = float(value)

    return {
        "userId": row["user_id"],
        "calculatedAt": calculated_at,
        "windowDays": 30,
        "features": features,
    }


# ========================================
# 6. 실행
# ========================================

if __name__ == "__main__":
    features_df = load_features()
    labels_df = load_labels()

    insufficient = int((features_df["booking_count"] < MIN_BOOKING_COUNT).sum())
    print(f"전체 사용자 수: {len(features_df)}")
    print(
        f"최소 데이터 기준(예매 {MIN_BOOKING_COUNT}건 이상) 미충족: "
        f"{insufficient}명 → INSUFFICIENT_DATA 처리"
    )
    print(f"전처리 설정: {PREPROCESS}, 점수 정규화: {SCORE_METHOD}")

    # ---- Feature 구성별 실험 ----
    comparison_rows = []
    trained = {}

    for set_name, columns in FEATURE_SETS.items():
        model, normalizer, results = train_and_score(features_df, columns)
        trained[set_name] = (model, normalizer, results, columns)

        row = {
            "feature_set": set_name,
            "n_features": len(columns),
            "anomaly_count": int(results["is_anomaly"].sum()),
        }
        row.update(evaluate(results, labels_df))
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")

    print("\n========== Feature 구성 비교 (MVP 5개 vs 전체 7개) ==========")
    print(comparison_df.round(3).T.to_string(header=False))

    # ---- 선택한 Feature 구성으로 결과 정리 ----
    model, normalizer, results, feature_columns = trained[SELECTED_FEATURE_SET]
    print(f"\n최종 선택 Feature 구성: {SELECTED_FEATURE_SET} {feature_columns}")

    results["anomaly_rank"] = (
        results["anomaly_score"].rank(method="min", ascending=False)
    )

    results = results.merge(
        features_df[["user_id"] + FEATURE_SETS["full"]], on="user_id", how="left"
    )

    if labels_df is not None:
        results = results.merge(
            labels_df, on="user_id", how="left", validate="one_to_one"
        )

    results = results.sort_values(
        "anomaly_score", ascending=False, na_position="last"
    ).reset_index(drop=True)

    # ---- 모델 및 결과 저장 ----
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    bundle = make_bundle(model, normalizer, feature_columns)
    anomaly_threshold = bundle["anomaly_threshold"]
    save_bundle(bundle, MODEL_PATH)
    results.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    # 저장한 파일을 다시 불러와도 같은 점수가 나오는지 확인한다 (직렬화 검증).
    reloaded = load_bundle(MODEL_PATH)
    eligible = features_df[features_df["booking_count"] >= MIN_BOOKING_COUNT]
    X_check = eligible[feature_columns]
    before = normalizer.transform(raw_anomaly_score(model, X_check))
    after = reloaded["normalizer"].transform(raw_anomaly_score(reloaded["model"], X_check))
    max_diff = float(np.abs(before - after).max())
    if max_diff > 1e-9:
        raise RuntimeError(f"저장한 모델이 원본과 다른 점수를 냅니다 (최대 차이 {max_diff})")
    print(f"\n직렬화 검증 통과: 재로드 후 점수 최대 차이 {max_diff:.1e}")

    # ---- 결과 출력 ----
    scored = results[results["status"] == "OK"]
    anomaly_count = int(scored["is_anomaly"].sum())

    print("\n========== Isolation Forest 결과 ==========")
    print(f"점수 산출 대상: {len(scored)}명")
    print(f"이상치로 분류: {anomaly_count}명 ({anomaly_count / len(scored) * 100:.2f}%)")
    print(
        f"정규화 점수 범위: {scored['anomaly_score'].min():.3f} ~ "
        f"{scored['anomaly_score'].max():.3f} "
        f"(이상치 경계 {anomaly_threshold:.3f}, "
        f"원점수 {scored['raw_score'].min():.3f} ~ {scored['raw_score'].max():.3f})"
    )

    show_columns = ["user_id", "anomaly_score", "raw_score", "is_anomaly"] + feature_columns
    if "user_type" in results.columns:
        show_columns.insert(1, "user_type")

    print("\n이상도 점수가 높은 사용자 상위 10명:")
    print(scored[show_columns].head(10).round(3).to_string(index=False))

    if "user_type" in scored.columns:
        print("\n합성 데이터 유형별 탐지 결과:")
        print(
            pd.crosstab(
                scored["user_type"],
                scored["is_anomaly"],
                rownames=["생성된 사용자 유형"],
                colnames=["이상치 탐지 여부"],
            )
        )

    print("\n========== 정상/이상치 특징 평균 비교 ==========")
    print(
        scored.groupby("is_anomaly")[feature_columns]
        .mean()
        .round(3)
        .rename(index={False: "정상", True: "이상치"})
        .T
    )

    # ---- 명세 7 입출력 형식 데모 ----
    print("\n========== 입출력 형식 데모 ==========")
    demo_rows = pd.concat([
        scored.head(1),
        scored.iloc[[len(scored) // 2]],
        scored.tail(1),
    ])
    calculated_at = datetime.now(timezone.utc).isoformat()
    for _, demo_row in demo_rows.iterrows():
        print(assess(to_payload(demo_row, calculated_at), bundle))

    # 데이터 부족(신규 사용자) 요청
    new_user = {
        "userId": "new_user",
        "calculatedAt": calculated_at,
        "windowDays": 30,
        "features": {
            "bookingCount": 1, "cancelCount": 0, "cancelRate": 0.0,
            "avgBookingInterval": None, "shortIntervalBookingCount": 0,
            "attendanceRate": None, "avgBookingDuration": 60.0,
        },
    }
    print(assess(new_user, bundle))

    # 형식 오류 요청
    try:
        assess({**new_user, "features": {**new_user["features"], "cancelRate": 1.7}}, bundle)
    except PayloadValidationError as error:
        print("검증 오류:", error.errors)

    print(f"\n모델 저장 위치: {MODEL_PATH}")
    print(f"분석 결과 저장 위치: {RESULT_PATH}")
    print(f"Feature 구성 비교 저장 위치: {COMPARISON_PATH}")