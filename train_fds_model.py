"""최종 FDS 모델(6개 Feature, Isolation Forest)을 학습해 서빙 번들을 만든다.

    python train_fds_model.py                      # ml_experiment/data/fds_v4_training_features.csv → models/behavior_model.joblib
    python train_fds_model.py --out some/path.joblib

같은 CSV 와 같은 패키지 버전이면 같은 모델이 나온다 (모델 seed 고정). Docker 이미지도 이 스크립트로 모델을 만든다.
모델 파일(models/)은 git 에 올리지 않는다. 학습용 CSV 와 이 스크립트가 재현 수단이다.

설정은 Issue #17 의 v4 평가와 같다 (experiments/issue17-event-based-dummy/REPORT_V4.md, REPORT_V4_DECISION.md).
    Feature       6개. booking_interval_variance 는 간격이 5개 이상일 때만 값이 있고 아니면 null (Feature 정의 v2)
    전처리        결측 → 학습(train) 계정의 중앙값, 그다음 log1p (avg_booking_interval, booking_interval_variance, booking_count)
    모델          IsolationForest(n_estimators=100, contamination="auto")
    원시 점수     s = -score_samples (높을수록 이상)
    임계값        validation 정상 계정의 오탐률이 5% 이하가 되는 가장 낮은 원시 점수. 판정은 s >= 임계값
    정규화 점수   임계값을 0.5 에 고정하는 구간 선형 변환 (preprocessing.ThresholdAnchoredNormalizer)

학습 데이터는 합성 데이터다. 번들에 기록되는 지표는 합성 데이터 안의 값이고 실제 운영 성능이 아니다.
이전 MVP 모델(train_model.py, pack_bundle.py)은 이 스크립트와 무관하다.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from inference import build_bundle, raw_anomaly_score, save_bundle
from preprocessing import FEATURE_SPECS, ThresholdAnchoredNormalizer, build_preprocessor

BASE_DIR = Path(__file__).resolve().parent
TRAINING_CSV = BASE_DIR / "ml_experiment" / "data" / "fds_v4_training_features.csv"
TRAINING_META = TRAINING_CSV.with_suffix(".meta.json")
MODEL_PATH = BASE_DIR / "models" / "behavior_model.joblib"

FEATURES = list(FEATURE_SPECS)  # 모델 입력 순서. 요청 필드 순서와 무관하게 이 순서로 맞춘다
FEATURE_DEFINITION_VERSION = "fds-features-v2"  # v1: 분산을 간격 1개부터 계산. v2: 간격 5개 미만이면 null
MIN_INTERVALS_FOR_VARIANCE = 5
MODEL_VERSION = "fds-if-v4.0.0"
N_ESTIMATORS = 100
MODEL_SEED = 42
TARGET_FPR = 0.05
MIN_BOOKING_COUNT = 1  # bookingCount 가 0 이면 INSUFFICIENT_DATA


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_model(seed=MODEL_SEED):
    """결측 중앙값 대체 → log1p → Isolation Forest. 서빙도 이 파이프라인 하나만 호출한다."""
    steps = build_preprocessor(FEATURES, imputation="median", scaling="none", log_transform=True).steps
    forest = IsolationForest(n_estimators=N_ESTIMATORS, contamination="auto", random_state=seed)
    return Pipeline(steps + [("isolation_forest", forest)])


def raw_threshold_for_fpr(normal_scores, target=TARGET_FPR):
    """정상 계정 중 s >= 임계값 인 비율이 target 이하가 되는 가장 낮은 임계값.

    정상 점수를 큰 순서로 놓고 허용 건수(floor(target × n)) 다음 점수의 바로 위 값을 쓴다.
    (#17 의 "s > k번째 점수" 와 같은 판정이다. 판정을 s >= 임계값 으로 쓰기 위해 한 칸 올린 것이다.)
    """
    scores = np.sort(np.asarray(normal_scores, dtype=float))[::-1]
    allowed = int(np.floor(target * len(scores) + 1e-9))
    if len(scores) <= allowed:
        raise ValueError("임계값을 정할 정상 계정이 부족합니다.")
    return float(np.nextafter(scores[allowed], np.inf))


def fit(train, validation, seed=MODEL_SEED, target=TARGET_FPR):
    """(모델, 원시 임계값, 정규화기). 전처리와 정규화 구간은 train 에서만, 임계값은 validation 정상 계정에서만 계산한다."""
    model = build_model(seed).fit(train[FEATURES])
    train_scores = raw_anomaly_score(model, train[FEATURES])
    normal = validation[validation["is_anomaly"] == 0]
    threshold = raw_threshold_for_fpr(raw_anomaly_score(model, normal[FEATURES]), target)
    normalizer = ThresholdAnchoredNormalizer(float(train_scores.min()), threshold, float(train_scores.max()))
    return model, threshold, normalizer


def confusion(frame, flagged):
    y = frame["is_anomaly"].to_numpy() == 1
    tp, fp, fn, tn = int((flagged & y).sum()), int((flagged & ~y).sum()), int((~flagged & y).sum()), int((~flagged & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def train(csv_path=TRAINING_CSV, out_path=MODEL_PATH, seed=MODEL_SEED):
    data = pd.read_csv(csv_path)
    missing = [c for c in FEATURES + ["split", "is_anomaly"] if c not in data.columns]
    if missing:
        raise ValueError(f"학습용 CSV 에 필요한 열이 없습니다: {missing}")
    train_rows, validation = data[data["split"] == "train"], data[data["split"] == "validation"]
    model, threshold, normalizer = fit(train_rows, validation, seed)

    imputer = model.named_steps["imputer"]
    log_columns = [FEATURES[i] for i in model.named_steps["log1p"].indices]
    validation_flagged = raw_anomaly_score(model, validation[FEATURES]) >= threshold
    meta = json.loads(TRAINING_META.read_text(encoding="utf-8")) if Path(csv_path) == TRAINING_CSV and TRAINING_META.exists() else {}
    bundle = build_bundle(
        model=model,
        normalizer=normalizer,
        feature_columns=FEATURES,
        preprocess={"imputation": "median", "imputation_values": dict(zip(FEATURES, map(float, imputer.statistics_))),
                    "log1p_columns": log_columns, "scaling": "none", "fitted_on": "train split"},
        score_method="-score_samples",
        anomaly_threshold=0.5,  # 정규화 점수 기준. 원시 임계값(raw_threshold)과 같은 판정이다
        model_version=MODEL_VERSION,
        min_booking_count=MIN_BOOKING_COUNT,
        raw_threshold=threshold,
        feature_definition_version=FEATURE_DEFINITION_VERSION,
        normalization=normalizer.describe(),
        training={
            "data_version": meta.get("data_version"), "data_file": Path(csv_path).name, "data_sha256": sha256(csv_path),
            "data_generation": meta.get("generation"), "split": meta.get("split"),
            "rows": {"train": int(len(train_rows)), "validation": int(len(validation)),
                     "validation_normal": int((validation["is_anomaly"] == 0).sum())},
            "train_anomaly_share": float(train_rows["is_anomaly"].mean()),
            "model": {"type": "IsolationForest", "n_estimators": N_ESTIMATORS, "contamination": "auto", "max_samples": "auto", "random_state": seed},
            "threshold_rule": f"validation 정상 계정 오탐률 <= {TARGET_FPR}", "target_fpr": TARGET_FPR,
            "validation_metrics": confusion(validation, validation_flagged),
            "min_intervals_for_variance": MIN_INTERVALS_FOR_VARIANCE,
            "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                         "pandas": pd.__version__, "joblib": joblib.__version__},
            "note": "합성 데이터로 학습했다. 지표는 합성 데이터 안의 값이고 실제 운영 성능이 아니다.",
        },
    )
    save_bundle(bundle, out_path)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=str(TRAINING_CSV))
    parser.add_argument("--out", default=str(MODEL_PATH))
    args = parser.parse_args()
    bundle = train(Path(args.csv), Path(args.out))
    shown = {k: v for k, v in bundle.items() if k not in ("model", "normalizer")}
    print(json.dumps(shown, ensure_ascii=False, indent=2))
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
