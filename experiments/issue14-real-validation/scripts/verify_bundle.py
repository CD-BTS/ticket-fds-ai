"""재생성한 모델 번들 검증 + 빌드 기록 작성 (실험용, 기존 코드는 import만 한다).

실행 (실험 폴더에서):
    .venv/Scripts/python scripts/verify_bundle.py

하는 일
1. inference.load_bundle 로 번들을 읽는다.
2. 더미 학습 데이터 전체를 inference.assess_batch 로 추론하고,
   모델을 직접 호출한 점수와 일치하는지 확인한다.
3. 결측(null) Feature가 들어왔을 때의 동작을 확인한다.
4. 학습 조건(데이터, Feature, 하이퍼파라미터, 버전, 파일 해시)을 model_build/BUILD_RECORD.json 에 기록한다.
"""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
AI = EXP.parent.parent  # 이 폴더는 ai/experiments/<실험명> 이므로 두 단계 위가 AI 레포 루트
sys.path.insert(0, str(AI))  # 기존 inference.py / preprocessing.py 를 그대로 사용

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

from inference import assess_batch, load_bundle, raw_anomaly_score  # noqa: E402
from preprocessing import CAMEL_TO_SNAKE, FEATURE_SPECS  # noqa: E402

BUILD = EXP / "model_build"
BUNDLE_PATH = BUILD / "models" / "behavior_model.joblib"
MODEL_PATH = BUILD / "ml_experiment" / "models" / "best_isolation_forest.joblib"
DATA_PATH = BUILD / "ml_experiment" / "data" / "fairticket_synthetic_features.csv"
CALCULATED_AT = "2026-11-19T00:00:00+00:00"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def to_payload(row):
    features = {}
    for camel, snake in CAMEL_TO_SNAKE.items():
        value = row[snake]
        if pd.isna(value):
            features[camel] = None
        elif FEATURE_SPECS[snake]["integer"]:
            features[camel] = int(value)
        else:
            features[camel] = float(value)
    return {"userId": str(row["user_id"]), "calculatedAt": CALCULATED_AT,
            "windowDays": 30, "features": features}


def main():
    bundle = load_bundle(BUNDLE_PATH)
    model = bundle["model"]
    columns = bundle["feature_columns"]
    df = pd.read_csv(DATA_PATH)
    print(f"번들 로드 OK: version={bundle['model_version']} features={columns}")

    # ---- 1) 더미 전체 추론: assess_batch vs 모델 직접 호출 ----
    results = assess_batch([to_payload(r) for _, r in df.iterrows()], bundle)
    status = pd.Series([r["status"] for r in results]).value_counts().to_dict()
    api_scores = np.array([np.nan if r["anomalyScore"] is None else r["anomalyScore"] for r in results])
    direct = bundle["normalizer"].transform(raw_anomaly_score(model, df[columns]))
    max_diff = float(np.nanmax(np.abs(api_scores - direct)))
    print(f"assess_batch 상태: {status}, 직접 계산과 최대 차이 {max_diff:.2e} (반올림 4자리 허용 5e-5)")
    assert status == {"OK": len(df)}, status
    assert max_diff <= 6e-5, max_diff

    df["anomaly_score"] = api_scores
    df["model_predict_anomaly"] = model.predict(df[columns]) == -1
    by_scenario = df.groupby("scenario").agg(
        n=("user_id", "size"), score_mean=("anomaly_score", "mean"),
        score_median=("anomaly_score", "median"),
        predict_rate=("model_predict_anomaly", "mean")).round(4)
    print("\n시나리오별 점수(더미 전체 1000건, 학습에 쓰인 700건 포함):")
    print(by_scenario.to_string())

    # predict() 경계가 정규화 점수로 얼마인지 (참고용, 변경하지 않음)
    predict_boundary = float(bundle["normalizer"].transform([-model.offset_])[0])
    print(f"\n모델 predict() 경계의 정규화 점수: {predict_boundary:.4f} / 번들 anomaly_threshold: {bundle['anomaly_threshold']}")

    # ---- 2) 결측 동작 확인 ----
    base = to_payload(df.iloc[0])
    null_checks = {}
    for camel in CAMEL_TO_SNAKE:
        payload = json.loads(json.dumps(base))
        payload["features"][camel] = None
        try:
            r = assess_batch([payload], bundle)[0]
            null_checks[camel] = {"status": r["status"], "anomalyScore": r["anomalyScore"],
                                  "errors": r.get("errors")}
        except Exception as error:  # 모델이 NaN을 받지 못하는 경우
            null_checks[camel] = {"status": "EXCEPTION", "error": f"{type(error).__name__}: {str(error)[:160]}"}
    print("\nFeature 하나를 null로 보냈을 때:")
    for key, value in null_checks.items():
        print(f"  {key}: {value}")

    # ---- 3) 빌드 기록 ----
    iforest = model
    ai_commit = subprocess.run(["git", "-C", str(AI), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ai_branch = subprocess.run(["git", "-C", str(AI), "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
    record = {
        "purpose": "이슈 #14 실측 검증용. 기존 스크립트를 수정 없이 재실행해 만든 모델이며, 팀이 처음 학습한 원본 파일은 아니다.",
        "built_at_utc": bundle["trained_at"],
        "ai_repo": {"commit": ai_commit, "branch": ai_branch,
                    "scripts": ["ml_experiment/train_and_evaluate_if.py", "pack_bundle.py"],
                    "script_sha256": {p: sha256(AI / p) for p in ["ml_experiment/train_and_evaluate_if.py", "pack_bundle.py",
                                                                    "inference.py", "preprocessing.py"]}},
        "training_data": {"file": "ml_experiment/data/fairticket_synthetic_features.csv (git에 커밋된 파일 그대로, 재생성 안 함)",
                          "sha256": sha256(DATA_PATH), "rows": int(len(df)),
                          "is_anomaly_counts": {str(k): int(v) for k, v in df["is_anomaly"].value_counts().items()},
                          "scenario_counts": {k: int(v) for k, v in df["scenario"].value_counts().items()},
                          "split": "train_test_split(test_size=0.3, random_state=42, stratify=is_anomaly) -> 학습 700 / 평가 300"},
        "features": list(iforest.feature_names_in_),
        "excluded_feature": "seat_selection_latency",
        "hyperparameters": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                            for k, v in iforest.get_params().items()},
        "fitted": {"n_features_in": int(iforest.n_features_in_), "max_samples_": int(iforest.max_samples_),
                   "offset_": float(iforest.offset_), "n_estimators_fitted": len(iforest.estimators_)},
        "bundle": {"model_version": bundle["model_version"], "score_method": bundle["score_method"],
                   "normalizer": "ScoreNormalizer(percentile), 더미 1000건 전체의 -score_samples 로 fit",
                   "anomaly_threshold": bundle["anomaly_threshold"], "min_booking_count": bundle["min_booking_count"],
                   "preprocess": bundle["preprocess"], "predict_boundary_normalized": predict_boundary},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__,
                     "scipy": scipy.__version__, "platform": platform.platform()},
        "artifacts_sha256": {"best_isolation_forest.joblib": sha256(MODEL_PATH),
                             "behavior_model.joblib": sha256(BUNDLE_PATH)},
        "reported_metrics_test300": "logs/01_train_and_evaluate_if.log 참고",
        "inference_check": {"assess_batch_status": status, "max_abs_diff_vs_direct": max_diff,
                            "null_feature_behavior": null_checks},
        "score_by_scenario_all1000": json.loads(by_scenario.reset_index().to_json(orient="records")),
    }
    out = BUILD / "BUILD_RECORD.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n빌드 기록 저장: {out}")


if __name__ == "__main__":
    main()
