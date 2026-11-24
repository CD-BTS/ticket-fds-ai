import json

import joblib
import numpy as np
import pandas as pd
import pytest

import train_fds_model as fds
from inference import (
    BUNDLE_SCHEMA_VERSION,
    assess_batch,
    build_bundle,
    load_bundle,
    meta_path_for,
    raw_anomaly_score,
    save_bundle,
)
from preprocessing import ScoreNormalizer, ThresholdAnchoredNormalizer
from tests.conftest import normal_payload, table_payloads


@pytest.fixture(scope="module")
def bundle(model_path):
    return load_bundle(model_path)


@pytest.fixture(scope="module")
def training():
    return pd.read_csv(fds.TRAINING_CSV)


# ---------- 번들 저장·로드 ----------

def test_bundle_contains_everything_needed_to_reproduce_and_serve(bundle):
    assert bundle["feature_columns"] == fds.FEATURES
    assert bundle["feature_definition_version"] == fds.FEATURE_DEFINITION_VERSION
    assert bundle["model_version"] == fds.MODEL_VERSION
    assert bundle["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert bundle["min_booking_count"] == 1 and bundle["anomaly_threshold"] == 0.5
    assert set(bundle["preprocess"]["imputation_values"]) == set(fds.FEATURES)
    assert bundle["preprocess"]["log1p_columns"] == ["avg_booking_interval", "booking_interval_variance", "booking_count"]
    assert bundle["normalization"]["low"] < bundle["raw_threshold"] < bundle["normalization"]["high"]
    training = bundle["training"]
    assert training["data_sha256"] == fds.sha256(fds.TRAINING_CSV)
    assert training["model"] == {"type": "IsolationForest", "n_estimators": 100, "contamination": "auto",
                                 "max_samples": "auto", "random_state": fds.MODEL_SEED}
    assert training["rows"] == {"train": 490, "validation": 210, "validation_normal": 168}
    assert training["data_version"] and training["data_generation"]["generation_seed"] == 2026
    assert {"python", "scikit-learn", "numpy", "pandas"} <= set(training["versions"])


def test_training_is_reproducible(tmp_path, bundle, training):
    again = fds.train(fds.TRAINING_CSV, tmp_path / "again.joblib")
    assert again["raw_threshold"] == bundle["raw_threshold"]
    assert again["normalization"] == bundle["normalization"]
    X = training[fds.FEATURES]
    assert np.array_equal(raw_anomaly_score(again["model"], X), raw_anomaly_score(bundle["model"], X))


def test_saved_bundle_reproduces_scores(bundle, training):
    X = training[fds.FEATURES]
    fresh, threshold, normalizer = fds.fit(training[training["split"] == "train"], training[training["split"] == "validation"])
    assert threshold == bundle["raw_threshold"]
    assert np.array_equal(raw_anomaly_score(fresh, X), raw_anomaly_score(bundle["model"], X))
    assert np.array_equal(normalizer.transform(raw_anomaly_score(fresh, X)), bundle["normalizer"].transform(raw_anomaly_score(bundle["model"], X)))


def test_meta_json_written_next_to_model(model_path):
    meta = json.loads(meta_path_for(model_path).read_text(encoding="utf-8"))
    assert meta["model_version"] == fds.MODEL_VERSION
    assert meta["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert meta["raw_threshold"] > 0 and meta["training"]["data_sha256"]
    assert "model" not in meta and "normalizer" not in meta


def test_save_leaves_no_temp_file(model_path):
    assert not list(model_path.parent.glob("*.tmp"))


def test_missing_model_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_fds_model.py"):
        load_bundle(tmp_path / "nope.joblib")


def test_old_format_model_is_rejected(tmp_path, bundle):
    # 예전 형식(파이프라인만 저장)은 명확한 오류로 거절한다
    path = tmp_path / "old.joblib"
    joblib.dump(bundle["model"], path)
    with pytest.raises(ValueError, match="지원하지 않는"):
        load_bundle(path)


def test_legacy_mvp_bundle_is_rejected(tmp_path, bundle):
    # 이전 MVP 스크립트(pack_bundle.py, train_model.py)가 만든 번들: 스키마 버전 1, 원시 임계값 없음
    legacy = {k: v for k, v in bundle.items() if k not in ("raw_threshold", "feature_definition_version", "normalization", "training")}
    legacy["schema_version"] = 1
    path = tmp_path / "legacy.joblib"
    joblib.dump(legacy, path)
    with pytest.raises(ValueError, match="스키마 버전"):
        load_bundle(path)


def test_bundle_built_without_raw_threshold_is_rejected(tmp_path, bundle):
    built = build_bundle(model=bundle["model"], normalizer=ScoreNormalizer("percentile").fit([0.4, 0.5]), feature_columns=fds.FEATURES,
                         preprocess={}, score_method="-score_samples", anomaly_threshold=0.5, model_version="legacy", min_booking_count=1)
    with pytest.raises(ValueError, match="raw_threshold"):
        load_bundle(save_bundle(built, tmp_path / "no_threshold.joblib"))


def test_bundle_with_other_feature_set_is_rejected(tmp_path, bundle):
    other = dict(bundle, feature_columns=["booking_count", "cancel_count", "cancel_rate", "avg_booking_interval", "short_interval_booking_count"])
    path = tmp_path / "mvp_features.joblib"
    joblib.dump(other, path)
    with pytest.raises(ValueError, match="Feature 구성"):
        load_bundle(path)


def test_incomplete_bundle_is_rejected(tmp_path):
    path = tmp_path / "bad.joblib"
    joblib.dump({"model": object(), "schema_version": BUNDLE_SCHEMA_VERSION}, path)
    with pytest.raises(ValueError, match="필요한 항목"):
        load_bundle(path)


def test_app_refuses_to_start_without_model(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with pytest.raises(FileNotFoundError, match="모델 파일이 없습니다"):
        with TestClient(app):
            pass


# ---------- 학습 코드와 서빙 코드의 전처리 일치 ----------

def test_serving_path_gives_the_same_raw_scores_as_training_code(bundle, training):
    """요청(JSON) → 검증 → Feature 순서 → 번들 파이프라인의 결과가, 학습 코드가 표에서 직접 계산한 점수와 같다."""
    direct = raw_anomaly_score(bundle["model"], training[fds.FEATURES])
    served = assess_batch(table_payloads(training), bundle)
    assert all(r["status"] == "OK" for r in served)
    assert np.allclose([r["rawAnomalyScore"] for r in served], direct, rtol=0, atol=1e-12)
    assert [r["isAnomaly"] for r in served] == list(direct >= bundle["raw_threshold"])


def test_recorded_preprocessing_reproduces_the_pipeline(bundle, training):
    """번들에 적힌 중앙값과 log1p 열만으로 손으로 전처리해도 파이프라인과 같은 점수가 나온다."""
    X = training[fds.FEATURES].to_numpy(dtype=float)
    medians = np.array([bundle["preprocess"]["imputation_values"][name] for name in fds.FEATURES])
    filled = np.where(np.isnan(X), medians, X)
    for name in bundle["preprocess"]["log1p_columns"]:
        column = fds.FEATURES.index(name)
        filled[:, column] = np.log1p(filled[:, column])
    by_hand = -bundle["model"].named_steps["isolation_forest"].score_samples(filled)
    assert np.allclose(by_hand, raw_anomaly_score(bundle["model"], training[fds.FEATURES]), rtol=0, atol=1e-12)


def test_imputation_values_come_from_the_train_split_only(bundle, training):
    train = training[training["split"] == "train"]
    for name in fds.FEATURES:
        assert bundle["preprocess"]["imputation_values"][name] == pytest.approx(float(train[name].median()))


def test_threshold_keeps_validation_false_positive_rate_within_target(bundle, training):
    normal = training[(training["split"] == "validation") & (training["is_anomaly"] == 0)]
    flagged = raw_anomaly_score(bundle["model"], normal[fds.FEATURES]) >= bundle["raw_threshold"]
    assert flagged.mean() <= fds.TARGET_FPR
    assert (flagged.sum() + 1) / len(normal) > fds.TARGET_FPR  # 허용 범위 안에서 가장 낮은 임계값


def test_normalization_anchors_come_from_the_train_split_only(bundle, training):
    train_scores = raw_anomaly_score(bundle["model"], training.loc[training["split"] == "train", fds.FEATURES])
    assert bundle["normalization"]["low"] == float(train_scores.min())
    assert bundle["normalization"]["high"] == float(train_scores.max())
    assert bundle["normalization"]["threshold"] == bundle["raw_threshold"]


# ---------- 정규화 점수와 판정 ----------

def test_normalizer_formula_and_anchors():
    normalizer = ThresholdAnchoredNormalizer(low=0.30, threshold=0.50, high=0.70)
    assert normalizer.transform([0.30, 0.40, 0.50, 0.60, 0.70]).tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert normalizer.transform([0.10, 0.90]).tolist() == [0.0, 1.0]  # 구간 밖은 잘린다
    with pytest.raises(ValueError):
        ThresholdAnchoredNormalizer(low=0.5, threshold=0.5, high=0.7)


def test_normalizer_is_monotone_and_strict_inside_the_range(bundle):
    normalizer = bundle["normalizer"]
    raw = np.linspace(normalizer.low - 0.05, normalizer.high + 0.05, 20001)
    scores = normalizer.transform(raw)
    assert (np.diff(scores) >= 0).all()
    inside = (raw > normalizer.low) & (raw < normalizer.high)
    assert (np.diff(scores[inside]) > 0).all()
    assert scores.min() == 0.0 and scores.max() == 1.0


def test_normalized_score_and_decision_agree_at_the_boundary(bundle):
    normalizer, threshold = bundle["normalizer"], bundle["raw_threshold"]
    just_below, just_above = np.nextafter(threshold, -np.inf), np.nextafter(threshold, np.inf)
    scores = normalizer.transform([just_below, threshold, just_above])
    assert scores[0] < 0.5 and scores[1] == 0.5 and scores[2] >= 0.5
    grid = np.concatenate([np.linspace(normalizer.low, normalizer.high, 100001), [just_below, threshold, just_above]])
    assert ((normalizer.transform(grid) >= 0.5) == (grid >= threshold)).all()


class FixedScores:
    """score_samples가 정해 둔 값을 돌려주는 가짜 모델 (경계값의 응답을 확인하는 용도)."""

    def __init__(self, raw):
        self.raw = np.asarray(raw, dtype=float)

    def score_samples(self, X):
        return -self.raw[: len(X)]


def test_response_rounding_never_contradicts_the_decision(bundle):
    threshold, normalizer = bundle["raw_threshold"], bundle["normalizer"]
    raw = [np.nextafter(threshold, -np.inf), threshold - 1e-9, threshold, threshold + 1e-9, normalizer.low - 0.1, normalizer.high + 0.1]
    fake = dict(bundle, model=FixedScores(raw))
    results = assess_batch([normal_payload(str(i)) for i in range(len(raw))], fake)
    assert [r["isAnomaly"] for r in results] == [False, False, True, True, False, True]
    for r in results:
        assert r["isAnomaly"] == (r["anomalyScore"] >= 0.5) == (r["rawAnomalyScore"] >= threshold)
    assert results[0]["anomalyScore"] < 0.5 and results[2]["anomalyScore"] == 0.5
    assert results[4]["anomalyScore"] == 0.0 and results[5]["anomalyScore"] == 1.0
    assert [r["rawAnomalyScore"] for r in results] == [float(v) for v in raw]  # 원시 점수는 반올림하지 않는다


def test_insufficient_data_is_not_scored(bundle):
    payload = normal_payload()
    payload["features"].update(bookingCount=0)
    result = assess_batch([payload], bundle)[0]
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["rawAnomalyScore"] is None and result["anomalyScore"] is None and result["isAnomaly"] is None
