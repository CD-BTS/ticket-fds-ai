import json

import numpy as np
import pytest
import joblib

import train_model as tm
from inference import (
    BUNDLE_SCHEMA_VERSION,
    load_bundle,
    meta_path_for,
    raw_anomaly_score,
)
from tests.conftest import make_features


def test_saved_bundle_reproduces_scores(model_path):
    features = make_features()
    eligible = features[features["booking_count"] >= tm.MIN_BOOKING_COUNT]
    X = eligible[tm.MVP_FEATURES]

    model, normalizer, _ = tm.train_and_score(features, tm.MVP_FEATURES)
    reloaded = load_bundle(model_path)

    before = normalizer.transform(raw_anomaly_score(model, X))
    after = reloaded["normalizer"].transform(raw_anomaly_score(reloaded["model"], X))
    assert np.allclose(before, after)


def test_meta_json_written_next_to_model(model_path):
    meta = json.loads(meta_path_for(model_path).read_text(encoding="utf-8"))
    assert meta["model_version"] == "iforest-v1"
    assert meta["schema_version"] == BUNDLE_SCHEMA_VERSION
    assert "model" not in meta and "normalizer" not in meta


def test_save_leaves_no_temp_file(model_path):
    assert not list(model_path.parent.glob("*.tmp"))


def test_missing_model_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_model.py"):
        load_bundle(tmp_path / "nope.joblib")


def test_old_format_model_is_rejected(tmp_path):
    # 예전 형식(파이프라인만 저장)은 명확한 오류로 거절한다
    model, _, _ = tm.train_and_score(make_features(), tm.MVP_FEATURES)
    path = tmp_path / "old.joblib"
    joblib.dump(model, path)
    with pytest.raises(ValueError, match="지원하지 않는"):
        load_bundle(path)


def test_incomplete_bundle_is_rejected(tmp_path):
    path = tmp_path / "bad.joblib"
    joblib.dump({"model": object()}, path)
    with pytest.raises(ValueError, match="필요한 항목"):
        load_bundle(path)


def test_app_refuses_to_start_without_model(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass