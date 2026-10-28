import copy

import pytest

from tests.conftest import extreme_payload, normal_payload


# ---------- /health ----------

def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["modelVersion"] == "iforest-v1"
    assert len(body["features"]) == 5


# ---------- POST /predict (단일) ----------

def test_predict_ok(client):
    response = client.post("/predict", json=normal_payload())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"userId", "status", "anomalyScore", "modelVersion", "assessedAt"}
    assert body["userId"] == "user_normal"
    assert body["status"] == "OK"
    assert 0.0 <= body["anomalyScore"] <= 1.0


def test_extreme_user_scores_higher_than_normal(client):
    normal = client.post("/predict", json=normal_payload()).json()["anomalyScore"]
    extreme = client.post("/predict", json=extreme_payload()).json()["anomalyScore"]
    assert extreme > normal
    assert extreme >= 0.95  # 학습 사용자 상위 5% 경계 이상


def test_insufficient_data_returns_null_score(client):
    payload = normal_payload()
    payload["features"].update(bookingCount=1, avgBookingInterval=None)
    body = client.post("/predict", json=payload).json()
    assert body["status"] == "INSUFFICIENT_DATA"
    assert body["anomalyScore"] is None


def test_nullable_features_are_accepted(client):
    payload = normal_payload()
    payload["features"].update(
        cancelRate=None, avgBookingInterval=None, attendanceRate=None, avgBookingDuration=None
    )
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "OK"


def test_missing_optional_keys_are_treated_as_null(client):
    payload = normal_payload()
    del payload["features"]["attendanceRate"]
    del payload["features"]["avgBookingDuration"]
    assert client.post("/predict", json=payload).status_code == 200


def test_extra_fields_are_ignored(client):
    payload = normal_payload()
    payload["features"]["bookingBurstCount"] = 3
    assert client.post("/predict", json=payload).status_code == 200


def test_same_request_gives_same_score(client):
    first = client.post("/predict", json=normal_payload()).json()["anomalyScore"]
    second = client.post("/predict", json=normal_payload()).json()["anomalyScore"]
    assert first == second


# ---------- userId 타입 (Spring의 Long 대응) ----------

def test_numeric_user_id_is_accepted_and_returned_as_string(client):
    payload = normal_payload()
    payload["userId"] = 12345
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["userId"] == "12345"


def test_numeric_user_id_in_batch(client):
    payload = normal_payload()
    payload["userId"] = 7
    body = client.post("/predict/batch", json={"requests": [payload]}).json()
    assert body["results"][0]["userId"] == "7"
    assert body["results"][0]["status"] == "OK"


@pytest.mark.parametrize("bad_id", [True, 1.5, [], {}, "   "])
def test_invalid_user_id_types_are_rejected(client, bad_id):
    payload = normal_payload()
    payload["userId"] = bad_id
    assert client.post("/predict", json=payload).status_code == 422


# ---------- 422 검증 오류 ----------

def _mutate(path, value):
    payload = copy.deepcopy(normal_payload())
    target = payload
    *parents, last = path.split(".")
    for key in parents:
        target = target[key]
    if value == "__delete__":
        del target[last]
    else:
        target[last] = value
    return payload


@pytest.mark.parametrize(
    "path,value,expected",
    [
        ("features.cancelRate", 1.7, "cancelRate"),               # 범위 초과
        ("features.bookingCount", -1, "bookingCount"),            # 음수
        ("features.bookingCount", 3.5, "정수"),                    # 정수 아님
        ("features.bookingCount", None, "bookingCount"),          # null 불가
        ("features.shortIntervalBookingCount", None, "null"),     # null 불가
        ("features.bookingCount", "6", "bookingCount"),           # 문자열
        ("features.bookingCount", True, "bookingCount"),          # boolean
        ("windowDays", 7, "windowDays"),                          # 30만 허용
        ("calculatedAt", "2026-10-08T15:00:00", "UTC"),           # 오프셋 없음
        ("calculatedAt", "yesterday", "calculatedAt"),            # 형식 오류
        ("userId", "", "userId"),                                 # 빈 값
        ("userId", "__delete__", "userId"),                       # 누락
        ("features", "__delete__", "features"),                   # 누락
    ],
)
def test_validation_errors_return_422(client, path, value, expected):
    response = client.post("/predict", json=_mutate(path, value))
    assert response.status_code == 422
    errors = response.json()["errors"]
    assert isinstance(errors, list) and errors
    assert any(expected in message for message in errors)


def test_malformed_json_returns_422(client):
    response = client.post(
        "/predict", content="{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert "errors" in response.json()


# ---------- POST /predict/batch ----------

def test_batch_mixed_results_keep_order(client):
    invalid = normal_payload("user_bad")
    invalid["features"]["cancelRate"] = 5
    short = normal_payload("user_new")
    short["features"]["bookingCount"] = 1

    body = client.post(
        "/predict/batch",
        json={"requests": [normal_payload("a"), invalid, extreme_payload("b"), short]},
    ).json()

    results = body["results"]
    assert [r["userId"] for r in results] == ["a", "user_bad", "b", "user_new"]
    assert [r["status"] for r in results] == ["OK", "INVALID", "OK", "INSUFFICIENT_DATA"]
    assert results[1]["anomalyScore"] is None and results[1]["errors"]
    assert results[3]["anomalyScore"] is None
    assert results[2]["anomalyScore"] > results[0]["anomalyScore"]


def test_batch_scores_match_single_scores(client):
    payloads = [normal_payload("a"), extreme_payload("b")]
    single = [client.post("/predict", json=p).json()["anomalyScore"] for p in payloads]
    batch = client.post("/predict/batch", json={"requests": payloads}).json()["results"]
    assert [r["anomalyScore"] for r in batch] == single


def test_batch_item_with_wrong_shape_does_not_fail_whole_batch(client):
    body = client.post(
        "/predict/batch", json={"requests": [normal_payload("a"), {"userId": 123}]}
    ).json()
    assert [r["status"] for r in body["results"]] == ["OK", "INVALID"]
    assert body["results"][1]["userId"] == "123"  # 오류 항목도 userId는 돌려줘 구분할 수 있다


def test_batch_item_without_valid_user_id_returns_null_user_id(client):
    body = client.post(
        "/predict/batch", json={"requests": [{"userId": True}, {"userId": ""}, {}]}
    ).json()
    assert [r["status"] for r in body["results"]] == ["INVALID"] * 3
    assert [r["userId"] for r in body["results"]] == [None, None, None]


def test_batch_empty_is_rejected(client):
    assert client.post("/predict/batch", json={"requests": []}).status_code == 422


def test_batch_over_limit_is_rejected(client):
    from app.schemas import MAX_BATCH_SIZE

    payloads = [normal_payload(f"u{i}") for i in range(MAX_BATCH_SIZE + 1)]
    assert client.post("/predict/batch", json={"requests": payloads}).status_code == 422


def test_batch_of_max_size_is_accepted(client):
    from app.schemas import MAX_BATCH_SIZE

    payloads = [normal_payload(f"u{i}") for i in range(MAX_BATCH_SIZE)]
    response = client.post("/predict/batch", json={"requests": payloads})
    assert response.status_code == 200
    assert len(response.json()["results"]) == MAX_BATCH_SIZE