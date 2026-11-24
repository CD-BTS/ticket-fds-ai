import copy
import random

import pandas as pd
import pytest

import train_fds_model as fds
from tests.conftest import extreme_payload, normal_payload, table_payloads

RESPONSE_KEYS = {"userId", "status", "rawAnomalyScore", "anomalyScore", "isAnomaly", "modelVersion", "assessedAt"}


def training_payloads(n):
    return table_payloads(pd.read_csv(fds.TRAINING_CSV).head(n))


# ---------- /health ----------

def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["modelVersion"] == fds.MODEL_VERSION
    assert body["features"] == fds.FEATURES and len(body["features"]) == 6
    assert body["featureDefinitionVersion"] == fds.FEATURE_DEFINITION_VERSION
    assert body["anomalyThreshold"] == 0.5
    assert 0.0 < body["rawThreshold"] < 1.0
    assert body["scoreNormalization"]["low"] < body["rawThreshold"] < body["scoreNormalization"]["high"]
    assert body["minBookingCount"] == 1
    assert body["trainingDataVersion"]


# ---------- POST /predict (단일) ----------

def test_predict_ok(client):
    response = client.post("/predict", json=normal_payload())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESPONSE_KEYS
    assert body["userId"] == "user_normal"
    assert body["status"] == "OK"
    assert 0.0 < body["rawAnomalyScore"] < 1.0
    assert 0.0 <= body["anomalyScore"] <= 1.0
    assert isinstance(body["isAnomaly"], bool)
    assert body["modelVersion"] == fds.MODEL_VERSION


def test_extreme_user_is_flagged_and_normal_user_is_not(client):
    normal = client.post("/predict", json=normal_payload()).json()
    extreme = client.post("/predict", json=extreme_payload()).json()
    assert extreme["rawAnomalyScore"] > normal["rawAnomalyScore"]
    assert extreme["anomalyScore"] > normal["anomalyScore"]
    assert extreme["isAnomaly"] is True and extreme["anomalyScore"] >= 0.5
    assert normal["isAnomaly"] is False and normal["anomalyScore"] < 0.5


def test_zero_booking_count_is_insufficient_data_not_a_score(client):
    payload = normal_payload()
    payload["features"].update(bookingCount=0, avgBookingInterval=None, cancelRate=None)
    body = client.post("/predict", json=payload).json()
    assert body["status"] == "INSUFFICIENT_DATA"
    # 정상 판정이나 0점이 아니라 "판정하지 않음"이다
    assert body["rawAnomalyScore"] is None and body["anomalyScore"] is None and body["isAnomaly"] is None
    assert body["modelVersion"] == fds.MODEL_VERSION


def test_single_attempt_is_scored(client):
    payload = normal_payload()
    payload["features"].update(bookingCount=1, avgBookingInterval=None)
    body = client.post("/predict", json=payload).json()
    assert body["status"] == "OK" and body["anomalyScore"] is not None


def test_null_variance_is_accepted(client):
    payload = extreme_payload()
    payload["features"]["bookingIntervalVariance"] = None
    response = client.post("/predict", json=payload)
    assert response.status_code == 200 and response.json()["status"] == "OK"


def test_all_nullable_features_null_is_scored_with_training_medians(client):
    payload = normal_payload()
    payload["features"].update(avgBookingInterval=None, bookingIntervalVariance=None, cancelRate=None)
    body = client.post("/predict", json=payload).json()
    assert body["status"] == "OK" and body["rawAnomalyScore"] is not None


def test_only_booking_count_given_is_scored(client):
    # 나머지 Feature를 모두 생략: 간격·분산·취소율은 null(학습 중앙값으로 대체), 카드·IP는 기본 1
    payload = normal_payload()
    payload["features"] = {"bookingCount": 4}
    explicit = normal_payload()
    explicit["features"].update(bookingCount=4, avgBookingInterval=None, bookingIntervalVariance=None, cancelRate=None)
    a, b = client.post("/predict", json=payload).json(), client.post("/predict", json=explicit).json()
    assert a["status"] == "OK" and a["rawAnomalyScore"] == b["rawAnomalyScore"]


def test_all_features_missing_is_rejected(client):
    payload = normal_payload()
    payload["features"] = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert any("bookingCount" in message for message in response.json()["errors"])


@pytest.mark.parametrize("name", ["bookingBurstCount", "cancelCount", "shortIntervalBookingCount", "booking_count", "bookingcount"])
def test_unknown_feature_name_is_rejected(client, name):
    payload = normal_payload()
    payload["features"][name] = 3
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert any(name in message and "알 수 없는" in message for message in response.json()["errors"])


def test_unknown_top_level_field_is_ignored(client):
    payload = normal_payload()
    payload["requestId"] = "abc"
    assert client.post("/predict", json=payload).status_code == 200


def test_same_request_gives_same_result_every_time(client):
    results = [client.post("/predict", json=extreme_payload()).json() for _ in range(5)]
    for key in ("rawAnomalyScore", "anomalyScore", "isAnomaly", "modelVersion"):
        assert len({r[key] for r in results}) == 1


def test_feature_key_order_does_not_change_the_result(client):
    payload = extreme_payload()
    shuffled = copy.deepcopy(payload)
    items = list(shuffled["features"].items())
    random.Random(7).shuffle(items)
    shuffled["features"] = dict(items)
    assert list(shuffled["features"]) != list(payload["features"])
    assert client.post("/predict", json=shuffled).json()["rawAnomalyScore"] == client.post("/predict", json=payload).json()["rawAnomalyScore"]


def test_is_anomaly_matches_raw_threshold_and_normalized_score(client):
    health = client.get("/health").json()
    results = client.post("/predict/batch", json={"requests": training_payloads(300)}).json()["results"]
    assert all(r["status"] == "OK" for r in results)
    flagged = [r for r in results if r["isAnomaly"]]
    assert flagged and len(flagged) < len(results)
    for r in results:
        assert r["isAnomaly"] == (r["rawAnomalyScore"] >= health["rawThreshold"])
        assert r["isAnomaly"] == (r["anomalyScore"] >= health["anomalyThreshold"])
    ordered = sorted(results, key=lambda r: r["rawAnomalyScore"])
    scores = [r["anomalyScore"] for r in ordered]
    assert scores == sorted(scores)  # 원시 점수가 크면 정규화 점수도 작지 않다


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
        ("features.cancelRate", 1.7, "cancelRate"),                    # 범위 초과
        ("features.cancelRate", -0.1, "cancelRate"),                   # 범위 미만
        ("features.bookingCount", -1, "bookingCount"),                 # 음수
        ("features.bookingCount", 3.5, "정수"),                         # 정수 아님
        ("features.bookingCount", None, "bookingCount"),               # null 불가
        ("features.bookingCount", "__delete__", "bookingCount"),       # 누락 (null 불가)
        ("features.cardReuseCount", None, "null"),                     # null 불가
        ("features.cardReuseCount", 0, "1 이상"),                       # 본인 포함이라 최소 1
        ("features.ipBookingDensity", 0, "1 이상"),
        ("features.ipBookingDensity", 2.5, "정수"),
        ("features.avgBookingInterval", -1.0, "avgBookingInterval"),   # 음수 간격
        ("features.bookingIntervalVariance", -0.5, "bookingIntervalVariance"),
        ("features.bookingIntervalVariance", "0.3", "bookingIntervalVariance"),  # 문자열
        ("features.bookingCount", "6", "bookingCount"),                # 문자열
        ("features.bookingCount", True, "bookingCount"),               # boolean
        ("features.avgBookingInterval", [1, 2], "avgBookingInterval"), # 배열
        ("features", [1, 2, 3], "features"),                           # 객체 아님
        ("windowDays", 7, "windowDays"),                               # 30만 허용
        ("calculatedAt", "2026-10-08T15:00:00", "UTC"),                # 오프셋 없음
        ("calculatedAt", "yesterday", "calculatedAt"),                 # 형식 오류
        ("userId", "", "userId"),                                      # 빈 값
        ("userId", "__delete__", "userId"),                            # 누락
        ("features", "__delete__", "features"),                        # 누락
    ],
)
def test_validation_errors_return_422(client, path, value, expected):
    response = client.post("/predict", json=_mutate(path, value))
    assert response.status_code == 422
    errors = response.json()["errors"]
    assert isinstance(errors, list) and errors
    assert any(expected in message for message in errors)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_rejected(client, literal):
    text = ('{"userId":"u","calculatedAt":"2026-10-08T15:00:00+09:00","windowDays":30,'
            '"features":{"bookingCount":3,"avgBookingInterval":%s}}' % literal)
    response = client.post("/predict", content=text, headers={"Content-Type": "application/json"})
    assert response.status_code == 422 and response.json()["errors"]


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
    unknown = normal_payload("user_old_field")
    unknown["features"]["cancelCount"] = 1
    empty = normal_payload("user_new")
    empty["features"].update(bookingCount=0, avgBookingInterval=None, cancelRate=None)

    body = client.post(
        "/predict/batch",
        json={"requests": [normal_payload("a"), invalid, extreme_payload("b"), empty, unknown]},
    ).json()

    results = body["results"]
    assert [r["userId"] for r in results] == ["a", "user_bad", "b", "user_new", "user_old_field"]
    assert [r["status"] for r in results] == ["OK", "INVALID", "OK", "INSUFFICIENT_DATA", "INVALID"]
    assert results[1]["anomalyScore"] is None and results[1]["isAnomaly"] is None and results[1]["errors"]
    assert results[3]["anomalyScore"] is None and results[3]["rawAnomalyScore"] is None and results[3]["isAnomaly"] is None
    assert any("cancelCount" in message for message in results[4]["errors"])
    assert results[2]["anomalyScore"] > results[0]["anomalyScore"]


def test_batch_results_match_single_results(client):
    payloads = [normal_payload("a"), extreme_payload("b")] + training_payloads(20)
    single = [client.post("/predict", json=p).json() for p in payloads]
    batch = client.post("/predict/batch", json={"requests": payloads}).json()["results"]
    for key in ("rawAnomalyScore", "anomalyScore", "isAnomaly"):
        assert [r[key] for r in batch] == [r[key] for r in single]


def test_omitted_card_and_ip_default_to_one_in_single_and_batch(client):
    payload = normal_payload("only_count")
    payload["features"] = {"bookingCount": 4}
    explicit = normal_payload("explicit")
    explicit["features"] = {"bookingCount": 4, "cardReuseCount": 1, "ipBookingDensity": 1}
    single = client.post("/predict", json=payload).json()
    batch = client.post("/predict/batch", json={"requests": [payload, explicit]}).json()["results"]
    assert [r["status"] for r in batch] == ["OK", "OK"]
    assert batch[0]["rawAnomalyScore"] == batch[1]["rawAnomalyScore"] == single["rawAnomalyScore"]
    # 생략은 기본값이지만 명시적 null 은 단건·배치 모두 거절한다
    null_card = normal_payload("null_card")
    null_card["features"]["cardReuseCount"] = None
    assert client.post("/predict", json=null_card).status_code == 422
    assert client.post("/predict/batch", json={"requests": [null_card]}).json()["results"][0]["status"] == "INVALID"


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
