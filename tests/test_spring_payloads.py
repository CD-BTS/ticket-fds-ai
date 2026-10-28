"""Spring Boot(Jackson)가 실제로 만들어 내는 JSON 형태가 그대로 받아지는지 확인한다.

파이썬 dict가 아니라 JSON 문자열을 그대로 보내서, Java 쪽 직렬화 특징을 재현한다.
- Instant/OffsetDateTime: 소수점 이하 초(마이크로/나노초)와 'Z' 접미사
- Long userId: 숫자로 전달
- Double: 지수 표기(1.0E-4), 정수값 Double(20.0)
- null 필드를 생략하지 않고 그대로 포함
- Content-Type에 charset 포함
"""

import json

import pytest

HEADERS = {"Content-Type": "application/json;charset=UTF-8"}


def post_raw(client, raw, path="/predict"):
    return client.post(path, content=raw.encode("utf-8"), headers=HEADERS)


def jackson_body(user_id="12345", calculated_at='"2026-10-15T01:23:45.123456Z"', features=None):
    default_features = (
        '{"bookingCount":6,"cancelCount":1,"cancelRate":0.17,'
        '"avgBookingInterval":80000.0,"shortIntervalBookingCount":0,'
        '"attendanceRate":null,"avgBookingDuration":null}'
    )
    return (
        '{"userId":%s,"calculatedAt":%s,"windowDays":30,"features":%s}'
        % (user_id, calculated_at, features or default_features)
    )


@pytest.mark.parametrize(
    "calculated_at",
    [
        '"2026-10-15T01:23:45Z"',                       # Instant, 소수점 없음
        '"2026-10-15T01:23:45.123Z"',                   # 밀리초
        '"2026-10-15T01:23:45.123456Z"',                # 마이크로초 (Java 9+ 기본)
        '"2026-10-15T01:23:45.123456789Z"',             # 나노초
        '"2026-10-15T10:23:45.123456+09:00"',           # OffsetDateTime (KST)
        '"2026-10-15T10:23:45+09:00[Asia/Seoul]"',      # ZonedDateTime 기본 직렬화 (zone id 포함)
    ],
)
def test_java_time_formats(client, calculated_at):
    response = post_raw(client, jackson_body(calculated_at=calculated_at))
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "OK"


def test_long_user_id_as_number(client):
    response = post_raw(client, jackson_body(user_id="9007199254740993"))  # JS 안전 정수 범위를 넘는 Long
    assert response.status_code == 200
    assert response.json()["userId"] == "9007199254740993"  # 정밀도 손실 없이 문자열로 반환


def test_exponent_and_integral_doubles(client):
    features = (
        '{"bookingCount":6.0,"cancelCount":1.0,"cancelRate":1.0E-4,'
        '"avgBookingInterval":8.0E4,"shortIntervalBookingCount":0.0,'
        '"attendanceRate":0.9,"avgBookingDuration":1.2E2}'
    )
    response = post_raw(client, jackson_body(features=features))
    assert response.status_code == 200, response.text


def test_all_nullable_features_explicit_null(client):
    features = (
        '{"bookingCount":6,"cancelCount":1,"cancelRate":null,'
        '"avgBookingInterval":null,"shortIntervalBookingCount":0,'
        '"attendanceRate":null,"avgBookingDuration":null}'
    )
    assert post_raw(client, jackson_body(features=features)).status_code == 200


def test_non_null_fields_omitted_by_non_null_inclusion(client):
    # @JsonInclude(NON_NULL) 설정이면 null인 필드가 아예 빠진다. 허용 필드는 통과해야 한다.
    features = '{"bookingCount":6,"cancelCount":1,"shortIntervalBookingCount":0}'
    assert post_raw(client, jackson_body(features=features)).status_code == 200


def test_unknown_fields_are_ignored(client):
    raw = (
        '{"userId":"u1","calculatedAt":"2026-10-15T01:23:45Z","windowDays":30,"requestId":"abc",'
        '"features":{"bookingCount":6,"cancelCount":1,"cancelRate":0.17,"avgBookingInterval":80000.0,'
        '"shortIntervalBookingCount":0,"bookingBurstCount":3}}'
    )
    assert post_raw(client, raw).status_code == 200


def test_non_ascii_user_id(client):
    response = post_raw(client, jackson_body(user_id='"사용자-한글"'))
    assert response.status_code == 200
    assert response.json()["userId"] == "사용자-한글"


def test_response_is_valid_utf8_json_with_expected_content_type(client):
    response = post_raw(client, jackson_body())
    assert response.headers["content-type"].startswith("application/json")
    assert set(json.loads(response.content)) == {
        "userId", "status", "anomalyScore", "modelVersion", "assessedAt"
    }


def test_local_date_time_without_offset_is_rejected_with_clear_message(client):
    # Java LocalDateTime 직렬화: 오프셋이 없다. 422와 함께 이유를 알려줘야 한다.
    response = post_raw(client, jackson_body(calculated_at='"2026-10-15T10:23:45.123456"'))
    assert response.status_code == 422
    assert any("UTC" in message for message in response.json()["errors"])


def test_timestamp_number_for_calculated_at_is_rejected(client):
    # WRITE_DATES_AS_TIMESTAMPS가 켜진 경우: 숫자(epoch)로 직렬화된다
    response = post_raw(client, jackson_body(calculated_at="1792027425.123456"))
    assert response.status_code == 422
    assert response.json()["errors"]


def test_array_form_of_local_date_time_is_rejected(client):
    # 구 버전 Jackson이 LocalDateTime을 [2026,10,15,10,23,45]로 직렬화하는 경우
    response = post_raw(client, jackson_body(calculated_at="[2026,10,15,10,23,45]"))
    assert response.status_code == 422


def test_batch_with_jackson_style_items(client):
    item = jackson_body(user_id="1")
    raw = '{"requests":[%s,%s]}' % (item, jackson_body(user_id="2", calculated_at='"2026-10-15T01:23:45Z"'))
    response = post_raw(client, raw, path="/predict/batch")
    assert response.status_code == 200
    assert [r["status"] for r in response.json()["results"]] == ["OK", "OK"]


def test_zone_id_without_offset_is_still_rejected(client):
    # 지역 이름만 있고 오프셋이 없으면 시각을 확정할 수 없으므로 거절한다
    response = post_raw(client, jackson_body(calculated_at='"2026-10-15T10:23:45[Asia/Seoul]"'))
    assert response.status_code == 422
    assert any("UTC" in message for message in response.json()["errors"])