"""실행 중인 AI 서버(로컬 또는 컨테이너)가 Spring이 보낼 요청에 올바르게 응답하는지 확인한다.

사용법 (서버를 먼저 실행한 뒤):
    python scripts/smoke_test.py                       # 기본 http://localhost:8000
    python scripts/smoke_test.py http://localhost:8001 # 포트를 바꿔 매핑했을 때

외부 패키지 없이 파이썬 기본 라이브러리만 사용한다. 하나라도 실패하면 종료 코드가 1이다.
"""

import copy
import json
import sys
import urllib.error
import urllib.request

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")

NORMAL = {
    "userId": "smoke_normal",
    "calculatedAt": "2026-10-08T15:00:00+09:00",
    "windowDays": 30,
    "features": {
        "bookingCount": 6, "cancelCount": 1, "cancelRate": 0.17,
        "avgBookingInterval": 80000.0, "shortIntervalBookingCount": 0,
        "attendanceRate": 0.9, "avgBookingDuration": 120.0,
    },
}


def variant(**changes):
    """NORMAL을 복사해 값을 바꾼다. 'features.키' 형태로 features 안의 값도 바꿀 수 있다."""
    payload = copy.deepcopy(NORMAL)
    for key, value in changes.items():
        if key.startswith("features_"):
            payload["features"][key[len("features_"):]] = value
        else:
            payload[key] = value
    return payload


def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        BASE_URL + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8") or "null")


results = []


def check(name, condition, detail=""):
    condition = bool(condition)
    results.append(condition)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  → {detail}" if not condition and detail else ""))


def main():
    print(f"대상 서버: {BASE_URL}\n")

    print("1. 서버 상태")
    status, body = call("GET", "/health")
    check("GET /health 200", status == 200, status)
    check("모델 버전과 Feature 목록 포함", bool(body) and "modelVersion" in body and "features" in body, body)

    print("\n2. 단일 예측 POST /predict")
    status, body = call("POST", "/predict", NORMAL)
    check("200 OK", status == 200, status)
    check("status == OK", body.get("status") == "OK", body)
    check("anomalyScore가 0~1", isinstance(body.get("anomalyScore"), float) and 0 <= body["anomalyScore"] <= 1, body)
    check("userId가 그대로 돌아옴", body.get("userId") == "smoke_normal", body)
    normal_score = body.get("anomalyScore")

    status, body = call("POST", "/predict", variant(
        userId="smoke_extreme", features_bookingCount=29, features_cancelCount=9,
        features_cancelRate=0.31, features_avgBookingInterval=190.0,
        features_shortIntervalBookingCount=19,
    ))
    check("극단적인 사용자의 점수가 평범한 사용자보다 높음",
          status == 200 and normal_score is not None and body.get("anomalyScore", 0) > normal_score, body)

    status, body = call("POST", "/predict", variant(
        features_cancelRate=None, features_avgBookingInterval=None,
        features_attendanceRate=None, features_avgBookingDuration=None,
    ))
    check("null Feature 허용", status == 200 and body.get("status") == "OK", body)

    status, body = call("POST", "/predict", variant(userId=12345))
    check("숫자(Long) userId 허용, 응답은 문자열", status == 200 and body.get("userId") == "12345", body)

    status, body = call("POST", "/predict", variant(features_bookingCount=1))
    check("예매 1건 → INSUFFICIENT_DATA, 점수 null",
          status == 200 and body.get("status") == "INSUFFICIENT_DATA" and body.get("anomalyScore") is None, body)

    print("\n3. 잘못된 요청 (422 {\"errors\": [...]} 여야 함)")
    for name, payload in {
        "cancelRate 범위 초과": variant(features_cancelRate=1.7),
        "windowDays가 30이 아님": variant(windowDays=7),
        "bookingCount가 null": variant(features_bookingCount=None),
        "calculatedAt에 UTC 오프셋 없음 (Java LocalDateTime)": variant(calculatedAt="2026-10-08T15:00:00"),
    }.items():
        status, body = call("POST", "/predict", payload)
        check(name, status == 422 and isinstance(body.get("errors"), list) and body["errors"], (status, body))

    print("\n4. 배치 POST /predict/batch")
    batch = [
        variant(userId="a"),
        variant(userId="b", features_cancelRate=9),
        variant(userId="c", features_bookingCount=1),
    ]
    status, body = call("POST", "/predict/batch", {"requests": batch})
    statuses = [r.get("status") for r in body.get("results", [])] if status == 200 else []
    check("200 + 요청 순서대로 OK / INVALID / INSUFFICIENT_DATA",
          statuses == ["OK", "INVALID", "INSUFFICIENT_DATA"], (status, statuses))
    status, body = call("POST", "/predict/batch", {"requests": []})
    check("빈 배치는 422", status == 422, status)

    passed = sum(results)
    print(f"\n결과: {passed}/{len(results)} 통과")
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.URLError as error:
        print(f"\n서버에 연결하지 못했습니다: {error.reason}")
        print("서버가 실행 중인지, 주소와 포트가 맞는지 확인하세요.")
        sys.exit(2)