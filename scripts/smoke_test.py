"""실행 중인 AI 서버(로컬 또는 컨테이너)에 실제 HTTP 요청을 보내 응답을 확인한다 (최종 6개 Feature 모델).

사용법 (서버를 먼저 실행한 뒤):
    python scripts/smoke_test.py                                   # 기본 http://localhost:8000
    python scripts/smoke_test.py http://localhost:8001              # 포트를 바꿔 매핑했을 때
    python scripts/smoke_test.py http://localhost:8000 --dump a.json  # 고정된 입력의 응답을 파일로 저장

--dump 로 저장한 파일 두 개(예: 로컬 서버와 Docker 컨테이너)를 비교하면 같은 모델인지 확인할 수 있다.
외부 패키지 없이 파이썬 기본 라이브러리만 사용한다. 하나라도 실패하면 종료 코드가 1이다.
"""

import copy
import json
import sys
import urllib.error
import urllib.request

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
DUMP = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None
if DUMP in ARGS:
    ARGS.remove(DUMP)
BASE_URL = (ARGS[0] if ARGS else "http://localhost:8000").rstrip("/")
RESPONSE_KEYS = {"userId", "status", "rawAnomalyScore", "anomalyScore", "isAnomaly", "modelVersion", "assessedAt"}

NORMAL = {
    "userId": "smoke_normal",
    "calculatedAt": "2026-10-08T15:00:00+09:00",
    "windowDays": 30,
    "features": {
        "bookingCount": 3, "avgBookingInterval": 250000.0, "bookingIntervalVariance": None,
        "cancelRate": 0.0, "cardReuseCount": 1, "ipBookingDensity": 1,
    },
}
EXTREME = {"bookingCount": 30, "avgBookingInterval": 3.0, "bookingIntervalVariance": 0.05,
           "cancelRate": 0.9, "cardReuseCount": 6, "ipBookingDensity": 8}

# --dump 에 쓰는 고정 입력: 결측 조합과 값의 범위를 고르게 넣었다
FIXED = [
    {"bookingCount": 1, "avgBookingInterval": None, "bookingIntervalVariance": None, "cancelRate": None, "cardReuseCount": 1, "ipBookingDensity": 1},
    {"bookingCount": 2, "avgBookingInterval": 86400.0, "bookingIntervalVariance": None, "cancelRate": 0.0, "cardReuseCount": 1, "ipBookingDensity": 1},
    {"bookingCount": 3, "avgBookingInterval": 250000.0, "bookingIntervalVariance": None, "cancelRate": 0.0, "cardReuseCount": 1, "ipBookingDensity": 2},
    {"bookingCount": 4, "avgBookingInterval": 9.7, "bookingIntervalVariance": None, "cancelRate": 0.0, "cardReuseCount": 1, "ipBookingDensity": 1},
    {"bookingCount": 5, "avgBookingInterval": 61.4, "bookingIntervalVariance": None, "cancelRate": 0.25, "cardReuseCount": 1, "ipBookingDensity": 5},
    {"bookingCount": 7, "avgBookingInterval": 8.3, "bookingIntervalVariance": 5.02, "cancelRate": 0.0, "cardReuseCount": 1, "ipBookingDensity": 1},
    {"bookingCount": 8, "avgBookingInterval": 120000.0, "bookingIntervalVariance": 3.1e10, "cancelRate": 0.6, "cardReuseCount": 1, "ipBookingDensity": 1},
    {"bookingCount": 12, "avgBookingInterval": 4.6, "bookingIntervalVariance": 0.12, "cancelRate": 0.2, "cardReuseCount": 2, "ipBookingDensity": 2},
    {"bookingCount": 20, "avgBookingInterval": 4.0, "bookingIntervalVariance": 1.0, "cancelRate": 1.0, "cardReuseCount": 1, "ipBookingDensity": 1},
    {"bookingCount": 25, "avgBookingInterval": 2.9, "bookingIntervalVariance": 0.03, "cancelRate": 0.3, "cardReuseCount": 1, "ipBookingDensity": 3},
    {"bookingCount": 6, "avgBookingInterval": 13.0, "bookingIntervalVariance": 1.4, "cancelRate": 0.5, "cardReuseCount": 7, "ipBookingDensity": 9},
    {"bookingCount": 4},
    EXTREME,
]


def variant(**changes):
    """NORMAL을 복사해 값을 바꾼다. 'features_키' 형태로 features 안의 값도 바꿀 수 있다."""
    payload = copy.deepcopy(NORMAL)
    for key, value in changes.items():
        if key.startswith("features_"):
            payload["features"][key[len("features_"):]] = value
        elif key == "features":
            payload["features"] = value
        else:
            payload[key] = value
    return payload


def call(method, path, body=None, raw=None):
    data = raw.encode("utf-8") if raw is not None else (None if body is None else json.dumps(body).encode("utf-8"))
    request = urllib.request.Request(BASE_URL + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8") or "null")


results = []


def check(name, condition, detail=""):
    condition = bool(condition)
    results.append(condition)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  → {detail}" if not condition and detail else ""))


def rejected(status, body, word=None):
    ok = status == 422 and isinstance(body, dict) and isinstance(body.get("errors"), list) and body["errors"]
    return ok and (word is None or any(word in message for message in body["errors"]))


def main():
    print(f"대상 서버: {BASE_URL}\n")

    print("1. 서버 상태 GET /health")
    status, health = call("GET", "/health")
    check("200", status == 200, status)
    check("모델 버전 포함", bool(health.get("modelVersion")), health)
    check("Feature 6개와 입력 순서", health.get("features") == ["avg_booking_interval", "booking_interval_variance", "cancel_rate",
                                                             "card_reuse_count", "ip_booking_density", "booking_count"], health.get("features"))
    check("Feature 정의 버전, 원시 임계값, 정규화 설정 포함",
          health.get("featureDefinitionVersion") and isinstance(health.get("rawThreshold"), float) and health.get("anomalyThreshold") == 0.5
          and health.get("scoreNormalization", {}).get("low") < health["rawThreshold"] < health["scoreNormalization"]["high"], health)
    version, threshold = health.get("modelVersion"), health.get("rawThreshold")

    print("\n2. 단일 예측 POST /predict")
    status, body = call("POST", "/predict", NORMAL)
    check("Feature 6개가 모두 있는 요청 → 200, status OK", status == 200 and body.get("status") == "OK", (status, body))
    check("응답 필드 7개", set(body) == RESPONSE_KEYS, sorted(body))
    check("rawAnomalyScore, anomalyScore(0~1), isAnomaly(bool) 반환",
          isinstance(body.get("rawAnomalyScore"), float) and isinstance(body.get("anomalyScore"), float) and 0 <= body["anomalyScore"] <= 1
          and isinstance(body.get("isAnomaly"), bool), body)
    check("모델 버전이 /health 와 같음", body.get("modelVersion") == version, body.get("modelVersion"))
    check("userId가 그대로 돌아옴", body.get("userId") == "smoke_normal", body)
    normal = body

    status, extreme = call("POST", "/predict", variant(userId="smoke_extreme", features=EXTREME))
    check("극단적인 사용자: 점수가 더 높고 isAnomaly true",
          status == 200 and extreme.get("isAnomaly") is True and extreme["anomalyScore"] >= 0.5 and extreme["rawAnomalyScore"] > normal["rawAnomalyScore"], extreme)
    check("평범한 사용자: isAnomaly false, 점수 0.5 미만", normal.get("isAnomaly") is False and normal["anomalyScore"] < 0.5, normal)

    status, body = call("POST", "/predict", variant(features=dict(EXTREME, bookingIntervalVariance=None)))
    check("bookingIntervalVariance 가 null 인 요청 → 200 OK", status == 200 and body.get("status") == "OK", (status, body))
    status, body = call("POST", "/predict", variant(features_avgBookingInterval=None, features_bookingIntervalVariance=None, features_cancelRate=None))
    check("허용된 결측(간격·분산·취소율 null) → 200 OK", status == 200 and body.get("status") == "OK" and body.get("rawAnomalyScore") is not None, (status, body))
    status, body = call("POST", "/predict", variant(features={"bookingCount": 4}))
    check("bookingCount 만 있는 요청 → 200 OK (나머지는 중앙값·기본값)", status == 200 and body.get("status") == "OK", (status, body))
    status, body = call("POST", "/predict", variant(features_bookingCount=0, features_avgBookingInterval=None, features_cancelRate=None))
    check("bookingCount 0 → INSUFFICIENT_DATA, 점수·판정 모두 null",
          status == 200 and body.get("status") == "INSUFFICIENT_DATA" and body.get("rawAnomalyScore") is None and body.get("anomalyScore") is None
          and body.get("isAnomaly") is None, (status, body))
    status, body = call("POST", "/predict", variant(userId=12345))
    check("숫자(Long) userId 허용, 응답은 문자열", status == 200 and body.get("userId") == "12345", body)

    repeated = [call("POST", "/predict", variant(features=EXTREME))[1] for _ in range(5)]
    check("같은 요청 5번 → 원시 점수·정규화 점수·판정이 모두 같음",
          len({(r.get("rawAnomalyScore"), r.get("anomalyScore"), r.get("isAnomaly")) for r in repeated}) == 1, repeated)

    print("\n3. 잘못된 요청 (422 {\"errors\": [...]} 여야 함)")
    status, body = call("POST", "/predict", variant(features={}))
    check("모든 Feature 가 없는 요청 (bookingCount 필수)", rejected(status, body, "bookingCount"), (status, body))
    for name, payload, word in [
        ("알 수 없는 Feature 이름", variant(features_bookingBurstCount=3), "bookingBurstCount"),
        ("이전 MVP 필드 (cancelCount)", variant(features_cancelCount=1), "cancelCount"),
        ("snake_case 이름 (booking_count)", variant(features_booking_count=3), "booking_count"),
        ("문자열 값", variant(features_bookingCount="6"), "bookingCount"),
        ("boolean 값", variant(features_cancelRate=True), "cancelRate"),
        ("정수여야 하는 값이 소수", variant(features_cardReuseCount=1.5), "정수"),
        ("cancelRate 범위 초과 (1.7)", variant(features_cancelRate=1.7), "cancelRate"),
        ("음수 간격", variant(features_avgBookingInterval=-1.0), "avgBookingInterval"),
        ("cardReuseCount 0 (본인 포함이라 최소 1)", variant(features_cardReuseCount=0), "1 이상"),
        ("bookingCount 가 null", variant(features_bookingCount=None), "bookingCount"),
        ("windowDays 가 30이 아님", variant(windowDays=7), "windowDays"),
        ("calculatedAt 에 UTC 오프셋 없음", variant(calculatedAt="2026-10-08T15:00:00"), "UTC"),
    ]:
        status, body = call("POST", "/predict", payload)
        check(name, rejected(status, body, word), (status, body))
    status, body = call("POST", "/predict", raw="{not json")
    check("JSON 이 아닌 본문", rejected(status, body), (status, body))

    print("\n4. 배치 POST /predict/batch")
    batch = [variant(userId="a"), variant(userId="b", features_cancelRate=9),
             variant(userId="c", features_bookingCount=0, features_avgBookingInterval=None, features_cancelRate=None),
             variant(userId="d", features=EXTREME)]
    status, body = call("POST", "/predict/batch", {"requests": batch})
    items = body.get("results", []) if status == 200 else []
    check("200 + 요청 순서대로 OK / INVALID / INSUFFICIENT_DATA / OK",
          [r.get("status") for r in items] == ["OK", "INVALID", "INSUFFICIENT_DATA", "OK"], (status, items))
    check("배치 결과가 단건 결과와 같음", len(items) == 4 and items[0].get("rawAnomalyScore") == normal["rawAnomalyScore"]
          and items[3].get("rawAnomalyScore") == extreme["rawAnomalyScore"], items)
    status, body = call("POST", "/predict/batch", {"requests": []})
    check("빈 배치는 422", status == 422, status)

    print("\n5. 원시 점수·임계값·판정의 일치 (고정 입력 %d건)" % len(FIXED))
    status, body = call("POST", "/predict/batch", {"requests": [variant(userId=f"fixed-{i}", features=f) for i, f in enumerate(FIXED)]})
    fixed = body.get("results", []) if status == 200 else []
    check("모두 OK", len(fixed) == len(FIXED) and all(r.get("status") == "OK" for r in fixed), (status, fixed))
    check("isAnomaly == (rawAnomalyScore >= rawThreshold)", fixed and all(r["isAnomaly"] == (r["rawAnomalyScore"] >= threshold) for r in fixed))
    check("isAnomaly == (anomalyScore >= 0.5)", fixed and all(r["isAnomaly"] == (r["anomalyScore"] >= 0.5) for r in fixed))
    ordered = sorted(fixed, key=lambda r: r["rawAnomalyScore"])
    check("원시 점수 순서와 정규화 점수 순서가 같음", [r["anomalyScore"] for r in ordered] == sorted(r["anomalyScore"] for r in fixed))
    check("탐지된 입력과 탐지되지 않은 입력이 모두 있음", len({r["isAnomaly"] for r in fixed}) == 2, [r.get("isAnomaly") for r in fixed])

    if DUMP:
        record = {"modelVersion": version, "featureDefinitionVersion": health.get("featureDefinitionVersion"), "rawThreshold": threshold,
                  "scoreNormalization": {k: health["scoreNormalization"].get(k) for k in ("method", "low", "threshold", "high")},
                  "results": [{"features": f, "rawAnomalyScore": r["rawAnomalyScore"], "anomalyScore": r["anomalyScore"], "isAnomaly": r["isAnomaly"]}
                              for f, r in zip(FIXED, fixed)]}
        with open(DUMP, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        print(f"\n고정 입력의 응답을 저장했습니다: {DUMP}")

    passed = sum(results)
    print(f"\n결과: {passed}/{len(results)} 통과")
    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.URLError as error:
        print(f"\n서버에 연결하지 못했습니다: {error.reason}")
        print("서버가 실행 중인지, 주소와 포트가 맞는지 확인하세요.")
        sys.exit(1)
