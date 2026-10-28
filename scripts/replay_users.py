"""저장된 학습 결과와 실행 중인 API의 응답이 사용자 전원에서 같은지 비교한다.

학습 때 계산한 점수(data/user_predictions.csv)와, 같은 Feature를 POST /predict/batch로
보냈을 때의 점수가 일치해야 한다. 서빙 코드(검증, 변환, 정규화)가 학습 코드와 어긋났는지
전수로 확인하는 용도다.

사용법 (서버를 먼저 실행한 뒤, 프로젝트 루트에서):
    python scripts/replay_users.py                       # 기본 http://localhost:8000
    python scripts/replay_users.py http://localhost:8001

전제: 서버가 읽는 모델(models/)과 data/user_features.csv, data/user_predictions.csv가
      같은 train_model.py 실행에서 만들어진 것이어야 한다.
      (다른 모델을 쓰는 서버라면 불일치가 나오는 것이 정상이다.)
하나라도 다르면 종료 코드가 1이다.
"""

import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing import CAMEL_TO_SNAKE, FEATURE_SPECS  # noqa: E402

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
FEATURES_PATH = ROOT / "data" / "user_features.csv"
PREDICTIONS_PATH = ROOT / "data" / "user_predictions.csv"

CALCULATED_AT = "2026-10-10T00:00:00+00:00"  # main.py의 ASSESSED_AT과 같은 시각
BATCH_SIZE = 200
TOLERANCE = 6e-5  # API는 점수를 소수 4자리로 반올림해서 돌려준다


def to_payload(row):
    features = {}
    for camel, snake in CAMEL_TO_SNAKE.items():
        raw = row[snake]
        if raw in ("", "null", "NaN", "nan"):
            features[camel] = None
        elif FEATURE_SPECS[snake]["integer"]:
            features[camel] = int(float(raw))
        else:
            features[camel] = float(raw)
    return {"userId": row["user_id"], "calculatedAt": CALCULATED_AT, "windowDays": 30, "features": features}


def post_batch(payloads):
    request = urllib.request.Request(
        BASE_URL + "/predict/batch",
        data=json.dumps({"requests": payloads}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))["results"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main():
    for path in (FEATURES_PATH, PREDICTIONS_PATH):
        if not path.exists():
            print(f"파일이 없습니다: {path}\n먼저 python main.py, python train_model.py를 실행하세요.")
            return 2

    features = read_csv(FEATURES_PATH)
    expected = {row["user_id"]: row for row in read_csv(PREDICTIONS_PATH)}
    print(f"대상 서버: {BASE_URL}")
    print(f"비교 대상: 사용자 {len(features)}명\n")

    api = {}
    for start in range(0, len(features), BATCH_SIZE):
        chunk = features[start:start + BATCH_SIZE]
        for result in post_batch([to_payload(row) for row in chunk]):
            api[result["userId"]] = result

    mismatches, compared, insufficient, max_diff = [], 0, 0, 0.0
    for row in features:
        user_id = row["user_id"]
        result, truth = api.get(user_id), expected.get(user_id)
        if result is None or truth is None:
            mismatches.append((user_id, "응답 또는 학습 결과에 없음"))
            continue

        if result["status"] != truth["status"]:
            mismatches.append((user_id, f"status 불일치: API {result['status']} / 학습 {truth['status']}"))
        elif truth["status"] == "OK":
            diff = abs(result["anomalyScore"] - float(truth["anomaly_score"]))
            max_diff = max(max_diff, diff)
            compared += 1
            if diff > TOLERANCE:
                mismatches.append((user_id, f"점수 불일치: API {result['anomalyScore']} / 학습 {float(truth['anomaly_score']):.4f}"))
        else:
            insufficient += 1

    print(f"점수 비교: {compared}명, 최대 차이 {max_diff:.6f} (허용 {TOLERANCE})")
    print(f"INSUFFICIENT_DATA 일치: {insufficient}명")
    if mismatches:
        print(f"\n불일치 {len(mismatches)}건 (최대 10건 표시)")
        for user_id, reason in mismatches[:10]:
            print(f"  - {user_id}: {reason}")
        return 1

    print("\n전원 일치: 서빙 응답이 학습 결과와 같습니다.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.URLError as error:
        print(f"서버에 연결하지 못했습니다: {error.reason}")
        sys.exit(2)