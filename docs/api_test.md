# API 실행 및 테스트 방법

## 1. 서버 실행 (프로젝트 루트에서)

```
python -m venv .venv
.venv\Scripts\activate                    # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

python train_fds_model.py                 # 모델 번들 생성 → models/behavior_model.joblib (처음 한 번, 몇 초)
uvicorn app.main:app --port 8000
```

- 브라우저에서 http://localhost:8000/docs 를 열면 Swagger UI에서 바로 요청을 보내 볼 수 있습니다.
- 모델 파일이 없으면 서버가 시작되지 않고 `모델 파일이 없습니다: ... (python train_fds_model.py 로 만들거나 MODEL_PATH 를 확인하세요)` 가 나옵니다.
- 다른 모델 파일을 쓰려면 환경 변수 `MODEL_PATH`를 지정합니다. 이전 MVP 번들(스키마 버전 1)은 받지 않습니다.
- `train_model.py`, `main.py`, `pack_bundle.py` 는 이전 MVP 버전입니다. 지금은 쓰지 않습니다.

## 2. 자동 테스트

```
pip install pytest httpx
python -m pytest -q
```

테스트가 `train_fds_model.py` 와 저장소의 학습용 CSV 로 모델을 직접 만들므로 `models/` 폴더가 없어도 동작합니다.

## 3. 실행 중인 서버 점검 (실제 HTTP 요청)

```
python scripts/smoke_test.py                                   # 기본 http://localhost:8000
python scripts/smoke_test.py http://localhost:8001              # 다른 포트
python scripts/smoke_test.py http://localhost:8000 --dump a.json  # 고정된 입력 13건의 응답을 저장
```

파이썬 기본 라이브러리만 씁니다. 39개 항목을 확인하고 하나라도 실패하면 종료 코드가 1입니다.
`--dump` 로 저장한 파일 두 개를 비교하면 두 서버(예: 로컬과 Docker)가 같은 모델인지 확인할 수 있습니다.

## 4. cURL (Windows PowerShell)

PowerShell의 `curl`은 다른 명령의 별칭이라 반드시 **`curl.exe`** 로 입력하고, JSON은 파일로 보냅니다.

```
curl.exe http://localhost:8000/health

curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@docs/samples/predict_single.json"

curl.exe -i -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@docs/samples/predict_invalid.json"

curl.exe -X POST http://localhost:8000/predict/batch -H "Content-Type: application/json" -d "@docs/samples/predict_batch.json"
```

(`-i`는 HTTP 상태 코드를 같이 보여 줍니다. 오류 요청은 `422`가 나와야 합니다.)

## 5. 기대 결과

| 요청 | HTTP | 응답 요약 |
|---|---|---|
| `predict_single.json` | 200 | `status: "OK"`, `rawAnomalyScore` 0.5569…, `anomalyScore` 0.604425, `isAnomaly: true` |
| `bookingCount` 가 0 인 요청 | 200 | `status: "INSUFFICIENT_DATA"`, 점수와 판정이 모두 `null` (정상 판정이 아님) |
| `predict_invalid.json` | 422 | `{"errors": [...]}`: `windowDays`, 알 수 없는 Feature `cancelCount`, `cancelRate` 범위 |
| `predict_batch.json` | 200 | 4건 결과: OK(false), OK(true), INSUFFICIENT_DATA, INVALID (요청 순서 그대로) |

응답 필드와 점수의 뜻은 [POST /predict 명세](predict_feature_spec.md)를 참고합니다.

## 6. 추론 성능 벤치마크

`scripts/benchmark_inference.py` 와 [추론 성능 측정](inference_benchmark.md)은 이전 MVP 모델로 잰 결과입니다. 현재 모델로 다시 재지 않았습니다.
