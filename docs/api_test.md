# API 실행 및 테스트 방법

## 1. 서버 실행 (프로젝트 루트에서)

```
python main.py            # 1) 합성 데이터와 Feature 생성 (처음 한 번)
python train_model.py     # 2) 학습 → models/behavior_model.joblib 생성
uvicorn app.main:app --port 8000
```

- 브라우저에서 http://localhost:8000/docs 를 열면 Swagger UI에서 바로 요청을 보내 볼 수 있습니다.
- 모델 파일이 없으면 서버가 시작되지 않습니다 (2번을 먼저 실행하세요).
- 다른 모델 파일을 쓰려면 환경 변수 `MODEL_PATH`를 지정합니다.

## 2. 자동 테스트

```
python -m pytest -q
```

모델을 테스트 안에서 직접 학습하므로 `data/`, `models/` 폴더가 없어도 동작합니다.

## 3. 추론 성능 벤치마크

단건 순차 추론과 10/50/100건 배치 추론을 비교하려면 다음 명령을 실행합니다.

```
python scripts/benchmark_inference.py --iterations 100 --warmup 10
```

결과의 `latency_ms`는 p50/p95/p99, `throughput_items_per_second`는 초당 처리 사용자
수입니다. 실행 주기 선택 기준과 해석 방법은 [추론 성능 측정](inference_benchmark.md)을
참고합니다.

## 4. cURL (Windows PowerShell)

PowerShell의 `curl`은 다른 명령의 별칭이라 반드시 **`curl.exe`** 로 입력하고, JSON은 파일로 보냅니다
(따옴표 이스케이프 문제를 피할 수 있습니다).

```
curl.exe http://localhost:8000/health

curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@docs/samples/predict_single.json"

curl.exe -i -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@docs/samples/predict_invalid.json"

curl.exe -X POST http://localhost:8000/predict/batch -H "Content-Type: application/json" -d "@docs/samples/predict_batch.json"
```

(`-i`는 HTTP 상태 코드를 같이 보여 줍니다. 오류 요청은 `422`가 나와야 합니다.)

## 5. Postman

1. `New → HTTP Request`, 메서드 `POST`, 주소 `http://localhost:8000/predict`
2. `Body → raw → JSON`을 선택하고 `docs/samples/predict_single.json` 내용을 붙여 넣은 뒤 `Send`
3. 배치는 주소를 `/predict/batch`로 바꾸고 `predict_batch.json` 내용을 사용합니다.

## 6. 기대 결과

| 요청 | HTTP | 응답 요약 |
|---|---|---|
| `predict_single.json` | 200 | `status: "OK"`, `anomalyScore` 0~1 |
| 예매 1건 사용자 | 200 | `status: "INSUFFICIENT_DATA"`, `anomalyScore: null` |
| `predict_invalid.json` | 422 | `{"errors": [...]}` |
| `predict_batch.json` | 200 | 4건 결과: OK, OK, INSUFFICIENT_DATA, INVALID (요청 순서 그대로) |