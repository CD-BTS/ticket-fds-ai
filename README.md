# Ticket FDS AI Serving Server (`ticket-fds-ai`)

백엔드가 계산한 사용자 행동 Feature 6개를 받아 Isolation Forest 로 이상 점수와 탐지 여부를 반환하는 서버입니다.

- 현재 모델: `fds-if-v4.0.0` (Feature 정의 `fds-features-v2`)
- 모델은 **합성 데이터**로 학습·검증했습니다. 문서의 성능 수치는 실제 운영 성능이 아닙니다.
- 근거와 검증 결과: [experiments/issue20-final-model/REPORT_FINAL_MODEL.md](experiments/issue20-final-model/REPORT_FINAL_MODEL.md)

## 실행

```bash
python -m venv .venv
.\.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

python train_fds_model.py           # 모델 번들 생성 → models/behavior_model.joblib (git 에 없으므로 처음 한 번)
uvicorn app.main:app --port 8000    # http://localhost:8000/docs
```

Docker (이미지 안에서 같은 모델을 학습합니다):

```bash
docker build -t ticket-fds-ai .
docker run -d --name bts-ai-server -p 8000:8000 ticket-fds-ai
```

## 점검

```bash
python scripts/smoke_test.py        # 실행 중인 서버에 실제 HTTP 요청 (39개 항목)
pip install pytest httpx
python -m pytest -q                 # 자동 테스트
```

## API

`POST /predict`

```json
{
  "userId": "user_001",
  "calculatedAt": "2026-10-08T15:00:00+09:00",
  "windowDays": 30,
  "features": {
    "bookingCount": 20,
    "avgBookingInterval": 4.6,
    "bookingIntervalVariance": 0.12,
    "cancelRate": 0.25,
    "cardReuseCount": 1,
    "ipBookingDensity": 2
  }
}
```

```json
{
  "userId": "user_001",
  "status": "OK",
  "rawAnomalyScore": 0.5569191291804622,
  "anomalyScore": 0.604425,
  "isAnomaly": true,
  "modelVersion": "fds-if-v4.0.0",
  "assessedAt": "2026-11-24T06:57:41.514168+00:00"
}
```

| 필드 | 뜻 |
|---|---|
| `rawAnomalyScore` | Isolation Forest 원시 이상 점수 (높을수록 이상) |
| `anomalyScore` | 0~1 정규화 점수. **0.5 가 탐지 임계값**입니다. 매크로일 확률이 아닙니다 |
| `isAnomaly` | 원시 점수가 임계값 이상이면 `true` (`anomalyScore >= 0.5` 와 같은 뜻) |
| `status` | `OK`. `bookingCount` 가 0 이면 `INSUFFICIENT_DATA` 이고 점수와 판정이 `null` 입니다 (정상 판정이 아님) |

- 계산할 수 없는 Feature 는 `null` 로 보냅니다. `bookingIntervalVariance` 는 간격이 5개 미만이면 `null` 입니다.
- 6개 외의 Feature 이름, 잘못된 타입·범위는 HTTP 422 `{"errors": [...]}` 입니다.
- 여러 건은 `POST /predict/batch`, 모델 정보는 `GET /health` 입니다.

## 문서

| 문서 | 내용 |
|---|---|
| [docs/predict_feature_spec.md](docs/predict_feature_spec.md) | Feature 6개의 계산식, 결측 처리, 응답 필드, 점수의 정의 |
| [docs/api_test.md](docs/api_test.md) | 실행과 테스트 방법 |
| [docs/docker_and_integration.md](docs/docker_and_integration.md) | Docker 실행, 백엔드 연동 시 주의 사항 |

## 참고

- BTS 백엔드에는 아직 이 서버를 호출하는 코드가 없습니다. 연동 시 필요한 변경은 보고서 13절에 있습니다.
- `train_model.py`, `main.py`, `pack_bundle.py`, `experiment_*.py`, `ml_experiment/` 의 스크립트는 이전 MVP 버전입니다. 현재 실행 경로에서 쓰지 않으며, 각 파일 맨 위에 표시가 있습니다.
