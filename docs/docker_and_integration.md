# Docker 실행 및 백엔드 연동 가이드

## 1. 한눈에 보기

```
docker build -t ticket-fds-ai .                               # 이미지 빌드 (모델 학습 포함)
docker run -d --name bts-ai-server -p 8000:8000 ticket-fds-ai   # 서버 실행
python scripts/smoke_test.py                                   # 실제 HTTP 요청으로 39개 항목 확인
docker rm -f bts-ai-server                                     # 종료
```

- 이미지를 만들 때 저장소의 학습용 Feature 표(`ml_experiment/data/fds_v4_training_features.csv`)로 **최종 6개 Feature 모델을 학습**한다. 모델 파일(`models/`)은 git 에 없다.
- 학습 데이터, 학습 스크립트(`train_fds_model.py`), 패키지 버전(`requirements.txt` 에 고정)이 같으므로 로컬에서 만든 모델과 같은 모델이 나온다.
- 빌드는 첫 실행에 몇 분 걸린다 (패키지 설치). 이후에는 캐시로 빨라진다.
- 저장소에 `docker-compose.yml` 은 없다. 위의 `docker build` / `docker run` 을 쓴다.

## 2. Docker 없이 로컬에서 실행하기

```
python -m venv .venv
.venv\Scripts\activate                              # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_fds_model.py                           # 모델 번들 생성 (처음 한 번)
uvicorn app.main:app --host 0.0.0.0 --port 8000
python scripts/smoke_test.py                        # 새 터미널에서 서버 점검
```

| 누가 접속하나 | 필요한 것 |
|---|---|
| 같은 PC의 백엔드 | `--host`는 생략해도 되고 주소는 `http://localhost:8000` |
| 다른 PC의 백엔드 | 서버를 `--host 0.0.0.0`으로 실행하고 `http://내_PC_IP:8000`으로 접속 (방화벽에서 8000번 허용) |

## 3. 이미지가 하는 일 (Dockerfile 3단계)

| 단계 | 내용 |
|---|---|
| `deps` | `python:3.11-slim` 에 `requirements.txt` 설치 (버전 고정) |
| `trainer` | `train_fds_model.py` 가 학습용 CSV 로 학습 → `models/behavior_model.joblib` 과 `.meta.json` 생성 |
| `runtime` | 서빙 코드(`app/`, `inference.py`, `preprocessing.py`)와 학습된 모델만 복사. 일반 사용자(`app`, uid 10001)로 실행 |

- 최종 이미지에는 학습 데이터와 학습 코드가 들어가지 않는다.
- 이전 MVP 학습 경로(`main.py` + `train_model.py`)는 이미지에서 쓰지 않는다. `.dockerignore` 가 빌드 컨텍스트에서 제외한다.

## 4. 확인

| 확인 | 명령 / 주소 | 정상 결과 |
|---|---|---|
| 컨테이너 실행 | `docker ps` | `bts-ai-server`, STATUS가 `Up ... (healthy)` |
| 포트 매핑 | `docker ps`의 PORTS 열 | `0.0.0.0:8000->8000/tcp` |
| 서버 응답 | `http://localhost:8000/health` | `{"status":"ok","modelVersion":"fds-if-v4.0.0", ...}` |
| API 문서 | `http://localhost:8000/docs` | Swagger UI |
| 로그 | `docker logs bts-ai-server` | `모델 로드 완료: fds-if-v4.0.0`, `Application startup complete` |

- `(healthy)`가 되기까지 20~30초 걸린다. 처음에는 `(health: starting)`으로 보인다.
- 8000번이 이미 사용 중이면 `-p 8001:8000` 으로 바꾸고 `http://localhost:8001` 로 접속한다.
- `/health` 는 모델 버전, Feature 6개의 입력 순서, Feature 정의 버전, 원시 임계값, 정규화 파라미터를 돌려준다.

### 로컬 모델과 Docker 모델이 같은지 확인하기

```
python scripts/smoke_test.py http://localhost:8000 --dump local.json     # 로컬 서버
python scripts/smoke_test.py http://localhost:8001 --dump docker.json    # 컨테이너
```

두 파일의 `rawThreshold`, `scoreNormalization`, 13개 입력의 `rawAnomalyScore`·`anomalyScore`·`isAnomaly` 가 같으면 같은 모델이다.

## 5. 직접 만든 모델로 교체하기 (선택)

```
docker run -d --name bts-ai-server -p 8000:8000 -v "%cd%\models:/app/models:ro" ticket-fds-ai
```

`./models/behavior_model.joblib`이 있어야 서버가 시작된다. 파일이 없거나 이전 MVP 번들이면 컨테이너가 바로 종료되고 `docker logs`에 이유가 나온다.
로컬 모델은 이미지와 같은 패키지 버전(`requirements.txt`)으로 만든 파일만 쓴다.

## 6. 백엔드(Spring Boot) 연동

**현재 BTS 백엔드(`ticket-fds-admin/backend`)에는 AI 서버를 호출하는 코드가 없다.** 아래는 후속 연동 이슈에서 쓸 참고 내용이다.

### 6-1. AI 서버 주소

| 백엔드가 실행되는 곳 | AI 서버 주소 |
|---|---|
| 같은 PC에서 직접 실행 | `http://localhost:8000` |
| 같은 Docker 네트워크의 컨테이너 | `http://bts-ai-server:8000` (컨테이너 이름) |
| Docker 컨테이너 → 호스트 PC의 AI 서버 | `http://host.docker.internal:8000` |
| 다른 PC | `http://AI_서버_PC_IP:8000` |

### 6-2. 호출 예시 (참고용. 컴파일·실행 검증을 하지 않았다)

```java
record AiFeatures(Integer bookingCount, Double avgBookingInterval, Double bookingIntervalVariance,
                  Double cancelRate, Integer cardReuseCount, Integer ipBookingDensity) {}

record AiPredictRequest(String userId, OffsetDateTime calculatedAt, int windowDays, AiFeatures features) {}

record AiPredictResponse(String userId, String status, Double rawAnomalyScore, Double anomalyScore,
                         Boolean isAnomaly, String modelVersion, String assessedAt) {}
```

`AiFeatures` 의 필드는 `UserCalculatedFeatures` 와 1:1 이다.

### 6-3. 자주 틀리는 부분

| 항목 | 올바른 방식 | 틀리면 |
|---|---|---|
| `calculatedAt` | UTC 오프셋이 있는 시각 (`Instant`, `OffsetDateTime`, `ZonedDateTime`) | `LocalDateTime`은 오프셋이 없어 422 |
| 날짜 직렬화 | 문자열 ISO-8601 (`WRITE_DATES_AS_TIMESTAMPS` 꺼짐) | 숫자 타임스탬프는 422 |
| `windowDays` | `30` | 다른 값은 422 |
| 계산 불가 Feature | `null` 로 전달 (`0` 으로 바꾸지 않기) | `0` 은 실제 값으로 해석되어 점수가 왜곡됨 |
| Feature 이름 | 6개만 (`bookingCount`, `avgBookingInterval`, `bookingIntervalVariance`, `cancelRate`, `cardReuseCount`, `ipBookingDensity`) | 다른 이름이 있으면 422 (이전 MVP 필드 포함) |
| null 불가 Feature | `bookingCount`, `cardReuseCount`, `ipBookingDensity` | `null` 이면 422 |
| `bookingIntervalVariance` | 간격 5개 미만이면 `null` (Feature 정의 v2) | 백엔드가 아직 v1 이다. 연동 전에 바꿔야 한다 |
| `cancelRate` | 0~1 비율 (백분율 아님) | 범위 초과 시 422 |
| `userId` | Long, String 모두 가능. 응답은 **항상 문자열** | 응답 DTO를 Long으로 받으면 변환 실패할 수 있음 |
| `status: INSUFFICIENT_DATA` | 점수·판정이 `null`. **정상이나 0점으로 저장하지 않는다** | 판정하지 않은 계정이 정상으로 기록됨 |
| `anomalyScore` | 0.5 가 모델의 탐지 임계값. 확률이 아니다 | 기존 정책 기준(0.40 / 0.70 / 0.85)에 그대로 넣으면 의미가 달라진다 |

요청·응답 필드와 점수의 정의는 [POST /predict 명세](predict_feature_spec.md)에 있다.

### 6-4. 연동 테스트 시나리오 (후속 이슈)

| # | 시나리오 | 기대 결과 |
|---|---|---|
| 1 | 정상 사용자 | AI 서버 로그에 `POST /predict ... 200`, `isAnomaly: false` |
| 2 | 짧고 일정한 간격으로 많이 시도한 사용자 | `isAnomaly: true`, `anomalyScore >= 0.5` |
| 3 | 시도가 없는 사용자 (`bookingCount` 0) | `INSUFFICIENT_DATA`, 점수 `null`. 백엔드가 이를 어떻게 저장할지 결정 필요 |
| 4 | 저장 확인 | `risk_assessment` 에 점수와 `modelVersion` 저장 |
| 5 | AI 서버 중지 | 백엔드가 타임아웃·연결 오류를 별도로 처리 |
| 6 | 잘못된 요청 | AI는 422. 백엔드 로그에 `errors` 내용이 남는지 |

## 7. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop이 실행 중이 아님 |
| `port is already allocated` | 8000번 사용 중. 로컬 `uvicorn`을 끄거나 `-p 8001:8000` |
| 컨테이너가 바로 종료됨 | `docker logs` 확인. `모델 파일이 없습니다` 면 볼륨 마운트 경로(5절) |
| `번들 스키마 버전이 다릅니다` | 이전 MVP 번들이다. `python train_fds_model.py` 로 다시 만든다 |
| 빌드 중 패키지 설치 실패 | 네트워크·프록시 문제 |
| 코드를 고쳤는데 반영이 안 됨 | `docker build` 를 다시 해야 한다 (코드가 이미지에 복사되어 있다) |
| 422 | 응답 본문의 `errors` 에 이유가 있다. 6-3 표 확인 |
| Windows 에서 `pip install -r requirements.txt` 가 `UnicodeDecodeError` | 이전 `requirements.txt` 의 한글 주석 때문이었다. 지금은 주석이 ASCII 다 |

## 8. 검증 현황 (2026-11-24, Issue #20)

| 항목 | 상태 |
|---|---|
| 자동 테스트 | 102 통과, 1 건너뜀 (이전 MVP Feature 용 API 테스트) |
| 로컬 서버에 실제 HTTP 요청 (`smoke_test.py`) | 39 / 39 통과 |
| `docker build` (이미지 안에서 모델 학습) | 성공 |
| `docker run` 후 컨테이너에 실제 HTTP 요청 | 39 / 39 통과, `(healthy)`, uid 10001 로 실행 |
| 로컬 모델과 Docker 모델 | 임계값·정규화 파라미터 동일, 고정 입력 13건의 원시 점수·정규화 점수·판정 동일 (차이 0) |
| 모델 파일이 없을 때 | 서버가 시작되지 않고 원인을 출력 |
| 백엔드와의 실제 연동 | 미확인 (백엔드에 호출 코드가 없다) |

세부 결과는 `experiments/issue20-final-model/REPORT_FINAL_MODEL.md` 와 `experiments/issue20-final-model/output/` 에 있다.
