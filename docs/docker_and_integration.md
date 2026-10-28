# Docker 실행 및 Spring 연동 테스트 가이드

## 1. 한눈에 보기

```
docker compose up --build        # 이미지 빌드(모델 학습 포함) + 서버 실행
python scripts/smoke_test.py     # 서버가 Spring 요청에 맞게 응답하는지 16개 항목 확인
docker compose down              # 종료
```

- 이미지를 만들 때 **합성 데이터로 모델을 학습**하므로, 저장소를 받자마자 별도 준비 없이 실행됩니다. (`data/`, `models/`는 `.gitignore` 대상이라 저장소에 없습니다.)
- 학습과 서빙이 같은 환경에서 이뤄져서 scikit-learn 버전 불일치 문제가 없습니다.
- 빌드는 첫 실행에 몇 분 걸립니다(패키지 설치). 이후에는 캐시로 빨라집니다.

## 2. 사전 준비

- Docker Desktop을 설치하고 실행해 둡니다 (Windows는 WSL2 백엔드 권장).
- 확인: `docker --version`, `docker compose version`
- 프로젝트 루트(`BTStest`)에서 명령을 실행합니다.

## 2-1. Docker 없이 로컬 서버로 실행하기 (Docker Desktop이 안 될 때)

Docker 없이도 같은 서버를 그대로 쓸 수 있고, Spring 연동 테스트에는 차이가 없습니다.

```
python main.py                                          # 처음 한 번 (합성 데이터 생성)
python train_model.py                                   # 처음 한 번 (모델 학습)
uvicorn app.main:app --host 0.0.0.0 --port 8000
python scripts/smoke_test.py                            # 새 터미널에서 서버 점검
```

| 누가 접속하나 | 필요한 것 |
|---|---|
| 같은 PC의 Spring | `--host`는 생략해도 되고 주소는 `http://localhost:8000` |
| **다른 PC의 Spring (인영님)** | 서버를 `--host 0.0.0.0`으로 실행하고, 인영님이 `http://내_PC_IP:8000`으로 접속 |

**다른 PC에서 접속하게 할 때**
1. 내 PC의 IP 확인: PowerShell에서 `ipconfig` → 같은 네트워크(같은 Wi-Fi/사내망) 어댑터의 `IPv4 주소`
2. 방화벽: 서버를 처음 켤 때 뜨는 Windows 보안 경고에서 **개인 네트워크 허용**을 누릅니다. 창을 놓쳤다면 관리자 PowerShell에서
   ```
   New-NetFirewallRule -DisplayName "BTS AI 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
   ```
   테스트가 끝나면 규칙을 지웁니다: `Remove-NetFirewallRule -DisplayName "BTS AI 8000"`
3. 인영님이 브라우저에서 `http://내_PC_IP:8000/health`를 열어 `{"status":"ok", ...}`가 나오는지 확인한 뒤, Spring의 `ai-server.base-url`을 그 주소로 바꿉니다.

접속이 안 되면: 같은 네트워크가 맞는지, 방화벽 규칙이 있는지, 공용 Wi-Fi가 기기 간 통신을 막고 있지 않은지(카페·학교 등) 확인하세요. 막혀 있다면 인영님 PC에서 같은 코드를 받아 위 명령으로 서버를 직접 실행하는 것이 가장 간단합니다.

## 3. 이미지 빌드와 실행

**compose 사용 (권장)**
```
docker compose up --build
```
백그라운드로 실행하려면 `-d`를 붙입니다: `docker compose up --build -d`

**docker 명령 직접 사용**
```
docker build -t bts-ai-server:local .
docker run -d --name bts-ai-server -p 8000:8000 bts-ai-server:local
```

### 이미지가 하는 일 (Dockerfile 3단계)

| 단계 | 내용 |
|---|---|
| `deps` | `requirements.txt` 설치 (서빙 패키지만. pytest 등 테스트 도구는 제외) |
| `trainer` | `main.py`로 합성 데이터 생성 → `train_model.py`로 학습 → `models/behavior_model.joblib` 생성 |
| `runtime` | 서빙 코드(`app/`, `inference.py`, `preprocessing.py`)와 학습된 모델만 복사. 일반 사용자(`app`, uid 10001)로 실행 |

최종 이미지에는 학습 데이터와 학습 코드가 들어가지 않습니다.

## 4. 포트 매핑 확인

`ports: "8000:8000"`은 **"내 PC 포트 : 컨테이너 포트"** 입니다.

| 확인 | 명령 / 주소 | 정상 결과 |
|---|---|---|
| 컨테이너 실행 | `docker ps` | `bts-ai-server`, STATUS가 `Up ... (healthy)` |
| 포트 매핑 | `docker ps`의 PORTS 열 | `0.0.0.0:8000->8000/tcp` |
| 서버 응답 | 브라우저 `http://localhost:8000/health` | `{"status":"ok", ...}` |
| API 문서 | 브라우저 `http://localhost:8000/docs` | Swagger UI |
| 로그 | `docker logs bts-ai-server` | `Application startup complete` |

- `(healthy)`가 되기까지 최대 20초 정도 걸립니다. 처음에는 `(health: starting)`으로 보입니다.
- **8000번이 이미 사용 중**이면(예: 로컬에서 `uvicorn`을 켜 둔 경우) 오류가 납니다. `docker-compose.yml`의 `"8000:8000"`을 `"8001:8000"`으로 바꾸면 내 PC의 8001번으로 접속합니다. 컨테이너 쪽 포트(오른쪽)는 그대로 둡니다.

### 컨테이너 안에서 `--host 0.0.0.0`이 필요한 이유
`uvicorn`은 `--host`를 생략하면 `127.0.0.1`에서만 요청을 받습니다. 컨테이너 안의 `127.0.0.1`은 컨테이너 자신이라, 포트를 매핑해도 바깥에서 접속할 수 없습니다. Dockerfile의 `CMD`에 `--host 0.0.0.0`이 들어 있는 이유입니다. (같은 구성으로 외부 주소 접속이 되는 것과 안 되는 것을 비교해 확인했습니다.)

## 5. 서버 동작 테스트

```
python scripts/smoke_test.py                        # 기본 http://localhost:8000
python scripts/smoke_test.py http://localhost:8001  # 포트를 바꿨다면
```

파이썬 기본 라이브러리만 써서 추가 설치가 필요 없고, 하나라도 실패하면 종료 코드가 1입니다. 확인하는 것:

- `GET /health`, 단일 예측(`OK` / `INSUFFICIENT_DATA`), 숫자 `userId`, `null` Feature
- 잘못된 요청이 422 `{"errors": [...]}`로 거절되는지 (`cancelRate` 범위, `windowDays`, `calculatedAt`의 UTC 오프셋 등)
- 배치 응답이 요청 순서대로 `OK` / `INVALID` / `INSUFFICIENT_DATA`인지

## 6. 직접 만든 모델로 교체하기 (선택)

이미지에 들어 있는 모델 대신 로컬에서 학습한 모델을 쓰려면 `docker-compose.yml`의 `volumes` 주석을 풉니다.
```yaml
volumes:
  - ./models:/app/models:ro
```
`./models/behavior_model.joblib`이 있어야 서버가 시작됩니다. 파일이 없으면 컨테이너가 바로 종료되고 `docker logs`에 `모델 파일이 없습니다`가 나옵니다. 로컬 모델은 이미지와 scikit-learn 버전이 다를 수 있으니, 같은 버전으로 학습한 파일만 사용하세요.

## 7. Spring Boot 연동 테스트 (인영님과 함께)

### 7-1. Spring에서 AI 서버 주소를 어떻게 쓰나

| Spring이 실행되는 곳 | AI 서버 주소 |
|---|---|
| 내 PC에서 직접 실행 (IDE 등) | `http://localhost:8000` |
| 같은 Docker Compose 안의 컨테이너 | `http://ai-server:8000` (서비스 이름) |
| Docker 컨테이너 → 호스트 PC의 AI 서버 | `http://host.docker.internal:8000` |
| 다른 사람 PC에서 내 PC의 AI 서버 | `http://내_PC_IP:8000` (방화벽에서 8000번 허용 필요) |

`application.yml` 예시
```yaml
ai-server:
  base-url: http://localhost:8000
  timeout-ms: 3000
```

### 7-2. Spring 쪽 호출 예시 (Spring Boot 3.2+, 참고용)
이 코드는 컴파일/실행 검증을 하지 않은 예시입니다. 프로젝트 구조에 맞게 고쳐 쓰세요.

```java
record AiFeatures(Integer bookingCount, Integer cancelCount, Double cancelRate,
                  Double avgBookingInterval, Integer shortIntervalBookingCount,
                  Double attendanceRate, Double avgBookingDuration) {}

record AiPredictRequest(Long userId, OffsetDateTime calculatedAt, int windowDays, AiFeatures features) {}

record AiPredictResponse(String userId, String status, Double anomalyScore,
                         String modelVersion, String assessedAt) {}

var factory = new SimpleClientHttpRequestFactory();
factory.setConnectTimeout(1000);
factory.setReadTimeout(3000);

RestClient ai = RestClient.builder().baseUrl("http://localhost:8000").requestFactory(factory).build();

AiPredictResponse result = ai.post().uri("/predict")
        .contentType(MediaType.APPLICATION_JSON)
        .body(new AiPredictRequest(userId, OffsetDateTime.now(ZoneOffset.UTC), 30, features))
        .retrieve()
        .body(AiPredictResponse.class);
```

### 7-3. 자주 틀리는 부분 (연동 전에 확인)

| 항목 | 올바른 방식 | 틀리면 |
|---|---|---|
| `calculatedAt` | **UTC 오프셋이 있는** 시각 (`Instant`, `OffsetDateTime`, `ZonedDateTime` 모두 가능. 소수점 이하 초와 `Z` 접미사도 가능) | `LocalDateTime`은 오프셋이 없어 422 |
| 날짜 직렬화 | 문자열 ISO-8601 (`WRITE_DATES_AS_TIMESTAMPS` 꺼짐. Spring Boot 기본값) | 숫자 타임스탬프로 나가면 422 |
| `windowDays` | `30` | 다른 값은 422 |
| 계산 불가 Feature | `null`로 전달 (`0`으로 바꾸지 않기) | `0`으로 보내면 "행동이 없다"로 해석되어 점수가 왜곡됨 |
| null 불가 Feature | `bookingCount`, `cancelCount`, `shortIntervalBookingCount`는 항상 숫자 (없으면 0) | null이면 422 |
| `cancelRate`, `attendanceRate` | 0~1 비율 (백분율 아님) | 범위 초과 시 422 |
| `userId` | Long, String 모두 가능. 응답은 **항상 문자열** | 응답 DTO를 Long으로 받으면 변환 실패할 수 있음 |
| `Content-Type` | `application/json` | 그 외는 422 |

### 7-4. 연동 테스트 시나리오

AI 서버 단독 점검(`smoke_test.py` 통과)이 끝난 뒤, Spring과 함께 아래를 확인합니다.

| # | 시나리오 | 호출 | 기대 결과 |
|---|---|---|---|
| 1 | 정상 사용자 | `POST /api/v1/users/{userId}/assessments` | Spring 응답에 `anomalyScore`가 있고, AI 서버 로그에 `POST /predict ... 200` |
| 2 | 이상 행동 사용자 (예매 많고 간격 짧음) | 위와 동일 | 정상 사용자보다 높은 점수 (0.95 이상이 상위 5% 경계) |
| 3 | 예매 1건 사용자 | 위와 동일 | AI는 `INSUFFICIENT_DATA`, `anomalyScore: null`. Spring이 이 경우를 어떻게 저장하고 응답할지 확인 |
| 4 | 저장 확인 | DB 조회 | `risk_assessment`에 `anomalyScore`, `modelVersion`, `assessedAt` 저장 |
| 5 | AI 서버 중지 | `docker compose stop` 후 같은 API 호출 | Spring이 타임아웃/연결 오류를 별도 에러 코드로 응답 (500으로 터지지 않는지) |
| 6 | 잘못된 요청 | 일부러 `cancelRate=1.7`로 호출 | AI는 422. Spring 로그에 AI의 `errors` 내용이 남는지 |

Spring이 보낸 요청을 AI 서버 쪽에서 보려면 `docker logs -f bts-ai-server`를 켜 둡니다. 요청마다 경로와 상태 코드가 한 줄씩 나옵니다.

## 8. 문제 해결

| 증상 | 원인 / 해결 |
|---|---|
| `docker: command not found` | Docker Desktop 미설치 또는 터미널을 다시 열지 않음 |
| `Cannot connect to the Docker daemon` | Docker Desktop이 실행 중이 아님. 켜고 초록불이 될 때까지 대기 |
| `port is already allocated` | 8000번 사용 중. 로컬 `uvicorn`을 끄거나 `"8001:8000"`으로 변경 |
| `http://localhost:8000` 접속 안 됨 | `docker ps`에서 `Up`인지 확인. 종료됐다면 `docker logs bts-ai-server`로 원인 확인 |
| 컨테이너가 바로 종료됨 | 로그에 `모델 파일이 없습니다`면 볼륨 마운트 경로 확인 (6장) |
| 빌드 중 패키지 설치 실패 | 네트워크/프록시 문제. 사내망이면 pip 프록시 설정 필요 |
| 이미지 빌드가 매번 오래 걸림 | `requirements.txt`를 바꾸면 설치 단계가 다시 실행됨. 코드만 바꾸면 캐시 사용 |
| 코드를 고쳤는데 반영이 안 됨 | `docker compose up --build`로 다시 빌드해야 함 (코드가 이미지에 복사되어 있음) |
| Spring에서 422 | 7-3 표 확인. 응답 본문의 `errors`에 이유가 적혀 있음 |
| 서버 로그에 `Unsupported upgrade request` 경고 | Java `HttpClient`가 HTTP/2 업그레이드를 시도해서 나오는 경고. 요청은 정상 처리되므로 무시해도 됨. 보기 싫으면 클라이언트를 HTTP/1.1로 고정 |
| 컨테이너 안 Spring이 `localhost:8000` 접속 실패 | 컨테이너의 `localhost`는 자기 자신. 7-1 표의 주소 사용 |

## 9. 검증 현황 (작성 시점)

| 항목 | 상태 |
|---|---|
| 자동 테스트 64개 | 통과 |
| `requirements.txt`만으로 설치, 학습, 직렬화 검증 (클린 환경) | 통과 |
| 런타임 단계 구성(`app/`, `inference.py`, `preprocessing.py`, `models/`)만으로 서버 시작 | 통과 |
| `--host 0.0.0.0`: 외부 주소 접속 성공, 생략 시 실패 | 확인 |
| `HEALTHCHECK` 명령 | 종료 코드 0 |
| 스모크 테스트 16개 (외부 주소 경유) | 통과 |
| 실제 Java 21 클라이언트 호출 (`HttpClient` 기본/HTTP 1.1, `HttpURLConnection`, 200회 연속) | 통과 (평균 약 19ms) |
| Jackson 스타일 JSON 호환 테스트 (Long `userId`, 나노초 `Z`, 지수 표기, null 포함 등) | 통과 |
| **`docker build` / `docker run` 실제 실행** | **미확인**: 작성 환경에 Docker 데몬이 없어 위 단계를 직접 재현해서 검증함. 팀원 PC에서 처음 실행할 때 확인 필요 |
| 일반 사용자(uid 10001)로 실행 | 미확인 (같은 사유) |
| Spring Boot 실제 연동 | 미확인. 7-4 시나리오로 인영님과 확인 필요 |