# 🧠 Ticket FDS AI Serving Server (`ticket-fds-ai`)

> **비지도학습(Isolation Forest) 기반 비정상 사용자 탐지 & 추론 서버**
> Spring Boot 백엔드로부터 전달받은 사용자 Feature 벡터를 기반으로 전처리 및 모델 추론을 수행하고 정규화된 이상치 스코어(0~1)를 반환하는 경량 서빙 서버입니다.

---

## 🛠 Tech Stack

* **Language**: Python 3.10+
* **Framework**: FastAPI, Uvicorn
* **ML Library**: scikit-learn, pandas, numpy, joblib
* **Deployment**: Docker

---

## 🚀 Quick Start (Local Setup)

### 1. 가상환경 세팅 및 패키지 설치
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate
# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt

```

### 2. FastAPI 서버 로컬 실행
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Docs(Swagger): http://localhost:8000/docs

### 3. Docker 실행 (배포/통합 테스트용)
```bash
docker build -t ticket-fds-ai .
docker run -p 8000:8000 ticket-fds-ai
```
📡 API Contract (Spring Boot ↔ FastAPI 연동 스펙)
### `POST /predict`

- **Request Body (JSON)**:JSON
```bash
{
  "userId": 1,
  "features": {
    "bookingCount": 15,
    "cancelCount": 12,
    "cancelRate": 0.8,
    "avgBookingInterval": 1.25,
    "shortIntervalBookingCount": 8
  }
}
```
- Response Body (JSON):JSON
```bash

{
  "userId": 1,
  "anomalyScore": 0.87,
  "isAnomaly": true,
  "evaluatedAt": "2026-10-10T15:30:00"
}
```
## 🌿 Git 브랜치 전략

- `main`: 안정화 서빙 브랜치
- `dev`: 모델 개선 및 API 개발 브랜치
- `feat/{issue_no}-{기능명}`: 단위 기능 브랜치 (예: `feat/3-iforest-pipeline`)

## Docker 로컬 실행

Docker 이미지에는 API 실행에 필요한 코드와 학습된 모델 파일만 포함합니다. 먼저 프로젝트 루트에서 모델 파일이 있는지 확인하세요.

```bash
# models/behavior_model.joblib 파일이 없다면 먼저 로컬에서 데이터와 모델 생성
python main.py
python train_model.py

# 이미지 빌드
 docker build -t bts-ai:local .

# 컨테이너 실행 (호스트 8000번 포트를 컨테이너 8000번에 연결)
docker run --rm --name bts-ai -p 8000:8000 bts-ai:local
```

별도 터미널에서 확인합니다.

```bash
curl http://localhost:8000/health
```

브라우저에서 `http://localhost:8000/docs`를 열어 `/predict` 요청도 시험할 수 있습니다. `docker run`이 실행 중인 터미널은 서버 로그를 보여주므로 테스트 중에는 종료하지 마세요. 종료하려면 `Ctrl+C`를 누릅니다.

### Spring Boot 연동

- Spring Boot가 호스트 PC에서 실행되는 경우: AI 서버 주소는 `http://localhost:8000`입니다.
- Spring Boot도 Docker 컨테이너 안에서 실행되는 경우: 두 컨테이너를 같은 Docker 네트워크에 연결하고 AI 서버 주소를 `http://bts-ai:8000`처럼 컨테이너 이름으로 설정해야 합니다. 컨테이너 내부의 `localhost`는 해당 컨테이너 자신을 가리킵니다.
- 예측 호출: `POST /predict`, 여러 건은 `POST /predict/batch`입니다. 정확한 JSON 필드와 필수값은 `http://localhost:8000/docs`에서 확인하세요.
- `/health`가 성공하면 서버와 모델이 로드된 것입니다. 실제 연동 완료 여부는 Spring Boot 담당자의 Assessment 요청 JSON과 응답 처리까지 함께 테스트해야 합니다.

> 참고: Docker 이미지 빌드는 Docker Desktop 또는 Docker Engine이 설치·실행 중인 환경에서 수행해야 합니다. 이 개발 환경에서는 Docker CLI를 사용할 수 없어 실제 이미지 빌드 및 컨테이너 구동까지 검증하지는 못했습니다.
