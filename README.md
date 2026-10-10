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
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
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
