# POST /predict 요청 Feature 정의서 (확정안)

기준 문서: BTS AI Feature 명세서 (4. Feature별 계산 조건, 6. 전처리, 7. 입출력 형식)
구현: `preprocessing.py`의 `FEATURE_SPECS`, `validate_payload`, `train_model.py`의 `assess`

## 1. 요청 (Spring Boot → AI)

```json
{
  "userId": "user_001",
  "calculatedAt": "2026-10-08T15:00:00+09:00",
  "windowDays": 30,
  "features": {
    "bookingCount": 20,
    "cancelCount": 15,
    "cancelRate": 0.75,
    "avgBookingInterval": 12.5,
    "shortIntervalBookingCount": 8,
    "attendanceRate": 0.25,
    "avgBookingDuration": 37.45
  }
}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `userId` | string | O | 사용자 식별자 |
| `calculatedAt` | string | O | Feature 계산 기준 시각 (`assessed_at`). ISO-8601, UTC 오프셋 필수 |
| `windowDays` | int | O | 분석 기간(일). 현재 `30`만 허용 |
| `features` | object | O | 아래 Feature 표 참고. **7개 키를 모두 전달**하고, 계산 불가는 `null` |

요청에는 7개를 모두 담고, 실제 모델이 사용하는 Feature는 모델 버전이 정한다. 1차 MVP(`iforest-v1`)는 앞 5개만 사용하고 나머지 2개는 무시한다. 나중에 모델이 바뀌어도 Spring 쪽 요청 형식은 바꾸지 않아도 된다.

## 2. Feature 정의

| 요청 필드 | 의미 | 단위/타입 | 허용 범위 | null 허용 | 계산 불가 시 | 모델 사용 |
|---|---|---|---|---|---|---|
| `bookingCount` | 최근 30일 고유 예매 건수 | 정수 | 0 이상 | X | 0 | MVP |
| `cancelCount` | 최근 30일 안에 취소가 발생한 고유 예매 건수 | 정수 | 0 이상 | X | 0 | MVP |
| `cancelRate` | 최근 30일 예매 중 `calculatedAt`까지 취소된 비율 | 비율 | 0~1 | O | 예매 0건이면 null | MVP |
| `avgBookingInterval` | 연속된 예매 완료 시각 차이의 평균 | 초 | 0 이상 | O | 예매 2건 미만이면 null | MVP |
| `shortIntervalBookingCount` | 연속 예매 간격이 5초 이하인 횟수 | 정수 | 0 이상 | X | 예매 2건 미만이면 0 | MVP |
| `attendanceRate` | 관람 완료 티켓 / 관람 여부가 확정된 유효 티켓 | 비율 | 0~1 | O | 확정된 티켓이 없으면 null | 추후 |
| `avgBookingDuration` | 예매 시작~완료 소요 시간 평균 | 초 | 0 이상 | O | 유효 데이터가 없으면 null | 추후 |

계산 규칙 (명세 4장 기준)
- 예매와 취소는 `serviceId + bookingId` 조합 기준으로 한 번만 집계합니다.
- 이벤트 정렬은 `occurredAt` 오름차순이고, 같은 시각이면 `eventId` 순입니다.
- `calculatedAt`보다 미래의 이벤트는 집계에서 제외합니다.
- `cancelCount`는 취소 발생 시각 기준, `cancelRate`는 예매 발생 시각 기준이라 둘의 집계 대상이 다를 수 있습니다. 따라서 `cancelCount > bookingCount`도 가능하며 오류가 아닙니다.
- 짧은 간격 기준(5초)은 실제 예매 시스템의 최소 예매 단위를 확인한 뒤 조정합니다.

## 3. 결측 처리

| 상황 | 처리 |
|---|---|
| Feature가 `null` | AI가 학습 데이터의 중앙값으로 대체 (실험에서 평균과 동급, 0 대체는 오탐 증가) |
| `bookingCount < 2` (신규 사용자 등) | 점수를 만들지 않고 `status: "INSUFFICIENT_DATA"`, `anomalyScore: null` 반환 |
| `null`을 허용하지 않는 Feature가 `null` | 검증 오류 (422) |

주의: 데이터 누락을 정상 행동으로 해석하지 않는다는 명세 6장 원칙에 따라, `null`은 Spring 쪽에서 `0` 등으로 미리 바꾸지 말고 그대로 전달해야 합니다.

## 4. 검증 규칙 (위반 시 422)

- `userId`가 없거나 비어 있으면 오류
- `calculatedAt`이 ISO-8601이 아니거나 UTC 오프셋이 없으면 오류
- `windowDays != 30`이면 오류
- `features`가 객체가 아니면 오류
- 각 Feature가 숫자가 아니거나(문자열, boolean, NaN, Infinity) 범위를 벗어나면 오류
- 정수 Feature에 소수가 들어오면 오류
- null 불가 Feature가 null이면 오류

오류 응답 예시 (구현은 `PayloadValidationError.errors` 목록을 그대로 사용)
```json
{ "errors": ["features.cancelRate은 1 이하여야 합니다."] }
```

## 5. 응답 (AI → Spring Boot)

```json
{
  "userId": "user_001",
  "status": "OK",
  "anomalyScore": 0.82,
  "modelVersion": "iforest-v1",
  "assessedAt": "2026-10-08T06:00:01+00:00"
}
```

| 필드 | 설명 |
|---|---|
| `status` | `OK` 또는 `INSUFFICIENT_DATA` (명세 예시에는 없는 추가 필드) |
| `anomalyScore` | 0~1, 높을수록 이상. `INSUFFICIENT_DATA`면 `null` |
| `modelVersion` | 사용한 모델 버전 (`risk_assessment` 저장용) |
| `assessedAt` | 평가 시각 (UTC) |

`anomalyScore` 해석
- 학습 사용자 점수 분포에서의 백분위입니다. 0.82는 "학습 사용자의 약 82%보다 패턴이 특이하다"는 뜻입니다.
- 암표상일 확률이 아니라 행동 패턴의 특이성입니다. (명세 7장)
- 현재 설정은 학습 사용자의 상위 5%(0.95 이상)를 이상치로 분류하는 경계입니다. 경계값은 모델 번들의 `anomaly_threshold`에 저장됩니다.
- 사용한 Feature는 `feature_snapshot`에, 분석 결과는 `risk_assessment`에 Spring 쪽에서 저장합니다.

## 6. 확정이 필요한 항목

- [ ] `windowDays` 외 다른 분석 기간을 허용할지 (현재 30 고정)
- [ ] `status` 필드 추가에 대한 Spring 쪽 합의
- [ ] 최소 데이터 기준 (현재 예매 2건)
- [ ] 이상치 경계 (현재 상위 5%)와 `anomalyScore` 사용처 (알림 기준 등)