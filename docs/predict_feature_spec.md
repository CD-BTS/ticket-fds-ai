# POST /predict 명세 (Feature 정의 v2, 모델 `fds-if-v4.0.0`)

BTS 백엔드가 사용자 행동 Feature 6개를 보내면 AI 서버가 원시 이상 점수, 0~1 정규화 점수, 탐지 여부를 돌려준다.
실제 스키마는 서버의 `/docs` (Swagger) 와 `app/schemas.py`, 검증 규칙은 `preprocessing.FEATURE_SPECS` 가 기준이다.

> 이 문서의 이전 판은 MVP Feature 7개(`cancelCount`, `shortIntervalBookingCount`, `attendanceRate` 등)를 설명했다. 그 필드는 더 이상 받지 않는다 (보내면 422).

## 1. 요청

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

| 필드 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `userId` | string 또는 정수(Long) | O | 응답에는 항상 문자열로 돌려준다 |
| `calculatedAt` | string | O | ISO-8601, **UTC 오프셋 필수** (`Z`, `+09:00`). 오프셋이 없으면 422 |
| `windowDays` | 정수 | O | `30` 만 허용 |
| `features` | object | O | 아래 6개. 여기 없는 이름이 있으면 422 |
| 그 밖의 최상위 필드 | | | 무시한다 |

## 2. Feature 6개

모두 `calculatedAt` 기준 최근 30일(`start <= occurredAt <= calculatedAt`)의 `behavior_event` 에서 계산한다.
"시도" 는 `BOOKING_ATTEMPT` 이벤트를 `serviceId + attemptId` 로 묶은 것이고, 시도의 시작 시각은 그 시도의 가장 이른 이벤트 시각이다.

모델 입력 순서는 아래 표의 순서다. 요청 JSON 안의 키 순서는 상관없다 (서버가 이름으로 맞춘다).

| # | 요청 필드 (모델 열 이름) | 계산식 | 단위·타입 | 허용 범위 | null | 계산할 수 없을 때 |
|---|---|---|---|---|---|---|
| 1 | `avgBookingInterval` (`avg_booking_interval`) | 시작 시각 순으로 연속한 두 시도의 시작 간격 `max(0, t[i+1] − t[i])` 의 평균 | 초, 실수 | 0 이상 | 허용 | 시도 2건 미만(간격 없음)이면 null |
| 2 | `bookingIntervalVariance` (`booking_interval_variance`) | 같은 간격들의 **모분산** `Σ(x − 평균)² / n` | 초², 실수 | 0 이상 | 허용 | **간격이 5개 미만(시도 6건 미만)이면 null** |
| 3 | `cancelRate` (`cancel_rate`) | 완료 예매(`BOOKING`, `serviceId + bookingId` 로 중복 제거) 중 `CANCEL` 된 비율 | 비율, 실수 | 0~1 | 허용 | 완료 예매 0건이면 null |
| 4 | `cardReuseCount` (`card_reuse_count`) | 사용자가 예매에 쓴 결제수단마다 같은 서비스에서 그 수단으로 예매한 고유 계정 수를 세어 그중 최댓값 (본인 포함) | 정수 | 1 이상 | 불가 | 완료 예매가 없으면 1. 키를 생략해도 1 |
| 5 | `ipBookingDensity` (`ip_booking_density`) | 같은 방식으로 IP 기준 | 정수 | 1 이상 | 불가 | 완료 예매가 없으면 1. 키를 생략해도 1 |
| 6 | `bookingCount` (`booking_count`) | 고유 예매 시도(`attemptId`) 수 | 정수 | 0 이상 | 불가 | 시도가 없으면 0 (생략·null 은 422) |

**Feature 정의 v2 에서 바뀐 것은 2번 하나다.**

| | v1 (현재 BTS 백엔드) | v2 (이 모델) |
|---|---|---|
| `bookingIntervalVariance` | 간격이 1개 이상이면 계산 (간격 1개면 0) | 간격이 **5개 이상**일 때만 계산, 아니면 null |

- 이유: 시도가 2~5건인 계정의 분산은 값이 0 이거나 매우 작게 나와, 반복 시도를 한 계정의 "작은 분산" 과 구별되지 않는다.
- **BTS 백엔드(`FeatureAggregationService`)는 아직 v1 로 계산한다.** 연동할 때 백엔드를 v2 로 바꿔야 한다 (5절).
  바꾸기 전에 v1 값을 그대로 보내면 요청은 받아들여지지만, 모델이 학습 때 보지 못한 값(시도가 적은 계정의 0 에 가까운 분산)을 받게 된다.

## 3. 결측과 데이터 부족

| 상황 | 처리 |
|---|---|
| `bookingCount` 가 0 | 점수를 내지 않는다. `status: "INSUFFICIENT_DATA"`, `rawAnomalyScore`·`anomalyScore`·`isAnomaly` 모두 `null` |
| `bookingCount` 가 1 이상이고 1~3번 Feature 가 `null` | 학습(train) 데이터의 중앙값으로 대체한 뒤 점수를 낸다 |
| `cardReuseCount`, `ipBookingDensity` 키 생략 | 1 로 본다 (단건·배치 동일). 명시적 `null` 은 422 |
| `bookingCount` 생략 또는 `null` | 422 |
| `features` 가 빈 객체 | 422 (`bookingCount` 필수) |

- **`INSUFFICIENT_DATA` 는 "판정하지 않음" 이다.** 정상 판정이 아니고 이상 점수 0 도 아니다. 호출자는 이 응답을 정상으로 저장하거나 0 점으로 바꾸지 않는다.
- `null` 은 백엔드에서 `0` 으로 바꾸지 말고 그대로 보낸다. `0` 은 "간격이 0초" 처럼 실제 값으로 해석된다.
- **`bookingCount` 외의 값이 모두 null 인 요청도 점수는 나오지만, 그 점수는 사실상 시도 수 하나와 학습 데이터의 중앙값으로 만든 값이다.** 관측 정보가 적은 계정(시도 1건 등)의 판정은 신뢰도가 낮다. 이런 응답을 강한 제재의 근거로 쓰지 않는다.
- 대체에 쓰는 중앙값은 `/health` 에는 없고 모델 번들의 `preprocess.imputation_values` 에 있다 (`models/behavior_model.meta.json`).

## 4. 응답

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

| 필드 | 설명 |
|---|---|
| `userId` | 요청의 값을 문자열로 |
| `status` | `OK` 또는 `INSUFFICIENT_DATA` (배치에서는 `INVALID` 도 있다) |
| `rawAnomalyScore` | Isolation Forest 원시 이상 점수 `s = −score_samples`. 높을수록 이상. 반올림하지 않는다 |
| `anomalyScore` | 0~1 정규화 점수 (소수 6자리). **0.5 이상이면 `isAnomaly` 가 true 와 같은 뜻**이다 |
| `isAnomaly` | `rawAnomalyScore >= 원시 임계값` 이면 true. 원시 임계값은 `/health` 의 `rawThreshold` |
| `modelVersion` | 사용한 모델 버전 |
| `assessedAt` | 서버가 판정한 시각 (UTC) |

이전 응답(`userId`, `status`, `anomalyScore`, `modelVersion`, `assessedAt`)에 `rawAnomalyScore`, `isAnomaly` 가 추가됐다. 기존 필드의 이름과 타입은 그대로다.
다만 **`anomalyScore` 의 뜻이 바뀌었다** (이전: 학습 사용자 대비 백분위. 지금: 임계값을 0.5 에 맞춘 값).

### 4.1 정규화 점수의 정의

```
low, threshold, high = /health 의 scoreNormalization.low / threshold / high

s <  threshold :  anomalyScore = 0.5 × (s − low) / (threshold − low)
s >= threshold :  anomalyScore = 0.5 + 0.5 × (s − threshold) / (high − threshold)
그 뒤 [0, 1] 로 자른다  (s <= low → 0, s >= high → 1)
```

| 항목 | 값 |
|---|---|
| `threshold` | validation 정상 계정의 오탐률이 5% 이하가 되는 가장 낮은 원시 점수 |
| `low`, `high` | 학습(train) 계정 원시 점수의 최솟값, 최댓값. 학습 뒤에는 다시 계산하지 않는다 |
| 단조성 | `low`~`high` 안에서는 원시 점수가 크면 정규화 점수도 반드시 크다. 구간 밖은 0 또는 1 로 잘려 같은 점수가 된다 |
| 판정과의 관계 | `anomalyScore >= 0.5` ⇔ `rawAnomalyScore >= threshold` ⇔ `isAnomaly == true`. 경계에서 반올림이 판정을 뒤집지 않도록 보정한다 |
| 순서가 필요할 때 | `rawAnomalyScore` 를 쓴다 (잘림과 반올림이 없다) |

- **`anomalyScore` 는 매크로·암표상일 확률이 아니다.** "이 모델의 임계값을 기준으로 어디에 있는가" 를 0~1 로 옮긴 값이다. 0.8 은 "80% 확률로 매크로" 가 아니다.
- 모델을 다시 학습하면 `low`, `threshold`, `high` 가 바뀐다. 서로 다른 `modelVersion` 의 점수를 직접 비교하지 않는다.

### 4.2 오류 응답

형식 오류는 HTTP 422 와 `{"errors": ["...", ...]}` 다.

| 원인 | 예 |
|---|---|
| 알 수 없는 Feature 이름 | `cancelCount`, `shortIntervalBookingCount`, `booking_count`(snake_case), 오타 |
| 타입 | 문자열 `"6"`, boolean, 배열, `NaN`/`Infinity` |
| 범위 | `cancelRate` 1.7, 음수 간격, `cardReuseCount` 0 |
| 정수 아님 | `bookingCount` 3.5 |
| null 불가 | `bookingCount`, `cardReuseCount`, `ipBookingDensity` 가 `null` |
| 그 밖 | `windowDays` 가 30 이 아님, `calculatedAt` 에 오프셋 없음, JSON 이 아님 |

`POST /predict/batch` 는 `{"requests": [...]}` (1~1000건)를 받아 요청 순서대로 결과를 돌려준다. 형식이 잘못된 항목은 전체를 실패시키지 않고 그 항목만 `status: "INVALID"` 와 `errors` 를 담는다.

## 5. BTS 백엔드 연동 시 주의 (후속 이슈)

이번 변경은 AI 서버(`ai` 저장소)까지다. 아래는 백엔드 쪽에서 해야 하는 일이다.

1. **Feature 정의 v2**: `ticket-fds-admin/backend` 의 `FeatureAggregationService.calculate` 에서 간격이 5개 미만이면 `bookingIntervalVariance` 를 null 로 둔다.
2. **AI 호출과 저장**: 백엔드에는 `/predict` 를 호출하는 코드와 `risk_assessment` 에 저장하는 코드가 아직 없다.
3. **점수의 뜻이 바뀌었다**: 현재 정책(`PolicyDecisionService`)은 `anomalyScore` 0.40 / 0.70 / 0.85 를 기준으로 쓰고, Credit 은 `anomalyScore × 40` 을 감점한다. 이 값들은 이전 점수 체계를 전제로 한 것이다. 새 점수에서 0.5 가 모델의 탐지 임계값이므로, 0.40 은 "탐지되지 않은 계정의 일부" 까지 포함한다 (합성 데이터에서 정상 계정의 약 14% 가 0.40 이상이었다. `experiments/issue20-final-model/REPORT_FINAL_MODEL.md` 6절).
4. **`INSUFFICIENT_DATA`** 를 정상이나 0 점으로 저장하지 않는다.
5. `isAnomaly` 를 쓸지 `anomalyScore` 를 쓸지, 어느 값을 저장할지는 정책 개선 작업에서 정한다. `risk_assessment` 테이블에는 지금 `anomaly_score` 와 `model_version` 만 있다.
