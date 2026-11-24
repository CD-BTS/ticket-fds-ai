# 최종 FDS 모델과 서빙 파이프라인 (Issue #20)

작성일 2026-11-24. 모델 `fds-if-v4.0.0`, Feature 정의 `fds-features-v2`, 학습 데이터 `v4-mixed-seed2026`.

> 학습과 검증에 쓴 데이터는 모두 **합성 데이터**다 (#17 의 v4 생성기). 이 문서의 수치는 합성 데이터 안의 값이고 실제 운영 성능이 아니다.
> D1·D2·D3 는 정상 사용자 속도에 대한 가정이며 어느 것도 실제 분포가 아니다.

수치의 출처는 `output/` 의 파일이다. 범위는 `ai` 저장소까지이고 BTS 백엔드와 demo-ticketing 은 수정하지 않았다.

## 요약

| 항목 | 결과 |
|---|---|
| Feature | 6개 유지. `booking_interval_variance` 만 간격 5개 미만이면 null |
| 학습 데이터 | **Mixed** (D1·D2·D3 를 계정 단위로 3등분). 네 후보의 차이는 매우 작았다 |
| 모델 | Isolation Forest (n_estimators 100, contamination auto), 결측 중앙값 대체 + log1p |
| 원시 임계값 | 0.518815 (validation 정상 계정 오탐률 4.8%) |
| 독립 검증 (새 seed 30개 데이터셋) | F1 0.789~0.792, Recall 0.801, Precision 0.78, ROC-AUC 0.92 |
| 독립 검증 오탐률 | **0.055~0.057 (목표 0.05 초과)**. 30개 중 27개 데이터셋에서 0.05 를 넘었다 |
| 서빙 | 응답에 `rawAnomalyScore`, `isAnomaly` 추가. `anomalyScore` 는 임계값을 0.5 에 맞춘 값 |
| 검증 | 자동 테스트 102 통과, 로컬·Docker 서버에 실제 HTTP 요청 39/39 통과, 로컬과 Docker 모델 일치 |

## 1. 최종 Feature 정의 (v2)

`calculatedAt` 기준 최근 30일의 `behavior_event` 에서 계산한다. 시도 = `BOOKING_ATTEMPT` 를 `serviceId + attemptId` 로 묶은 것, 시작 시각 = 그 시도의 가장 이른 이벤트.

| 순서 | Feature | 계산식 | 단위 | 결측 |
|---|---|---|---|---|
| 1 | `avg_booking_interval` | 연속한 두 시도의 시작 간격 `max(0, t[i+1] − t[i])` 의 평균 | 초 | 시도 2건 미만이면 null |
| 2 | `booking_interval_variance` | 같은 간격의 모분산 `Σ(x − 평균)² / n` | 초² | **간격 5개 미만(시도 6건 미만)이면 null** |
| 3 | `cancel_rate` | 완료 예매 중 취소된 비율 | 0~1 | 완료 예매 0건이면 null |
| 4 | `card_reuse_count` | 같은 결제수단으로 예매한 고유 계정 수의 최댓값 (본인 포함) | 정수 ≥ 1 | null 불가, 기본 1 |
| 5 | `ip_booking_density` | 같은 IP 로 예매한 고유 계정 수의 최댓값 (본인 포함) | 정수 ≥ 1 | null 불가, 기본 1 |
| 6 | `booking_count` | 고유 예매 시도 수 | 정수 ≥ 0 | null 불가 |

- 위 순서가 모델 입력 순서다. 번들의 `feature_columns` 에 저장되고 서버가 이름으로 맞춘다.
- v1(현재 BTS 백엔드)과 다른 것은 2번뿐이다. 근거는 #17 의 `REPORT_V4.md` 5.3절, `REPORT_V4_DECISION.md` 2~4절.
- 모델에서 null 은 학습(train) 데이터의 중앙값으로 대체한다.

## 2. 학습 데이터 후보 비교

**방법** (`scripts/select_training_data.py`, 결과를 보기 전에 고정)

- 데이터: #17 이 만든 기준 조건의 계정 표를 다시 썼다 (생성 seed 0~9, 가정당 10개). 새로 생성하지 않았다.
- 분할: 계정 단위로 한 번 (test 300 / train 490 / validation 210). 네 후보가 같은 계정 분할을 쓴다.
- 후보: D1, D2, D3 는 train·validation 계정의 값을 그 가정의 데이터에서 가져온다. Mixed 는 계정을 시나리오별로 3등분해 세 가정에서 하나씩 가져온다.
- 임계값: 후보마다 같은 방식으로 만든 validation 의 정상 계정으로 정한다.
- 평가: 같은 test 계정을 D1·D2·D3 의 값으로 각각 평가한다.
- 학습 120회 (후보 4 × 생성 seed 10 × 모델 seed 3).

**확인한 것** (`selection_decision.json`, 10개 seed 모두)

| 확인 | 결과 |
|---|---|
| 후보별 학습 계정 수 | 모두 490 |
| 후보별 학습 이상 계정 비율 | 모두 19.8% |
| 같은 seed 의 D1·D2·D3 가 같은 계정·레이블 | 예 |
| 이상 계정의 Feature 가 세 데이터에서 같음 (같은 매크로 이벤트) | 예. 그래서 Mixed 는 계정을 한 번씩만 넣는다 |
| 후보 안에서 계정 중복 | 없음 |
| train / validation / test 계정이 겹치지 않음 | 예 |
| Mixed 의 구성 (seed 0) | D1 169, D2 161, D3 160 계정 |

**결과** (생성 seed 10개 평균. 출처: `selection_summary.csv`, `selection_worst_case.csv`)

| 학습 후보 | F1: D1 에서 | D2 에서 | D3 에서 | **최저 F1** | 평균 F1 | 최고 오탐률 | 최저 Recall |
|---|---|---|---|---|---|---|---|
| D1 | 0.769 | 0.768 | 0.763 | 0.763 | 0.767 | 0.053 | 0.747 |
| D2 | 0.764 | 0.771 | 0.773 | 0.764 | 0.769 | 0.054 | 0.753 |
| D3 | 0.763 | 0.771 | 0.779 | 0.763 | 0.771 | 0.057 | 0.757 |
| **Mixed** | 0.766 | 0.772 | 0.776 | **0.766** | 0.771 | **0.050** | 0.745 |

| 학습 후보 | 정상 유형별 오탐률 (세 평가 분포 중 최고) | | | | 매크로 유형별 Recall (최저) | | | |
|---|---|---|---|---|---|---|---|---|
| | normal | fast_normal | shared_ip | frequent_cancel | fast | jitter | multi_account | hold_cancel |
| D1 | 0.023 | 0.032 | 0.190 | 0.141 | 0.558 | 0.523 | 0.994 | 0.803 |
| D2 | 0.027 | 0.033 | 0.180 | 0.144 | 0.554 | 0.519 | 0.994 | 0.827 |
| D3 | 0.029 | 0.032 | 0.190 | 0.141 | 0.550 | 0.512 | 0.994 | 0.845 |
| Mixed | 0.026 | 0.033 | 0.158 | 0.119 | 0.525 | 0.506 | 0.994 | 0.825 |

생성 seed 사이의 F1 표준편차는 0.04~0.05 이고, 가장 낮은 seed 의 F1 은 D1 0.692, D2 0.707, D3 0.670, Mixed 0.701 이다.

## 3. 최종 학습 데이터 선정 근거

| 기준 (실행 전에 고정) | 적용 |
|---|---|
| 1. 세 평가 분포 중 최저 F1 이 가장 높은 후보 | Mixed 0.766 (D2 0.764, D1·D3 0.763) |
| 2. 최고와 0.01 이내인 후보끼리는 최악 분포의 오탐률이 낮은 쪽 | 네 후보가 모두 0.01 이내. 오탐률은 Mixed 0.050 이 가장 낮다 |
| 3. 그래도 같으면 Mixed | 적용할 필요 없음 |

- **Mixed 를 골랐다.** 다만 **네 후보의 차이(최저 F1 0.003)는 생성 seed 사이의 흔들림(0.04~0.05)보다 훨씬 작다.** 어느 후보를 써도 결과는 사실상 같다.
- 이유는 #17 에서 확인한 대로다: 30일 평균 간격은 예매 속도가 아니라 오픈 사이의 공백으로 정해져서, 정상 사용자의 속도 가정이 이 6개 Feature 에 거의 영향을 주지 않는다.
- Mixed 는 `shared_ip`, `frequent_cancel` 오탐이 가장 낮고, 단독 fast·jitter 매크로의 Recall 은 가장 낮다 (0.525, 0.506). 어느 쪽도 seed 흔들림 안의 차이다.
- 특정 가정을 실제 분포로 보지 않는다는 원칙과도 맞는다.

**최종 학습 데이터** (`ml_experiment/data/fds_v4_training_features.csv`, `scripts/build_datasets.py`)

| 항목 | 값 |
|---|---|
| 생성 | v4 생성기와 설정 그대로 (`scenarios_v4.json` 의 기준 조건). 생성 seed **2026** (이전에 쓰지 않은 seed) |
| 구성 | 계정 1,000개를 시나리오별로 3등분: D1 332, D2 333, D3 335. 계정은 한 번씩만 |
| 분할 | 계정 단위, seed 42: train 490 (이상 19.8%) / validation 210 (20%) / test 300 (20%) |
| 열 | `row_id`, `split`, `source_assumption`, `scenario`, `is_anomaly`, Feature 6개 |
| 내용 | 합성 데이터만. 실측 원본, 계정 ID, 개인정보 없음 |
| 결측 비율 | 간격 20.2%, 분산 70.0%, 취소율 3.7% |
| 해시 (SHA-256) | `5c367a5b0fb2d6a5…` (전체 값과 생성 정보는 옆의 `.meta.json`) |

## 4. 모델 설정과 전처리

#17 의 v4 평가와 같다. 하이퍼파라미터 탐색은 하지 않았다.

| 항목 | 값 |
|---|---|
| 모델 | `IsolationForest(n_estimators=100, contamination="auto", max_samples="auto", random_state=42)` |
| 전처리 1 | 결측 → train 계정의 중앙값 |
| 전처리 2 | log1p: `avg_booking_interval`, `booking_interval_variance`, `booking_count` |
| 스케일링 | 없음 |
| 원시 점수 | `s = −score_samples` (높을수록 이상) |
| 임계값 | validation 정상 계정(168개)의 오탐률이 5% 이하가 되는 가장 낮은 원시 점수. 판정은 `s >= 임계값` |
| 최소 데이터 기준 | `bookingCount` 가 0 이면 점수를 내지 않는다 |

| 대체에 쓰는 중앙값 (train 490계정) | 값 |
|---|---|
| `avg_booking_interval` | 129,609초 |
| `booking_interval_variance` | 5.38 × 10¹⁰ 초² |
| `cancel_rate` | 0 |
| `card_reuse_count`, `ip_booking_density` | 1 |
| `booking_count` | 3 |

- 전처리·모델은 sklearn 파이프라인 하나로 번들에 들어간다. 서버는 그 파이프라인만 호출하므로 학습과 서빙의 전처리가 같다 (테스트로 확인, 11절).
- 분산의 대체값이 큰 것은 분산이 있는 계정(시도 6건 이상)이 주로 여러 오픈에 참여한 계정이기 때문이다. 분산이 null 인 계정은 "간격이 며칠씩 벌어진 평범한 계정" 과 같은 값으로 들어간다.

## 5. 독립 검증

**방법** (`scripts/independent_validation.py`)

- 모델: `models/behavior_model.joblib` (SHA-256 `7c9fcafb7b8e6e51…`). 서버가 로드한 파일과 같은 파일이다.
- 추론: 서버와 같은 코드 경로 (`inference.assess_batch`: 요청 검증 → Feature 순서 → 번들 파이프라인 → 판정·정규화).
- 데이터: 생성 seed **100~109** × D1·D2·D3 = 30개 데이터셋, 각 1,000계정 전부. 선정(seed 0~9)과 학습(seed 2026)에 쓰지 않았다.
- 임계값과 정규화 파라미터는 번들의 값이다. 이 데이터로 다시 계산하지 않았다.
- **이 결과를 본 뒤 Feature 정의, 모델 설정, 임계값을 바꾸지 않았고 재학습하지 않았다.**

**결과** (데이터셋 10개의 평균. 출처: `independent_summary.csv`, `independent_runs.csv`)

| 평가 분포 | Precision | Recall | F1 (표준편차, 최소) | ROC-AUC | 목표 오탐률 | **실제 오탐률** (표준편차, 최대) |
|---|---|---|---|---|---|---|
| D1 | 0.780 | 0.801 | 0.790 (0.019, 0.762) | 0.920 | 0.05 | **0.056** (0.006, 0.065) |
| D2 | 0.785 | 0.801 | 0.792 (0.018, 0.760) | 0.921 | 0.05 | **0.055** (0.004, 0.062) |
| D3 | 0.778 | 0.801 | 0.789 (0.018, 0.760) | 0.920 | 0.05 | **0.057** (0.006, 0.071) |

| 정상 유형별 오탐률 | D1 | D2 | D3 |
|---|---|---|---|
| normal | 0.027 | 0.023 | 0.024 |
| fast_normal | 0.038 | 0.038 | 0.038 |
| shared_ip | **0.190** | **0.190** | **0.205** |
| frequent_cancel | **0.131** | **0.148** | **0.138** |

| 매크로 유형별 Recall (세 분포에서 같다) | 값 | 미탐률 |
|---|---|---|
| fast | 0.595 | 0.405 |
| jitter | 0.619 | 0.381 |
| multi_account | 0.986 | 0.014 |
| hold_cancel | 0.918 | 0.082 |
| 도구 프로필별: fast / jitter / human | 0.670 / 0.748 / 0.990 | |

참고로 학습용 CSV 의 test 분할(seed 2026 의 300계정, 학습·임계값에 쓰지 않음): F1 0.768, Recall 0.717, Precision 0.827, 오탐률 0.0375 (240개 중 9개), ROC-AUC 0.933.

**해석**

- **오탐률이 목표를 넘는다.** 30개 데이터셋 중 27개에서 0.05 를 넘었고 범위는 0.047~0.071 이다. 임계값을 validation 정상 계정 168개로 정했기 때문에(허용 8건) 그 정밀도에 한계가 있다. validation 에서는 0.048 이었다. **목표 오탐률은 보장값이 아니다.**
- F1 은 선정 실험(0.766)과 같은 수준에서 조금 높다 (0.79). 이상 계정이 세 평가 분포에서 같으므로 Recall 은 세 분포가 같고, 정상 계정만 달라 오탐률이 조금씩 다르다.
- 세 분포의 차이는 데이터셋 사이의 흔들림보다 작다.
- **단독 fast·jitter 매크로의 약 40% 는 탐지하지 못한다.** 탐지되는 것은 카드·IP 를 여러 계정이 같이 쓰는 경우(`multi_account` 98.6%)와 취소가 많은 경우(`hold_cancel` 91.8%)다.
- **`shared_ip` 정상 사용자의 19~21%, `frequent_cancel` 의 13~15% 가 오탐된다.** #17 에서 본 구조 그대로다.

## 6. 탐지 임계값과 정규화 점수

**값** (번들과 `/health` 에 있다)

| 파라미터 | 값 | 계산한 데이터 |
|---|---|---|
| `threshold` (원시 임계값) | 0.5188149554589429 | validation 정상 계정 168개 |
| `low` | 0.3636158540814233 | train 490계정 원시 점수의 최솟값 |
| `high` | 0.7012622341470274 | train 490계정 원시 점수의 최댓값 |

**수식** (`preprocessing.ThresholdAnchoredNormalizer`)

```
s <  threshold :  anomalyScore = 0.5 × (s − low) / (threshold − low)
s >= threshold :  anomalyScore = 0.5 + 0.5 × (s − threshold) / (high − threshold)
[0, 1] 로 자른다  (s <= low → 0, s >= high → 1)
isAnomaly = (s >= threshold)
```

| 항목 | 정의와 확인 |
|---|---|
| 임계값 | 정상 점수를 큰 순서로 놓고 허용 건수(floor(0.05 × 168) = 8) 다음 점수의 바로 위 부동소수점 값. #17 의 "점수 > k번째 값" 과 같은 판정을 `>=` 로 쓰기 위한 것이다 |
| `low`, `high` 의 근거 | 학습 데이터에서 실제로 관측된 원시 점수의 범위. 이론적 범위 (0, 1) 을 쓰면 값이 0.35~0.75 에 몰려 0~1 을 쓰지 못한다 |
| 파라미터를 계산하는 데이터 | train(`low`, `high`)과 validation 정상 계정(`threshold`)뿐. 독립 검증 데이터로 계산하지 않았다 |
| 범위 밖의 원시 점수 | 0 또는 1 로 자른다. 독립 검증 3만 계정 중 0 으로 잘린 것 0.39%, 1 로 잘린 것 0.06% |
| 같은 점수 | 잘린 계정끼리는 같은 점수(0 또는 1)가 된다. 응답은 소수 6자리로 반올림해 원시 점수 5,372종이 5,342종이 됐다. 순서가 필요하면 `rawAnomalyScore` 를 쓴다 |
| 단조성 | `low`~`high` 안에서는 순증가, 전체로는 단조 비감소. 독립 검증 3만 계정과 2만 점 격자에서 확인 |
| `anomalyScore >= 0.5` 와 `isAnomaly` | 독립 검증 3만 계정 전부에서 일치. 판정은 반올림하지 않은 원시 점수와 원시 임계값으로 한다 |
| 경계와 부동소수점 | `s = threshold` 면 정확히 0.5. 임계값 바로 아래의 값은 계산·반올림 뒤에도 0.5 미만이 되게 보정한다 (0.499999 이하). 임계값의 바로 아래·위 부동소수점 값으로 테스트했다. 독립 검증에서 임계값에 가장 가까운 두 점수는 0.5188149554589427(미탐지)과 0.518815381843285(탐지)다 |

**점수의 뜻**

- **`anomalyScore` 는 매크로·암표상일 확률이 아니다.** 이 모델의 임계값을 기준으로 한 상대 위치다.
- 이전 방식(학습 사용자 대비 백분위)과 비교

| | 백분위 (이전) | 임계값 고정 구간 선형 (채택) |
|---|---|---|
| 뜻 | 학습 사용자의 몇 % 보다 특이한가 | 탐지 임계값 대비 위치 (0.5 = 임계값) |
| 판정과의 관계 | 없음. 점수만으로 탐지 여부를 알 수 없다 | `>= 0.5` 가 탐지와 같다 |
| 분포 | 균등: 사용자 절반이 0.5 이상 | 정상 계정의 중앙값 0.17, 95% 지점 0.52 |
| 순서 보존 | 학습 최댓값 위는 모두 1, 같은 순위는 같은 점수 | 구간 안에서는 순증가 |
| 단점 | 이상 계정이 섞인 학습 데이터의 백분위라 해석이 어렵다 | `low`, `high` 가 학습 데이터의 극단값 하나에 좌우된다. 모델이 바뀌면 점수의 눈금도 바뀐다 |

**기존 BTS 정책 기준을 넘는 계정의 비율** (독립 검증, 기록만 했고 정책은 바꾸지 않았다. 출처: `independent_score_checks.json`)

| `anomalyScore` 기준 (현재 `PolicyDecisionService`) | 정상 계정: D1 | D2 | D3 | 이상 계정 |
|---|---|---|---|---|
| 0.40 이상 (CAPTCHA) | 13.6% | 13.8% | 14.7% | 85.7% |
| 0.70 이상 (DELAY) | 0.74% | 0.59% | 0.55% | 40.4% |
| 0.85 이상 (Credit 20 미만이면 RESTRICT) | 0.01% | 0.01% | 0.01% | 16.8% |

- 지금의 정책 기준을 이 점수에 그대로 쓰면 **정상 계정의 약 14% 가 CAPTCHA 대상**이 된다 (모델의 탐지 기준 0.5 보다 낮은 0.40 을 쓰기 때문이다).
- Credit 감점(`anomalyScore × 40`)도 그대로 적용하면 정상 계정 중앙값(0.17)에서 약 7점, 95% 지점(0.52)에서 약 21점이 깎인다.
- 이 값들은 합성 데이터의 것이다. 정책 기준은 후속 정책 개선 작업에서 다시 정해야 한다 (13절).

## 7. 모델 번들 구성

`models/behavior_model.joblib` (git 제외) 과 사람이 읽을 수 있는 `behavior_model.meta.json`. 스키마 버전 2.

| 키 | 내용 |
|---|---|
| `model` | 파이프라인: 중앙값 대체 → log1p → Isolation Forest |
| `feature_columns` | Feature 6개의 이름과 입력 순서 |
| `feature_definition_version` | `fds-features-v2` |
| `preprocess` | 대체 방식, **Feature 별 중앙값**, log1p 대상 열, 계산한 분할(train) |
| `raw_threshold` | 원시 점수 임계값 |
| `anomaly_threshold` | 정규화 점수 기준 (0.5) |
| `normalizer`, `normalization` | 정규화 객체와 파라미터(`low`, `threshold`, `high`), 수식 |
| `score_method` | `-score_samples` |
| `model_version` | `fds-if-v4.0.0` |
| `min_booking_count` | 1 |
| `training` | 학습 데이터 버전·파일·해시, 생성 설정과 seed, 분할 정보, 행 수, 모델 설정과 seed, 임계값 규칙, validation 지표, Python·라이브러리 버전 |
| `schema_version`, `trained_at`, `sklearn_version`, `python_version` | |

- 서버는 스키마 버전이 다르거나, `raw_threshold` 가 없거나, Feature 구성이 서버의 스펙과 다른 번들을 받지 않는다 (이전 MVP 번들 포함).

## 8. 모델 재학습 방법

**같은 모델을 다시 만들기** (학습용 CSV 가 저장소에 있다)

```
pip install -r requirements.txt
python train_fds_model.py
```

같은 CSV 와 같은 패키지 버전이면 같은 임계값·같은 점수가 나온다 (테스트와 Docker 로 확인).

**학습용 CSV 부터 다시 만들기** (#17 의 생성기와 #14 의 추출기가 필요하다)

```
cd experiments\issue20-final-model
..\..\.venv\Scripts\python scripts\build_datasets.py          # 생성 seed 2026, 100~109 → CSV 와 .meta.json
..\..\.venv\Scripts\python scripts\select_training_data.py    # (참고) 후보 비교
cd ..\..
python train_fds_model.py
.venv\Scripts\python experiments\issue20-final-model\scripts\independent_validation.py   # 독립 검증 (한 번만)
```

| 기록 | 위치 |
|---|---|
| 생성 설정과 seed, 분할, CSV 해시, 버전 | `ml_experiment/data/fds_v4_training_features.meta.json` |
| 모델 seed, 임계값, 중앙값, 버전 | `models/behavior_model.meta.json` (생성 후) |
| 패키지 버전 | `requirements.txt` (scikit-learn 1.9.1, numpy 2.4.6, pandas 3.1.1, scipy 1.17.1, joblib 1.6.0) |

모델 파일은 기존 정책대로 git 에 올리지 않는다 (`.gitignore` 의 `models/`). 학습용 CSV 와 메타 파일만 예외로 올린다.

## 9. AI 서버 실행 방법

```
python train_fds_model.py
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```
docker build -t ticket-fds-ai .
docker run -d --name bts-ai-server -p 8000:8000 ticket-fds-ai
```

- 모델 경로는 환경 변수 `MODEL_PATH` (기본 `models/behavior_model.joblib`).
- 모델 파일이 없으면 서버가 시작되지 않고 `모델 파일이 없습니다: … (python train_fds_model.py 로 만들거나 MODEL_PATH 를 확인하세요)` 를 출력한다.
- Docker 이미지는 안에서 `train_fds_model.py` 로 모델을 만든다. 이전 MVP 경로(`main.py` + `train_model.py`)는 쓰지 않는다.
- 자세한 내용: `docs/api_test.md`, `docs/docker_and_integration.md`.

## 10. API 요청과 응답 예시

실행 중인 서버의 실제 응답이다. 전체 명세는 `docs/predict_feature_spec.md`.

**탐지되는 요청**

```json
{"userId":"user_001","calculatedAt":"2026-10-08T15:00:00+09:00","windowDays":30,
 "features":{"bookingCount":20,"avgBookingInterval":4.6,"bookingIntervalVariance":0.12,"cancelRate":0.25,"cardReuseCount":1,"ipBookingDensity":2}}
```
```json
{"userId":"user_001","status":"OK","rawAnomalyScore":0.5569191291804622,"anomalyScore":0.604425,"isAnomaly":true,"modelVersion":"fds-if-v4.0.0","assessedAt":"2026-11-24T06:57:41.514168+00:00"}
```

**분산이 null 인 평범한 요청**

```json
{"userId":"user_002","calculatedAt":"2026-10-08T15:00:00+09:00","windowDays":30,
 "features":{"bookingCount":3,"avgBookingInterval":250000.0,"bookingIntervalVariance":null,"cancelRate":0.0,"cardReuseCount":1,"ipBookingDensity":1}}
```
```json
{"userId":"user_002","status":"OK","rawAnomalyScore":0.3677115475270744,"anomalyScore":0.013195,"isAnomaly":false,"modelVersion":"fds-if-v4.0.0","assessedAt":"…"}
```

**시도가 없는 사용자** (`bookingCount` 0)

```json
{"userId":"user_003","status":"INSUFFICIENT_DATA","rawAnomalyScore":null,"anomalyScore":null,"isAnomaly":null,"modelVersion":"fds-if-v4.0.0","assessedAt":"…"}
```

**잘못된 요청** (HTTP 422)

```json
{"errors":["features에 알 수 없는 Feature가 있습니다: cancelCount (허용: avgBookingInterval, bookingIntervalVariance, cancelRate, cardReuseCount, ipBookingDensity, bookingCount)","features.cancelRate은 1 이하여야 합니다."]}
```

| 호환성 | |
|---|---|
| 응답 | 기존 5개 필드는 이름·타입 그대로. `rawAnomalyScore`, `isAnomaly` 추가 |
| `anomalyScore` 의 뜻 | 바뀌었다 (백분위 → 임계값 기준). 값을 저장·비교하는 쪽은 `modelVersion` 과 함께 다뤄야 한다 |
| 요청 | 6개 외의 Feature 이름은 422 (이전에는 무시). 이전 MVP 필드를 보내는 호출자는 고쳐야 한다 |
| `INSUFFICIENT_DATA` | `bookingCount` 0 일 때만. 정상 판정이나 0점이 아니다 |
| `/health` | `featureDefinitionVersion`, `rawThreshold`, `scoreNormalization`, `minBookingCount`, `trainingDataVersion` 추가 |

## 11. 테스트 결과

| 항목 | 결과 |
|---|---|
| 변경 전 기존 테스트 | **8 실패 / 56 통과**. 실패는 모두 이전 MVP 필드로 요청을 보내는 테스트이고, 이번 변경 전부터 있던 것이다 (`output/baseline_tests_before_changes.txt`) |
| 변경 후 자동 테스트 | **102 통과, 1 건너뜀, 실패 0**. 건너뛴 1건은 이전 MVP Feature 로 API 를 호출하는 테스트다 (사유 표시) |
| 로컬 서버에 실제 HTTP 요청 | **39 / 39 통과** (`output/smoke_test_local.txt`) |
| Docker 빌드 | 성공. 이미지 안에서 모델 학습 (`output/docker_build.log`) |
| Docker 컨테이너에 실제 HTTP 요청 | **39 / 39 통과**, `(healthy)`, uid 10001 로 실행 (`output/smoke_test_docker.txt`) |
| 로컬 모델과 Docker 모델 | 임계값·정규화 파라미터 동일, 고정 입력 13건의 원시 점수·정규화 점수·판정 동일 (차이 0, `output/local_vs_docker.json`) |

| 요청하신 검증 항목 | 자동 테스트 | 실제 HTTP |
|---|---|---|
| 모델 번들 생성과 로딩 | ✓ | ✓ |
| `/health`, `/predict` 정상 응답 | ✓ | ✓ |
| Feature 6개가 모두 있는 입력 | ✓ | ✓ |
| `booking_interval_variance` 가 null 인 입력 | ✓ | ✓ |
| 허용된 결측이 있는 입력 | ✓ | ✓ |
| 모든 Feature 가 결측인 입력 | ✓ 422 (`bookingCount` 필수). `bookingCount` 만 있으면 점수 산출 | ✓ |
| 잘못된 Feature 이름과 자료형 | ✓ | ✓ |
| 허용 범위를 벗어난 입력 | ✓ | ✓ |
| 학습 코드와 서빙 코드의 전처리 일치 | ✓ 1,000행에서 원시 점수 일치 (오차 1e-12 이내). 번들에 적힌 중앙값·log1p 로 손으로 계산한 값과도 일치 | |
| 같은 입력의 반복 추론 일치 | ✓ | ✓ 5회 |
| 원시 점수·임계값과 `isAnomaly` 일치 | ✓ 경계값 포함 | ✓ |
| 모델 버전 반환 | ✓ | ✓ |
| 모델 파일 누락 시 오류 | ✓ | ✓ 서버가 시작되지 않고 원인 출력 |
| Docker 에서 실행 | | ✓ |

- 실제 HTTP 검증에서 문제를 하나 찾아 고쳤다: 카드·IP 필드를 생략하면 단건에서는 기본값 1 이었는데 배치에서는 `INVALID` 였다. 두 경로가 같은 규칙을 쓰게 했다.
- Windows 에서 `pip install -r requirements.txt` 가 한글 주석 때문에 실패하던 것도 고쳤다 (주석을 ASCII 로).

## 12. 한계

- **합성 데이터에서의 검증이다.** 실제 사용자에 대한 Precision·Recall·오탐률은 알 수 없다.
- **오탐률이 목표(5%)를 조금 넘는다** (독립 검증 5.5~5.7%). validation 정상 계정 168개로 정한 임계값의 한계다.
- 단독으로 움직이는 fast·jitter 매크로의 약 40% 를 놓친다. 공유 IP 정상 사용자의 약 20%, 취소가 잦은 정상 사용자의 약 14% 가 오탐된다.
- 탐지와 오탐의 상당 부분이 IP·카드 공유에서 나온다. 합성 데이터의 공유 구조는 실측이 없는 가정이다.
- 독립 검증은 "새 seed" 의 독립이다. 같은 생성기와 같은 가정에서 나온 데이터라, 생성 가정 자체가 틀린 경우는 걸러내지 못한다. #17 에서 결과를 가장 크게 바꾼 가정(오픈 참여 횟수, 매크로 시도 수, 대기열 대기)은 기준 조건 하나만 썼다.
- 분산의 최소 간격 수 5 는 #17 에서 test 결과를 보고 고른 정의다. 이번 독립 검증은 새 seed 에서 그 정의를 확인했지만 3, 10 같은 다른 값은 비교하지 않았다.
- 학습 데이터의 이상 계정 비율은 20% 다. 실제 비율이 다르면 점수 분포와 임계값이 달라진다.
- `bookingCount` 외의 값이 모두 null 인 계정(시도 1건 등)은 점수가 나오지만 사실상 중앙값으로 만든 값이다. 판정의 신뢰도가 낮다.
- 정규화의 `high`, `low` 는 학습 데이터의 극단값 하나씩이다. 재학습하면 눈금이 바뀐다.
- 추론 속도는 새 모델로 다시 재지 않았다 (`docs/inference_benchmark.md` 는 이전 모델의 값).

## 13. 후속 백엔드 연동 시 수정해야 할 사항

이번 이슈에서 하지 않은 것이다 (`ticket-fds-admin`, `demo-ticketing` 은 수정하지 않았다).

| # | 대상 | 내용 |
|---|---|---|
| 1 | `FeatureAggregationService.calculate` | 간격이 5개 미만이면 `bookingIntervalVariance` 를 null 로 (Feature 정의 v2). 테스트와 `UserCalculatedFeatures` 주석도 함께. **이 변경 없이 v1 값을 보내면 모델이 학습 때 보지 못한 값을 받는다** |
| 2 | AI 호출 | `/predict` 를 호출하는 클라이언트와 AI 서버 주소 설정이 없다. 호출 시점(이벤트마다 / 주기적 / 정책 결정 직전)도 정해야 한다 |
| 3 | `RiskAssessment` 저장 | 저장하는 코드가 없다. 테이블에는 `anomaly_score`, `model_version` 만 있어 `rawAnomalyScore`, `isAnomaly` 를 저장하려면 스키마를 바꿔야 한다 |
| 4 | `INSUFFICIENT_DATA` | 정상이나 0점으로 저장하지 않는다. 저장하지 않거나 별도 상태로 둔다 |
| 5 | `PolicyDecisionService` 의 기준 0.40 / 0.70 / 0.85 | 이전 점수 체계를 전제로 한 값이다. 새 점수에서는 정상 계정의 약 14% 가 0.40 을 넘는다 (6절). `isAnomaly` 를 쓸지, 기준을 다시 정할지 결정이 필요하다 |
| 6 | `CreditCalculator` 의 `anomalyScore × 40` | 탐지되지 않은 계정도 점수에 비례해 감점된다. 감점 대상을 `isAnomaly` 로 제한할지 등을 정해야 한다 |
| 7 | 규칙 감점의 "간격 분산 1.0 미만" | Feature 정의 v2 에서는 분산이 null 인 계정이 많다. null 의 처리를 정해야 한다 |
| 8 | 임계값 | 운영 데이터에는 "정상" 레이블이 없다. 운영 전에 섀도 실행으로 탐지 비율과 탐지된 계정을 확인해 임계값을 다시 정해야 한다 |
| 9 | `backend/` 저장소 | 이전 Feature 정의의 사본이다. 배포 대상이 아님을 확인한다 |

`REPORT_V4_DECISION.md` (#17) 8~10절에 수정 파일과 적용 전 검증 절차가 더 자세히 있다.

## 부록: 파일

| 구분 | 파일 |
|---|---|
| 최종 모델 학습 | `train_fds_model.py`, `ml_experiment/data/fds_v4_training_features.csv`, `.meta.json` |
| 서빙 | `inference.py`, `preprocessing.py`, `app/schemas.py`, `app/main.py`, `Dockerfile`, `.dockerignore`, `requirements.txt` |
| 테스트 | `tests/conftest.py`, `test_api.py`, `test_inference.py`, `test_spring_payloads.py`, `test_golden_features.py`, `scripts/smoke_test.py` |
| 문서 | `README.md`, `docs/predict_feature_spec.md`, `docs/api_test.md`, `docs/docker_and_integration.md`, `docs/samples/*.json` |
| 실험 | `experiments/issue20-final-model/scripts/` (`common.py`, `select_training_data.py`, `build_datasets.py`, `independent_validation.py`), `output/` |
| 이전 MVP (표시만 추가, 삭제하지 않음) | `train_model.py`, `main.py`, `pack_bundle.py`, `experiment_*.py`, `ml_experiment/*.py`, `scripts/replay_users.py`, `build_golden.py`, `benchmark_inference.py` |

| 실행 횟수 | 계획 | 실제 |
|---|---|---|
| 데이터 생성 | 33 | 33 (학습용 3, 독립 검증용 30) |
| 모델 학습 | 121 | 선정 120 + 최종 모델 1. 그 밖에 테스트가 같은 CSV 로 같은 모델을 다시 만들고, Docker 빌드가 한 번 만든다 (실험이 아니라 재현 확인) |
