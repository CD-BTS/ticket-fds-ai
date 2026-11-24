# 이슈 #17 이벤트 기반 더미데이터 생성 및 이상탐지 성능 비교

위치: `ai/experiments/issue17-event-based-dummy/`. 아래 명령의 상대 경로는 이 폴더 기준이다.

기존 더미데이터는 계정 1행마다 Feature 값을 서로 독립인 분포에서 직접 뽑는다.
이 실험은 **계정의 행동 이벤트를 먼저 만들고, 그 이벤트에서 기존 추출 로직으로 Feature 를 계산해** 별도의 데이터셋을 만든 뒤 기존 방식과 비교한다.
결과와 해석은 [`REPORT.md`](REPORT.md) 에 있다.

이후 v1 을 보존한 채 이벤트 생성 규칙을 보정한 **v2** 를 추가했다. v2 의 근거·검증·모델 비교는 [`REPORT_V2.md`](REPORT_V2.md) 에 있다 (아래 "v2" 절).
v2 데이터와 Feature 6개를 그대로 두고 Isolation Forest 의 결측 처리·변환·contamination 을 검증 데이터로 고른 결과는 [`REPORT_V2_TUNING.md`](REPORT_V2_TUNING.md) 에 있다 (`scripts/tune_if_v2.py`).
오탐률(FPR) 5% 이하를 조건으로 임계값과 학습 데이터 구성(혼합 / 정상만)을 비교한 결과는 [`REPORT_V2_FPR.md`](REPORT_V2_FPR.md) 에 있다 (`scripts/tune_if_v2_fpr.py`).

v2 를 보존한 채 이벤트 생성 규칙 6개(오픈 참여 횟수, 시도 수, 매크로 도구 프로필, 선점 거절, 결제 오류)를 바꾼 **v3** 와 v2 / v3 비교는 [`REPORT_V3.md`](REPORT_V3.md) 에 있다 (`config/scenarios_v3.json`, `scripts/generate_events_v3.py`, `scripts/compare_v3.py`, `scripts/evaluate_v3.py`, `tests/test_v3_dataset.py`).
v3 이벤트를 그대로 두고 예매 간격 Feature 2개를 같은 세션 안의 간격으로 다시 계산한 실험과 BTS 적용 검토는 [`REPORT_V3_SESSION.md`](REPORT_V3_SESSION.md) 에 있다 (`scripts/extract_session_features.py`, `scripts/evaluate_session.py`, `data/v3_session/`, `output/v3_session/`).
예매 간격 생성 규칙을 실측 이벤트(#14, #16)와 비교한 분석과 개선안은 [`REPORT_INTERVAL_VALIDATION.md`](REPORT_INTERVAL_VALIDATION.md) 에 있다 (`scripts/analyze_interval_validation.py`, 읽기 전용. 데이터셋 생성·모델 학습 없음).
정상 사용자 예매를 추가로 실측하기 위한 절차와 수집·분석 스크립트는 [`measurement/README.md`](measurement/README.md) 에 있다 (실측 원본은 `measurement/data/` 에 저장하고 git 에 올리지 않는다).
첫 수집(참가자 1명, 실험 A·B·C)의 분석 결과는 [`REPORT_MEASUREMENT.md`](REPORT_MEASUREMENT.md) 에 있다.
추가 수집을 합친 통합 분석과 v4 생성 규칙 설계는 [`REPORT_MEASUREMENT_ADD.md`](REPORT_MEASUREMENT_ADD.md) 에 있다.
그 설계로 만든 v4 데이터(정상 사용자 속도에 대한 세 가정 D1·D2·D3)와 분포·민감도 분석, Isolation Forest 평가는 [`REPORT_V4.md`](REPORT_V4.md) 에 있다. 실제 사용자에 대한 성능 검증이 아니라 합성 데이터에서의 강건성 평가다.
그 결과를 BTS 적용 관점에서 정리한 의사결정 문서는 [`REPORT_V4_DECISION.md`](REPORT_V4_DECISION.md) 다 (실험과 서빙 구성의 일치 여부, 권장 정의, 수정 범위, 적용 전 검증 절차).

- 기존 더미데이터, #14·#16 의 결과, 학습·서빙 코드는 읽기만 한다. 스크립트는 실행 전후에 기존 데이터셋의 sha256 을 확인한다.
- Feature 계산은 #14 의 `extract_features.py`(BTS `FeatureAggregationService.calculate` 재현)를 import 해서 그대로 쓴다.
- 모델 비교는 #16 의 `compare_if_models.py` 에 옮겨 둔 학습 조건(함수·상수)을 import 해서 그대로 쓴다.
- 실제 서비스 DB 에는 접속하지 않는다. 모든 입력은 저장된 파일이다.

## 폴더

| 경로 | 내용 |
|---|---|
| `config/scenarios.json` | 생성 설정과 시나리오 정의. 값마다 근거(`basis`: measured / code / inherited / assumption)를 적었다 |
| `reference/macro_attempt_cycles.csv` | #16 실측 매크로 6계정의 시도별 단계 소요 시간과 다음 시도까지의 간격 (120건) |
| `scripts/build_reference_timings.py` | 위 참조 파일을 #16 의 실측 원본에서 다시 만든다 (실측 원본이 있는 PC 에서만) |
| `scripts/generate_events.py` | 이벤트 생성 |
| `scripts/extract_dataset.py` | 이벤트 → Feature 데이터셋, 추출 검증 |
| `scripts/compare_distributions.py` | 기존 방식과의 분포 비교 |
| `scripts/compare_if.py` | Isolation Forest 비교 |
| `scripts/assumption_sensitivity.py` | 생성 가정을 하나씩 바꿨을 때의 모델 평가 |
| `tests/test_event_dataset.py` | 재현성·일관성 테스트 |
| `data/single_session/` | 기본 데이터셋 (계정당 티켓 오픈 1회 참여) |
| `data/multi_session/` | 민감도 확인용 변형 (오픈 여러 번 참여) |
| `output/feature_extraction/` | 추출 검증 결과, 결측 보고 |
| `output/distribution_comparison/` | 분포 비교 CSV 와 그래프 |
| `output/model_comparison/` | 모델 비교 CSV 와 그래프 |

`data/<profile>/` 의 파일

| 파일 | 내용 |
|---|---|
| `events.csv` | 이벤트 원본. #14 `export_events.py` 가 BTS `behavior_event` 를 내보낼 때와 같은 컬럼 (약 5.7만 행, 16MB) |
| `accounts.csv` | 계정별 시나리오, 공유 묶음(IP·카드), 행동 성향(속도, 완료 확률, 취소 성향), 배정된 실측 계정 |
| `attempts.csv` | 시도별 결과(완료·이탈·선점 만료), 시작·예매·취소 시각 |
| `fairticket_event_based_features.csv` | **Feature 데이터셋.** 기존 더미와 같은 컬럼·순서·레이블 |
| `features_full.csv` | 추출기의 원래 출력(반올림 없음)과 보조 컬럼 |
| `generation.json` | seed, 설정·참조 파일 해시, 건수, 출력 파일 해시 |

## 실행

Python 환경은 #14 의 가상환경을 같이 쓴다 (`..\issue14-real-validation\.venv`, 버전은 그쪽 `requirements.lock.txt`).

```powershell
cd ai\experiments\issue17-event-based-dummy
$py = "..\issue14-real-validation\.venv\Scripts\python"

# 1. 이벤트 생성 → Feature 추출 (기본 데이터셋). 이미 있는 폴더는 --force 가 있어야 다시 쓴다
& $py scripts\generate_events.py --force
& $py scripts\extract_dataset.py

# 2. 민감도 확인용 변형
& $py scripts\generate_events.py --profile multi_session --force
& $py scripts\extract_dataset.py --profile multi_session

# 3. 비교
& $py scripts\compare_distributions.py
& $py scripts\compare_if.py                 # 생성 seed 10회 반복 포함, 수 분 걸림. --generation-seeds 0 이면 생략
& $py scripts\assumption_sensitivity.py     # 수 분 걸림

# 4. 테스트 (1~2분)
& $py tests\test_event_dataset.py
```

- seed 는 `config/scenarios.json` 의 `seed`(42)다. 같은 seed·같은 설정·같은 참조 파일이면 `events.csv`, `accounts.csv`, `attempts.csv`, Feature 데이터셋이 바이트 단위로 같다 (테스트가 확인한다).
- `--seed N` 으로 다른 seed 의 데이터셋을 만들 수 있다. 이때는 `--out` 으로 다른 폴더를 지정한다.
- `build_reference_timings.py` 는 #16 의 `runs/<run>/raw_events.csv`(git 에 올리지 않는 실측 원본)가 있어야 한다. 없는 PC 에서는 저장된 `reference/macro_attempt_cycles.csv` 를 그대로 쓰면 된다.

## 데이터셋 읽는 법

| 컬럼 | 계산 |
|---|---|
| `avg_booking_interval`, `booking_interval_variance`, `cancel_rate`, `card_reuse_count`, `ip_booking_density`, `booking_count` | BTS 계산식 그대로 (#14 추출기). 모델 입력 6개 |
| `seat_selection_latency` | BTS 가 계산하지 않는 컬럼. 컬럼 형식을 맞추려고 '시도별 (선점 성공 − 좌석 선택 진입) 시간의 계정 평균'으로 넣었다. **이 정의는 이 실험의 가정**이고 기준 모델은 쓰지 않는다 |

- 계산할 수 없는 값은 빈 칸이다: 시도 1건이면 간격·분산, 완료 예매가 0건이면 취소율. 기존 더미에는 빈 칸이 없다.
- 시도 2건이면 간격이 1개라 분산이 0 이다 (BTS 식의 결과).
- 소수 자릿수는 기존 생성 코드와 같게 반올림했다 (간격 2자리, 분산·취소율 4자리). 기존 코드의 하한(`max(0.1, …)`)은 계산값을 바꾸므로 적용하지 않았다.
- `scenario` 이름은 #16 v2 와 같다 (`macro` 대신 `fast` / `jitter`). 계정·행 순서·`is_anomaly` 는 원본과 같다.

## v2 (보정 버전)

v1 의 데이터셋(`data/single_session/`, `data/multi_session/`)과 설정(`config/scenarios.json`)은 그대로 두고, 별도의 설정과 폴더로 만들었다.
Feature 값을 직접 고치지 않고 이벤트 생성 규칙만 바꾼 뒤 같은 추출 로직으로 다시 계산했다.

| 경로 | 내용 |
|---|---|
| `config/scenarios_v2.json` | v2 생성 설정. `changes_from_v1` 에 바꾼 항목 4개와 근거를 적었다 |
| `data/v2/` | v2 이벤트 원본, 계정, 시도, Feature 데이터셋 (파일 구성은 v1 과 같다) |
| `scripts/compare_v2.py` | A(기존) · B(v1) · C(v2) 분포 비교, 기존 평가 기준 충족 비율, 실측 비교, Leave-One-Account-Out |
| `scripts/compare_if_v2.py` | A · B · C 모델 비교, seed 변동, `INSUFFICIENT_DATA` 분리 |
| `tests/test_v2_dataset.py` | v2 재현성·일관성, v1 파일이 그대로인지 확인 |
| `output/feature_extraction/v2/`, `output/v2/` | v2 추출 검증, 비교 결과 |

```powershell
& $py scripts\generate_events.py --config config\scenarios_v2.json --out data2 --force
& $py scripts\extract_dataset.py --profile v2 --config config\scenarios_v2.json
& $py scripts\compare_v2.py
& $py scripts\compare_if_v2.py      # scripts\compare_if.py 의 결과(output/model_comparison)가 먼저 있어야 한다
& $py tests	est_v2_dataset.py
```

`generate_events.py` 에 추가한 기능(매크로 계정 수준 효과, 단계 시간 하한)은 설정에 해당 항목이 있을 때만 동작한다.
기본 설정으로 돌리면 v1 과 바이트 단위로 같은 파일이 나온다 (`tests/test_event_dataset.py` 가 확인).
기본 설정이 아닌 설정을 쓸 때는 `--out` 을 반드시 지정해야 한다 (v1 폴더를 덮어쓰지 않기 위함).
