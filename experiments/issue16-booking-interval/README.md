# 이슈 #16 실제 예매 데이터 재수집 및 예매 간격 분포 분석

위치: `ai/experiments/issue16-booking-interval/`. 아래 명령의 상대 경로는 이 폴더 기준이다.

데모 결제 화면이 "카드번호 직접 입력"에서 "저장된 테스트 카드 선택"으로 바뀐 뒤(demo-ticketing #56),
매크로 예매 데이터를 다시 수집해 **기존 #14 실측**, **기존 더미데이터**와 예매 간격 분포를 비교한다.

- 범위: 실측 재수집과 분포 분석까지. 더미 생성 로직, Isolation Forest 모델은 바꾸지 않는다.
- #14 폴더(`../issue14-real-validation`)의 데이터·결과는 읽기만 한다. Feature 계산과 이벤트 내보내기는 #14 스크립트
  (`extract_features.py`, `export_events.py`)를 import 해서 그대로 쓴다.
- 사람(실제 사용자) 데이터는 새로 수집하지 않는다. 비교에 나오는 사람 계정은 #14 의 것이다.

## 폴더

| 경로 | 내용 |
|---|---|
| `scripts/collect.py` | 매크로 실행 → BTS 이벤트 내보내기 → 분석까지 한 번에 |
| `scripts/analyze.py` | CSV·그래프·`REPORT.md` 생성. 기존 CSV 로 그래프만 다시 그릴 수도 있다 |
| `scripts/compare_remeasure.py` | 기존 실측 · 새 실측 · 보정 더미의 유형별 3자 비교 |
| `FINAL_REPORT.md` | 최종 보고서: 실측 기반 더미데이터 검증 및 보정 |
| `scripts/validation_stats.py` | 최종 보고서용 계정 단위 비교·분포 거리 계산 (읽기만 함) → `final_validation/` |
| `scripts/compare_if_models.py` | 원본 · 보정 버전으로 Isolation Forest 를 같은 조건으로 학습·평가해 비교 |
| `model_comparison/<name>/` | 모델 비교 1회의 설정·지표·그래프·보고서 |
| `scripts/make_adjusted_dummy.py` | 기존 더미를 복사해 매크로를 Fast/Jitter 로 나누고 예매 간격 Feature 를 실측으로 조정한 버전 생성 |
| `dummy_adjusted/<version>/` | 수정 더미데이터 1개 버전과 그 비교 결과 |
| `runs/<run-id>/` | 실행 1회의 원본·CSV·그래프·보고서. 실행마다 새 폴더가 생기고 덮어쓰지 않는다 |

Python 환경은 #14 의 가상환경을 같이 쓴다 (`..\issue14-real-validation\.venv`, 버전은 그쪽 `requirements.lock.txt`).

## 실행

### 1. 스택 기동 (#14 와 같은 격리 환경)

Docker Desktop 을 켠 뒤:

```powershell
cd ..\..\..\demo-ticketing\e2e
.\Start-E2E.ps1 -DemoPort 8080        # 일회성 DB 3종 + BTS(18080) + Demo(8080)
cd ..\frontend
npm run dev                            # http://localhost:5173 (/api → 8080 프록시)
```

- `frontend/.env` 에서 `VITE_USE_MOCK_TICKETING=false`, `VITE_USE_MOCK_BTS=false` 인지 확인한다 (collect.py 가 값을 기록한다).
- 이 환경의 DB 는 메모리(tmpfs)에만 있고, 기존 개발용 컨테이너(`fairticket-mysql`, `demo-ticketing-*`)와 분리돼 있다.
- BTS 는 `..\..\ticket-fds-admin` 의 현재 체크아웃으로 뜬다. 커밋과 `origin/dev` 와의 일치 여부가 보고서에 기록된다.

### 2. 수집 + 분석

```powershell
cd ..\..\ai\experiments\issue16-booking-interval
$py = "..\issue14-real-validation\.venv\Scripts\python"
& $py scripts\collect.py                       # run-id 는 UTC 시각으로 자동 생성
& $py scripts\collect.py --run-id my-run --note "메모"
```

기본 계획은 #14 본 수집과 같다: `macro-fast-01` 3회, `macro-fast-02`·`macro-jitter-01`·`macro-human-01` 각 20회,
좌석 1석, 공연 0번, 창을 띄운 상태(headed), 기본 테스트 카드 A. 약 12분 걸린다.

| 옵션 | 설명 |
|---|---|
| `--plan 계정:모드:횟수,...` | 계정 구성 변경. 모드는 `fast` / `jitter` / `human` |
| `--card` | `1`=테스트 카드 A(기본), `2`=테스트 카드 B, 숫자 15자리=직접 입력(#14 방식 재현) |
| `--headless` | 창 없이 실행 (#14 와 조건이 달라진다) |
| `--allow-existing` | 계정에 이미 이벤트가 있어도 진행. 기본은 중단 (간격이 섞이기 때문) |
| `--skip-analyze` | 수집만 하고 분석은 나중에 |

같은 DB 에서 다시 수집하려면 계정 이름을 바꾸거나(`--plan`), 스택을 내렸다 올려 DB 를 비운다.
한 공연의 좌석은 96석이므로 한 DB 에서 기본 계획(63석)은 한 번만 돌릴 수 있다 (`--concert 1` 로 바꾸면 더 가능).

### 2-1. 연속 예매 (브라우저·로그인 재사용)

기본 방식은 예매 1건마다 `node macro.js` 를 새로 실행하므로 브라우저 실행과 로그인 시간이 예매 간격에 들어간다.
`--continuous` 를 주면 계정마다 `node macro.js --repeat N` 을 한 번 실행해, 한 브라우저·한 로그인으로 이어서 예매한다.

```powershell
& $py scripts\collect.py --continuous --baseline-run 20261122-main --run-id my-continuous
```

- `--baseline-run <run-id>`: 비교 기준을 #14 대신 이 폴더의 다른 실행으로 바꾼다 (예: 재실행 방식 `20261122-main`). 기준 실행의 파일은 읽기만 한다.
- 예매가 실패하면 매크로가 실패 종류·단계·화면 오류·주소를 기록하고 그 계정의 남은 예매를 실행하지 않는다 (`macro_runs.csv` 에 `NOT_RUN`). 계속 시도하려면 `--continue-on-fail`.
- 연속 예매에서 추가로 기록되는 값: `ms_browser_launch`(브라우저 실행), `ms_context_open`, `ms_login`(첫 건만), `ms_prev_paid_to_begin`(앞 예매 완료 → 다음 예매하기 클릭).
- 이 모드의 보고서는 `comparison_vs_baseline.csv` 를 쓰고, 구간별 소요 시간(예매 흐름 / 결제 / 재기동 / 다음 예매 준비)을 두 방식으로 나눠 보여 준다.

### 3. 분석만 다시 / 그래프만 다시

```powershell
& $py scripts\analyze.py --run-id <run-id>                  # CSV·그래프·보고서 전부 다시 생성
& $py scripts\analyze.py --run-id <run-id> --figures-only   # 기존 CSV 로 그래프만 다시 그림
```

스택이 꺼져 있어도 된다. 수집 원본(`raw_events.csv`, `macro_runs.csv` 등)은 건드리지 않는다.

### 4. 수정 더미데이터 만들기 (매크로를 Fast/Jitter 로 나누고 예매 간격을 실측으로 조정)

```powershell
& $py scripts\make_adjusted_dummy.py                                   # dummy_adjusted/v1 (실행 20261122-continuous, fast 44 + jitter 43, seed 42)
& $py scripts\make_adjusted_dummy.py --version v2 --run-id <새 run-id>  # 새 실측으로 새 버전
& $py scripts\make_adjusted_dummy.py --version v3 --split fast:30,jitter:30,human:27
```

- 원본 `ml_experiment/data/fairticket_synthetic_features.csv` 와 `generate_dummy_data.py` 는 읽기만 한다.
- 원본의 `scenario = macro` 행을 `--split` 의 행 수대로 유형에 배정하고(seed 로 섞은 순서), `scenario` 값을 유형 이름(`fast`, `jitter`)으로 바꾼다.
  수정본에는 `macro` 값이 없다. `is_anomaly` 는 그대로다.
- 이 행들의 `avg_booking_interval`, `booking_interval_variance` 만 다시 계산한다. 행마다 `booking_count − 1` 개의 간격을
  배정된 유형의 실측 간격에서만 복원추출해 평균과 모분산을 넣는다.
- 쓰기 전에 나머지 값이 원본과 같은지 확인하고, 다르면 아무것도 쓰지 않는다.
- 이미 있는 버전은 `--force` 없이는 덮어쓰지 않는다. `--force` 는 폴더를 비우고 새로 쓰므로 이전 결과가 섞이지 않는다.
- 결과: `dummy_adjusted/<version>/` 에 `fairticket_synthetic_features_adjusted.csv`, `generation.json`(방식·seed·해시·확인 결과),
  `macro_rows_before_after.csv`(행별 유형 배정과 변경 전후 값), `real_interval_pool.csv`, 통계 CSV, `figures/`, `REPORT.md`. 스택이 꺼져 있어도 된다.
- 모델 학습·평가는 하지 않는다.

#### v2: 두 번의 실측을 계정 단위로 반영 (`--method account-hierarchical`)

```powershell
& $py scripts\make_adjusted_dummy.py --version v2 --method account-hierarchical `
    --run-id 20261122-continuous --run-id 20261123-remeasure --compare-version v1
```

- v1 방식(`typed-split`, 기본값)은 유형의 실측 간격을 하나로 합쳐 복원추출하므로 같은 유형의 더미 계정 평균이 거의 같아진다.
- v2 방식은 더미 계정마다 같은 유형의 실측 계정 하나를 붙이고(계정별 행 수가 고르게), 그 계정의 간격에서만 복원추출한다.
  실측 계정 사이의 평균·분산 차이가 더미에 남는다. 분포 모양은 가정하지 않는다.
- 간격이 `--min-account-intervals`(기본 5)개 미만인 실측 계정은 쓰지 않는다.
- 유형 배정은 v1 과 같은 seed·절차라 어느 행이 fast/jitter 인지는 v1 과 같다.
- 보고서(`dummy_adjusted/v2/REPORT.md`)는 원본 · `--compare-version` · 이번 버전 · 실측 계정을 함께 비교하고,
  실측 계정 하나를 빼고 만드는 검증(`leave_one_account_out.csv`)을 포함한다. 보고서 코드는 `scripts/adjusted_dummy_v2_report.py`.
- v1 은 기존 명령(`make_adjusted_dummy.py --version v1 --force`)으로 그대로 재현된다.

### 5. 재측정과 보정 더미 비교

사이트가 바뀐 뒤 다시 측정해 기존 실측·보정 더미와 비교한다. 더미를 다시 만들거나 모델을 학습하지 않는다.

```powershell
# 수집 (Fast 2계정, Jitter 2계정, 계정당 20회 연속 예매). --macro-debug 는 브라우저 오류(HTTP 409 등)를 macro_stdout.log 에 남긴다
& $py scripts\collect.py --run-id <새 run-id> --continuous --macro-debug --baseline-run 20261122-continuous `
    --plan macro-fast-11:fast:20,macro-fast-12:fast:20,macro-jitter-11:jitter:20,macro-jitter-12:jitter:20
# 기존 실측 · 새 실측 · 보정 더미 3자 비교 → runs/<새 run-id>/dummy_comparison/
& $py scripts\compare_remeasure.py --run-id <새 run-id> --baseline-run 20261122-continuous --dummy-version v1
```

- 스택을 `Start-E2E.ps1` 기본값(Demo 18081)으로 띄웠다면 프론트는 `DEMO_API_TARGET=http://localhost:18081 npm run dev` 로 실행한다.
- 계정 이름은 그 DB 에서 처음 쓰는 이름이어야 한다 (이미 이벤트가 있으면 collect.py 가 중단한다).
- `compare_remeasure.py` 는 기존 실행과 `dummy_adjusted/<version>/` 을 읽기만 하고, 새 실행 폴더의 `dummy_comparison/` 만 쓴다.

### 6. Isolation Forest 비교 (원본 · v1 · v2)

```powershell
& $py scripts\compare_if_models.py                       # model_comparison/if-original-v1-v2/
& $py scripts\compare_if_models.py --name <이름> --versions v1 v2 --force
```

- 조건은 `ml_experiment/train_and_evaluate_if.py` 와 같다 (Feature, 전처리 없음, 7:3 계층 분할 seed 42, 트리 100개, contamination 0.20 등).
  그 파일과 서비스용 모델은 건드리지 않고, 모델 파일도 저장하지 않는다.
- 세 데이터셋의 행 순서·레이블이 같은지 먼저 확인하고 같은 행 번호로 학습/평가한다.
- 결과: `metrics.csv`, `confusion_matrix.csv`, `scenario_detection.csv`, `score_distribution.csv`, `test_predictions_<데이터셋>.csv`,
  `seed_sensitivity_*.csv`(모델 seed 20개 보조 확인), `experiment_config.json`(설정·해시), `figures/`, `REPORT.md`.
- 합성 데이터 안에서의 비교다. 실제 환경의 탐지 성능으로 읽지 않는다.

### 7. 정리

```powershell
cd ..\..\..\demo-ticketing\e2e
.\Stop-E2E.ps1          # 백엔드 종료 + e2e 컨테이너 삭제 (collect.py 가 끝난 뒤에)
```

## 결과 파일 (`runs/<run-id>/`)

| 파일 | 내용 |
|---|---|
| `REPORT.md` | 요약 보고서 (그래프 포함). 자동 생성 |
| `environment.json` | 세 레포 커밋, 매크로 옵션, 프론트 mock 설정, 버전 |
| `notes.md` | 사람이 적는 실행 메모 (`--note`). 보고서에 그대로 실린다 |
| `macro_runs.csv` | 매크로 실행 1건당 한 줄: macro.js 결과 + 실행 시작·종료 시각 |
| `raw_events.csv` | BTS `behavior_event` 내보내기 (BOOKING, BOOKING_ATTEMPT) |
| `bookings.csv` | 예매 1건당 한 줄 (#14, #16): 시작·완료 시각, 단계별 소요 시간, 카드, 실패 여부 |
| `booking_intervals.csv` | 연속 예매 간격 1개당 한 줄 + 구성요소 분해 + 긴 간격 표시 |
| `feature_summary.csv` | 계정별 Feature (BTS 계산식) + 긴 간격 제외 민감도(가상) |
| `distribution_stats.csv` | 집단별 평균·중앙값·표준편차·최소·최대·사분위수 |
| `comparison_14_vs_16.csv` 또는 `comparison_vs_baseline.csv` | 모드별 기준 대 신규 차이 (`_base` = 기준, `_new` = 이번 실행) |
| `figures/interval_distribution.png`, `_log.png` | #14 대 #16 예매 간격 히스토그램 (선형 / 로그) |
| `figures/interval_by_macro.png`, `_log.png` | Fast / Jitter / Human-like 별 간격 상자 그림 |
| `figures/feature_mean_comparison.png` | 더미 대 실측 `avg_booking_interval` |
| `figures/feature_variance_comparison.png` | 더미 대 실측 `booking_interval_variance` |
| `figures/payment_duration.png` | 결제 단계·전체 예매 소요 시간 |
| `figures/account_comparison.png` | 계정별 평균 간격·분산 |
| `figures/gap_breakdown.png` | 간격의 구성 (예매 흐름 / 결제 / 브라우저 재기동) |

## 용어

| 이름 | 뜻 |
|---|---|
| 예매 간격 `gap_seconds` | 같은 계정의 연속한 **예매 시도 시작 시각** 사이의 간격(초). attemptId 별 첫 `BOOKING_ATTEMPT` 의 `occurredAt` 기준 |
| `avg_booking_interval` | 계정의 30일 내 예매 간격의 평균. BTS `FeatureAggregationService.calculate` 와 같은 식 |
| `booking_interval_variance` | 같은 간격의 **모분산** (n 으로 나눔). 시도 2건이면 0, 1건이면 값 없음 |
| `ms_booking` | 예매 1건 안에서 "예매하기" 클릭 → 결제 완료. 간격이 아니다 |
| `ms_paid` | macro.js 실행 시작 → 결제 완료까지의 누적 시간 |
| `payment_step_ms` | `ms_paid − ms_hold`. 결제 화면 진입 → 완료 화면 (시연용 결제 지연 1,200ms 포함) |
| `after_paid_to_next_begin_s` | 앞 예매 결제 완료 → 다음 예매하기 클릭. 재실행 방식에서는 재기동 + 로그인이 포함된다 |
| `launch_overhead_s` | 간격에서 앞 예매의 `ms_booking` 과 다음 예매의 클릭 전 구간을 뺀 나머지. 브라우저 종료·node·브라우저 재기동 시간 |
| 긴 간격 `long_gap` | 계정의 간격이 5개 이상일 때 `Q3 + 3×IQR` 와 중앙값의 2배를 모두 넘는 간격. 표시만 하고 지우지 않는다 |
| "가상" | 관측값이 아니라 긴 간격을 빼고 다시 계산한 값 |

## git

이 폴더의 `.gitignore` 는 #14 관례(실측 원본은 올리지 않음)에 맞춰 `runs/*/` 의 이벤트 원본과 매크로 실행 기록을 제외한다.
보고서, 그래프, 요약 CSV 는 올릴 수 있다.
