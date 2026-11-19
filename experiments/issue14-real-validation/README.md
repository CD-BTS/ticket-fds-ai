# 이슈 #14 실측 데이터 검증 실험

위치: `ai/experiments/issue14-real-validation/` (AI 레포 안). 아래 명령의 상대 경로는 이 폴더 기준이다.

사람이 직접 한 예매와 Playwright 매크로 예매의 행동 이벤트를 모아, 기존 6개 Feature를 계산하고
(1) 기존 Isolation Forest 점수, (2) 기존 더미데이터 분포와 비교한다.

기존 코드(AI 모델·학습 코드·더미 생성 코드, BTS, Demo)는 수정하지 않는다. 이 폴더의 스크립트는
기존 코드를 import 하거나 SELECT 로 읽기만 한다.

## 폴더

| 경로 | 내용 |
|---|---|
| `model_build/` | 기존 학습 스크립트를 그대로 재실행해 만든 모델·번들, `BUILD_RECORD.json`(학습 조건 기록) |
| `scripts/` | 실험용 스크립트 |
| `data/<tag>/` | 내보낸 이벤트, 계산한 Feature, 점수 (`<tag>` 예: `pilot`, `main`) |
| `output/<tag>/` | 비교 통계 CSV와 그림 |
| `logs/` | 모델 학습·패키징·검증 로그 |
| `.venv/` | 실험 전용 Python 환경 (`requirements.lock.txt` 에 버전 기록) |

git 에 올리지 않는 것 (이 폴더의 `.gitignore`): `.venv/`, `data/`(실측 이벤트·Feature·점수), 모델 파일(`*.joblib`, `model_build/models/`, `model_build/ml_experiment/`), `__pycache__/`.
가상환경을 다시 만들 때: `python -m venv .venv` 후 `.venv\Scripts\python -m pip install -r requirements.lock.txt`

관련 폴더 (이 실험이 읽기만 함)

- `../..` — AI 레포 루트 (이 폴더는 `ai/experiments/issue14-real-validation`). `inference.py`, `preprocessing.py` 를 import 한다
- `../../../ticket-fds-admin` — BTS `feat/#40/risk-assessment-credit-pipeline` 별도 체크아웃 (기존 `../../../backend` 는 그대로)
- `../../../demo-ticketing` — Demo, 매크로(`macro/macro.js`), 통합 환경(`e2e/`)

## 모델

`model_build/BUILD_RECORD.json` 과 `logs/` 참고. 요약:

- 학습 데이터: `ml_experiment/data/fairticket_synthetic_features.csv` (커밋된 파일 그대로, 재생성 안 함)
- 스크립트: `ml_experiment/train_and_evaluate_if.py` → `pack_bundle.py` (수정 없음)
- 이 모델은 같은 코드·데이터·시드로 **다시 학습한 것**이며, 팀이 처음 만든 원본 파일과 동일하다는 보장은 없다.

다시 만들려면 (`model_build/` 에서 실행해야 산출물이 이 폴더에 남는다):

```powershell
cd model_build
..\.venv\Scripts\python ..\..\..\ml_experiment\train_and_evaluate_if.py
..\.venv\Scripts\python ..\..\..\pack_bundle.py
cd ..
.venv\Scripts\python scripts\verify_bundle.py
```

## 실험 실행 순서

아래에서 `<tag>` 는 `pilot` 또는 `main`.

### 1. 스택 기동

Docker Desktop 을 켠 뒤:

```powershell
cd ..\..\..\demo-ticketing\e2e
.\Start-E2E.ps1 -DemoPort 8080        # DB 3종 + BTS(18080) + Demo(8080). BTS 폴더 기본값이 ..\..\ticket-fds-admin
cd ..\frontend
npm run dev                            # http://localhost:5173 (/api → 8080 프록시)
```

- 이 환경의 DB는 메모리(tmpfs)에만 있다. **`Stop-E2E.ps1`, Docker 재시작, PC 재부팅을 하면 데이터가 사라진다.**
  수집이 끝나면 4번(내보내기)을 먼저 한다.
- 기존 개발용 컨테이너·볼륨(`fairticket-mysql`, `demo-ticketing-*`)은 쓰지 않고 건드리지 않는다.

### 2. 사람 예매

- `http://localhost:5173` 에서 계정 ID `human-01`, `human-02`, … 로 로그인한다.
- 계정마다 정해진 횟수만큼 연속으로 예매한다 (끝나면 바로 다음). 좌석 1석, 결제는 "직접 입력".
- 카드번호는 매크로 기본값과 같은 `123456789012345` 를 쓴다 (같은 결제 식별자로 맞추기 위함).
- **같은 공연을 다시 예매할 때는 새 탭을 열어 주소를 직접 입력한다** (탭 복제는 안 됨). 같은 탭에서는
  끝난 시도 정보가 브라우저 sessionStorage 에 남아 "요청이 현재 리소스 상태와 충돌합니다"(409)가 난다.
  새 탭 대신 개발자 도구 → Application → Session Storage 에서 `fds.attempt.v1.<계정>.<공연>` 키를 지워도 된다.
  서로 다른 공연(c1, c2, c3)은 같은 탭에서 이어서 예매할 수 있다.
- 계정별 시작·종료 시각과 특이사항을 `data/<tag>/manifest.csv` 에 적는다.

### 3. 매크로 예매

`--runs` 는 계정을 바꾸므로 쓰지 않고, 같은 `--user` 로 명령을 반복한다 (한 번 실행 = 예매 시도 1건).

```powershell
cd ..\..\..\demo-ticketing\macro
$out = "..\..\ai\experiments\issue14-real-validation\data\<tag>\macro-results.csv"
1..5 | ForEach-Object { node macro.js --user macro-fast-01   --mode fast   --seats 1 --concert 0 --out $out }
1..5 | ForEach-Object { node macro.js --user macro-jitter-01 --mode jitter --seats 1 --concert 0 --out $out }
1..5 | ForEach-Object { node macro.js --user macro-human-01  --mode human  --seats 1 --concert 0 --out $out }
```

- `--cancel` 은 그 계정의 확정 예매를 전부 취소하므로 기본 조건에서는 붙이지 않는다.
- 사람 계정과 같은 공연·좌석 수·시도 횟수로 맞춘다.

### 4. 내보내기 (스택을 끄기 전에)

```powershell
cd ..\..\ai\experiments\issue14-real-validation
.venv\Scripts\python scripts\export_events.py --tag <tag>
```

`export_summary.json` 에서 그룹별 계정 수와 이벤트 종류(BOOKING_ATTEMPT / BOOKING / CANCEL)를 확인한다.
Outbox 전송 주기가 5초이므로 마지막 예매 후 10초쯤 기다렸다가 실행한다.

### 5. Feature 추출 → 점수 → 비교

```powershell
$now = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")   # 내보낸 직후의 UTC 시각, 기록해 둔다
.venv\Scripts\python scripts\extract_features.py --tag <tag> --calculated-at $now
.venv\Scripts\python scripts\score_model.py --tag <tag>
.venv\Scripts\python scripts\compare_with_dummy.py --tag <tag>
```

### 6. 정리

```powershell
cd ..\..\..\demo-ticketing\e2e
.\Stop-E2E.ps1          # 백엔드 종료 + e2e 컨테이너 삭제 (데이터 삭제됨 — 4번을 먼저 했는지 확인)
```

## 파일럿에서 확인할 것

1. 이벤트 3종이 저장되고 그룹(`human` / `macro_*`)이 맞게 나뉘는지
2. `occurred_at_utc` 가 실제 UTC 시각과 맞는지 (9시간 밀림 없음), 밀리초가 남는지
3. BOOKING metadata 의 `paymentIdentifier`, `ipAddress` 실제 값
4. 한 계정의 Feature 를 손으로 계산한 값과 `real_features.csv` 가 같은지
5. 매크로 한 번 실행에 시도(`attemptId`)가 1건씩 생기는지, 연속 실행 간격이 몇 초인지
6. 예매 도중 CAPTCHA·지연 같은 정책 개입이 없었는지

## 컬럼 읽는 법 (`real_features.csv`)

| 컬럼 | 의미 |
|---|---|
| `booking_count` … `ip_booking_density` (6개) | BTS `FeatureAggregationService.calculate` 와 같은 식. 모델 입력 |
| `avg_booking_interval`, `booking_interval_variance` | **30일 전체** 시도 간격 기준 (BTS 방식) |
| `session_interval_mean`, `session_interval_variance`, `session_interval_cv` | **세션 내** 간격 기준. 실험용 가정이며 모델 입력이 아님 |
| `variance_degenerate` | 시도 2건이라 BTS 식의 분산이 0으로 나온 경우 |
| `has_null_model_input` | 모델 입력 6개 중 결측이 있는 경우 |
| `card_env_distorted`, `ip_env_distorted` | 데모 환경 때문에 카드·IP 값이 왜곡된 경우 |

세션 기준은 "연속 시도 시작 간격 30분 이하"가 기본이며 제안값이다 (`--session-gap-minutes`).

## 해석할 때 주의

- 사람 참가자가 1명이므로 사람 계정들은 독립 표본이 아니라 한 사람의 반복 측정이다.
- 카드·IP 는 모든 계정이 같은 결제 식별자·같은 IP 를 써서 계정 수만큼 커진다. 이 값이 그대로 들어간 점수
  (`anomaly_score`)는 그룹 구분 없이 높게 나올 수 있어, 카드·IP 를 1로 둔 가정 점수
  (`anomaly_score_card_ip_1`)를 함께 본다.
- 결측 Feature 가 있어도 현재 scikit-learn 버전에서는 점수가 계산된다. 다만 모델에 결측 대체 단계가 있는 것은
  아니므로, `has_null_model_input` 이 참인 계정의 점수는 따로 구분해서 본다.

## 스크립트 점검

```powershell
.venv\Scripts\python scripts\test_extract_features.py     # BTS 단위 테스트 사례와 같은 값이 나오는지
.venv\Scripts\python scripts\make_selftest_events.py      # 가짜 이벤트로 전체 흐름 점검 (--tag _selftest, 결과는 의미 없음)
```

## 예매 횟수·모드별 실험 (누적 3/10/20회)

- 매크로는 중간에 멈추지 않고 끝까지 돌린다 (멈추면 그 대기 시간이 시도 간격에 섞인다).
- 누적 N회 시점 결과는 사후에 잘라서 만든다:
  `.venv\Scripts\python scripts\make_snapshots.py --source-tag <tag> --account <계정> --counts 3 10 20`
  → `data/<tag>__<계정>__nNN/` 태그가 생기고, 이어서 실행할 추출·점수·비교 명령이 출력된다.
- 시도 간격에 브라우저 실행 시간이 얼마나 들어 있는지는 계정별로 기록한다:
  `.venv\Scripts\python scripts\macro_gap_breakdown.py --tag <tag> --results data\<tag>\macro-results-<계정>.csv`
- 카드·IP 공유 수는 계정이 늘수록 커지므로, 모드·횟수 비교는 `anomaly_score_card_ip_1` 로 한다 (실제 값 점수도 함께 기록).
