# 정상 사용자 예매 실측 (Issue #17)

위치: `ai/experiments/issue17-event-based-dummy/measurement/`. 아래 명령은 별도 표시가 없으면 `ai\experiments\issue17-event-based-dummy` 에서 실행한다.

[`REPORT_INTERVAL_VALIDATION.md`](../REPORT_INTERVAL_VALIDATION.md) 에서 확인한 빈칸을 메우기 위한 수집이다: 기존 #14 실측은 사람이 1명이고, 같은 공연 안의 연속 시도 간격이 하나도 없다.
이 폴더에는 **수집 환경 점검·내보내기·분석 스크립트와 실험 절차**만 있다. 실제 예매는 사람이 직접 한다.

- 기존 코드(Demo, BTS, AI 서빙, #14·#16 스크립트, v1~v3 데이터와 생성기)는 수정하지 않는다. 스크립트는 기존 코드를 import 하거나 DB 를 SELECT 로 읽기만 한다.
- v4 생성, 모델 학습은 하지 않는다.
- 실측 원본은 `measurement/data/` 에만 저장하고 git 에 올리지 않는다 (`measurement/.gitignore`). #14 (`issue14-real-validation/data/`), #16 (`issue16-booking-interval/runs/`) 의 파일은 읽기만 한다.

## 폴더

| 경로 | 내용 | git |
|---|---|---|
| `scripts/collect.py` | `check`: 수집 전 점검 / `export`: BTS 의 `behavior_event` 를 CSV 로 내보내기 (SELECT 만) | 올림 |
| `scripts/analyze.py` | 시도 수, 단계별 소요 시간, 연속 시도 간격, 6개 Feature, 기존 실측·v3 와의 비교 | 올림 |
| `scripts/make_selftest_events.py` | 분석 스크립트 점검용 가짜 이벤트 (v3 합성 이벤트의 이름만 바꾼 것) | 올림 |
| `manifest_template.csv` | 수집 기록 양식. 내보낼 때 `data/<run-id>/manifest.csv` 로 복사된다 | 올림 |
| `data/<run-id>/` | 내보낸 이벤트(`raw_events.csv`), `users.csv`, `manifest.csv`, 환경 기록 | **제외** |
| `data/_analysis/<이름>/` | 시도 1행·간격 1행·계정 1행 표 | **제외** |
| `output/<이름>/` | 집계표와 그래프 (참가자·실험 유형별 통계) | 내용을 본 뒤 결정 |

---

## 1. 기존 환경에서 확인한 것

### 1.1 이벤트가 저장되는 경로

```
브라우저(프론트 5173) → Demo 백엔드 → Demo DB (booking_attempt_events, bookings, outbox_events)
                                    → 5초마다 Outbox 전송 → BTS POST /api/v1/events → BTS DB behavior_event
```

- BTS 로 가는 이벤트는 `BOOKING_ATTEMPT`(예매 시도의 단계 전이), `BOOKING`(예매 완료), `CANCEL`(취소), `CHECK_IN` 이다.
- `occurredAt` 은 서버가 기록한 시각이다 (기존 실측 1,564건에서 `recordedAt` 과 1ms 이내로 같음).
- Demo 백엔드를 그냥 띄우면 `BTS_ENABLED=false` 라 BTS 로 아무것도 가지 않는다. **격리 E2E 환경(`demo-ticketing/e2e`)으로 띄워야 한다.** #14·#16 도 이 방식이었다.
- 프론트의 `.env` 가 `VITE_USE_MOCK_TICKETING=false` 여야 한다 (현재 값 false). true 면 예매가 브라우저 안에서만 처리된다.

### 1.2 예매 중 기록되는 이벤트

| 행동 | 기록되는 이벤트 | 비고 |
|---|---|---|
| "예매하기" 클릭 | `STARTED/ENTERED` | 시도(attemptId)가 새로 생긴다. 이 시각이 "시도의 시작" 이다 |
| 대기열 통과 | `QUEUE/SUCCEEDED` | 정원 100명이라 바로 통과한다 |
| 좌석 화면 진입 | `SEAT_SELECTION/ENTERED` | |
| "선점 후 결제" → 성공 | `SEAT_HOLD/SUCCEEDED` (holdId) | |
| "선점 후 결제" → 이미 선점된 좌석 | `SEAT_HOLD/REJECTED`, `reasonCode=SEAT_ALREADY_TAKEN` | **같은 시도 안**에 남는다. 다른 좌석을 고르면 그대로 이어진다 |
| 결제 화면 진입 | `PAYMENT/ENTERED` | |
| 결제 완료 | `BOOKING` (공연, 결제 식별자, IP, 티켓) + `COMPLETED/SUCCEEDED` | |
| 좌석 화면의 "예매 시도 취소" | `SEAT_SELECTION/ABANDONED`, `reasonCode=USER_CANCELLED` | |
| 결제 화면의 "이전으로 (좌석 다시 선택)" | `PAYMENT/ABANDONED`, `reasonCode=USER_CANCELLED` | 선점이 먼저 해제된다 |
| 결제 화면에서 선점 시간이 끝남 | `SEAT_HOLD/EXPIRED`, `reasonCode=HOLD_EXPIRED` | 서버 작업이 5초 주기라 최대 5초 늦게 기록된다 |
| 마이페이지의 "예매 취소" | `CANCEL` | 좌석이 다시 풀린다 |

### 1.3 기록되지 않는 것 (만들어 넣지 않는다)

| 요청 항목 | 기록 여부 | 대신 쓸 수 있는 것 |
|---|---|---|
| 익명화된 사용자 ID | 가능 | 계정 ID 규칙(3절)으로 참가자 번호를 넣는다 |
| 실험 유형 (A/B/C) | 가능 | 계정 ID 에 넣는다 |
| 공연 ID | 가능 | `metadata.concertId` |
| **세션 식별 정보** | **없음** | 이벤트에 세션·브라우저·탭 식별자가 없다. 세션은 시각과 공연으로 **추정**한다 (분석에서 두 가지 정의를 같이 계산) |
| 예매 시도 시작 시각 | 가능 | `STARTED` 의 `occurredAt` |
| 좌석 선택 시작 시각 | 가능 | `SEAT_SELECTION/ENTERED` |
| 좌석 선택 완료 시각 | 선점 요청 시각으로만 | 좌석을 **클릭한** 시각은 서버에 남지 않는다. `SEAT_HOLD` 이벤트(선점 요청 처리 시각)를 쓴다 |
| 결제 단계 진입·완료 시각 | 가능 | `PAYMENT/ENTERED`, `BOOKING` |
| 예매 성공·이탈·만료·취소 | 가능 | 위 표 |
| **결제 실패** | **없음** | UI 에서 결제를 실패시킬 방법이 없고(카드 번호는 항상 유효한 식별자로 바뀐다), 실패해도 이벤트가 남지 않는다 |
| 실패 사유 | 일부 | `reasonCode`: `USER_CANCELLED`, `HOLD_EXPIRED`, `SEAT_ALREADY_TAKEN`, `QUEUE_NOT_ADMITTED` |
| 탭 닫기, 브라우저 뒤로 가기 | **없음** | 이벤트가 없다. 그 시도는 "종료 이벤트 없음" 으로만 보인다 |
| 같은 세션 안의 연속 시도 간격 | 계산 가능 | 시도 시작 시각의 차이 |

프론트가 브라우저 안에서 계산하는 값(`step_stay_time`, `recent_booking_count` 등)은 localStorage 에만 있고 서버로 가지 않는다.

### 1.4 기존 6개 Feature 를 계산할 수 있는가

| Feature | 계산 | 실측 환경에서의 주의 |
|---|---|---|
| `booking_count` | 가능 | |
| `avg_booking_interval`, `booking_interval_variance` | 가능 (시도 2건 이상) | |
| `cancel_rate` | 가능 (완료 예매 1건 이상) | 취소를 하지 않으면 0 |
| `card_reuse_count` | 계산은 되지만 **왜곡** | 데모의 결제 식별자가 3개뿐이라 계정 수만큼 커진다 |
| `ip_booking_density` | 계산은 되지만 **왜곡** | 프록시를 거쳐 모든 계정이 같은 IP 로 기록된다 |

계산은 #14 의 추출기(`extract_features.py`, BTS `FeatureAggregationService` 재현)를 그대로 쓴다. 카드·IP 는 #14 와 같은 이유로 해석하지 않는다.

---

## 2. 서버 실행

Docker Desktop 을 먼저 켠다. PowerShell 창 두 개가 필요하다.

**창 1: DB + BTS + Demo 백엔드**

```powershell
cd <작업 폴더>\demo-ticketing\e2e
.\Start-E2E.ps1 -DemoPort 8080
# "준비 완료: BTS http://localhost:18080, Demo http://localhost:8080" 이 나올 때까지 기다린다 (처음에는 몇 분)
```

**창 2: 프론트**

```powershell
cd <작업 폴더>\demo-ticketing\frontend
npm install      # 처음 한 번
npm run dev      # http://localhost:5173
```

- 8080 포트를 다른 프로그램이 쓰고 있으면 `.\Start-E2E.ps1` (Demo 18081)로 띄우고, 프론트를 `$env:DEMO_API_TARGET = "http://localhost:18081"; npm run dev` 로 실행한다.
- **이 환경의 DB 는 메모리(tmpfs)에만 있다.** `Stop-E2E.ps1`, Docker 재시작, PC 재부팅을 하면 데이터가 사라진다. 5절의 내보내기를 먼저 한다.
- 기존 개발용 DB(`fairticket-mysql`, `demo-ticketing-mysql`, `demo-ticketing-redis` 와 그 볼륨)는 쓰지 않고 건드리지 않는다. E2E 환경은 별도 컨테이너(`cdbts-e2e-*`)를 새로 만든다.
- 공연과 좌석은 컨테이너가 만들어질 때 자동으로 들어간다: 공연 3개(`c1`, `c2`, `c3`), 공연마다 700석. 따로 준비할 것이 없다.

**점검** (예매를 시작하기 전)

```powershell
cd <작업 폴더>\ai\experiments\issue17-event-based-dummy
$py = "..\issue14-real-validation\.venv\Scripts\python"
& $py measurement\scripts\collect.py check
```

프론트와 Demo API 가 모두 200 이고, `VITE_USE_MOCK_TICKETING=false` 이면 된다.

---

## 3. 계정

로그인은 ID 만 입력한다 (비밀번호 없음, 처음 입력한 ID 는 자동 등록). ID 가 곧 BTS 의 `external_user_id` 다.

```
u<참가자 2자리>-<실험 a|b|c>-<회차 2자리>        u01-a-01   u01-b-01   u01-c-01   u02-a-01 ...
helper-<2자리>                                  helper-01  (실험 C 에서 좌석을 먼저 잡는 보조 계정. 분석에서 뺀다)
```

- **실험마다 계정을 바꾼다.** Feature 는 계정 단위로 계산되므로 A·B·C 를 한 계정에서 하면 섞인다.
- 같은 실험을 다시 하면 회차를 올린다 (`u01-b-02`). 이미 쓴 계정 ID 는 다시 쓰지 않는다 (`collect.py check` 가 사용된 계정을 보여 준다).
- 참가자 번호와 실제 사람의 **이름**은 이 폴더 어디에도 적지 않는다.
- **계정 ID 의 참가자 번호는 사람을 보장하지 않는다.** 한 사람이 `u01`, `u02`, `u03` 을 모두 썼다면 그것은 1명의 반복 측정이다. 내보낸 뒤 `manifest.csv` 의 `person` 열에 실제로 수행한 사람을 구분하는 값(`p01`, `p02` …)을 적는다. 분석은 이 열을 기준으로 참가자를 센다. 첫 수집(2026-11-24)은 번호 세 개가 모두 같은 사람이었다.
- 로그인 정보는 브라우저 프로필 단위로 저장된다. **한 브라우저 창에는 한 계정만** 로그인된다. 계정을 바꿀 때는 로그아웃 후 다시 로그인하고, 두 계정을 동시에 쓸 때(실험 C3)는 시크릿 창을 하나 더 연다.

---

## 4. 실험 절차

공통

- 좌석 수와 좌석 위치는 참가자가 고른다. 결제는 저장된 테스트 카드 중 하나를 고르거나 15자리 숫자를 직접 입력한다.
- 진행자는 속도에 대해 말하지 않는다 ("빨리", "천천히" 모두).
- 실험을 시작·종료한 시각과 특이사항(실수, 중단, 오류 메시지)을 적어 둔다. 내보낸 뒤 `manifest.csv` 에 옮긴다.
- 참가자 한 명의 순서는 A → B → C 다 (A 는 화면을 처음 보는 상태여야 한다).
- 좌석 화면에 5분 넘게 머물면 대기열 입장 시간이 끝나 선점이 거절된다 (`QUEUE_NOT_ADMITTED`). 그렇게 됐다면 그대로 두고 메모만 남긴다.

### A. 자연스러운 예매 — 계정 `uNN-a-01`

화면을 처음 보는 참가자가 한다. 사전 설명은 "공연 티켓을 예매해 보세요" 까지만 한다.

1. `http://localhost:5173` 에서 `uNN-a-01` 로 로그인.
2. 공연 `c1` 을 골라 예매를 끝까지 한다 (예매하기 → 좌석 선택 → 선점 후 결제 → 결제).
3. 이어서 `c2`, `c3` 을 한 번씩 예매한다.

얻는 것: 익숙하지 않은 상태의 단계별 소요 시간, 다른 공연 사이의 연속 시도 간격.

### B. 숙련된 사용자 예매 — 계정 `uNN-b-01`

A 를 마친 참가자가 한다. "같은 공연을 여러 번 예매해 주세요. 편한 속도로 하시면 됩니다" 라고만 말한다.

1. 로그아웃 후 `uNN-b-01` 로 로그인.
2. 공연 `c1` 을 예매한다.
3. 결제 완료 화면에서 "공연 목록으로" → 다시 `c1` → "예매하기". 같은 탭에서 이어서 한다.
4. 2~3 을 반복해 **같은 공연을 6번** 예매한다 (간격 5개).

얻는 것: **같은 공연 안의 연속 시도 간격** (지금까지 실측 0개), 익숙해진 뒤의 단계별 소요 시간.

### C. 예매 실패 후 재시도 — 계정 `uNN-c-01`, 공연 `c2`

각 경우를 표의 횟수만큼, 위에서부터 순서대로 한다. 실패 뒤에는 **같은 공연을 다시 예매해 끝까지 완료**한다.

| | 실패 경로 | 하는 방법 | 횟수 | 기록되는 이벤트 |
|---|---|---|---|---|
| C1 | 좌석 선택에서 나감 | 좌석 화면에서 "예매 시도 취소" → 다시 `c2` → 예매하기 → 완료 | 2 | `SEAT_SELECTION/ABANDONED` |
| C2 | 결제 화면에서 나감 | 좌석을 선점하고 결제 화면에서 "이전으로 (좌석 다시 선택)" → 다시 예매 → 완료 | 2 | `PAYMENT/ABANDONED` |
| C3 | 좌석 선점 거절 | 아래 절차 | 2 | `SEAT_HOLD/REJECTED` (`SEAT_ALREADY_TAKEN`) |
| C4 | 선점 만료 | 좌석을 선점하고 결제 화면에서 시간이 끝날 때까지 기다림 → 상단 메뉴로 공연 목록 → `c2` → 예매하기 → 완료 | 1 | `SEAT_HOLD/EXPIRED` |

**C3 의 절차** (창 두 개)

1. 참가자 창: `c2` 의 좌석 화면까지 들어가 좌석 배치도를 띄워 둔다. 아직 좌석을 누르지 않는다.
2. 시크릿 창: `helper-01` 로 로그인 → `c2` → 예매하기 → 좌석 하나(예: 앞줄 가운데)를 골라 "선점 후 결제" → 결제 화면에 그대로 둔다.
3. 참가자 창: **같은 좌석**을 골라 "선점 후 결제". "이미 선택된 좌석입니다" 가 나오고 배치도가 새로 고쳐진다.
4. 참가자 창: 다른 좌석을 골라 예매를 완료한다 (같은 시도 안에서 이어진다).
5. 시크릿 창: "이전으로 (좌석 다시 선택)" → "예매 시도 취소" 로 보조 계정의 선점을 푼다.

좌석 배치도는 자동으로 새로 고쳐지지 않으므로 1번을 먼저 해야 거절이 난다.

**C4 에 대해**

- 선점 시간은 기본 5분이다. 그대로 두면 한 번에 5분을 기다린다.
- 줄이고 싶으면 `Start-E2E.ps1` 을 실행하기 **전에** 같은 창에서 `$env:HOLD_DURATION = "PT1M"` 을 설정한다. 스크립트가 이 값을 덮어쓰지 않으므로 Demo 백엔드에 전달될 것으로 보이지만 **실행해서 확인하지는 않았다**. 결제 화면의 남은 시간이 1분으로 표시되는지로 확인한다. 이렇게 띄웠다면 A·B 를 포함한 모든 실험이 그 조건이므로 `manifest.csv` 의 `hold_duration` 에 적는다.
- 만료 뒤 "좌석 다시 선택" 버튼을 누르면 "선점 해제 상태를 확인하지 못했습니다" 가 나올 수 있다 (코드상 예상되는 동작). 그 버튼 대신 상단 메뉴로 공연 목록에 간다.

**하지 않는 것**

- 결제 실패: UI 에서 만들 수 없고 이벤트도 남지 않는다 (1.3절).
- 탭 닫기: 이벤트가 남지 않아 측정할 것이 없다. 실수로 닫았다면 메모만 남긴다.

얻는 것: 실패 유형별 이벤트, **실패 뒤 다시 시작하기까지의 시간**, 선점 거절 뒤 다른 좌석을 고르는 시간.

### 본 수집 전에 한 번 확인할 것 (파일럿)

진행자가 `u99-b-01`, `u99-c-01` 같은 연습 계정으로 먼저 해 본다. 코드를 읽고 판단한 내용이라 실제 화면에서 확인이 필요하다.

1. **같은 탭에서 같은 공연을 다시 예매할 수 있는가** (B 의 3번). #14 때는 "요청이 현재 리소스 상태와 충돌합니다"(409)가 나서 새 탭을 열어야 했다. 지금 코드로는 같은 탭에서 될 것으로 보인다. 409 가 나면 새 탭에서 주소를 직접 입력하는 방식으로 바꾸고 `manifest.csv` 에 적는다.
2. C3 에서 거절 메시지가 나오고 같은 화면에서 다른 좌석으로 이어지는가.
3. C4 에서 만료 뒤 상단 메뉴로 나가 다시 예매할 수 있는가.
4. 5절대로 내보내고 6절대로 분석했을 때 연습 계정의 시도 수가 실제로 한 횟수와 같은가.

연습 계정의 데이터는 본 수집과 섞이지 않게 `--run-id pilot-...` 로 따로 내보낸다. 본 수집은 스택을 다시 띄운 뒤에 해도 되고 이어서 해도 된다 (계정 ID 가 다르면 분석에서 구분된다).

---

## 5. 이벤트 내보내기 (스택을 끄기 전에)

마지막 예매 후 10초쯤 기다린다 (Outbox 전송 주기 5초). 스크립트도 이벤트 수가 더 늘지 않을 때까지 기다린다.

```powershell
cd <작업 폴더>\ai\experiments\issue17-event-based-dummy
$py = "..\issue14-real-validation\.venv\Scripts\python"
& $py measurement\scripts\collect.py export --run-id 20261125-u01 --note "참가자 u01, A B C"
```

- `--run-id` 는 날짜와 참가자 번호로 정한다. 이미 있는 이름은 덮어쓰지 않고 멈춘다.
- 저장되는 곳: `measurement\data\<run-id>\raw_events.csv`, `users.csv`, `export_summary.json`, `environment.json`, `manifest.csv`(양식).
- `manifest.csv` 를 열어 계정별 시작·종료 시각, 브라우저, 특이사항을 채운다.
- 한 번 띄운 스택에서 여러 참가자를 이어서 해도 된다. 그때는 마지막에 한 번 내보내면 모든 참가자가 들어간다 (중간중간 내보내 두는 것이 안전하다. 나중 파일이 앞의 내용을 포함한다).

**정상적으로 수집됐는지 확인** (`export_summary.json` 또는 화면 출력)

| 확인 | 기대 |
|---|---|
| `accounts_by_experiment` | 실험한 계정이 A·B·C 에 모두 있다 |
| `events_by_account_and_type` 의 `BOOKING` | 계정별로 완료한 예매 수와 같다 (A 3건, B 6건, C 7건) |
| `events_by_account_and_type` 의 `BOOKING_ATTEMPT` | 완료한 시도 1건에 이벤트 6개(STARTED, QUEUE, SEAT_SELECTION, SEAT_HOLD, PAYMENT, COMPLETED) 정도 |
| `flush_stable` | true |
| `accounts_outside_naming_rule` | 비어 있다 (오타 난 계정 ID 가 없다) |
| `occurred_at_min` / `max` | 실험한 시각(UTC = 한국 시각 − 9시간)과 맞다 |

모두 확인한 뒤에 스택을 끈다.

```powershell
cd <작업 폴더>\demo-ticketing\e2e
.\Stop-E2E.ps1          # 백엔드 종료 + e2e 컨테이너 삭제. 이 환경의 데이터가 사라진다
```

---

## 6. 분석

```powershell
& $py measurement\scripts\analyze.py --run-id 20261125-u01
# 여러 수집을 합쳐서 볼 때
& $py measurement\scripts\analyze.py --run-id 20261125-u01 --run-id 20261126-u02 --name all
```

| 출력 (`measurement\output\<이름>\`) | 내용 |
|---|---|
| `attempt_counts.csv` (git 제외: `data/_analysis/<이름>/`) | 참가자·실험·계정별 시도 수, 결과별 건수, 선점 거절 수, 간격 수 |
| `stage_times_by_experiment.csv`, `stage_times_by_participant.csv` | 단계별 소요 시간의 건수, 평균, 중앙값, 표준편차, 최소, 5·25·75·95%, 최대 |
| `gap_summary.csv` | 연속 시도 간격. 세션 정의 두 가지: **같은 공연 안**(공연+30분), **활동 세션**(30분만) |
| `retry_gap_by_previous_outcome.csv` | 앞 시도의 결과(완료, 좌석 이탈, 결제 이탈, 만료, 선점 거절 여부)별 재시도 간격 |
| `recorded_event_kinds.csv` | 실험 유형별로 실제 기록된 이벤트 종류와 사유 |
| `feature_summary.csv` | 6개 Feature(BTS 계산식)와 세션 기준 간격의 분포, 결측 계정 수 |
| `comparison_with_existing.csv` | 이번 실측 / #14 사람 / #16 매크로(연속 실행) / v3 합성 |
| `figures/gaps.png`, `figures/stage_times.png` | |
| `analysis_summary.json` | 계산 조건, 추출기 일치 확인, 측정할 수 없는 항목 |

계정·시도 단위 표(`attempts.csv`, `gaps.csv`, `features.csv`)는 `measurement\data\_analysis\<이름>\` 에 저장된다 (git 제외).

- **Feature 는 기존 방식과 같다.** #14 의 추출기를 import 해서 계산하고, 시도 표에서 직접 계산한 시도 수·평균 간격과 일치하는지 매번 확인한다 (`analysis_summary.json` 의 `consistency_checks`).
- **측정할 수 없는 값은 결측이다.** 해당 단계 이벤트가 없는 시도는 그 시각과 소요 시간이 빈 칸이고, 통계의 건수(n)에서 빠진다. 시도 2건 미만이면 간격 Feature 가 결측이다.
- #14·#16 의 실측 원본이 없는 PC 에서는 기존 실측과의 비교만 건너뛴다.

**스크립트 점검** (실측 없이 분석 스크립트가 도는지)

```powershell
& $py measurement\scripts\make_selftest_events.py     # v3 합성 이벤트의 이름만 바꾼 가짜 이벤트. 결과에 의미 없음
& $py measurement\scripts\analyze.py --run-id _selftest
```

---

## 7. 해석할 때 주의

- 대기열이 비어 있고 좌석 경쟁이 없는 로컬 환경이다. 실제 티켓 오픈의 시간과 다르다.
- 실험 B 는 "같은 공연을 여러 번 예매" 하라고 요청한 상황이다. 실제 사용자가 그렇게 하는 빈도는 이 수집으로 알 수 없다.
- 실험 C 는 실패를 일부러 만든다. 참가자가 실패를 예상하고 있으므로, 실패 뒤의 대기 시간은 실제보다 짧을 수 있다. C3 은 진행자와 호흡을 맞춰야 해서 좌석 선택 시간이 길게 나온다.
- 세션 식별자가 없어 "세션" 은 규칙으로 정한 것이다. 두 정의(같은 공연 안 / 활동 세션)의 결과를 함께 본다.
- 카드·IP Feature 는 환경 때문에 왜곡된다.
- 참가자 수가 적으면 사람 사이의 차이를 추정할 수 없다. 참가자별 표(`stage_times_by_participant.csv`)를 같이 본다.
