"""Issue #20 실험 스크립트의 공통 부분: Feature 표 만들기, 계정 분할, Mixed 구성, 지표.

모델·임계값·정규화는 저장소 루트의 train_fds_model.py (서빙 번들을 만드는 코드)를 그대로 쓴다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
ISSUE17 = EXP.parent / "issue17-event-based-dummy"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ISSUE17 / "scripts"))

from train_fds_model import FEATURES, MIN_INTERVALS_FOR_VARIANCE  # noqa: E402

DISTS = ("D1", "D2", "D3")
SCENARIOS = ["normal", "fast_normal", "shared_ip", "frequent_cancel", "fast", "jitter", "multi_account", "hold_cancel"]
NORMAL, ANOMALY = SCENARIOS[:4], SCENARIOS[4:]
TEST_SIZE, VALIDATION_SIZE, SPLIT_SEED = 0.3, 0.3, 42  # #17 과 같은 분할: test 300 / train 490 / validation 210
OUT = EXP / "output"
DATA = EXP / "data"  # 생성한 Feature 표 (저장소의 data/ 규칙으로 git 제외)


def feature_table(frame):
    """#17 의 계정 1행 표(BTS 추출기 출력 + 간격 정의별 값)에서 Feature 정의 v2 의 6개 Feature 를 뽑는다.

    avg_booking_interval      30일 창의 연속 시도 시작 간격 평균 (초). 간격이 없으면 null
    booking_interval_variance 같은 간격의 모분산 (초²). 간격이 5개 미만이면 null   ← v2 에서 바뀐 부분
    나머지 4개는 BTS 추출기의 값 그대로다.
    """
    assert MIN_INTERVALS_FOR_VARIANCE == 5
    out = frame[["user_id", "scenario", "is_anomaly"]].copy()
    out["avg_booking_interval"] = frame["avg_booking_interval__d30__mean_var"].to_numpy(dtype=float)
    out["booking_interval_variance"] = frame["booking_interval_variance__d30__mean_var__min5"].to_numpy(dtype=float)
    for column in ("cancel_rate", "card_reuse_count", "ip_booking_density", "booking_count"):
        out[column] = frame[column].to_numpy()
    out["tool_profile"] = frame["tool_profile"].fillna("").to_numpy() if "tool_profile" in frame else ""
    return out[["user_id", "scenario", "is_anomaly", "tool_profile"] + FEATURES]


def split_accounts(table, seed=SPLIT_SEED):
    """계정 위치(행 번호)의 train / validation / test. 계정 단위로 한 번만 나눈다."""
    index = np.arange(len(table))
    development, test = train_test_split(index, test_size=TEST_SIZE, random_state=seed, stratify=table["is_anomaly"])
    train, validation = train_test_split(development, test_size=VALIDATION_SIZE, random_state=seed, stratify=table["scenario"].iloc[development])
    return np.sort(train), np.sort(validation), np.sort(test)


def mixed_assignment(table, seed):
    """계정마다 D1·D2·D3 중 하나. 시나리오 안에서 섞은 뒤 차례로 배정해 세 분포가 같은 비중이 되게 한다."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 20]))
    assigned = np.empty(len(table), dtype=object)
    for scenario in SCENARIOS:
        rows = np.flatnonzero((table["scenario"] == scenario).to_numpy())
        rows = rng.permutation(rows)
        start = int(rng.integers(len(DISTS)))
        for k, row in enumerate(rows):
            assigned[row] = DISTS[(start + k) % len(DISTS)]
    return assigned


def take(tables, rows, assignment):
    """rows 의 계정을 assignment(분포 이름 하나 또는 계정별 배열)에 따라 가져온다. 한 계정은 한 번만 들어간다."""
    if isinstance(assignment, str):
        return tables[assignment].iloc[rows].assign(source_assumption=assignment)
    parts = [tables[dist].iloc[rows[assignment[rows] == dist]].assign(source_assumption=dist) for dist in DISTS]
    return pd.concat(parts).sort_index()


def metrics(frame, raw, threshold):
    """Precision, Recall, F1, 오탐률, ROC-AUC, 정상 유형별 오탐률, 매크로 유형별 Recall."""
    y = frame["is_anomaly"].to_numpy() == 1
    flagged = np.asarray(raw) >= threshold
    tp, fp, fn, tn = (flagged & y).sum(), (flagged & ~y).sum(), (~flagged & y).sum(), (~flagged & ~y).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    row = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
           "fpr": fp / (fp + tn) if fp + tn else 0.0, "roc_auc": roc_auc_score(y, raw), "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}
    scenario = frame["scenario"].to_numpy()
    for name in NORMAL:
        row[f"fpr_{name}"] = float(flagged[scenario == name].mean()) if (scenario == name).any() else np.nan
    for name in ANOMALY:
        row[f"recall_{name}"] = float(flagged[scenario == name].mean()) if (scenario == name).any() else np.nan
    profile, macro = frame["tool_profile"].fillna("").to_numpy(), np.isin(scenario, ["fast", "jitter", "multi_account"])
    for name in ("fast", "jitter", "human"):
        mask = macro & (profile == name)
        row[f"recall_tool_{name}"] = float(flagged[mask].mean()) if mask.any() else np.nan
    return row
