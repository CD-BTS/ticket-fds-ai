"""이벤트 생성의 임시 가정을 하나씩 바꿨을 때 Isolation Forest 평가가 얼마나 달라지는지 본다 (파일로 저장하는 데이터셋은 없음).

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\assumption_sensitivity.py

기본 설정(config/scenarios.json)에서 실측 근거가 없는 가정 하나만 바꿔 같은 seed 로 이벤트를 다시 만들고,
같은 추출기·같은 분할·같은 모델 설정(기준 구성)으로 평가한다. 출력: output/model_comparison/assumption_sensitivity.csv
"""
import copy
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP.parent / "issue16-booking-interval" / "scripts"))
from compare_if_models import BEST_CONTAMINATION, BEST_FEATURES, SPLIT_SEED, TEST_SIZE, fit_and_score, metrics  # noqa: E402
from extract_dataset import build_dataset  # noqa: E402
from generate_events import generate, load_config  # noqa: E402

HUMAN = ["normal", "fast_normal", "shared_ip", "frequent_cancel"]
SHOWN = ["normal", "shared_ip", "frequent_cancel", "fast", "jitter", "multi_account", "hold_cancel"]


def no_expired_wait(config):
    for name in HUMAN:
        outcomes = config["scenarios"][name]["outcomes"]
        outcomes["complete"] += outcomes["hold_expired_wait"]
        outcomes["hold_expired_wait"] = 0.0


def hold_cancel_always_pays(config):
    config["scenarios"]["hold_cancel"]["outcomes"]["complete"] = 1.0


def hold_cancel_rarely_pays(config):
    config["scenarios"]["hold_cancel"]["outcomes"]["complete"] = {"dist": "uniform", "low": 0.05, "high": 0.3}


def steadier_humans(config):
    for name in HUMAN:
        config["scenarios"][name]["within_sigma"]["value"] /= 2


def all_attempts_complete(config):
    for name in HUMAN + ["multi_account"]:
        outcomes = config["scenarios"][name]["outcomes"]
        for key in ("abandon_seat", "abandon_payment", "hold_expired_wait"):
            outcomes[key] = 0.0
        outcomes["complete"] = 1.0


VARIANTS = [
    ("기본 설정", None),
    ("정상 사용자: 선점 만료까지 기다리는 시도 없음 (3~4% → 0)", no_expired_wait),
    ("정상 사용자: 시도 사이 시간 변동 절반 (within_sigma ÷ 2)", steadier_humans),
    ("정상·다중 계정: 모든 시도가 예매 완료 (이탈 없음)", all_attempts_complete),
    ("선점 후 취소: 항상 결제 후 취소 (결제 비율 0.4~0.9 → 1.0)", hold_cancel_always_pays),
    ("선점 후 취소: 대부분 미결제 (결제 비율 0.4~0.9 → 0.05~0.3)", hold_cancel_rarely_pays),
]


def main():
    base = load_config()
    rows = []
    for label, change in VARIANTS:
        config = copy.deepcopy(base)
        if change:
            change(config)
        events, accounts, _ = generate(config, "single_session", base["seed"])
        dataset, full, _ = build_dataset(events, accounts, config)
        train_idx, test_idx = train_test_split(dataset.index, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=dataset["is_anomaly"])
        pred, _, normalized = fit_and_score(dataset, train_idx, test_idx, BEST_FEATURES, BEST_CONTAMINATION)
        test = dataset.loc[test_idx]
        hold = full[full["scenario"] == "hold_cancel"]
        row = {"variant": label, **{k: v for k, v in metrics(test["is_anomaly"].to_numpy(), pred, normalized).items()
                                    if k in ("precision", "recall", "f1", "fpr", "roc_auc", "TP", "FP", "FN")},
               "rows_with_null_model_input": int(dataset[BEST_FEATURES].isna().any(axis=1).sum()),
               "hold_cancel_cancel_rate_null": int(hold["cancel_rate"].isna().sum()),
               "hold_cancel_cancel_rate_median": hold["cancel_rate"].median(),
               "normal_variance_median": dataset.loc[dataset["scenario"] == "normal", "booking_interval_variance"].median(),
               "normal_variance_q95": dataset.loc[dataset["scenario"] == "normal", "booking_interval_variance"].quantile(0.95)}
        for name in SHOWN:
            row[f"flag_rate_{name}"] = pred[(test["scenario"] == name).to_numpy()].mean()
        rows.append(row)
        print(f"{label} 완료", flush=True)
    table = pd.DataFrame(rows)
    out = EXP / "output" / "model_comparison" / "assumption_sensitivity.csv"
    table.to_csv(out, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(table.round(3).T.to_string())
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
