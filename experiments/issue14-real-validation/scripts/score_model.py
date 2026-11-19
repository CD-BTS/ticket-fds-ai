"""실측 Feature에 기존 Isolation Forest 점수를 매긴다 (실험용, 모델·임계값은 바꾸지 않는다).

실행 (실험 폴더에서):
    .venv/Scripts/python scripts/score_model.py --tag pilot

입력 : data/<tag>/real_features.csv, model_build/models/behavior_model.joblib
출력 : data/<tag>/real_scores.csv

점수는 기존 함수 inference.assess_batch 로 낸다 (서빙 API와 같은 경로).
- anomaly_score              실제 계산된 6개 Feature 그대로 넣은 점수
- anomaly_score_card_ip_1    가정 분석: card_reuse_count, ip_booking_density 만 1로 바꿔 넣은 점수.
                             데모 환경 왜곡(결제 식별자 3개, 같은 IP)을 뺐을 때를 보는 참고값이다.
- model_predict_anomaly      모델 자체 판정 predict() (contamination 0.20 경계)
- ge_*                       기존 임계값별 판정 (번들 0.5, BTS 정책 0.40 / 0.70 / 0.85). 값은 그대로 가져다 쓴다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent.parent
AI = EXP.parent.parent  # 이 폴더는 ai/experiments/<실험명> 이므로 두 단계 위가 AI 레포 루트
sys.path.insert(0, str(AI))

from inference import assess_batch, load_bundle  # noqa: E402
from preprocessing import CAMEL_TO_SNAKE, FEATURE_SPECS  # noqa: E402

BUNDLE_PATH = EXP / "model_build" / "models" / "behavior_model.joblib"
BTS_POLICY_THRESHOLDS = (0.40, 0.70, 0.85)  # ticket-fds-admin PolicyDecisionService.evaluatePolicy


def to_payload(row, overrides=None):
    features = {}
    for camel, snake in CAMEL_TO_SNAKE.items():
        value = (overrides or {}).get(snake, row[snake])
        if pd.isna(value):
            features[camel] = None
        elif FEATURE_SPECS[snake]["integer"]:
            features[camel] = int(value)
        else:
            features[camel] = float(value)
    return {"userId": str(row["external_user_id"]), "calculatedAt": str(row["calculated_at_utc"]) + "+00:00",
            "windowDays": 30, "features": features}


def score(df, bundle, overrides=None):
    results = assess_batch([to_payload(row, overrides) for _, row in df.iterrows()], bundle)
    return ([r["status"] for r in results],
            [np.nan if r["anomalyScore"] is None else r["anomalyScore"] for r in results],
            ["; ".join(r.get("errors") or []) for r in results])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    data_dir = EXP / "data" / args.tag
    df = pd.read_csv(data_dir / "real_features.csv")
    bundle = load_bundle(BUNDLE_PATH)
    columns = bundle["feature_columns"]

    out = df[["user_id", "external_user_id", "group"] + columns
             + ["has_null_model_input", "variance_degenerate", "card_env_distorted", "ip_env_distorted"]].copy()
    out["status"], out["anomaly_score"], out["errors"] = score(df, bundle)
    _, out["anomaly_score_card_ip_1"], _ = score(df, bundle, {"card_reuse_count": 1, "ip_booking_density": 1})

    ok = out["status"] == "OK"
    out["model_predict_anomaly"] = pd.NA
    if ok.any():
        out.loc[ok, "model_predict_anomaly"] = bundle["model"].predict(df.loc[ok, columns]) == -1
    def at_least(threshold):
        # 점수가 없는 계정은 False가 아니라 빈 값으로 둔다
        return (out["anomaly_score"] >= threshold).astype("boolean").where(ok)

    out["ge_bundle_threshold"] = at_least(bundle["anomaly_threshold"])
    for threshold in BTS_POLICY_THRESHOLDS:
        out[f"ge_{threshold:.2f}"] = at_least(threshold)
    out["model_version"] = bundle["model_version"]

    out.to_csv(data_dir / "real_scores.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(out[["external_user_id", "group", "status", "anomaly_score", "anomaly_score_card_ip_1",
               "model_predict_anomaly", "has_null_model_input"]].to_string(index=False))
    if ok.any():
        print("\n그룹별 점수 요약 (계정 수가 적으면 범위만 참고):")
        print(out[ok].groupby("group")[["anomaly_score", "anomaly_score_card_ip_1"]]
              .agg(["count", "min", "median", "max"]).round(3).to_string())
    print(f"\n저장: {data_dir / 'real_scores.csv'}")


if __name__ == "__main__":
    main()
