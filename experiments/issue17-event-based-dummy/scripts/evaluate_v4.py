"""v4 에서 Isolation Forest 를 평가한다. 생성 규칙(config/scenarios_v4.json)과 아래 평가 조건은 결과를 보기 전에 고정했다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\evaluate_v4.py

이 실험은 실제 사용자에 대한 탐지 성능 검증이 아니다. 실측 일부를 반영한 합성 데이터에서, 생성 가정이 달라질 때 결과가 얼마나 달라지는지를 보는 강건성 평가다.
D1·D2·D3 (정상 사용자 사이의 속도 분포에 대한 가정) 각각의 결과를 따로 낸다. 평균을 내어 합치지 않는다.

고정한 조건
  Feature        6개 (종류를 바꾸지 않는다. Ablation 없음). 간격 2개의 계산 방식만 extract_v4 의 후보로 바꿔 본다
  모델           IsolationForest(n_estimators=100, contamination="auto"), 결측 중앙값 대체 + log1p  (REPORT_V2_TUNING.md 의 전처리. 추가 튜닝 없음)
  분할           test 300 (is_anomaly 층화) / train 490 / validation 210 (scenario 층화). 전처리는 train 에만 fit
  임계값         validation 정상 계정의 오탐률이 5% 이하가 되는 가장 낮은 점수. test 에 그대로 적용
  학습 데이터    혼합 (기본) / Credit 규칙 통과 계정 / 정상만 (레이블 사용, 상한)
  seed           모델 seed, 분할 seed, 생성 seed

실험
  E1 기준 데이터셋 (생성 seed 42, 분할 seed 42) × 모델 seed 0~19 × 정의 18개 × 학습 3가지
  E2 생성 seed 0~9 × 모델 seed 0~4 : 정의 18개 × 혼합, 경계 3개(평균·모분산) × 나머지 학습
  E3 분할 seed 0~19 (기준 데이터셋, 모델 seed 42, 경계 3개, 혼합)
  E4 민감도 조건 (sweep_v4.py) × 생성 seed 0~9 × 모델 seed 0~2 : 현재 BTS 정의와 활동 세션, 혼합
  E5 오탐·미탐이 어떤 계정에서 나오는지 (E2 의 test 예측, 현재 BTS 정의와 활동 세션)
  E6 실측한 1명의 8계정이 어떻게 판정되는지 (사례. 비율로 해석하지 않는다. 카드·IP 는 측정 환경의 값이라 1 로 둔다)

출력: output/v4/model_evaluation/
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tune_if_v2 import EXP, MODEL_SEED, N_ESTIMATORS, SCENARIOS, SPLIT_SEED, F, Scorer, evaluate, flag_rates, sha256  # noqa: E402
from tune_if_v2_fpr import TARGET_FPR, fpr_threshold  # noqa: E402
from evaluate_v3 import all_rows, credit_pass, normal_only, splits  # noqa: E402
from extract_v4 import BOUNDARIES, INTERVAL, REPRESENTATIVES, VARIANCE, definition_columns  # noqa: E402
from generate_events_v4 import CONFIG_PATH, DISTS  # noqa: E402
from analyze_v4 import PRIVATE, load_main  # noqa: E402
from sweep_v4 import OUT as SENS, SEEDS as GENERATION_SEEDS, run_path  # noqa: E402

OUT = EXP / "output" / "v4" / "model_evaluation"
TUNED = (1, "median", True)
MODEL_SEEDS, SPLIT_SEEDS, GEN_MODEL_SEEDS, CONDITION_MODEL_SEEDS = list(range(20)), list(range(20)), list(range(5)), list(range(3))
METRICS = ["precision", "recall", "f1", "fpr", "roc_auc"]
RATES = [f"flag_rate_{s}" for s in SCENARIOS]
MACRO = ["fast", "jitter", "multi_account"]
DEFINITIONS = [(b, r, m) for b in BOUNDARIES for r in REPRESENTATIVES for m in (False, True)]
BTS, ACTIVITY, CONCERT = ("d30", "mean_var", False), ("activity", "mean_var", False), ("concert", "mean_var", False)
TRAININGS = [("혼합", all_rows), ("저위험: Credit 규칙 통과", credit_pass), ("정상만 (레이블 사용, 상한)", normal_only)]


def definition_name(definition):
    boundary, representative, min5 = definition
    return f"{BOUNDARIES[boundary]} · {REPRESENTATIVES[representative]}" + (" · 분산은 간격 5개 이상" if min5 else "")


def with_definition(frame, definition):
    a, b = definition_columns(*definition)
    return frame.assign(**{INTERVAL: frame[a], VARIANCE: frame[b]})


def run(frame, train, val, test, selector, seed):
    y, scenario = frame["is_anomaly"], frame["scenario"]
    fit_idx = selector(frame, train, seed)
    scorer = Scorer(frame, fit_idx, TUNED, seed)
    val_scores, test_scores = scorer.score(val), scorer.score(test)
    threshold = fpr_threshold(y.loc[val].to_numpy(), val_scores)
    v, _ = evaluate(y.loc[val].to_numpy(), val_scores, threshold)
    t, pred = evaluate(y.loc[test].to_numpy(), test_scores, threshold)
    st = scenario.loc[test].to_numpy()
    macro = np.isin(st, MACRO)
    profile = frame.loc[test, "tool_profile"].fillna("").to_numpy()
    row = {"n_fit": len(fit_idx), "fit_anomaly_share": float(y.loc[fit_idx].mean()), "val_fpr": v["fpr"], "val_recall": v["recall"],
           **{k: t[k] for k in METRICS + ["TP", "FP", "TN", "FN"]}, **flag_rates(pred, st), "recall_macro": float(pred[macro].mean()),
           **{f"recall_tool_{p}": (float(pred[macro & (profile == p)].mean()) if (macro & (profile == p)).any() else np.nan) for p in ("fast", "jitter", "human")}}
    return row, pred, scorer, threshold


def summarize(runs, by):
    columns = [c for c in runs.columns if c not in by and c not in ("model_seed", "split_seed", "generation_seed") and pd.api.types.is_numeric_dtype(runs[c])]
    grouped = runs.groupby(by, sort=False)[columns]
    return grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_sd")).join(grouped.size().rename("runs")).reset_index()


def trait_bins(frame):
    out = pd.DataFrame(index=frame.index)
    out["정상: 속도 수준 s"] = pd.cut(frame["speed_s"], [0, 0.15, 0.3, 0.6, 1.2, np.inf], labels=["0.10~0.15 (실측한 1명 근처)", "0.15~0.3", "0.3~0.6", "0.6~1.2", "1.2 초과"]).astype(str)
    out["시도 수"] = pd.cut(frame["booking_count"], [0, 1, 2, 4, 9, np.inf], labels=["1", "2", "3~4", "5~9", "10 이상"]).astype(str)
    out["오픈 참여 횟수"] = frame["n_sessions"].astype(int).astype(str)
    out["같은 IP 의 계정 수"] = pd.cut(frame["ip_booking_density"], [0, 1, 2, 4, np.inf], labels=["1", "2", "3~4", "5 이상"]).astype(str)
    out["취소율"] = pd.cut(frame["cancel_rate"].fillna(-1), [-2, -0.5, 0, 0.5, 1], labels=["결측 (완료 예매 없음)", "0", "0 초과 0.5 이하", "0.5 초과"]).astype(str)
    out["간격 Feature"] = np.where(frame[INTERVAL].isna(), "결측 (간격 없음)", "값 있음")
    out["매크로 도구 프로필"] = frame["tool_profile"].fillna("").replace("", "(도구 아님)")
    return out


def for_assumption(dist):
    main = load_main(dist)
    tables = {name: [] for name in ("main", "generation", "split", "condition", "errors", "person")}

    # ---- E1: 기준 데이터셋 ----
    _, train, val, test = splits(main, SPLIT_SEED)
    kept = {}
    for definition in DEFINITIONS:
        frame = with_definition(main, definition)
        for training, selector in TRAININGS:
            for seed in MODEL_SEEDS:
                row, pred, scorer, threshold = run(frame, train, val, test, selector, seed)
                tables["main"].append({"assumption": dist, "definition": definition_name(definition), "training": training, "model_seed": seed, **row})
                if training == "혼합" and definition in (BTS, ACTIVITY):
                    kept.setdefault(definition, []).append((scorer, threshold))

    # ---- E6: 실측한 1명의 계정 (사례) ----
    person_path = PRIVATE / "person_features.csv"
    if person_path.exists():
        person = pd.read_csv(person_path)
        for definition, models in kept.items():
            x = with_definition(person, definition)[F].astype(float)
            x["card_reuse_count"], x["ip_booking_density"] = 1.0, 1.0  # 측정 환경의 값(계정 수)이라 해석할 수 없다. 공유 없음으로 둔다
            flagged = np.mean([scorer.raw(x.to_numpy()) > threshold for scorer, threshold in models], axis=0)
            for account, share, (_, values) in zip(person["account"], flagged, x.iterrows()):
                tables["person"].append({"assumption": dist, "definition": definition_name(definition), "account": account,
                                         "share_of_model_seeds_flagged": float(share), **values.round(3).to_dict()})

    # ---- E3: 분할 seed ----
    for split_seed in SPLIT_SEEDS:
        _, a, b, c = splits(main, split_seed)
        for definition in (BTS, CONCERT, ACTIVITY):
            row, *_ = run(with_definition(main, definition), a, b, c, all_rows, MODEL_SEED)
            tables["split"].append({"assumption": dist, "definition": definition_name(definition), "training": "혼합", "split_seed": split_seed, **row})

    # ---- E2: 생성 seed, E5: 오탐·미탐이 나오는 계정 ----
    for generation_seed in GENERATION_SEEDS:
        base = pd.read_csv(run_path(dist, "base", generation_seed))
        _, train, val, test = splits(base, SPLIT_SEED)
        for definition in DEFINITIONS:
            frame = with_definition(base, definition)
            for training, selector in TRAININGS:
                if training != "혼합" and definition not in (BTS, CONCERT, ACTIVITY):
                    continue
                preds = []
                for seed in GEN_MODEL_SEEDS:
                    row, pred, *_ = run(frame, train, val, test, selector, seed)
                    preds.append(pred)
                    tables["generation"].append({"assumption": dist, "definition": definition_name(definition), "training": training,
                                                 "generation_seed": generation_seed, "model_seed": seed, **row})
                if training == "혼합" and definition in (BTS, ACTIVITY):
                    part = frame.loc[test]
                    bins = trait_bins(part)
                    flagged = np.mean(preds, axis=0)
                    for trait in bins.columns:
                        for label in (0, 1):
                            if (label == 1 and trait.startswith("정상")) or (label == 0 and trait.startswith("매크로")):
                                continue
                            mask = (part["is_anomaly"] == label).to_numpy()
                            groups = pd.DataFrame({"value": bins.loc[mask, trait].to_numpy(), "flagged": flagged[mask], "scenario": part.loc[mask, "scenario"].to_numpy()})
                            for value, group in groups.groupby("value"):
                                tables["errors"].append({"assumption": dist, "definition": definition_name(definition), "label": "정상 (탐지 = 오탐)" if label == 0 else "이상 (미탐지 = 미탐)",
                                                         "trait": trait, "value": value, "generation_seed": generation_seed, "accounts": len(group),
                                                         "flagged_sum": float(group["flagged"].sum())})

    # ---- E4: 민감도 조건 ----
    listed = [c["condition"] for c in json.loads((SENS / "conditions.json").read_text(encoding="utf-8"))["one_at_a_time"]]
    choice_path = SENS / "combination_choice.json"
    if choice_path.exists():
        listed += [c["condition"] for c in json.loads(choice_path.read_text(encoding="utf-8"))["combinations"]]
    for condition in listed:
        for generation_seed in GENERATION_SEEDS:
            path = run_path(dist, condition, generation_seed)
            if not path.exists():
                continue
            data = pd.read_csv(path)
            _, train, val, test = splits(data, SPLIT_SEED)
            for definition in (BTS, ACTIVITY):
                frame = with_definition(data, definition)
                for seed in CONDITION_MODEL_SEEDS:
                    row, *_ = run(frame, train, val, test, all_rows, seed)
                    tables["condition"].append({"assumption": dist, "condition": condition, "definition": definition_name(definition), "training": "혼합",
                                                "generation_seed": generation_seed, "model_seed": seed, **row})
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(len(DISTS)) as pool:
        results = pool.map(for_assumption, DISTS)
    tables = {name: pd.concat([r[name] for r in results], ignore_index=True) for name in results[0]}
    csv = lambda frame, name: frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")

    csv(tables["main"], "main_runs.csv")
    main_summary = summarize(tables["main"], ["assumption", "definition", "training"])
    csv(main_summary, "main_summary.csv")
    csv(tables["generation"], "generation_runs.csv")
    # 생성 seed 마다 모델 seed 평균을 낸 뒤, 생성 seed 사이의 평균·표준편차·최소·최대
    per_seed = tables["generation"].groupby(["assumption", "definition", "training", "generation_seed"], sort=False)[METRICS + RATES + ["recall_macro", "recall_tool_fast", "recall_tool_jitter", "recall_tool_human", "val_fpr"]].mean().reset_index()
    generation_summary = summarize(per_seed, ["assumption", "definition", "training"])
    spread = per_seed.groupby(["assumption", "definition", "training"], sort=False)[METRICS].agg(["min", "max"])
    spread.columns = [f"{a}_{b}" for a, b in spread.columns]
    generation_summary = generation_summary.merge(spread.reset_index(), on=["assumption", "definition", "training"])
    csv(generation_summary, "generation_summary.csv")
    csv(tables["split"], "split_runs.csv")
    csv(summarize(tables["split"], ["assumption", "definition", "training"]), "split_summary.csv")
    csv(tables["condition"], "condition_runs.csv")
    per_condition = tables["condition"].groupby(["assumption", "condition", "definition", "generation_seed"], sort=False)[METRICS + RATES + ["recall_macro"]].mean().reset_index()
    condition_summary = summarize(per_condition, ["assumption", "condition", "definition"])
    csv(condition_summary, "condition_summary.csv")
    errors = tables["errors"].groupby(["assumption", "definition", "label", "trait", "value"], sort=False)[["accounts", "flagged_sum"]].sum().reset_index()
    errors["flag_rate"] = errors["flagged_sum"] / errors["accounts"]
    csv(errors.drop(columns="flagged_sum"), "error_by_trait.csv")
    if len(tables["person"]):
        PRIVATE.mkdir(parents=True, exist_ok=True)
        tables["person"].to_csv(PRIVATE / "person_case.csv", index=False, encoding="utf-8-sig")

    record = {"purpose": "합성 데이터에서의 강건성 평가. 실제 사용자에 대한 탐지 성능 검증이 아니다",
              "features": F, "model": {"n_estimators": N_ESTIMATORS, "contamination": "auto", "preprocessing": "결측 중앙값 대체 + log1p (train 에만 fit)", "tuning": "없음"},
              "threshold": f"validation 정상 계정 오탐률 ≤ {TARGET_FPR}", "split": {"test": 300, "train": 490, "validation": 210, "seed": SPLIT_SEED},
              "model_seeds": MODEL_SEEDS, "split_seeds": SPLIT_SEEDS, "generation_seeds": GENERATION_SEEDS, "generation_model_seeds": GEN_MODEL_SEEDS,
              "condition_model_seeds": CONDITION_MODEL_SEEDS, "definitions": [definition_name(d) for d in DEFINITIONS], "trainings": [t for t, _ in TRAININGS],
              "speed_assumptions": list(DISTS), "generation_config_sha256": sha256(CONFIG_PATH), "sklearn": sklearn.__version__,
              "runs": {name: int(len(frame)) for name, frame in tables.items()}}
    (OUT / "experiment_config.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pd.set_option("display.width", 260, "display.max_columns", 40, "display.max_rows", 500)
    cols = ["assumption", "definition", "training"] + [f"{m}_mean" for m in METRICS]
    print("[E1] 기준 데이터셋 (모델 seed 20개 평균)\n", main_summary[cols + ["f1_sd"]].round(3).to_string(index=False))
    print("\n[E2] 생성 seed 10개\n", generation_summary[cols + ["f1_sd", "f1_min", "f1_max", "recall_min", "recall_max"]].round(3).to_string(index=False))
    rate_cols = ["assumption", "definition", "training"] + [f"{r}_mean" for r in RATES] + ["recall_tool_fast_mean", "recall_tool_jitter_mean", "recall_tool_human_mean"]
    print("\n[E2] 시나리오별 탐지 비율\n", generation_summary.loc[generation_summary["definition"].isin([definition_name(d) for d in (BTS, CONCERT, ACTIVITY)]), rate_cols].round(3).to_string(index=False))
    print("\n[E4] 조건별\n", condition_summary[["assumption", "condition", "definition"] + [f"{m}_mean" for m in METRICS] + ["f1_sd"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
