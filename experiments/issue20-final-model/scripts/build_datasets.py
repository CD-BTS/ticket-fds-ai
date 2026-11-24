"""최종 학습용 데이터와 독립 검증용 데이터를 만든다 (v4 생성기, BTS 추출기 그대로. 생성 규칙은 바꾸지 않는다).

    ..\\..\\.venv\\Scripts\\python scripts\\build_datasets.py

생성 (합성 데이터만. 실측 원본은 쓰지 않는다)
  최종 학습용      생성 seed 2026 의 D1·D2·D3                    3개
  독립 검증용      생성 seed 100~109 의 D1·D2·D3                30개
  (#17 이 쓴 생성 seed 는 42 와 0~9 다. 위 seed 는 처음 쓴다.)
  각 데이터셋은 계정 1,000개의 Feature 표로 저장한다: data/<가정>__seed<k>.csv.gz (git 제외)

최종 학습용 CSV: ml_experiment/data/fds_v4_training_features.csv (git 에 올린다)
  선정된 구성(Mixed): seed 2026 의 계정 1,000개를 시나리오별로 3등분해 D1·D2·D3 에서 하나씩 가져온다. 계정은 한 번만 들어간다.
  계정 단위로 train 490 / validation 210 / test 300 으로 나눠 split 열에 적는다.
  열: row_id, split, source_assumption, scenario, is_anomaly, Feature 6개. 계정 ID 는 넣지 않는다 (합성 ID 도 넣지 않는다).
  옆에 .meta.json 으로 생성 설정·seed·분할·해시·패키지 버전을 남긴다.
"""
import json
import platform
from multiprocessing import Pool

import numpy as np
import pandas as pd
import sklearn

from common import DATA, DISTS, EXP, ISSUE17, OUT, ROOT, SPLIT_SEED, TEST_SIZE, VALIDATION_SIZE, feature_table, mixed_assignment, split_accounts, take
from train_fds_model import FEATURE_DEFINITION_VERSION, FEATURES, MIN_INTERVALS_FOR_VARIANCE, TRAINING_CSV, TRAINING_META, sha256

FINAL_SEED = 2026
INDEPENDENT_SEEDS = list(range(100, 110))
SEEDS_USED_BEFORE = [42] + list(range(10))
DATA_VERSION = f"v4-mixed-seed{FINAL_SEED}"


def dataset_path(dist, seed):
    return DATA / f"{dist}__seed{seed}.csv.gz"


def one(task):
    dist, seed = task
    path = dataset_path(dist, seed)
    if path.exists():
        return None
    from extract_v4 import build_all
    from generate_events_v4 import generate, load_v4
    configs = load_v4()
    events, accounts, attempts, params = generate(dist, None, seed, configs)
    dataset, _, definitions, _, _ = build_all(events, accounts, configs[0])
    frame = dataset.merge(definitions, on="user_id", validate="one_to_one").merge(accounts[["user_id", "tool_profile"]], on="user_id", validate="one_to_one")
    feature_table(frame).to_csv(path, index=False, encoding="utf-8", lineterminator="\n", compression="gzip")
    return {"assumption": dist, "seed": seed, "accounts": int(len(accounts)), "events": int(len(events)), "attempts": int(len(attempts)), "params": params}


def main():
    assert not set([FINAL_SEED] + INDEPENDENT_SEEDS) & set(SEEDS_USED_BEFORE)
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [(dist, seed) for seed in [FINAL_SEED] + INDEPENDENT_SEEDS for dist in DISTS]
    with Pool(6) as pool:
        generated = [row for row in pool.imap_unordered(one, tasks) if row is not None]
    log_path = OUT / "generated_datasets.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    log = sorted({(r["assumption"], r["seed"]): r for r in log + generated}.values(), key=lambda r: (r["seed"], r["assumption"]))
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- 최종 학습용 CSV (Mixed) ----
    tables = {dist: pd.read_csv(dataset_path(dist, FINAL_SEED), keep_default_na=True) for dist in DISTS}
    base = tables["D1"]
    assert all(tables[d][["user_id", "scenario", "is_anomaly"]].equals(base[["user_id", "scenario", "is_anomaly"]]) for d in DISTS)
    anomaly = (base["is_anomaly"] == 1).to_numpy()
    assert all(tables[d].loc[anomaly, FEATURES].equals(base.loc[anomaly, FEATURES]) for d in DISTS)  # 이상 계정은 세 데이터에서 같다
    train, validation, test = split_accounts(base)
    assignment = mixed_assignment(base, FINAL_SEED)
    mixed = take(tables, np.arange(len(base)), assignment)
    assert mixed["user_id"].is_unique and len(mixed) == len(base)
    split = np.full(len(base), "test", dtype=object)
    split[train], split[validation] = "train", "validation"
    out = pd.DataFrame({"row_id": np.arange(len(mixed)), "split": split, "source_assumption": mixed["source_assumption"].to_numpy(),
                        "scenario": mixed["scenario"].to_numpy(), "is_anomaly": mixed["is_anomaly"].to_numpy()})
    for column in FEATURES:
        out[column] = mixed[column].to_numpy()
    for column in ("card_reuse_count", "ip_booking_density", "booking_count"):
        out[column] = out[column].astype("int64")
    variance_present = out["booking_interval_variance"].notna()
    assert (out.loc[variance_present, "booking_count"] >= MIN_INTERVALS_FOR_VARIANCE + 1).all()  # 분산이 있으면 간격 5개 이상 = 시도 6건 이상
    assert (out.loc[~variance_present, "booking_count"] <= MIN_INTERVALS_FOR_VARIANCE).all()
    TRAINING_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(TRAINING_CSV, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g")

    config = json.loads((ISSUE17 / "config" / "scenarios_v4.json").read_text(encoding="utf-8"))
    by_split = out.groupby("split")
    meta = {
        "data_version": DATA_VERSION, "file": TRAINING_CSV.name, "sha256": sha256(TRAINING_CSV), "rows": int(len(out)),
        "content": "합성 데이터만. 실측 원본, 계정 ID, 개인정보 없음 (row_id 는 행 번호)",
        "feature_definition_version": FEATURE_DEFINITION_VERSION, "features_in_model_order": FEATURES,
        "generation": {
            "generator": "experiments/issue17-event-based-dummy/scripts/generate_events_v4.py (수정 없음)",
            "extractor": "experiments/issue14-real-validation/scripts/extract_features.py (BTS FeatureAggregationService 재현) + extract_v4.interval_definitions",
            "config": "experiments/issue17-event-based-dummy/config/scenarios_v4.json", "config_sha256": sha256(ISSUE17 / "config" / "scenarios_v4.json"),
            "base_config_sha256": sha256(ISSUE17 / "config" / "scenarios_v3.json"), "params": {k: v for k, v in config["baseline"].items() if k != "basis"},
            "generation_seed": FINAL_SEED, "assumptions": list(DISTS),
            "composition": "Mixed: 계정을 시나리오별로 3등분해 D1·D2·D3 에서 하나씩. 한 계정은 한 번만 들어간다",
            "mixed_assignment_seed": FINAL_SEED, "accounts_by_source": out["source_assumption"].value_counts().sort_index().to_dict(),
        },
        "split": {"unit": "계정", "seed": SPLIT_SEED, "test_size": TEST_SIZE, "validation_size_of_rest": VALIDATION_SIZE,
                  "stratify": "test 는 is_anomaly, train/validation 은 scenario", "rows": by_split.size().to_dict(),
                  "anomaly_share": by_split["is_anomaly"].mean().round(6).to_dict(),
                  "train_by_source": out[out["split"] == "train"]["source_assumption"].value_counts().sort_index().to_dict()},
        "null_share": out[FEATURES].isna().mean().round(4).to_dict(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "seeds_never_used_before": {"final_training": FINAL_SEED, "independent_validation": INDEPENDENT_SEEDS, "used_in_issue17": SEEDS_USED_BEFORE},
    }
    TRAINING_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"datasets_on_disk": len(list(DATA.glob("*.csv.gz"))), "generated_now": len(generated), **{k: meta[k] for k in ("data_version", "sha256", "rows", "split", "null_share")}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
