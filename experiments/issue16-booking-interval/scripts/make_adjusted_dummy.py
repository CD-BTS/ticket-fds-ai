"""기존 더미데이터를 복사해, 매크로 계정을 실측 유형으로 나누고 예매 간격 Feature 두 개를 실측 간격으로 다시 만든다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\make_adjusted_dummy.py
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\make_adjusted_dummy.py --version v2 --run-id <새 run-id>

하는 일
  1. 원본 더미(`ml_experiment/data/fairticket_synthetic_features.csv`)를 읽는다. 원본과 생성 코드는 건드리지 않는다.
  2. 실측 실행(`runs/<run-id>/booking_intervals.csv`)에서 유형(매크로 모드)별 예매 간격을 모은다.
  3. `scenario == macro` 인 행을 `--split` 의 개수대로 유형에 배정하고 (기본 fast 44, jitter 43), `scenario` 값을 그 유형 이름으로 바꾼다.
     배정은 seed 로 섞은 순서의 앞에서부터 차례로 한다. `is_anomaly` 는 바꾸지 않는다.
  4. 행마다 `booking_count - 1` 개의 간격을 **배정된 유형의 실측 간격에서만** 복원추출해
     평균 → `avg_booking_interval`, 모분산 → `booking_interval_variance` 로 넣는다 (BTS Feature 계산식과 같은 정의).
     `booking_count < 2` 인 행은 간격이 없으므로 두 값을 바꾸지 않고 보고만 한다.
  5. 그 밖의 컬럼·행은 그대로 둔 채 `dummy_adjusted/<version>/` 에 새 데이터셋, 통계 CSV, 그래프, REPORT.md 를 쓴다.

방식 (--method)
  typed-split (기본, v1)      : 유형의 실측 간격 전체를 하나로 합쳐 그 안에서 복원추출한다.
  account-hierarchical (v2)  : 더미 계정마다 같은 유형의 실측 계정 하나를 먼저 고르고, 그 계정의 간격에서만 복원추출한다.
                               실측 계정 사이의 평균·분산 차이가 더미 계정에 그대로 남는다. 분포를 따로 가정하지 않는다.
                               간격이 --min-account-intervals 개 미만인 실측 계정은 고르지 않는다 (비교에는 표시한다).

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\make_adjusted_dummy.py --version v2 --method account-hierarchical ^
        --run-id 20261122-continuous --run-id 20261123-remeasure --compare-version v1

모델 학습·평가는 하지 않는다. 이미 있는 버전 폴더는 덮어쓰지 않는다.
--force 를 주면 이 스크립트가 만든 폴더(generation.json 이 있는 폴더)에 한해 내용을 지우고 새로 쓴다.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
REPO_DIR = EXPERIMENT_DIR.parents[1]
ORIGINAL_PATH = REPO_DIR / "ml_experiment" / "data" / "fairticket_synthetic_features.csv"
DATASET_NAME = "fairticket_synthetic_features_adjusted.csv"
METHOD = "typed-split"  # 유형별 배정 + 유형별 실측 간격에서 복원추출 (v1)
METHOD_HIERARCHICAL = "account-hierarchical"  # 유형별 배정 + 실측 계정을 고른 뒤 그 계정의 간격에서 복원추출 (v2)

TARGET_SCENARIO = "macro"
MEAN_COL = "avg_booking_interval"
VAR_COL = "booking_interval_variance"
CHANGED_COLS = ("scenario", MEAN_COL, VAR_COL)
# 원본 생성 코드(generate_dummy_data.py)와 같은 자릿수·하한
MEAN_DECIMALS, VAR_DECIMALS = 2, 4
MEAN_FLOOR, VAR_FLOOR = 0.1, 0.01

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
C_ORIGINAL, C_ADJUSTED, C_REAL = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#1f2328", "#6b7280", "#e5e7eb", "#ffffff"
L_ORIGINAL, L_ADJUSTED = "기존 더미", "수정 더미"
MARKERS = ["o", "s", "D", "P", "X"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_split(text):
    split = {}
    for part in text.split(","):
        name, _, count = part.strip().partition(":")
        if not name or not count.isdigit() or name in split:
            sys.exit(f"--split 형식이 잘못됐습니다: {text!r} (예: fast:44,jitter:43)")
        split[name] = int(count)
    return split


def load_pool(run_ids, types):
    """실측 실행들에서 지정한 유형의 예매 간격을 모은다. 값은 고르거나 버리지 않는다."""
    frames = []
    for run_id in run_ids:
        path = EXPERIMENT_DIR / "runs" / run_id / "booking_intervals.csv"
        if not path.exists():
            sys.exit(f"실측 간격 파일이 없습니다: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        # 비교 기준 실행의 간격도 같은 파일에 들어 있으므로, 이 실행에서 수집한 것만 쓴다.
        frame = frame[(frame["dataset"] == run_id) & (frame["mode"].isin(types))]
        frames.append(frame[["dataset", "account", "mode", "interval_index", "from_attempt_utc",
                             "to_attempt_utc", "gap_seconds", "long_gap"]])
    pool = pd.concat(frames, ignore_index=True)
    for name in types:
        if not (pool["mode"] == name).any():
            sys.exit(f"유형 {name!r} 의 실측 간격이 없습니다 (run={run_ids}). 값을 지어내지 않고 중단합니다.")
    if pool["gap_seconds"].isna().any() or (pool["gap_seconds"] <= 0).any():
        sys.exit("실측 간격에 빈 값 또는 0 이하의 값이 있습니다. 원본을 확인하세요.")
    return pool


def donor_accounts(pool, name, min_intervals, exclude=None):
    """유형 name 의 실측 계정 중 복원추출에 쓸 계정 목록 (실행 순서, 계정 이름 순)."""
    part = pool[pool["mode"] == name]
    counts = part.groupby(["dataset", "account"], sort=False).size()
    order = {run: i for i, run in enumerate(pool["dataset"].unique())}
    keys = sorted(counts.index, key=lambda key: (order[key[0]], key[1]))
    return [key for key in keys if counts[key] >= min_intervals and key != exclude]


def adjust(original, pool, split, seed, method=METHOD, min_intervals=5, exclude=None):
    """macro 행을 유형에 배정하고 두 Feature 를 다시 만든다. seed 가 같으면 결과가 같다.

    난수 사용 순서
      typed-split          : (1) macro 행 번호를 한 번 섞어 --split 순서대로 잘라 유형 배정, (2) 파일의 행 순서대로 계정마다 복원추출.
      account-hierarchical : (1) 같음, (1-2) 유형마다 그 유형의 행을 한 번 더 섞어 실측 계정을 차례로 돌아가며 배정 (계정별 행 수가 고르게),
                             (2) 파일의 행 순서대로, 배정된 실측 계정의 간격에서 복원추출.
    exclude=(run, account) 는 그 실측 계정을 빼고 만들어 보는 검증용이다.
    """
    rng = np.random.default_rng(seed)
    macro_index = original.index[original["scenario"] == TARGET_SCENARIO].to_numpy()
    if sum(split.values()) != len(macro_index):
        sys.exit(f"--split 의 합({sum(split.values())})이 {TARGET_SCENARIO} 행 수({len(macro_index)})와 다릅니다.")
    collision = set(split) & set(original["scenario"].unique())
    if collision:
        sys.exit(f"유형 이름이 원본에 이미 있는 scenario 값과 겹칩니다: {sorted(collision)}")

    shuffled = rng.permutation(macro_index)
    assigned, start = {}, 0
    for name, count in split.items():
        assigned.update(dict.fromkeys(shuffled[start:start + count].tolist(), name))
        start += count

    gaps_by_type = {name: pool.loc[pool["mode"] == name, "gap_seconds"].to_numpy() for name in split}
    donor_of, gaps_by_donor = {}, {}
    if method == METHOD_HIERARCHICAL:
        for name in split:
            donors = donor_accounts(pool, name, min_intervals, exclude)
            if not donors:
                sys.exit(f"유형 {name!r} 에 간격이 {min_intervals}개 이상인 실측 계정이 없습니다. 값을 지어내지 않고 중단합니다.")
            for key in donors:
                gaps_by_donor[key] = pool.loc[(pool["dataset"] == key[0]) & (pool["account"] == key[1]), "gap_seconds"].to_numpy()
            rows = np.array([index for index in macro_index if assigned[int(index)] == name])
            for position, index in enumerate(rng.permutation(rows)):
                donor_of[int(index)] = donors[position % len(donors)]
    adjusted = original.copy()
    skipped = []
    for index in macro_index:
        name = assigned[int(index)]
        adjusted.at[index, "scenario"] = name
        n_intervals = int(original.at[index, "booking_count"]) - 1
        if n_intervals < 1:
            skipped.append(index)
            continue
        source = gaps_by_donor[donor_of[int(index)]] if method == METHOD_HIERARCHICAL else gaps_by_type[name]
        sample = rng.choice(source, size=n_intervals, replace=True)
        adjusted.at[index, MEAN_COL] = round(max(MEAN_FLOOR, float(sample.mean())), MEAN_DECIMALS)
        adjusted.at[index, VAR_COL] = round(max(VAR_FLOOR, float(sample.var(ddof=0))), VAR_DECIMALS)
    return adjusted, skipped, donor_of


def verify(original, adjusted, skipped, split):
    """원본과 수정본이 약속한 범위 밖에서는 완전히 같은지 확인한다."""
    macro = original["scenario"] == TARGET_SCENARIO
    other_cols = [c for c in original.columns if c not in CHANGED_COLS]
    counts_original = original.loc[~macro, "scenario"].value_counts().to_dict()
    counts_adjusted = adjusted["scenario"].value_counts().to_dict()
    checks = {
        "rows_original": int(len(original)),
        "rows_adjusted": int(len(adjusted)),
        "columns_identical": list(original.columns) == list(adjusted.columns),
        "dtypes_identical": bool((original.dtypes == adjusted.dtypes).all()),
        "other_columns_identical": bool(original[other_cols].equals(adjusted[other_cols])),
        "non_macro_rows_identical": bool(original[~macro].equals(adjusted[~macro])),
        "labels_identical": bool(original["is_anomaly"].equals(adjusted["is_anomaly"])),
        "label_ratio_original": float(original["is_anomaly"].mean()),
        "label_ratio_adjusted": float(adjusted["is_anomaly"].mean()),
        "other_scenario_counts_identical": all(counts_adjusted.get(k) == v for k, v in counts_original.items()),
        "macro_rows": int(macro.sum()),
        "macro_scenario_remaining": int((adjusted["scenario"] == TARGET_SCENARIO).sum()),
        "type_counts": {name: int(counts_adjusted.get(name, 0)) for name in split},
        "type_counts_match_split": all(counts_adjusted.get(name, 0) == count for name, count in split.items()),
        "type_rows_all_anomaly": bool((adjusted.loc[adjusted["scenario"].isin(list(split)), "is_anomaly"] == 1).all()),
        "macro_rows_changed_mean": int((original.loc[macro, MEAN_COL] != adjusted.loc[macro, MEAN_COL]).sum()),
        "macro_rows_changed_variance": int((original.loc[macro, VAR_COL] != adjusted.loc[macro, VAR_COL]).sum()),
        "macro_rows_skipped_booking_count_lt_2": [str(original.at[i, "user_id"]) for i in skipped],
        "floor_applied_mean": int((adjusted.loc[macro, MEAN_COL] <= MEAN_FLOOR).sum()),
        "floor_applied_variance": int((adjusted.loc[macro, VAR_COL] <= VAR_FLOOR).sum()),
    }
    checks["all_passed"] = all([
        checks["rows_original"] == checks["rows_adjusted"], checks["columns_identical"], checks["dtypes_identical"],
        checks["other_columns_identical"], checks["non_macro_rows_identical"], checks["labels_identical"],
        checks["other_scenario_counts_identical"], checks["macro_scenario_remaining"] == 0,
        checks["type_counts_match_split"], checks["type_rows_all_anomaly"],
        set(counts_adjusted) == set(counts_original) | set(split),
    ])
    return checks


def describe(values, label, unit, feature):
    values = np.asarray(values, dtype=float)
    row = {"feature": feature, "group": label, "unit": unit, "n": len(values), "mean": values.mean(),
           "median": np.median(values), "std": values.std(ddof=1) if len(values) > 1 else np.nan,
           "variance": values.var(ddof=1) if len(values) > 1 else np.nan, "min": values.min()}
    for q in QUANTILES:
        row[f"q{int(q * 100):02d}"] = np.quantile(values, q)
    row["max"] = values.max()
    return row


def distance(values, reference, label, reference_label):
    values, reference = np.asarray(values, float), np.asarray(reference, float)
    lo, hi = reference.min(), reference.max()
    return {"feature": MEAN_COL, "group": label, "reference": reference_label,
            "mean_diff": values.mean() - reference.mean(), "median_diff": np.median(values) - np.median(reference),
            "wasserstein": sps.wasserstein_distance(values, reference),
            "ks_statistic": sps.ks_2samp(values, reference).statistic,
            "share_within_reference_range": float(((values >= lo) & (values <= hi)).mean())}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def hist_panel(ax, values, bins, color, title, ylabel):
    style(ax)
    ax.hist(values, bins=bins, color=color, edgecolor=SURFACE, linewidth=1)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)


def label_clusters(ax, groups):
    """같은 색으로 그린 유형별 덩어리 위에 이름과 평균을 적는다."""
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.22)
    for name, values in groups.items():
        ax.text(np.mean(values), top * 1.04, f"{name}\n평균 {np.mean(values):.3f}", ha="center", va="bottom",
                fontsize=9, color=INK)


def make_figures(out, original, adjusted, pool, real_accounts, types):
    macro = original["scenario"] == TARGET_SCENARIO
    gaps = pool["gap_seconds"].to_numpy()
    rows_of = {name: adjusted["scenario"] == name for name in types}
    gaps_of = {name: pool.loc[pool["mode"] == name, "gap_seconds"].to_numpy() for name in types}
    figures = out / "figures"
    figures.mkdir(exist_ok=True)

    # 1) 평균 예매 간격: 세 분포를 같은 x축에 위아래로
    hi = max(gaps.max(), adjusted.loc[macro, MEAN_COL].max()) * 1.05
    bins = np.linspace(0, hi, 61)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7.4), sharex=True, constrained_layout=True)
    values = original.loc[macro, MEAN_COL]
    hist_panel(axes[0], values, bins, C_ORIGINAL, f"{L_ORIGINAL}: scenario = macro 계정의 avg_booking_interval (n={len(values)})", "계정 수")
    label_clusters(axes[0], {"macro": values})
    hist_panel(axes[1], adjusted.loc[macro, MEAN_COL], bins, C_ADJUSTED,
               f"{L_ADJUSTED}: 유형별 계정의 avg_booking_interval (" + ", ".join(f"{n} {int(rows_of[n].sum())}" for n in types) + ")", "계정 수")
    label_clusters(axes[1], {name: adjusted.loc[rows_of[name], MEAN_COL] for name in types})
    hist_panel(axes[2], gaps, bins, C_REAL,
               "실측: 개별 예매 간격 (" + ", ".join(f"{n} {len(gaps_of[n])}" for n in types) + ")   ▲ 실측 계정별 평균", "간격 수")
    label_clusters(axes[2], gaps_of)
    for value in real_accounts[MEAN_COL]:
        axes[2].plot(value, 0, marker="^", markersize=9, color=INK, markeredgecolor=SURFACE, clip_on=False, zorder=5)
    axes[2].set_xlabel("초", fontsize=9, color=MUTED)
    fig.suptitle("평균 예매 간격: 기존 더미 · 수정 더미 · 실측", fontsize=12, color=INK)
    save(fig, figures / "avg_booking_interval_comparison.png")

    # 2) 분산: 기존 더미 + 유형별 수정 더미
    hi = max(original.loc[macro, VAR_COL].max(), adjusted.loc[macro, VAR_COL].max()) * 1.05
    bins = np.linspace(0, hi, 51)
    fig, axes = plt.subplots(1 + len(types), 1, figsize=(8, 2.5 * (1 + len(types))), sharex=True, constrained_layout=True)
    values = original.loc[macro, VAR_COL]
    hist_panel(axes[0], values, bins, C_ORIGINAL,
               f"{L_ORIGINAL}: macro 계정 (n={len(values)}, 중앙값 {values.median():.3f})", "계정 수")
    for ax, name in zip(axes[1:], types):
        values = adjusted.loc[rows_of[name], VAR_COL]
        pooled = gaps_of[name].var(ddof=0)
        hist_panel(ax, values, bins, C_ADJUSTED,
                   f"{L_ADJUSTED}: {name} 계정 (n={len(values)}, 중앙값 {values.median():.3f})   "
                   f"초록 선 = 실측 {name} 간격 전체의 모분산 {pooled:.3f}, ▲ = 실측 계정", "계정 수")
        ax.axvline(pooled, color=C_REAL, linewidth=1.5)
        for value in real_accounts.loc[real_accounts["mode"] == name, VAR_COL]:
            ax.plot(value, 0, marker="^", markersize=9, color=C_REAL, markeredgecolor=SURFACE, clip_on=False, zorder=5)
    axes[-1].set_xlabel("booking_interval_variance (초²)", fontsize=9, color=MUTED)
    fig.suptitle("예매 간격 분산: 변경 전후", fontsize=12, color=INK)
    save(fig, figures / "booking_interval_variance_comparison.png")

    # 3) 평균과 분산의 관계
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.scatter(original.loc[macro, MEAN_COL], original.loc[macro, VAR_COL], s=36, color=C_ORIGINAL,
               edgecolor=SURFACE, linewidth=1, label=f"{L_ORIGINAL} macro (계정)")
    for name, marker in zip(types, MARKERS):
        ax.scatter(adjusted.loc[rows_of[name], MEAN_COL], adjusted.loc[rows_of[name], VAR_COL], s=36, marker=marker,
                   color=C_ADJUSTED, edgecolor=SURFACE, linewidth=1, label=f"{L_ADJUSTED} {name} (계정)")
    ax.scatter(real_accounts[MEAN_COL], real_accounts[VAR_COL], s=100, marker="^", color=C_REAL,
               edgecolor=SURFACE, linewidth=1, label="실측 계정", zorder=4)
    for _, row in real_accounts.iterrows():
        # 간격이 많은 실측 계정은 더미 덩어리 안에 있으므로 이름을 위로 빼서 선으로 잇는다.
        inside = row["n_intervals"] >= 5
        ax.annotate(f"{row['account']} (간격 {int(row['n_intervals'])}개)", (row[MEAN_COL], row[VAR_COL]),
                    xytext=(0, 70) if inside else (0, -14), textcoords="offset points", ha="center", fontsize=8, color=MUTED,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8} if inside else None)
    ax.set_ylim(bottom=-0.04)
    ax.set_xlabel("avg_booking_interval (초)", fontsize=9, color=MUTED)
    ax.set_ylabel("booking_interval_variance (초²)", fontsize=9, color=MUTED)
    ax.set_title("계정별 평균 간격과 분산의 관계", loc="left", fontsize=12, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, figures / "mean_variance_relation.png")

    # 4) 전체 시나리오 안에서의 위치 (로그 축)
    order = original.groupby("scenario")[MEAN_COL].median().sort_values().index.tolist()
    labels = [f"{s} (기존)" if s == TARGET_SCENARIO else s for s in order]
    at = order.index(TARGET_SCENARIO) + 1
    entries = ([(labels[i], original.loc[original["scenario"] == order[i]], C_ORIGINAL) for i in range(at)]
               + [(f"{name} (수정)", adjusted.loc[rows_of[name]], C_ADJUSTED) for name in types]
               + [(labels[i], original.loc[original["scenario"] == order[i]], C_ORIGINAL) for i in range(at, len(order))])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, col, unit in zip(axes, (MEAN_COL, VAR_COL), ("초", "초²")):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for pos, (_, frame, color) in enumerate(entries):
            box = ax.boxplot(frame[col], positions=[pos], orientation="horizontal", widths=0.5, patch_artist=True,
                             showfliers=False, medianprops={"color": SURFACE, "linewidth": 1.5})
            for patch in box["boxes"]:
                patch.set(facecolor=color, edgecolor=color)
            for key in ("whiskers", "caps"):
                for line in box[key]:
                    line.set(color=color, linewidth=1.2)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels([label for label, _, _ in entries], fontsize=9)
        ax.set_xlabel(f"{col} ({unit}, 로그 축)", fontsize=9, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_ORIGINAL), plt.Rectangle((0, 0), 1, 1, color=C_ADJUSTED)]
    axes[1].legend(handles, [L_ORIGINAL, L_ADJUSTED],
                   frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("시나리오별 분포 속에서 매크로의 위치 (상자 = 사분위, 수염 = 1.5×IQR 이내)", fontsize=12, color=INK)
    save(fig, figures / "scenario_position.png")


def table(frame, columns, digits=3):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append("—" if np.isnan(value) else f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(out, meta, checks, stats, distances, variance, overlap, real_accounts, types):
    stat_cols = ["group", "unit", "n", "mean", "median", "std", "variance", "min", "q05", "q25", "q75", "q95", "max"]
    mean_stats = stats[stats["feature"] == MEAN_COL]
    var_stats = stats[stats["feature"] == VAR_COL]
    split_text = ", ".join(f"{name} {count}행" for name, count in meta["split"].items())
    pool_text = ", ".join(f"{name} {meta['pool_by_type'][name]}개" for name in types)
    skipped = checks["macro_rows_skipped_booking_count_lt_2"]
    ok = lambda flag: "같음" if flag else "다름"
    narrow = []
    for name in types:
        adj = mean_stats.loc[mean_stats["group"] == f"{L_ADJUSTED} {name} (계정 평균)"].squeeze()
        real = mean_stats.loc[mean_stats["group"] == f"실측 {name} (개별 간격)"].squeeze()
        narrow.append(f"- {name}: 실측 개별 간격은 {real['min']:.2f}~{real['max']:.2f}초(표준편차 {real['std']:.3f})인데, "
                      f"수정 더미의 계정 평균은 {adj['min']:.2f}~{adj['max']:.2f}초(표준편차 {adj['std']:.3f})다.")
    lines = [
        f"# 수정 더미데이터 {meta['version']}: 매크로를 Fast/Jitter 로 나누고 예매 간격을 실측으로 조정",
        "",
        f"자동 생성 (`scripts/make_adjusted_dummy.py`, 방식 `{METHOD}`). 모델 학습·평가는 하지 않았다.",
        "",
        "## 무엇을 바꿨나",
        "",
        f"- 원본: `{meta['original_path']}` (sha256 `{meta['original_sha256'][:12]}…`, 수정하지 않음)",
        f"- 수정본: `{meta['dataset_path']}`",
        f"- 원본의 `scenario = {TARGET_SCENARIO}` {checks['macro_rows']}행을 {split_text}으로 나누고 `scenario` 값을 유형 이름으로 바꿨다. "
        f"수정본에는 `{TARGET_SCENARIO}` 값이 없다. `is_anomaly` 는 그대로 1이다.",
        f"- 이 {checks['macro_rows']}행의 `{MEAN_COL}`, `{VAR_COL}` 을 다시 계산했다. 바뀐 컬럼은 `scenario` 를 포함해 세 개뿐이다.",
        f"- 실측: 실행 {', '.join(meta['run_ids'])} 의 간격 — {pool_text}. 각 계정은 배정된 유형의 간격만 쓴다.",
        f"- 배정: seed {meta['seed']} (`numpy.random.default_rng`)로 {checks['macro_rows']}행의 순서를 한 번 섞어 앞에서부터 "
        f"{split_text} 순으로 잘랐다. 어느 행이 어느 유형인지는 `macro_rows_before_after.csv` 에 있다. "
        "배정은 무작위이며 원본의 다른 Feature 값과 무관하다.",
        "- 계산: 같은 난수 생성기로 이어서, 파일의 행 순서대로 `booking_count − 1` 개를 복원추출 → 평균과 모분산.",
        f"- 반올림·하한은 원본 생성 코드와 같다 (평균 소수 {MEAN_DECIMALS}자리·하한 {MEAN_FLOOR}, 분산 소수 {VAR_DECIMALS}자리·하한 {VAR_FLOOR}). "
        f"하한이 적용된 행: 평균 {checks['floor_applied_mean']}개, 분산 {checks['floor_applied_variance']}개.",
        f"- `booking_count < 2` 인 매크로 행: {'없음' if not skipped else ', '.join(skipped) + ' (두 Feature 는 바꾸지 않음)'}",
        "",
        "## 원본과의 동일성 확인",
        "",
        "| 항목 | 결과 |",
        "|---|---|",
        f"| 행 수 | {checks['rows_original']} → {checks['rows_adjusted']} |",
        f"| 컬럼 이름·순서·자료형 | {ok(checks['columns_identical'] and checks['dtypes_identical'])} |",
        f"| `scenario` 와 두 Feature 를 뺀 모든 컬럼 (전 행) | {ok(checks['other_columns_identical'])} |",
        f"| 원본에서 macro 가 아니던 행 (전 컬럼) | {ok(checks['non_macro_rows_identical'])} |",
        f"| `is_anomaly` (전 행) | {ok(checks['labels_identical'])}, 이상 비율 {checks['label_ratio_original']:.3f} → {checks['label_ratio_adjusted']:.3f} |",
        f"| 다른 시나리오의 행 수 | {ok(checks['other_scenario_counts_identical'])} |",
        f"| 유형별 행 수 | {', '.join(f'{k} {v}' for k, v in checks['type_counts'].items())} (남은 `{TARGET_SCENARIO}` {checks['macro_scenario_remaining']}) |",
        f"| 값이 바뀐 행 | 평균 {checks['macro_rows_changed_mean']} / 분산 {checks['macro_rows_changed_variance']} (대상 {checks['macro_rows']}) |",
        "",
        "## 평균 예매 간격",
        "",
        "![평균 예매 간격](figures/avg_booking_interval_comparison.png)",
        "",
        table(mean_stats, stat_cols),
        "",
        "더미의 한 값은 계정 1개의 평균이고, 실측 개별 간격의 한 값은 간격 1개다. 단위가 다르다는 점을 감안해 읽어야 한다.",
        "",
        "### 실측과의 거리",
        "",
        table(distances, ["group", "reference", "mean_diff", "median_diff", "wasserstein", "ks_statistic", "share_within_reference_range"]),
        "",
        "- `wasserstein`: 두 분포를 서로 맞추려면 값을 평균 몇 초 옮겨야 하는지. 작을수록 가깝다.",
        "- `share_within_reference_range`: 더미 계정 중 기준 실측값의 최소~최대 범위 안에 들어가는 비율.",
        "- 기존 더미에는 유형이 없으므로, 기존 더미 행은 macro 87행 전체를 각 유형의 실측과 비교한 것이다.",
        "",
        "### 계정 평균의 분포는 개별 간격의 분포보다 좁다",
        "",
        f"수정 더미의 계정 평균은 유형의 실측 간격에서 {meta['n_intervals_min']}~{meta['n_intervals_max']}개를 뽑아 평균한 값이다. "
        "표본 평균은 뽑는 개수가 많을수록 그 유형의 평균으로 모이므로, 계정 평균의 퍼짐은 개별 간격의 퍼짐보다 훨씬 좁다.",
        "",
        *narrow,
        "",
        "즉 유형별 평균 수준과 유형 안에서의 간격 변동은 실측을 따르지만, **같은 유형의 계정끼리는 평균이 거의 같다.** "
        "실제로 계정마다(PC, 네트워크, 매크로 설정) 평균이 얼마나 다른지는 계정이 유형당 1~2개뿐이라 측정되지 않았고, 수정 더미에도 들어 있지 않다.",
        "",
        "## 예매 간격 분산",
        "",
        "![분산](figures/booking_interval_variance_comparison.png)",
        "",
        table(var_stats, stat_cols),
        "",
        "유형별로 실측과 나란히 보면:",
        "",
        table(variance, ["type", "real_pool_variance", "real_account_variances", "original_median", "adjusted_median",
                         "adjusted_min", "adjusted_max", "adjusted_median_minus_real_pool"]),
        "",
        "- `real_pool_variance`: 그 유형의 실측 간격 전체의 모분산. `real_account_variances`: 실측 계정별 분산 (괄호 안은 간격 수).",
        "- 수정 더미의 분산은 그 유형 실측 분산의 근처에 모인다. 기존 더미(중앙값 "
        f"{var_stats.loc[var_stats['group'].str.startswith(L_ORIGINAL), 'median'].squeeze():.3f}초²)보다 작고 실측 계정의 범위와 같은 수준이다.",
        "- 복원추출한 표본의 모분산은 평균적으로 원래 분산의 (n−1)/n 배라서 실측보다 조금 작게 나온다 (n = 15~48 이라 2~7%).",
        "",
        "## 평균과 분산의 관계",
        "",
        "![관계](figures/mean_variance_relation.png)",
        "",
        f"- 기존 더미: 두 값을 서로 독립으로 뽑았다 (순위상관 {meta['spearman_original']:+.2f}).",
        "- 수정 더미: 같은 표본에서 평균과 분산을 함께 계산했다. 유형 안의 순위상관은 "
        + ", ".join(f"{name} {value:+.2f}" for name, value in meta["spearman_adjusted_by_type"].items())
        + ". 부호는 그 유형 실측 간격의 치우침을 따른다: 긴 쪽으로 꼬리가 있으면 긴 간격을 많이 뽑은 계정의 평균과 분산이 함께 커진다(양수).",
        "- 분산이 음수이거나 유형의 실측 간격 범위로 가능한 상한을 넘는 계정은 없다.",
        "- 간격이 19개인 실측 계정은 수정 더미의 덩어리 안에 놓인다. 간격이 2개뿐인 실측 계정은 분산이 훨씬 작게 나와 덩어리 밖에 있다 "
        "(수정 더미 계정은 간격이 최소 " + str(meta["n_intervals_min"]) + "개다).",
        "",
        "## 다른 시나리오와의 위치",
        "",
        "![시나리오](figures/scenario_position.png)",
        "",
        table(overlap, ["scenario", "is_anomaly", "n", "min", "median", "max"]
              + [f"share_within_{name}_range" for name in types]),
        "",
        "`share_within_<유형>_range` 는 그 시나리오의 계정 중 `" + MEAN_COL + "` 이 수정 더미 해당 유형의 범위("
        + ", ".join(f"{name} {meta['adjusted_mean_range'][name][0]:.2f}~{meta['adjusted_mean_range'][name][1]:.2f}초" for name in types)
        + ") 안에 드는 비율이다. 수정 더미의 유형별 범위가 좁아서 이 비율 자체는 작다. "
        "다만 기존 더미에서는 매크로만 0.1~1.3초로 떨어져 있었고, 수정 더미의 매크로는 `hold_cancel`·`multi_account` 가 있는 2~10초 구간 안에 놓인다. "
        "모델 성능에 어떤 영향을 주는지는 이번에 확인하지 않았다.",
        "",
        "## 한계",
        "",
        "- 실측은 같은 PC·같은 날 1회 수집이고 계정은 " + ", ".join(
            f"{name} {int((real_accounts['mode'] == name).sum())}개" for name in types) + "다. 계정 사이의 차이는 측정되지 않았다.",
        "- 유형별 행 수(" + split_text + ")는 실측에서 나온 값이 아니라 정해 준 값이다. 실제 매크로의 유형 비율은 알 수 없다.",
        "- 사람처럼 느린 매크로(Human-like, 평균 12.83초)는 이번 조정에 넣지 않았다.",
        "- 매크로 행의 나머지 Feature(`seat_selection_latency`, `cancel_rate`, `booking_count` 등)는 기존 더미 값 그대로다. "
        "예: 기존 `seat_selection_latency` 는 0.1~0.8초인데 평균 예매 간격은 이제 2.6초 / 4.3초 수준이다.",
        "- 정상 사용자와 다른 이상 시나리오의 값은 그대로다. 이들이 현실적인지는 이 작업의 범위가 아니다.",
        "",
        "## 다시 만들기 / 새 실측 반영",
        "",
        "```powershell",
        f"& $py scripts\\make_adjusted_dummy.py --version {meta['version']} --force      # 같은 결과를 다시 생성 (폴더를 비우고 새로 씀)",
        "& $py scripts\\make_adjusted_dummy.py --version v2 --run-id <새 run-id>       # 새 실측으로 새 버전",
        "& $py scripts\\make_adjusted_dummy.py --version v3 --run-id A --run-id B --split fast:30,jitter:30,human:27",
        "```",
        "",
        "`real_interval_pool.csv` 에 이 버전에 쓴 실측 간격이 그대로 들어 있다.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_output(out, force):
    """출력 폴더를 준비한다. --force 는 이 스크립트가 만든 폴더만 비운다 (이전 방식의 파일이 남지 않게)."""
    if out.exists():
        if not force:
            sys.exit(f"이미 있는 버전입니다: {out}\n다른 --version 을 쓰거나, 같은 버전을 다시 만들려면 --force 를 주세요.")
        if not (out / "generation.json").exists():
            sys.exit(f"{out} 에 generation.json 이 없습니다. 이 스크립트가 만든 폴더가 아닐 수 있어 지우지 않았습니다.")
        shutil.rmtree(out)
    out.mkdir(parents=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="v1", help="출력 폴더 이름 (dummy_adjusted/<version>)")
    parser.add_argument("--run-id", action="append", help="실측 실행 run-id. 여러 번 줄 수 있다 (기본 20261122-continuous)")
    parser.add_argument("--split", default="fast:44,jitter:43",
                        help="macro 행을 나눌 유형과 행 수. 유형 이름은 실측의 매크로 모드이자 새 scenario 값이다. 합이 macro 행 수와 같아야 한다")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="이 스크립트가 만든 기존 버전 폴더를 비우고 다시 쓴다")
    parser.add_argument("--method", choices=[METHOD, METHOD_HIERARCHICAL], default=METHOD,
                        help="typed-split = 유형의 간격을 합쳐서 복원추출 (v1). account-hierarchical = 실측 계정을 고른 뒤 그 계정의 간격에서 복원추출 (v2)")
    parser.add_argument("--min-account-intervals", type=int, default=5,
                        help="account-hierarchical 에서 실측 계정을 쓰기 위한 최소 간격 수")
    parser.add_argument("--compare-version", default=None,
                        help="account-hierarchical 의 보고서에서 함께 비교할 기존 버전 (예: v1). 읽기만 한다")
    args = parser.parse_args()
    run_ids = args.run_id or ["20261122-continuous"]
    split = parse_split(args.split)
    types = list(split)

    original_hash = sha256(ORIGINAL_PATH)
    original = pd.read_csv(ORIGINAL_PATH)
    pool = load_pool(run_ids, types)
    adjusted, skipped, donor_of = adjust(original, pool, split, args.seed, args.method, args.min_account_intervals)
    checks = verify(original, adjusted, skipped, split)
    if not checks["all_passed"]:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
        sys.exit("원본과의 동일성 확인에 실패했습니다. 아무것도 쓰지 않았습니다.")
    if sha256(ORIGINAL_PATH) != original_hash:
        sys.exit("실행 중 원본 파일이 바뀌었습니다.")

    out = EXPERIMENT_DIR / "dummy_adjusted" / args.version
    prepare_output(out, args.force)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    dataset_path = out / DATASET_NAME
    adjusted.to_csv(dataset_path, index=False)
    pool.to_csv(out / "real_interval_pool.csv", index=False, encoding="utf-8-sig")

    macro = original["scenario"] == TARGET_SCENARIO
    rows_of = {name: adjusted["scenario"] == name for name in types}
    gaps_of = {name: pool.loc[pool["mode"] == name, "gap_seconds"].to_numpy() for name in types}
    gaps = pool["gap_seconds"].to_numpy()
    real_accounts = (pool.groupby(["mode", "account"])["gap_seconds"]
                     .agg(n_intervals="count", **{MEAN_COL: "mean", VAR_COL: lambda s: s.var(ddof=0)}).reset_index())
    real_accounts.to_csv(out / "real_account_features.csv", index=False, encoding="utf-8-sig")

    rows = [describe(original.loc[macro, MEAN_COL], f"{L_ORIGINAL} macro (계정 평균)", "초", MEAN_COL)]
    for name in types:
        rows.append(describe(adjusted.loc[rows_of[name], MEAN_COL], f"{L_ADJUSTED} {name} (계정 평균)", "초", MEAN_COL))
        rows.append(describe(gaps_of[name], f"실측 {name} (개별 간격)", "초", MEAN_COL))
    rows.append(describe(adjusted.loc[macro, MEAN_COL], f"{L_ADJUSTED} " + "+".join(types) + " 전체 (계정 평균)", "초", MEAN_COL))
    rows.append(describe(gaps, "실측 " + "+".join(types) + " 전체 (개별 간격)", "초", MEAN_COL))
    rows.append(describe(original.loc[macro, VAR_COL], f"{L_ORIGINAL} macro (계정 분산)", "초²", VAR_COL))
    for name in types:
        rows.append(describe(adjusted.loc[rows_of[name], VAR_COL], f"{L_ADJUSTED} {name} (계정 분산)", "초²", VAR_COL))
    stats = pd.DataFrame(rows)
    stats.to_csv(out / "distribution_stats.csv", index=False, encoding="utf-8-sig")

    rows = []
    for name in types:
        reference = f"실측 {name} (개별 간격)"
        rows.append(distance(original.loc[macro, MEAN_COL], gaps_of[name], f"{L_ORIGINAL} macro", reference))
        rows.append(distance(adjusted.loc[rows_of[name], MEAN_COL], gaps_of[name], f"{L_ADJUSTED} {name}", reference))
    reference = "실측 " + "+".join(types) + " 전체 (개별 간격)"
    rows.append(distance(original.loc[macro, MEAN_COL], gaps, f"{L_ORIGINAL} macro", reference))
    rows.append(distance(adjusted.loc[macro, MEAN_COL], gaps, f"{L_ADJUSTED} 전체", reference))
    distances = pd.DataFrame(rows)
    distances.to_csv(out / "distance_to_real.csv", index=False, encoding="utf-8-sig")

    rows = []
    for name in types:
        accounts = real_accounts[real_accounts["mode"] == name]
        values = adjusted.loc[rows_of[name], VAR_COL]
        pooled = float(gaps_of[name].var(ddof=0))
        rows.append({"type": name, "real_pool_variance": pooled,
                     "real_account_variances": ", ".join(f"{v:.3f} ({int(n)})" for v, n in zip(accounts[VAR_COL], accounts["n_intervals"])),
                     "original_median": float(original.loc[macro, VAR_COL].median()),
                     "adjusted_median": float(values.median()), "adjusted_min": float(values.min()),
                     "adjusted_max": float(values.max()), "adjusted_median_minus_real_pool": float(values.median() - pooled)})
    variance = pd.DataFrame(rows)
    variance.to_csv(out / "variance_vs_real.csv", index=False, encoding="utf-8-sig")

    ranges = {name: [float(adjusted.loc[rows_of[name], MEAN_COL].min()), float(adjusted.loc[rows_of[name], MEAN_COL].max())]
              for name in types}
    others = adjusted[~macro]
    overlap = others.groupby("scenario").agg(is_anomaly=("is_anomaly", "first"), n=(MEAN_COL, "size"),
                                             min=(MEAN_COL, "min"), median=(MEAN_COL, "median"), max=(MEAN_COL, "max"))
    for name, (lo, hi) in ranges.items():
        overlap[f"share_within_{name}_range"] = others.groupby("scenario")[MEAN_COL].apply(
            lambda s: float(((s >= lo) & (s <= hi)).mean()))
    overlap = overlap.reset_index().sort_values("median")
    overlap.to_csv(out / "scenario_overlap.csv", index=False, encoding="utf-8-sig")

    # 변경 전후를 행 단위로 볼 수 있는 표 (유형 배정 포함)
    changes = original.loc[macro, ["user_id", "scenario", "booking_count", MEAN_COL, VAR_COL]].rename(
        columns={"scenario": "scenario_original", MEAN_COL: f"{MEAN_COL}_original", VAR_COL: f"{VAR_COL}_original"})
    changes.insert(2, "scenario_adjusted", adjusted.loc[macro, "scenario"])
    changes[f"{MEAN_COL}_adjusted"] = adjusted.loc[macro, MEAN_COL]
    changes[f"{VAR_COL}_adjusted"] = adjusted.loc[macro, VAR_COL]
    if donor_of:
        changes["real_run"] = [donor_of[int(i)][0] for i in changes.index]
        changes["real_account"] = [donor_of[int(i)][1] for i in changes.index]
    changes.to_csv(out / "macro_rows_before_after.csv", index=False, encoding="utf-8-sig")

    n_intervals = original.loc[macro, "booking_count"] - 1
    meta = {
        "version": args.version, "method": args.method, "seed": args.seed, "rng": "numpy.random.default_rng",
        "numpy": np.__version__, "pandas": pd.__version__, "run_ids": run_ids, "split": split,
        "original_path": ORIGINAL_PATH.relative_to(REPO_DIR).as_posix(), "original_sha256": original_hash,
        "dataset_path": dataset_path.relative_to(REPO_DIR).as_posix(), "dataset_sha256": sha256(dataset_path),
        "changed_columns": list(CHANGED_COLS),
        "pool_by_type": {name: int(len(gaps_of[name])) for name in types},
        "pool_mean_by_type": {name: float(gaps_of[name].mean()) for name in types},
        "pool_variance_population_by_type": {name: float(gaps_of[name].var(ddof=0)) for name in types},
        "n_intervals_min": int(n_intervals.min()), "n_intervals_max": int(n_intervals.max()),
        "adjusted_mean_range": ranges,
        "spearman_original": float(sps.spearmanr(original.loc[macro, MEAN_COL], original.loc[macro, VAR_COL]).statistic),
        "spearman_adjusted_by_type": {name: float(sps.spearmanr(adjusted.loc[rows_of[name], MEAN_COL],
                                                               adjusted.loc[rows_of[name], VAR_COL]).statistic) for name in types},
        "checks": checks,
    }
    if args.method == METHOD_HIERARCHICAL:
        meta["min_account_intervals"] = args.min_account_intervals
        meta["compare_version"] = args.compare_version
        meta["pool_by_run_and_type"] = {f"{run}/{mode}": int(n) for (run, mode), n in
                                        pool.groupby(["dataset", "mode"], sort=False).size().items()}
        meta["rows_per_real_account"] = {f"{run}/{account}": int(n) for (run, account), n in
                                         changes.groupby(["real_run", "real_account"], sort=False).size().items()}
    (out / "generation.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.method == METHOD_HIERARCHICAL:
        from adjusted_dummy_v2_report import report_hierarchical
        report_hierarchical(out, meta, checks, original, adjusted, pool, split, donor_of, overlap, args)
    else:
        make_figures(out, original, adjusted, pool, real_accounts, types)
        write_report(out, meta, checks, stats, distances, variance, overlap, real_accounts, types)

    print(f"수정 데이터셋: {dataset_path}")
    print(f"유형별 행 수 {checks['type_counts']}, 실측 간격 {meta['pool_by_type']}, 건너뜀 {len(skipped)}행")
    print(stats.round(4).to_string(index=False))
    print(distances.round(4).to_string(index=False))
    print(variance.round(4).to_string(index=False))
    print(f"보고서: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
