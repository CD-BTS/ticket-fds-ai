"""make_adjusted_dummy.py --method account-hierarchical 의 통계·그래프·보고서.

직접 실행하지 않는다. make_adjusted_dummy.py 가 데이터셋을 쓴 뒤 report_hierarchical() 을 부른다.
원본 더미, 비교할 기존 버전(--compare-version), 이번 버전, 실측 실행들을 유형별로 비교한다. 읽기만 하고 모델은 다루지 않는다.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import make_adjusted_dummy as base

MEAN_COL, VAR_COL = base.MEAN_COL, base.VAR_COL
C_ORIGINAL, C_PREVIOUS, C_THIS, C_REAL = "#2a78d6", "#6250d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = base.INK, base.MUTED, base.GRID, base.SURFACE
RUN_MARKERS = ["^", "v", "D", "s"]


def describe(values):
    values = np.asarray(values, dtype=float)
    row = {"n": len(values), "mean": values.mean(), "median": np.median(values),
           "std": values.std(ddof=1) if len(values) > 1 else np.nan, "min": values.min()}
    for q in base.QUANTILES:
        row[f"q{int(q * 100):02d}"] = np.quantile(values, q)
    row["max"] = values.max()
    return row


def within(value, values, lo=0.0, hi=1.0):
    return bool(np.quantile(values, lo) <= value <= np.quantile(values, hi))


def report_hierarchical(out, meta, checks, original, adjusted, pool, split, donor_of, overlap, args):
    types = list(split)
    version = meta["version"]
    run_ids = meta["run_ids"]
    run_label = dict(zip(run_ids, ["기존 실측", "새 실측"])) if len(run_ids) == 2 else {r: f"실측 {r}" for r in run_ids}
    min_n = args.min_account_intervals
    macro = original["scenario"] == base.TARGET_SCENARIO
    (out / "figures").mkdir(exist_ok=True)

    # 비교 대상 더미: 원본(유형 없음 → macro 전체), 기존 버전, 이번 버전
    versions = {"원본": (C_ORIGINAL, {name: original[macro] for name in types})}
    previous_label = None
    if args.compare_version:
        previous_dir = base.EXPERIMENT_DIR / "dummy_adjusted" / args.compare_version
        previous = pd.read_csv(previous_dir / base.DATASET_NAME)
        previous_meta = json.loads((previous_dir / "generation.json").read_text(encoding="utf-8"))
        previous_label = args.compare_version
        versions[previous_label] = (C_PREVIOUS, {name: previous[previous["scenario"] == name] for name in types})
    versions[version] = (C_THIS, {name: adjusted[adjusted["scenario"] == name] for name in types})

    # 실측 계정
    real = (pool.groupby(["mode", "dataset", "account"], sort=False)["gap_seconds"]
            .agg(n_intervals="count", **{MEAN_COL: "mean", VAR_COL: lambda s: s.var(ddof=0)},
                 median="median", std="std", min="min", max="max").reset_index())
    real["run"] = real["dataset"].map(run_label)
    real["used_for_generation"] = real["n_intervals"] >= min_n
    rows_per = pd.Series({key: 0 for key in zip(real["dataset"], real["account"])})
    for key in donor_of.values():
        rows_per[key] += 1
    real["dummy_rows_assigned"] = [int(rows_per[(d, a)]) for d, a in zip(real["dataset"], real["account"])]
    real.to_csv(out / "real_account_features.csv", index=False, encoding="utf-8-sig")
    used = real[real["used_for_generation"]]

    # 1) 통계표
    rows = []
    for name in types:
        for feature, unit in ((MEAN_COL, "초"), (VAR_COL, "초²")):
            for label, (_, frames) in versions.items():
                suffix = "macro 전체" if label == "원본" else name
                rows.append({"type": name, "feature": feature, "group": f"더미 {label} ({suffix}, 계정)", "unit": unit,
                             **describe(frames[name][feature])})
            rows.append({"type": name, "feature": feature, "group": f"실측 계정 (간격 {min_n}개 이상)", "unit": unit,
                         **describe(used.loc[used["mode"] == name, feature])})
        for run_id in run_ids:
            values = pool.loc[(pool["mode"] == name) & (pool["dataset"] == run_id), "gap_seconds"]
            rows.append({"type": name, "feature": "gap_seconds", "group": f"{run_label[run_id]} (개별 간격)", "unit": "초",
                         **describe(values)})
    stats = pd.DataFrame(rows)
    stats.to_csv(out / "distribution_stats.csv", index=False, encoding="utf-8-sig")

    # 2) 실측 계정이 더미 범위에 들어가는가
    rows = []
    for name in types:
        accounts = real[real["mode"] == name]
        for label, (_, frames) in versions.items():
            d = frames[name]
            for scope, part in (("전체", accounts), (f"간격 {min_n}개 이상", accounts[accounts["used_for_generation"]])):
                rows.append({
                    "type": name, "dummy": label, "real_accounts": scope, "n_real_accounts": len(part),
                    "mean_within_min_max": int(sum(within(v, d[MEAN_COL]) for v in part[MEAN_COL])),
                    "mean_within_q05_q95": int(sum(within(v, d[MEAN_COL], 0.05, 0.95) for v in part[MEAN_COL])),
                    "variance_within_min_max": int(sum(within(v, d[VAR_COL]) for v in part[VAR_COL])),
                    "variance_within_q05_q95": int(sum(within(v, d[VAR_COL], 0.05, 0.95) for v in part[VAR_COL])),
                    "dummy_mean_min": d[MEAN_COL].min(), "dummy_mean_max": d[MEAN_COL].max(),
                    "dummy_variance_min": d[VAR_COL].min(), "dummy_variance_max": d[VAR_COL].max()})
    coverage = pd.DataFrame(rows)
    for col in ("mean_within_min_max", "mean_within_q05_q95", "variance_within_min_max", "variance_within_q05_q95"):
        coverage[col + "_share"] = coverage[col] / coverage["n_real_accounts"]
    coverage.to_csv(out / "real_account_coverage.csv", index=False, encoding="utf-8-sig")

    # 3) 계정 간 변동성
    rows = []
    for name in types:
        part = used[used["mode"] == name]
        rows.append({"type": name, "group": f"실측 계정 (간격 {min_n}개 이상)", "n": len(part),
                     "account_mean_avg": part[MEAN_COL].mean(), "account_mean_std": part[MEAN_COL].std(ddof=1),
                     "account_mean_min": part[MEAN_COL].min(), "account_mean_max": part[MEAN_COL].max(),
                     "account_variance_median": part[VAR_COL].median(),
                     "account_variance_min": part[VAR_COL].min(), "account_variance_max": part[VAR_COL].max()})
        for label, (_, frames) in versions.items():
            d = frames[name]
            rows.append({"type": name, "group": f"더미 {label}", "n": len(d),
                         "account_mean_avg": d[MEAN_COL].mean(), "account_mean_std": d[MEAN_COL].std(ddof=1),
                         "account_mean_min": d[MEAN_COL].min(), "account_mean_max": d[MEAN_COL].max(),
                         "account_variance_median": d[VAR_COL].median(),
                         "account_variance_min": d[VAR_COL].min(), "account_variance_max": d[VAR_COL].max()})
    variability = pd.DataFrame(rows)
    variability["account_mean_range_width"] = variability["account_mean_max"] - variability["account_mean_min"]
    variability.to_csv(out / "between_account_variability.csv", index=False, encoding="utf-8-sig")

    # 4) 실측 계정 하나를 빼고 만들면 그 계정이 범위 안에 드는가 (같은 seed, 파일로 쓰지 않는 검증용 생성)
    rows = []
    for _, account in used.iterrows():
        key = (account["dataset"], account["account"])
        if len(used[used["mode"] == account["mode"]]) < 2:
            continue
        held, _, _ = base.adjust(original, pool, split, meta["seed"], base.METHOD_HIERARCHICAL, min_n, exclude=key)
        d = held[held["scenario"] == account["mode"]]
        rows.append({"type": account["mode"], "held_out_run": account["run"], "held_out_account": account["account"],
                     "held_out_mean": account[MEAN_COL], "dummy_mean_min": d[MEAN_COL].min(), "dummy_mean_max": d[MEAN_COL].max(),
                     "mean_within": within(account[MEAN_COL], d[MEAN_COL]),
                     "mean_gap_to_range": max(d[MEAN_COL].min() - account[MEAN_COL], account[MEAN_COL] - d[MEAN_COL].max(), 0),
                     "held_out_variance": account[VAR_COL], "dummy_variance_min": d[VAR_COL].min(),
                     "dummy_variance_max": d[VAR_COL].max(), "variance_within": within(account[VAR_COL], d[VAR_COL])})
    holdout = pd.DataFrame(rows)
    holdout.to_csv(out / "leave_one_account_out.csv", index=False, encoding="utf-8-sig")

    # ---------- 그래프 ----------
    def real_markers(ax, name, y, x_col=MEAN_COL, label_points=False, stagger=False):
        # 이름표가 겹치지 않게, x 순서대로 번갈아 높이를 바꾼다.
        rank = real[real["mode"] == name].sort_values(x_col)["account"].tolist()
        for marker, run_id in zip(RUN_MARKERS, run_ids):
            part = real[(real["mode"] == name) & (real["dataset"] == run_id)]
            ys = part[y] if isinstance(y, str) else np.full(len(part), y)
            ax.scatter(part[x_col], ys, s=110, marker=marker, color=C_REAL, edgecolor=INK, linewidth=0.8, zorder=5,
                       label=f"{run_label[run_id]} 계정")
            if label_points:
                for (_, row), yy in zip(part.iterrows(), ys):
                    lift = 16 * (rank.index(row["account"]) % 2) if stagger else 0
                    ax.annotate(f"{row['account']} ({int(row['n_intervals'])})", (row[x_col], yy),
                                xytext=(0, 12 + lift) if stagger else (7, 5), ha="center" if stagger else "left",
                                textcoords="offset points", fontsize=8, color=MUTED)

    # (1) 전체: 원본·기존 버전·이번 버전의 계정 평균 (같은 x축)
    labels = list(versions)
    hi = max(real[MEAN_COL].max(), adjusted.loc[macro, MEAN_COL].max()) * 1.06
    bins = np.linspace(0, hi, 71)
    fig, axes = plt.subplots(len(labels), 1, figsize=(8.5, 2.5 * len(labels)), sharex=True, constrained_layout=True)
    for ax, label in zip(axes, labels):
        base.style(ax)
        color, frames = versions[label]
        values = original.loc[macro, MEAN_COL] if label == "원본" else pd.concat([frames[n][MEAN_COL] for n in types])
        ax.hist(values, bins=bins, color=color, edgecolor=SURFACE, linewidth=1)
        ax.set_ylabel("계정 수", fontsize=9, color=MUTED)
        ax.set_title(f"더미 {label}: 매크로 계정의 {MEAN_COL} (n={len(values)})", loc="left", fontsize=10, color=INK)
        for marker, run_id in zip(RUN_MARKERS, run_ids):
            part = real[real["dataset"] == run_id]
            ax.scatter(part[MEAN_COL], np.zeros(len(part)), s=90, marker=marker, color=C_REAL, edgecolor=INK,
                       linewidth=0.8, zorder=5, clip_on=False, label=f"{run_label[run_id]} 계정 평균")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[-1].set_xlabel("초", fontsize=9, color=MUTED)
    fig.suptitle("매크로 계정의 평균 예매 간격: 더미 버전별 분포와 실측 계정", fontsize=12, color=INK)
    base.save(fig, out / "figures" / "avg_booking_interval_by_version.png")

    # (2) 유형별 확대: 계정 평균
    rng = np.random.default_rng(0)  # 점을 위아래로 흩는 용도 (값에는 영향 없음)
    shown = [label for label in labels if label != "원본"]
    fig, axes = plt.subplots(1, len(types), figsize=(6 * len(types), 1.6 + 1.1 * (len(shown) + 1)), constrained_layout=True)
    for ax, name in zip(np.atleast_1d(axes), types):
        base.style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for position, label in enumerate(shown):
            color, frames = versions[label]
            values = frames[name][MEAN_COL].to_numpy()
            ax.scatter(values, position + rng.uniform(-0.22, 0.22, len(values)), s=24, color=color, edgecolor=SURFACE, linewidth=0.7)
        real_markers(ax, name, len(shown), label_points=True, stagger=True)
        ax.set_yticks(range(len(shown) + 1))
        ax.set_yticklabels([f"더미 {label}" for label in shown] + ["실측 계정"], fontsize=9)
        ax.set_ylim(-0.6, len(shown) + 0.9)
        ax.margins(x=0.12)
        ax.set_xlabel(f"{MEAN_COL} (초)", fontsize=9, color=MUTED)
        ax.set_title(name, loc="left", fontsize=11, color=INK)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("유형별 계정 평균 (괄호 = 실측 계정의 간격 수, 원본 macro 는 0.1~1.3초라 이 범위 밖)", fontsize=12, color=INK)
    base.save(fig, out / "figures" / "account_mean_by_type.png")

    # (3) 유형별 Feature 평면
    fig, axes = plt.subplots(1, len(types), figsize=(6 * len(types), 5), constrained_layout=True)
    for ax, name in zip(np.atleast_1d(axes), types):
        base.style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        for label in shown:
            color, frames = versions[label]
            ax.scatter(frames[name][MEAN_COL], frames[name][VAR_COL], s=26, color=color, edgecolor=SURFACE, linewidth=0.7,
                       label=f"더미 {label} (n={len(frames[name])})")
        real_markers(ax, name, VAR_COL, label_points=True)
        ax.set_ylim(bottom=-0.012)
        ax.margins(x=0.14)
        ax.set_xlabel(f"{MEAN_COL} (초)", fontsize=9, color=MUTED)
        ax.set_ylabel(f"{VAR_COL} (초²)", fontsize=9, color=MUTED)
        ax.set_title(name, loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("계정별 평균 예매 간격과 분산 (괄호 = 실측 계정의 간격 수)", fontsize=12, color=INK)
    base.save(fig, out / "figures" / "mean_variance_by_type.png")

    # ---------- 보고서 ----------
    table = base.table
    ok = lambda flag: "같음" if flag else "다름"
    split_text = ", ".join(f"{name} {count}행" for name, count in split.items())
    stat_cols = ["type", "group", "unit", "n", "mean", "median", "std", "min", "q05", "q25", "q75", "q95", "max"]
    real_cols = ["mode", "run", "account", "n_intervals", MEAN_COL, VAR_COL, "median", "std", "min", "max",
                 "used_for_generation", "dummy_rows_assigned"]
    excluded = real[~real["used_for_generation"]]
    this = variability[variability["group"] == f"더미 {version}"].set_index("type")
    prev = variability[variability["group"] == f"더미 {previous_label}"].set_index("type") if previous_label else None
    real_var = variability[variability["group"].str.startswith("실측")].set_index("type")
    cov = coverage[coverage["real_accounts"] == f"간격 {min_n}개 이상"]

    def cov_text(label):
        part = cov[cov["dummy"] == label]
        return (f"평균 {int(part['mean_within_min_max'].sum())}/{int(part['n_real_accounts'].sum())}, "
                f"분산 {int(part['variance_within_min_max'].sum())}/{int(part['n_real_accounts'].sum())}")

    same_assignment = None
    if previous_label:
        same_assignment = bool((previous["scenario"] == adjusted["scenario"]).all())
    lines = [
        f"# 수정 더미데이터 {version}: 두 번의 실측을 계정 단위로 반영",
        "",
        f"자동 생성 (`scripts/make_adjusted_dummy.py --method {base.METHOD_HIERARCHICAL}`). 모델 학습·평가는 하지 않았다.",
        "",
        "## 무엇을 바꿨나",
        "",
        f"- 원본: `{meta['original_path']}` (sha256 `{meta['original_sha256'][:12]}…`, 수정하지 않음)",
        f"- 수정본: `{meta['dataset_path']}` (sha256 `{meta['dataset_sha256'][:12]}…`)",
        f"- 원본의 `scenario = macro` {checks['macro_rows']}행을 {split_text}으로 나누고, 이 행들의 `{MEAN_COL}`, `{VAR_COL}` 을 다시 계산했다. "
        "바뀐 컬럼은 `scenario` 를 포함해 세 개뿐이고 `is_anomaly` 는 그대로 1이다.",
        "- 실측 출처: " + "; ".join(
            f"`runs/{run_id}/booking_intervals.csv` ({run_label[run_id]}: "
            + ", ".join(f"{name} {int(((pool['dataset'] == run_id) & (pool['mode'] == name)).sum())}개" for name in types) + ")"
            for run_id in run_ids) + ". 쓴 간격은 `real_interval_pool.csv` 에 그대로 있다.",
        "",
        "## 생성 방법",
        "",
        f"1. **유형 배정**: seed {meta['seed']} 로 {checks['macro_rows']}행의 순서를 섞어 앞에서부터 {split_text}."
        + (f" {previous_label} 과 같은 절차라 행별 유형이 {previous_label} 과 {'같다' if same_assignment else '다르다'}." if previous_label else ""),
        f"2. **실측 계정 배정**: 유형마다 그 유형의 더미 행을 한 번 더 섞어, 같은 유형의 실측 계정(간격 {min_n}개 이상)을 차례로 돌아가며 붙인다. "
        "실측 계정마다 붙는 더미 행 수가 고르게 된다.",
        "3. **간격 복원추출**: 더미 행마다 `booking_count − 1` 개를 **그 실측 계정의 간격에서만** 복원추출해 평균과 모분산을 넣는다.",
        "",
        "실측 계정의 평균 수준과 분산이 더미 계정에 그대로 옮겨지므로, 실측 계정 사이의 차이(실행 간 차이 포함)가 더미에 남는다. "
        "정규분포 같은 분포 모양이나 계정 간 분산의 크기를 따로 가정하지 않았다. 더미 계정의 평균은 실측 계정 평균 주변에만 생긴다.",
        "",
        table(real, real_cols),
        "",
        f"- `{MEAN_COL}` 은 그 계정 간격의 평균, `{VAR_COL}` 은 모분산이다.",
        "- 실측 계정마다 같은 비중을 줬다. 그 결과 더미 행의 실행별 비중은 "
        + ", ".join(f"{run_label[r]} {int(real.loc[real['dataset'] == r, 'dummy_rows_assigned'].sum())}행" for r in run_ids) + "이다.",
        ("- 제외한 실측 계정: " + ", ".join(f"{r['account']} (간격 {int(r['n_intervals'])}개)" for _, r in excluded.iterrows())
         + f". 간격이 {min_n}개 미만이면 복원추출한 분산이 그 몇 개의 값으로만 정해지기 때문이다. 아래 비교에는 표시한다."
         if len(excluded) else "- 제외한 실측 계정: 없음"),
        f"- 반올림·하한은 원본 생성 코드와 같다. 하한이 적용된 행: 평균 {checks['floor_applied_mean']}개, 분산 {checks['floor_applied_variance']}개. "
        f"`booking_count < 2` 인 매크로 행: {len(checks['macro_rows_skipped_booking_count_lt_2'])}개.",
        "- 행별 유형과 실측 계정 배정은 `macro_rows_before_after.csv` 에 있다 (데이터셋에는 컬럼을 추가하지 않았다).",
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
        f"| 유형별 행 수 | {', '.join(f'{k} {v}' for k, v in checks['type_counts'].items())} (남은 `macro` {checks['macro_scenario_remaining']}) |",
        f"| 값이 바뀐 행 | 평균 {checks['macro_rows_changed_mean']} / 분산 {checks['macro_rows_changed_variance']} (대상 {checks['macro_rows']}) |",
        "",
        "## 분포 비교",
        "",
        "![버전별](figures/avg_booking_interval_by_version.png)",
        "",
        "![유형별 계정 평균](figures/account_mean_by_type.png)",
        "",
        "### 평균 예매 간격 (계정 평균)",
        "",
        table(stats[stats["feature"] == MEAN_COL], stat_cols),
        "",
        "### 예매 간격 분산 (계정 분산)",
        "",
        table(stats[stats["feature"] == VAR_COL], stat_cols),
        "",
        "### 실측 개별 간격 (참고)",
        "",
        table(stats[stats["feature"] == "gap_seconds"], stat_cols),
        "",
        "원본에는 유형이 없어 macro 87행 전체를 각 유형 옆에 적었다. 실측 계정 통계는 계정이 3개씩이라 분위수는 참고용이다.",
        "",
        "![Feature 평면](figures/mean_variance_by_type.png)",
        "",
        "## 실측 계정이 더미 범위에 포함되는 비율",
        "",
        table(coverage, ["type", "dummy", "real_accounts", "n_real_accounts", "mean_within_min_max", "mean_within_q05_q95",
                         "variance_within_min_max", "variance_within_q05_q95", "dummy_mean_min", "dummy_mean_max",
                         "dummy_variance_min", "dummy_variance_max"]),
        "",
        f"- 값은 계정 수다. 간격 {min_n}개 이상인 실측 계정 기준으로 더미 범위(최소~최대) 안에 드는 계정: "
        + "; ".join(f"{label} — {cov_text(label)}" for label in labels) + ".",
        f"- **{version} 의 포함 비율은 만든 방법에서 나오는 결과다.** 더미가 바로 그 실측 계정들에서 만들어졌으므로 포함되는 것이 당연하고, "
        "새로운 계정에도 맞는다는 근거가 아니다. 그 점은 아래 검증으로 따로 본다.",
        "",
        "### 실측 계정 하나를 빼고 만들면",
        "",
        "실측 계정 하나를 숨기고 나머지로 같은 방법·같은 seed 로 더미를 만든 뒤, 숨긴 계정이 그 더미 범위 안에 드는지 본 것이다 (파일로 저장하지 않음).",
        "",
        table(holdout, ["type", "held_out_run", "held_out_account", "held_out_mean", "dummy_mean_min", "dummy_mean_max",
                        "mean_within", "mean_gap_to_range", "held_out_variance", "dummy_variance_min", "dummy_variance_max",
                        "variance_within"]),
        "",
        f"숨긴 계정 {len(holdout)}개 중 평균이 범위 안에 든 계정은 {int(holdout['mean_within'].sum())}개, "
        f"분산이 범위 안에 든 계정은 {int(holdout['variance_within'].sum())}개다. "
        "이 방법은 본 적 있는 계정 수준 주변만 만들기 때문에, 본 적 없는 수준의 계정은 범위 밖에 놓일 수 있다. "
        "`mean_gap_to_range` 는 범위에서 벗어난 거리(초)다.",
        "",
        "## 계정 간 변동성",
        "",
        table(variability, ["type", "group", "n", "account_mean_avg", "account_mean_std", "account_mean_min", "account_mean_max",
                            "account_mean_range_width", "account_variance_median", "account_variance_min", "account_variance_max"]),
        "",
    ]
    for name in types:
        text = (f"- {name}: 계정 평균의 표준편차는 실측 계정 {real_var.loc[name, 'account_mean_std']:.3f}초, "
                + (f"{previous_label} {prev.loc[name, 'account_mean_std']:.3f}초, " if prev is not None else "")
                + f"{version} {this.loc[name, 'account_mean_std']:.3f}초. 범위 폭은 실측 {real_var.loc[name, 'account_mean_range_width']:.3f}초, "
                + (f"{previous_label} {prev.loc[name, 'account_mean_range_width']:.3f}초, " if prev is not None else "")
                + f"{version} {this.loc[name, 'account_mean_range_width']:.3f}초.")
        lines.append(text)
    lines += [
        "",
        (f"{previous_label} 에서 부족했던 계정 간 변동성은 {version} 에서 실측 계정 사이의 차이만큼 늘었다. " if previous_label else "")
        + "다만 그 변동은 연속적으로 퍼진 것이 아니라 실측 계정 수만큼의 덩어리로 나타난다 (그래프 참고). "
        "덩어리 사이의 빈 구간에 실제 계정이 없다는 뜻이 아니라, 그 구간을 채울 관측이 없다는 뜻이다.",
        "",
        "## 다른 시나리오와의 위치",
        "",
        table(overlap, ["scenario", "is_anomaly", "n", "min", "median", "max"] + [f"share_within_{name}_range" for name in types]),
        "",
        "`share_within_<유형>_range`: 그 시나리오의 계정 중 `" + MEAN_COL + "` 이 " + version + " 해당 유형의 범위("
        + ", ".join(f"{name} {meta['adjusted_mean_range'][name][0]:.2f}~{meta['adjusted_mean_range'][name][1]:.2f}초" for name in types)
        + ") 안에 드는 비율. 모델 성능에 어떤 영향을 주는지는 확인하지 않았다.",
        "",
        "## 한계",
        "",
        "- 실측 계정이 유형당 " + ", ".join(f"{name} {int((used['mode'] == name).sum())}개" for name in types)
        + "뿐이다. 계정 간 분포의 모양(연속적인지, 얼마나 넓은지)은 이 수로는 알 수 없고, 더미는 관측된 계정 수준만 재현한다.",
        "- 두 실측은 같은 PC 에서 하루 간격으로 한 번씩 수집했다. 실행 간 차이(약 0.4~0.6초)가 사이트 변경 때문인지 실행 조건 때문인지 분리되지 않은 채 섞여 들어갔다.",
        "- 실측 계정마다 같은 비중을 준 것은 선택이다. 최신 사이트만 반영하거나 실행마다 같은 비중을 주면 결과가 달라진다.",
        "- 유형별 행 수(" + split_text + ")는 정해 준 값이고, 사람처럼 느린 매크로(Human-like)는 넣지 않았다.",
        "- 매크로 행의 나머지 Feature 와 다른 시나리오의 값은 원본 그대로다.",
        "",
        "## 다시 만들기",
        "",
        "```powershell",
        f"& $py scripts\\make_adjusted_dummy.py --version {version} --method {base.METHOD_HIERARCHICAL} "
        + " ".join(f"--run-id {r}" for r in run_ids)
        + (f" --compare-version {args.compare_version}" if args.compare_version else "") + " --force",
        "& $py scripts\\make_adjusted_dummy.py --version v1 --force      # v1 (typed-split) 은 기존 명령 그대로 재현된다",
        "```",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(real[real_cols].round(4).to_string(index=False))
    print(variability.round(4).to_string(index=False))
    print(coverage[["type", "dummy", "real_accounts", "n_real_accounts", "mean_within_min_max", "variance_within_min_max"]].to_string(index=False))
    print(holdout.round(3).to_string(index=False))
