"""기존 실측 · 새 실측 · 보정 더미데이터의 예매 간격을 유형(Fast/Jitter)별로 비교한다.

    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\compare_remeasure.py --run-id 20261123-remeasure
    ..\\issue14-real-validation\\.venv\\Scripts\\python scripts\\compare_remeasure.py --run-id <새 run-id> --baseline-run <기준 run-id> --dummy-version v1

입력 (모두 읽기만 한다)
    runs/<baseline-run>/booking_intervals.csv, macro_runs.csv     기존 실측
    runs/<run-id>/booking_intervals.csv, macro_runs.csv, raw_events.csv, macro_stdout.log   새 실측
    dummy_adjusted/<version>/fairticket_synthetic_features_adjusted.csv, generation.json     보정 더미

출력: runs/<run-id>/dummy_comparison/ (CSV, figures/, REPORT.md). 다시 실행하면 이 폴더만 새로 쓴다.
더미데이터를 다시 만들거나 모델을 학습하지 않는다.

예매 간격 = 같은 계정의 연속한 예매 시도(BOOKING_ATTEMPT) 시작 시각의 차이(초). 예매 1건에 걸린 시간이 아니다.
계정 Feature: avg_booking_interval = 간격의 평균, booking_interval_variance = 간격의 모분산 (BTS 계산식).
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
MEAN_COL, VAR_COL = "avg_booking_interval", "booking_interval_variance"
QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
C_BASE, C_NEW, C_DUMMY = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#1f2328", "#6b7280", "#e5e7eb", "#ffffff"
L_BASE, L_NEW, L_DUMMY = "기존 실측", "새 실측", "보정 더미"
SEGMENTS = [("click_to_hold_s", "예매하기 클릭 → 좌석 선점"), ("payment_step_s", "결제 단계"),
            ("after_paid_to_next_begin_s", "결제 완료 → 다음 예매하기 클릭")]


def load_intervals(run_id, types):
    path = EXPERIMENT_DIR / "runs" / run_id / "booking_intervals.csv"
    if not path.exists():
        sys.exit(f"파일이 없습니다: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame[(frame["dataset"] == run_id) & frame["mode"].isin(types)].copy()


def describe(values):
    values = np.asarray(values, dtype=float)
    row = {"n": len(values), "mean": values.mean(), "median": np.median(values),
           "std": values.std(ddof=1) if len(values) > 1 else np.nan, "variance_population": values.var(ddof=0),
           "min": values.min()}
    for q in QUANTILES:
        row[f"q{int(q * 100):02d}"] = np.quantile(values, q)
    row["max"] = values.max()
    return row


def run_outcomes(run_id):
    """매크로 실행 결과 건수와 로그의 HTTP 409 흔적."""
    run_dir = EXPERIMENT_DIR / "runs" / run_id
    runs = pd.read_csv(run_dir / "macro_runs.csv", encoding="utf-8-sig")
    table = runs.groupby(["account", "mode"])["result"].value_counts().unstack(fill_value=0).reset_index()
    table.insert(2, "attempted", runs.groupby(["account", "mode"]).size().to_numpy())
    if "OK" not in table:
        table["OK"] = 0
    table["failed_or_not_run"] = table["attempted"] - table["OK"]
    tries = pd.to_numeric(runs.get("hold_tries"), errors="coerce")
    table["hold_retries"] = (tries - 1).clip(lower=0).groupby([runs["account"], runs["mode"]]).sum().to_numpy()
    log_path = run_dir / "macro_stdout.log"
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    errors = runs["error"].astype("string").fillna("")
    info = {
        "browser_error_lines": len(re.findall(r"! 브라우저 오류|! 페이지 오류", log)),
        "http_409_lines": len(re.findall(r"status of 409|\b409\b.*(Conflict|충돌)|충돌합니다", log)),
        "error_text_in_results": int(errors.str.strip().ne("").sum()),
        "conflict_text_in_results": int(errors.str.contains("409|충돌|CONFLICT").sum()),
    }
    return table, info


def stage_patterns(run_id):
    """예매 시도 1건이 BTS 에 남긴 단계 순서별 건수 (매크로 계정만)."""
    path = EXPERIMENT_DIR / "runs" / run_id / "raw_events.csv"
    if not path.exists():
        return {}
    events = pd.read_csv(path, encoding="utf-8-sig")
    events = events[(events["event_type"] == "BOOKING_ATTEMPT") & events["group"].astype(str).str.startswith("macro")]
    meta = events["metadata"].map(json.loads)
    frame = pd.DataFrame({"attempt": meta.map(lambda m: m.get("attemptId")), "stage": meta.map(lambda m: m.get("stage")),
                          "sequence": meta.map(lambda m: m.get("sequence", 0))}).sort_values(["attempt", "sequence"])
    return frame.groupby("attempt")["stage"].agg(" → ".join).value_counts().to_dict()


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


def figure_intervals(path, base, new, dummy, types):
    """유형별: 계정마다 개별 간격을 점으로, 계정 평균을 가로선으로. 더미 계정 평균의 범위를 띠로."""
    rng = np.random.default_rng(0)  # 점이 겹치지 않게 좌우로 흩는 용도 (값에는 영향 없음)
    fig, axes = plt.subplots(1, len(types), figsize=(5.6 * len(types), 5), constrained_layout=True)
    for ax, name in zip(np.atleast_1d(axes), types):
        style(ax)
        d = dummy.loc[dummy["scenario"] == name, MEAN_COL]
        ax.axhspan(d.min(), d.max(), color=C_DUMMY, alpha=0.22, linewidth=0)
        ax.axhline(d.mean(), color=C_DUMMY, linewidth=1.5)
        ticks, labels = [], []
        position = 0
        for frame, color, label in ((base, C_BASE, L_BASE), (new, C_NEW, L_NEW)):
            for account, part in frame[frame["mode"] == name].groupby("account"):
                values = part["gap_seconds"].to_numpy()
                ax.scatter(position + rng.uniform(-0.16, 0.16, len(values)), values, s=26, color=color,
                           edgecolor=SURFACE, linewidth=0.8, zorder=3)
                ax.hlines(values.mean(), position - 0.3, position + 0.3, color=INK, linewidth=1.5, zorder=4)
                ax.text(position + 0.34, values.mean(), f"{values.mean():.2f}", va="center", fontsize=8, color=INK)
                ticks.append(position)
                labels.append(f"{label}\n{account}\n(간격 {len(values)}개)")
                position += 1
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-0.6, position - 0.2)
        ax.set_ylabel("예매 간격 (초)", fontsize=9, color=MUTED)
        ax.set_title(f"{name}:  초록 띠 = {L_DUMMY} 계정 평균의 범위 {d.min():.2f}~{d.max():.2f}초 (n={len(d)})",
                     loc="left", fontsize=10, color=INK)
    fig.suptitle("개별 예매 간격과 계정 평균 (검은 가로선 = 계정 평균)", fontsize=12, color=INK)
    save(fig, path)


def figure_features(path, accounts, dummy, types):
    """계정 Feature 평면: 더미 계정(점)과 실측 계정(세모)."""
    fig, axes = plt.subplots(1, len(types), figsize=(5.6 * len(types), 4.8), constrained_layout=True)
    for ax, name in zip(np.atleast_1d(axes), types):
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        d = dummy[dummy["scenario"] == name]
        ax.scatter(d[MEAN_COL], d[VAR_COL], s=30, color=C_DUMMY, edgecolor=SURFACE, linewidth=0.8,
                   label=f"{L_DUMMY} {name} 계정 (n={len(d)})")
        for source, color, label in (("base", C_BASE, L_BASE), ("new", C_NEW, L_NEW)):
            part = accounts[(accounts["mode"] == name) & (accounts["source"] == source)]
            ax.scatter(part[MEAN_COL], part[VAR_COL], s=110, marker="^", color=color, edgecolor=SURFACE, linewidth=1,
                       label=f"{label} 계정", zorder=4)
            for _, row in part.iterrows():
                ax.annotate(f"{row['account']} ({int(row['n'])})", (row[MEAN_COL], row[VAR_COL]), xytext=(8, 4),
                            textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_ylim(bottom=-0.01)
        ax.margins(x=0.18)
        ax.set_xlabel(f"{MEAN_COL} (초)", fontsize=9, color=MUTED)
        ax.set_ylabel(f"{VAR_COL} (초²)", fontsize=9, color=MUTED)
        ax.set_title(name, loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("계정별 Feature: 평균 예매 간격과 분산 (괄호 = 간격 수)", fontsize=12, color=INK)
    save(fig, path)


def figure_segments(path, segments, types):
    """간격을 이루는 세 구간의 중앙값: 기존 실측과 새 실측."""
    fig, axes = plt.subplots(1, len(types), figsize=(5.6 * len(types), 4.2), constrained_layout=True)
    width = 0.36
    for ax, name in zip(np.atleast_1d(axes), types):
        style(ax)
        part = segments[segments["mode"] == name]
        x = np.arange(len(part))
        for offset, col, color, label in ((-width / 2, "median_base", C_BASE, L_BASE), (width / 2, "median_new", C_NEW, L_NEW)):
            bars = ax.bar(x + offset, part[col], width=width - 0.03, color=color, label=label)
            for bar, value in zip(bars, part[col]):
                ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace(" → ", " →\n") for s in part["segment_label"]], fontsize=8)
        ax.set_ylabel("중앙값 (초)", fontsize=9, color=MUTED)
        ax.set_title(name, loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.set_ylim(0, part[["median_base", "median_new"]].to_numpy().max() * 1.25)
    fig.suptitle("예매 간격을 이루는 구간별 소요 시간", fontsize=12, color=INK)
    save(fig, path)


def table(frame, columns, digits=3):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append("—" if np.isnan(value) else f"{value:.1e}" if col.endswith("_p") else f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="새 실측 실행")
    parser.add_argument("--baseline-run", default="20261122-continuous", help="기존 실측 실행")
    parser.add_argument("--dummy-version", default="v1")
    args = parser.parse_args()

    dummy_dir = EXPERIMENT_DIR / "dummy_adjusted" / args.dummy_version
    generation = json.loads((dummy_dir / "generation.json").read_text(encoding="utf-8"))
    types = list(generation["split"])
    dummy = pd.read_csv(dummy_dir / "fairticket_synthetic_features_adjusted.csv")
    dummy = dummy[dummy["scenario"].isin(types)]
    base, new = load_intervals(args.baseline_run, types), load_intervals(args.run_id, types)
    for name in types:
        if not (new["mode"] == name).any():
            sys.exit(f"새 실측에 {name} 간격이 없습니다. 값을 지어내지 않고 중단합니다.")

    out = EXPERIMENT_DIR / "runs" / args.run_id / "dummy_comparison"
    if out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 1) 계정별 통계 (Feature 포함)
    rows = []
    for source, run_id, frame in (("base", args.baseline_run, base), ("new", args.run_id, new)):
        for (mode, account), part in frame.groupby(["mode", "account"]):
            stats = describe(part["gap_seconds"])
            rows.append({"source": source, "run_id": run_id, "mode": mode, "account": account, **stats,
                         MEAN_COL: stats["mean"], VAR_COL: stats["variance_population"]})
    accounts = pd.DataFrame(rows)
    for name in types:
        d = dummy[dummy["scenario"] == name]
        mask = accounts["mode"] == name
        accounts.loc[mask, "mean_within_dummy_range"] = accounts.loc[mask, MEAN_COL].between(d[MEAN_COL].min(), d[MEAN_COL].max())
        accounts.loc[mask, "variance_within_dummy_range"] = accounts.loc[mask, VAR_COL].between(d[VAR_COL].min(), d[VAR_COL].max())
        accounts.loc[mask, "mean_minus_dummy_mean"] = accounts.loc[mask, MEAN_COL] - d[MEAN_COL].mean()
        accounts.loc[mask, "variance_minus_dummy_median"] = accounts.loc[mask, VAR_COL] - d[VAR_COL].median()
    accounts.to_csv(out / "account_stats.csv", index=False, encoding="utf-8-sig")

    # 2) 유형별 통계: 개별 간격(기존·새) + 더미 계정 Feature
    rows = []
    for name in types:
        rows.append({"mode": name, "group": f"{L_BASE} (개별 간격)", "unit": "초", **describe(base.loc[base["mode"] == name, "gap_seconds"])})
        rows.append({"mode": name, "group": f"{L_NEW} (개별 간격)", "unit": "초", **describe(new.loc[new["mode"] == name, "gap_seconds"])})
        d = dummy[dummy["scenario"] == name]
        rows.append({"mode": name, "group": f"{L_DUMMY} (계정 평균)", "unit": "초", **describe(d[MEAN_COL])})
        rows.append({"mode": name, "group": f"{L_DUMMY} (계정 분산)", "unit": "초²", **describe(d[VAR_COL])})
    type_stats = pd.DataFrame(rows)
    type_stats.to_csv(out / "type_stats.csv", index=False, encoding="utf-8-sig")

    # 3) 유형별 요약: 기존 대 새 실측, 새 실측 대 더미
    rows = []
    for name in types:
        b, n = base.loc[base["mode"] == name, "gap_seconds"].to_numpy(), new.loc[new["mode"] == name, "gap_seconds"].to_numpy()
        d = dummy[dummy["scenario"] == name]
        new_accounts = accounts[(accounts["mode"] == name) & (accounts["source"] == "new")]
        base_accounts = accounts[(accounts["mode"] == name) & (accounts["source"] == "base")]
        rows.append({
            "mode": name, "n_base": len(b), "n_new": len(n),
            "mean_base": b.mean(), "mean_new": n.mean(), "mean_diff": n.mean() - b.mean(),
            "mean_diff_pct": (n.mean() - b.mean()) / b.mean() * 100,
            "median_base": np.median(b), "median_new": np.median(n), "median_diff": np.median(n) - np.median(b),
            "std_base": b.std(ddof=1), "std_new": n.std(ddof=1),
            "min_new_vs_max_base": n.min() - b.max(),
            "share_new_above_base_max": float((n > b.max()).mean()),
            "mannwhitney_p": sps.mannwhitneyu(n, b, alternative="two-sided").pvalue,
            "ks_statistic": sps.ks_2samp(n, b).statistic, "wasserstein_s": sps.wasserstein_distance(n, b),
            "account_means_base": ", ".join(f"{v:.3f}" for v in base_accounts[MEAN_COL]),
            "account_means_new": ", ".join(f"{v:.3f}" for v in new_accounts[MEAN_COL]),
            "account_variances_base": ", ".join(f"{v:.3f}" for v in base_accounts[VAR_COL]),
            "account_variances_new": ", ".join(f"{v:.3f}" for v in new_accounts[VAR_COL]),
            "dummy_mean_avg": d[MEAN_COL].mean(), "dummy_mean_min": d[MEAN_COL].min(), "dummy_mean_max": d[MEAN_COL].max(),
            "dummy_variance_median": d[VAR_COL].median(), "dummy_variance_min": d[VAR_COL].min(), "dummy_variance_max": d[VAR_COL].max(),
            "new_mean_minus_dummy_mean": n.mean() - d[MEAN_COL].mean(),
            "new_account_variance_avg_minus_dummy_median": new_accounts[VAR_COL].mean() - d[VAR_COL].median(),
            "new_accounts_mean_within_dummy_range": f"{int(new_accounts['mean_within_dummy_range'].sum())}/{len(new_accounts)}",
            "new_accounts_variance_within_dummy_range": f"{int(new_accounts['variance_within_dummy_range'].sum())}/{len(new_accounts)}",
            "share_dummy_accounts_within_new_interval_range": float(d[MEAN_COL].between(n.min(), n.max()).mean()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary_by_type.csv", index=False, encoding="utf-8-sig")

    # 4) 간격을 이루는 구간
    rows = []
    for name in types:
        for col, label in SEGMENTS:
            b, n = base.loc[base["mode"] == name, col].dropna(), new.loc[new["mode"] == name, col].dropna()
            rows.append({"mode": name, "segment": col, "segment_label": label, "median_base": b.median(),
                         "median_new": n.median(), "median_diff": n.median() - b.median()})
    segments = pd.DataFrame(rows)
    segments.to_csv(out / "segment_comparison.csv", index=False, encoding="utf-8-sig")

    outcomes, log_info = run_outcomes(args.run_id)
    outcomes.to_csv(out / "run_outcomes.csv", index=False, encoding="utf-8-sig")
    patterns = {"base": stage_patterns(args.baseline_run), "new": stage_patterns(args.run_id)}
    env_new = json.loads((EXPERIMENT_DIR / "runs" / args.run_id / "environment.json").read_text(encoding="utf-8"))
    env_base = json.loads((EXPERIMENT_DIR / "runs" / args.baseline_run / "environment.json").read_text(encoding="utf-8"))
    commit = lambda env, repo: env.get("repos", {}).get(repo, {}).get("commit", "?")[:7]

    figure_intervals(out / "figures" / "intervals_by_account.png", base, new, dummy, types)
    figure_features(out / "figures" / "account_features_vs_dummy.png", accounts, dummy, types)
    figure_segments(out / "figures" / "segment_comparison.png", segments, types)

    # 재보정 판단: 새 실측 계정의 Feature 가 더미 계정의 범위 안에 있는가 (규칙은 보고서에 그대로 적는다)
    new_accounts = accounts[accounts["source"] == "new"]
    mean_in, var_in = int(new_accounts["mean_within_dummy_range"].sum()), int(new_accounts["variance_within_dummy_range"].sum())
    total = len(new_accounts)
    plan = env_new.get("plan", [])
    stat_cols = ["mode", "group", "unit", "n", "mean", "median", "std", "variance_population", "min", "q05", "q25", "q75", "q95", "max"]
    account_cols = ["source", "mode", "account", "n", "mean", "median", "std", "variance_population", "min", "q05", "q25", "q75", "q95", "max"]
    labelled = accounts.assign(source=accounts["source"].map({"base": L_BASE, "new": L_NEW}))
    lines = [
        f"# 재측정 비교: {args.baseline_run} · {args.run_id} · 보정 더미 {args.dummy_version}",
        "",
        "자동 생성 (`scripts/compare_remeasure.py`). 더미데이터 재생성과 모델 학습은 하지 않았다.",
        "",
        "## 실행 조건",
        "",
        "| | 기존 실측 | 새 실측 |",
        "|---|---|---|",
        f"| run-id | `{args.baseline_run}` | `{args.run_id}` |",
        f"| demo-ticketing 커밋 | `{commit(env_base, 'demo-ticketing')}` | `{commit(env_new, 'demo-ticketing')}` |",
        f"| ticket-fds-admin 커밋 | `{commit(env_base, 'ticket-fds-admin')}` | `{commit(env_new, 'ticket-fds-admin')}` |",
        f"| 계획 | {', '.join(str(p['account']) + '×' + str(p['runs']) for p in env_base.get('plan', []))} | {', '.join(str(p['account']) + '×' + str(p['runs']) for p in plan)} |",
        f"| macro.js `--debug` | {'예' if env_base.get('macro_debug') else '아니오'} | {'예' if env_new.get('macro_debug') else '아니오'} |",
        f"| 예매 시도 1건의 단계 기록 | {'; '.join(f'{k} ({v}건)' for k, v in patterns['base'].items())} | {'; '.join(f'{k} ({v}건)' for k, v in patterns['new'].items())} |",
        "",
        "## 성공·실패",
        "",
        table(outcomes, list(outcomes.columns)),
        "",
        f"- 결과 CSV 에 오류 문구가 있는 예매: {log_info['error_text_in_results']}건 (그중 409·충돌: {log_info['conflict_text_in_results']}건)",
        f"- 매크로 로그의 브라우저 오류 줄: {log_info['browser_error_lines']}개, HTTP 409 줄: {log_info['http_409_lines']}개"
        + ("" if env_new.get("macro_debug") else " (`--debug` 없이 수집해 브라우저 오류는 로그에 남지 않는다)"),
        "- `hold_retries`: 좌석 선점을 다시 시도한 횟수",
        "",
        "## 계정별 통계 (개별 간격, 초)",
        "",
        table(labelled, account_cols),
        "",
        f"계정 Feature 는 `{MEAN_COL}` = mean, `{VAR_COL}` = variance_population 이다. std 는 표본 표준편차.",
        "",
        "## 기존 실측 대 새 실측",
        "",
        "![개별 간격](figures/intervals_by_account.png)",
        "",
        table(summary, ["mode", "n_base", "n_new", "mean_base", "mean_new", "mean_diff", "mean_diff_pct", "median_base",
                        "median_new", "median_diff", "std_base", "std_new", "share_new_above_base_max", "mannwhitney_p",
                        "wasserstein_s"]),
        "",
        "- `mean_diff_pct`: 기존 대비 변화율(%). `share_new_above_base_max`: 새 간격 중 기존 최댓값보다 큰 비율.",
        "- `mannwhitney_p`: 두 실행의 간격이 같은 분포에서 나왔다고 볼 때 이만큼의 차이가 날 확률. "
        "같은 계정 안의 간격은 서로 독립이 아니므로 참고용이다.",
        "",
        "### 어느 구간이 달라졌나",
        "",
        "![구간](figures/segment_comparison.png)",
        "",
        table(segments, ["mode", "segment_label", "median_base", "median_new", "median_diff"]),
        "",
        "세 구간의 합이 예매 간격이다. 변화의 원인은 이 데이터만으로는 가려낼 수 없다. 두 실행 사이에 달라진 조건은 "
        "실행 조건 표의 커밋 차이(좌석 수, 좌석 배치도 UI, 연속 예매 수정으로 단계 기록이 1개 늘어남), `--debug` 사용 여부, "
        "매크로의 sessionStorage 정리 제거, 수집 시각과 PC 상태이며, 한 번씩만 측정했다.",
        "",
        "## 새 실측 대 보정 더미",
        "",
        "![Feature](figures/account_features_vs_dummy.png)",
        "",
        table(type_stats, stat_cols),
        "",
        table(summary, ["mode", "account_means_base", "account_means_new", "dummy_mean_avg", "dummy_mean_min", "dummy_mean_max",
                        "new_mean_minus_dummy_mean", "new_accounts_mean_within_dummy_range"]),
        "",
        table(summary, ["mode", "account_variances_base", "account_variances_new", "dummy_variance_median", "dummy_variance_min",
                        "dummy_variance_max", "new_account_variance_avg_minus_dummy_median", "new_accounts_variance_within_dummy_range"]),
        "",
        "- `..._within_dummy_range`: 새 실측 계정 중 Feature 값이 보정 더미 계정의 최소~최대 안에 드는 계정 수.",
        "- `share_dummy_accounts_within_new_interval_range` (summary_by_type.csv): 더미 계정 평균 중 새 실측 개별 간격의 범위 안에 드는 비율 — "
        + ", ".join(f"{r['mode']} {r['share_dummy_accounts_within_new_interval_range'] * 100:.0f}%" for _, r in summary.iterrows()) + ".",
        "",
        "## 재보정 판단",
        "",
        f"- 새 실측 계정 {total}개 중 평균이 더미 범위 안에 드는 계정: **{mean_in}개**, 분산이 더미 범위 안에 드는 계정: **{var_in}개**.",
        "- 유형별 평균 차이(새 실측 − 더미): "
        + ", ".join(f"{r['mode']} {r['new_mean_minus_dummy_mean']:+.3f}초" for _, r in summary.iterrows()) + ".",
        "- 같은 유형의 새 실측 계정끼리의 평균 차이: "
        + ", ".join(f"{name} {new_accounts.loc[new_accounts['mode'] == name, MEAN_COL].max() - new_accounts.loc[new_accounts['mode'] == name, MEAN_COL].min():.3f}초"
                    for name in types)
        + f". 보정 더미 {args.dummy_version} 의 같은 유형 계정 평균 범위 폭: "
        + ", ".join(f"{r['mode']} {r['dummy_mean_max'] - r['dummy_mean_min']:.3f}초" for _, r in summary.iterrows()) + ".",
        "",
        ("새 실측 계정의 평균이 하나도 더미 범위 안에 들지 않으므로, 보정 더미는 **현재 사이트의 매크로 간격을 그대로 반영하지는 못한다.** "
         if mean_in == 0 else
         f"새 실측 계정 {total}개 중 {mean_in}개의 평균이 더미 범위 안에 든다. ")
        + "더미의 계정 평균 범위가 좁은 것은 기존 실측 1회(유형당 계정 1~2개)의 간격만으로 만들었기 때문이고, "
        "이번 재측정으로 실행·계정 사이의 차이가 그 범위보다 크다는 것이 확인됐다.",
        "",
        "재보정을 한다면 한 실행의 값으로 옮기기보다, 두 실행의 간격을 함께 써서 계정 사이의 차이를 넣는 쪽이 근거가 있다 "
        "(`make_adjusted_dummy.py --run-id " + args.baseline_run + " --run-id " + args.run_id + " --version v2`). "
        "이 차이가 모델 판정에 영향을 주는지는 이번에 확인하지 않았다.",
        "",
        "## 한계",
        "",
        "- 두 실행 모두 같은 PC 에서 한 번씩 수집했다. 실행 사이의 차이와 사이트 변경의 효과를 분리할 수 없다.",
        "- 계정 수가 유형당 1~2개(기존), 2개(새)라 계정 사이의 분포를 추정하기에는 적다.",
        "- 기존 실행은 `--debug` 없이, 새 실행은 `--debug` 로 수집했다 (단계와 브라우저 오류를 로그에 남기는 옵션).",
        "",
        "## 다시 만들기",
        "",
        "```powershell",
        f"& $py scripts\\compare_remeasure.py --run-id {args.run_id} --baseline-run {args.baseline_run} --dummy-version {args.dummy_version}",
        "```",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(outcomes.to_string(index=False))
    print(log_info, patterns)
    print(accounts[["source", "mode", "account", "n", "mean", "median", "std", "variance_population", "min", "max",
                    "mean_within_dummy_range", "variance_within_dummy_range"]].round(3).to_string(index=False))
    print(summary.round(4).T.to_string())
    print(segments.round(3).to_string(index=False))
    print(f"보고서: {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
