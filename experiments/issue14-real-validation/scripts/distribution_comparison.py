"""더미데이터와 실측 데이터의 Feature 분포를 비교한다 (실험용, 오프라인 추가 분석).

목적은 모델의 탐지 성능 평가가 아니라, 학습용 더미데이터가 실제 사용자·매크로 행동을
얼마나 현실적으로 반영하는지 확인하는 것이다. 모델·점수는 이 스크립트에서 쓰지 않는다.

실행 (실험 폴더에서):
    .venv/Scripts/python scripts/distribution_comparison.py
    .venv/Scripts/python scripts/distribution_comparison.py --out-tag distribution_comparison_rerun

입력 (읽기 전용)
    model_build/ml_experiment/data/fairticket_synthetic_features.csv   학습에 쓴 더미 (계정 단위 Feature)
    model_build/BUILD_RECORD.json                                      학습 데이터 해시 기록
    data/count/real_features.csv                                       실측 6계정 Feature (BTS 계산식 재현)
    data/count/events.csv, attempt_intervals.csv                       개별 간격 계산
    data/count/macro_gap_breakdown__<계정>.csv                         브라우저 재기동 시간 분해

출력: output/<out-tag>/  (이미 있으면 덮어쓰지 않고 멈춘다)
    data_verification.json           더미·실측 파일 검증 기록
    feature_stats.csv                Feature × 집단 요약 통계
    distance_metrics.csv             KS 통계량, Wasserstein Distance (정상끼리, 매크로끼리)
    real_positions.csv               실측 계정 값이 더미 분포의 어디에 있는지
    fig_feature_<feature>.png        Feature별 분포 비교 그래프 6개
    individual_intervals.csv         매크로 개별 간격 (이벤트 단위)
    interval_stats.csv               매크로별 개별 간격 통계 (원본 / 가상 시나리오)
    fig_macro_intervals.png          매크로별 개별 예매 간격 분포
    dummy_generation_distributions.csv  더미 생성 코드의 간격 관련 확률분포와 실제 생성값 범위
    realism_assessment.csv           Feature별 현실성 평가표

비교 기준
- 더미 정상  : is_anomaly == 0 (normal, fast_normal, shared_ip, frequent_cancel)  801건
- 더미 매크로: scenario == "macro"                                               87건
  (hold_cancel, multi_account 는 매크로 시나리오가 아니므로 그래프·거리 계산에서 빼고 통계표에만 참고로 남긴다)
- 실측 정상  : human-01, human-02
- 실측 매크로: macro-fast-01, macro-fast-02, macro-jitter-01, macro-human-01

주의
- 그래프의 jitter 는 세로 위치에만 준다. 가로축(Feature 값)은 원본 그대로다.
- 개별 간격(이벤트 단위)과 더미 avg_booking_interval(계정 단위 평균)은 집계 단위가 달라 같은 축에 그리지 않는다.
- "가상" 이라고 적힌 값은 관측값이 아니라 긴 간격을 빼거나 바꿔 다시 계산한 값이다.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.stats import ks_2samp, wasserstein_distance  # noqa: E402

EXP = Path(__file__).resolve().parent.parent
AI = EXP.parent.parent
DUMMY_PATH = EXP / "model_build" / "ml_experiment" / "data" / "fairticket_synthetic_features.csv"
REPO_DUMMY_PATH = AI / "ml_experiment" / "data" / "fairticket_synthetic_features.csv"
BUILD_RECORD = EXP / "model_build" / "BUILD_RECORD.json"
REAL_TAG = "count"

LONG_GAP_SECONDS = 20.0  # sensitivity_fast_macro.py 와 같은 기준
LONG_GAP_ACCOUNTS = ("macro-fast-01", "macro-fast-02")  # 긴 간격 기준은 fast 모드에만 적용 (나머지 간격이 5~8초)

# feature: (단위, 로그 축 여부, 환경 메모)
FEATURES = {
    "avg_booking_interval": ("초", True, ""),
    "booking_interval_variance": ("초²", True, ""),
    "cancel_rate": ("비율 (0~1)", False, "실험에서 취소를 하지 않아 실측값은 모두 0"),
    "card_reuse_count": ("같은 결제 식별자를 쓴 계정 수", False, "환경 영향: 데모는 테스트 결제 식별자 3개를 전 계정이 공유"),
    "ip_booking_density": ("같은 IP로 예매한 계정 수", False, "환경 영향: 모든 계정이 로컬 IP(::1) 하나를 공유"),
    "booking_count": ("예매 시도 수 (회)", False, "실측 횟수는 실험자가 정한 값(3회, 20회)"),
}

# 실측 계정: (집단, 표시 이름, 색, 마커)
ACCOUNTS = {
    "human-01": ("human", "실측 사람 human-01", "#008300", "D"),
    "human-02": ("human", "실측 사람 human-02", "#008300", "s"),
    "macro-fast-01": ("macro", "실측 매크로 Fast-01 (3회)", "#6250d6", "^"),
    "macro-fast-02": ("macro", "실측 매크로 Fast-02 (20회)", "#6250d6", "v"),
    "macro-jitter-01": ("macro", "실측 매크로 Jitter-01 (20회)", "#e87ba4", "P"),
    "macro-human-01": ("macro", "실측 매크로 Human-like-01 (20회)", "#eda100", "X"),
}
MODE_LABEL = {"macro-fast-01": "Fast-01", "macro-fast-02": "Fast-02", "macro-jitter-01": "Jitter-01",
              "macro-human-01": "Human-like-01"}

C_NORMAL, C_MACRO, C_ALERT = "#2a78d6", "#eb6834", "#e34948"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# generate_dummy_data.py 의 시나리오별 생성식 (읽고 옮겨 적은 것. 생성 코드는 실행·수정하지 않는다)
GENERATION_SPEC = {
    "normal": ("lognormal(mean=4.1, sigma=0.5)", "uniform(5.0, 50.0)", "poisson(3) + 1"),
    "fast_normal": ("uniform(15.0, 40.0)", "uniform(3.0, 20.0)", "poisson(5) + 1"),
    "shared_ip": ("lognormal(mean=3.8, sigma=0.5)", "uniform(4.0, 30.0)", "poisson(4) + 1"),
    "frequent_cancel": ("lognormal(mean=4.0, sigma=0.5)", "uniform(5.0, 40.0)", "poisson(4) + 1"),
    "macro": ("exponential(scale=0.2) + 0.1", "uniform(0.01, 0.5)", "randint(15, 50)"),
    "hold_cancel": ("uniform(2.0, 8.0)", "uniform(0.5, 2.0)", "randint(10, 30)"),
    "multi_account": ("uniform(3.0, 10.0)", "uniform(1.0, 5.0)", "randint(10, 25)"),
}

# 현실성 평가 (분석 결과를 보고 사람이 내린 판단. 근거 수치는 실행 시 계산해 함께 적는다)
ASSESSMENT = {
    "avg_booking_interval": (
        "차이 확인 (매크로) / 검증 제한 (정상)",
        "실측 매크로 4계정이 모두 더미 매크로 최댓값보다 크고, 더미 정상 범위 쪽에 놓인다. 사람은 1명이 더미 정상 IQR 안, 1명이 더미 정상 최솟값 아래다.",
        "더미 매크로는 요청 간격(0.1~1.5초)을 가정했지만, 실측 간격은 한 번의 예매 전체(로그인→좌석→결제 1.2초 대기)와 브라우저 재기동 시간을 포함한 '시도 시작 간격'이다. 사람 2명은 표본이 작고 연속 예매를 의도적으로 수행했다.",
        "매크로 간격의 하한을 '예매 1건을 끝내는 데 걸리는 시간'(수 초 이상)으로 올리고, 정상과 겹치는 구간을 허용한다. 평균 간격만으로 매크로가 분리된다는 가정을 버린다.",
    ),
    "booking_interval_variance": (
        "차이 확인",
        "더미는 집단별로 좁은 균등분포(정상 3~50, 매크로 0.01~0.5)인데, 실측은 사람 0.05와 53.6, 매크로 0.07~52.6으로 집단 구분과 무관하게 넓게 퍼진다.",
        "더미는 분산을 평균 간격과 독립으로 뽑아, 평균 60초에 표준편차 2~7초 같은 조합이 나온다. 실측 분산은 시도 수가 적으면(간격 2개) 불안정하고, 긴 간격 하나에 크게 좌우된다(Fast-02: 52.6, 가상 대체 시 0.39).",
        "분산을 따로 뽑지 말고 개별 간격을 먼저 생성한 뒤 평균·분산을 계산한다. 그러면 평균과 분산의 관계, 시도 수에 따른 불안정성, 드문 긴 간격이 함께 반영된다.",
    ),
    "cancel_rate": (
        "검증 제한",
        "실측 6계정 모두 0이다. 더미 매크로는 0.1~0.4로 0이 한 건도 없고, 더미 정상은 0 근처에 몰려 있다.",
        "이번 실험은 취소를 수행하지 않았다. 취소 행동에 대한 관측이 없어 더미의 현실성을 판단할 수 없다.",
        "이번 결과로는 개선 방향을 정할 수 없다. 다만 '취소하지 않는 매크로'가 더미에 없다는 점은 생성 로직의 가정으로 기록해 둔다.",
    ),
    "card_reuse_count": (
        "검증 제한",
        "실측 6계정 모두 6이다(더미 정상은 전부 1, 더미 매크로는 1~2).",
        "데모 환경이 테스트 결제 식별자 3개만 허용해 모든 계정이 같은 식별자를 공유했다. 실제 카드 사용 행태가 아니다.",
        "판단 보류. 실제 결제 식별자가 계정마다 다른 환경에서 다시 수집해야 한다.",
    ),
    "ip_booking_density": (
        "검증 제한",
        "실측 6계정 모두 6이다(더미 정상 1~6, 더미 매크로 1~3).",
        "모든 예매가 한 PC의 로컬 IP(::1)에서 발생했다. 실제 네트워크 분포가 아니다.",
        "판단 보류. 서로 다른 네트워크에서 접속한 데이터가 필요하다.",
    ),
    "booking_count": (
        "검증 제한",
        "실측은 3회와 20회 두 값뿐이다. 20회는 더미 매크로 범위(15~49) 안, 3회는 더미 정상 범위 안이다.",
        "예매 횟수는 실험자가 정한 값이라 실제 사용자·매크로의 횟수 분포를 대표하지 않는다.",
        "이번 결과로는 개선 방향을 정할 수 없다. 다만 더미 매크로의 하한이 15회라, 소수 회만 시도하는 매크로는 더미에 없다.",
    ),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    s = pd.Series(values, dtype=float).dropna()
    if s.empty:
        return dict(n=0)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return dict(n=len(s), mean=s.mean(), median=s.median(), min=s.min(), max=s.max(), q1=q1, q3=q3, iqr=q3 - q1,
                std=s.std(ddof=1) if len(s) > 1 else np.nan)


def plain(value, _pos=None):
    return f"{value:g}"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def feature_figure(feature, dummy_groups, real, out_path, rng):
    unit, log, note = FEATURES[feature]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(plain))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    rows = {"더미 정상": 3, "더미 매크로": 2, "실측 정상": 1, "실측 매크로": 0}
    for (label, values), color in zip(dummy_groups.items(), (C_NORMAL, C_MACRO)):
        y0 = rows[label]
        # jitter 는 세로 위치에만 적용한다 (가로축 값은 원본)
        ax.scatter(values, y0 + rng.uniform(-0.3, 0.3, len(values)), s=11, color=color, alpha=0.4, linewidths=0, zorder=2)
        q1, med, q3 = np.percentile(values, [25, 50, 75])
        ax.add_patch(Rectangle((q1, y0 - 0.38), q3 - q1, 0.76, fill=False, edgecolor=INK, linewidth=1.3, zorder=3))
        ax.plot([med, med], [y0 - 0.38, y0 + 0.38], color=INK, linewidth=2.2, zorder=4)

    # 실측: 값이 겹쳐도 보이도록 계정마다 세로 위치를 다르게 둔다
    all_values = np.concatenate(list(dummy_groups.values()) + [real[feature].to_numpy(dtype=float)])
    lo, hi = all_values.min(), all_values.max()
    if log:
        ax.set_xlim(lo / 1.6, hi * 1.6)
    else:
        pad = (hi - lo) * 0.05 or 0.5
        ax.set_xlim(lo - pad, hi + pad)
    for group, row_label in (("human", "실측 정상"), ("macro", "실측 매크로")):
        names = [a for a in ACCOUNTS if ACCOUNTS[a][0] == group]
        offsets = np.linspace(0.3, -0.3, len(names)) if len(names) > 1 else [0.0]
        for name, dy in zip(names, offsets):
            _, _, color, marker = ACCOUNTS[name]
            value = float(real.loc[name, feature])
            y = rows[row_label] + dy
            ax.scatter([value], [y], s=120, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.5, zorder=5)
            frac = ax.transAxes.inverted().transform(ax.transData.transform((value, y)))[0]
            right = frac < 0.72
            short = MODE_LABEL.get(name, name)
            ax.annotate(f"{short}: {value:.4g}", (value, y), xytext=(9 if right else -9, 0), textcoords="offset points",
                        ha="left" if right else "right", va="center", fontsize=8.5, color=INK, zorder=6)

    ax.set_ylim(-0.6, 3.6)
    ax.set_yticks(list(rows.values()))
    ax.set_yticklabels([f"{label}\n(n={len(dummy_groups[label])})" if label in dummy_groups
                        else f"{label}\n(n={2 if '정상' in label else 4})" for label in rows], color=INK, fontsize=10)
    ax.axhline(1.5, color=GRID, linewidth=1.2)
    ax.set_xlabel(f"{feature} · {unit}" + (" · 로그 축(값은 원래 단위)" if log else " · 선형 축"), color=INK_2, fontsize=10)

    fig.suptitle(f"{feature}: 더미데이터와 실측 계정 비교", x=0.012, y=0.985, ha="left", fontsize=14, color=INK, fontweight="bold")
    subtitle = "점 하나 = 계정 하나. 더미의 세로 흩뿌림은 겹침을 줄이기 위한 것이며 가로축 값은 원본 그대로."
    if note:
        subtitle += f"\n{note}"
    fig.text(0.012, 0.925 if note else 0.94, subtitle, ha="left", va="top", fontsize=9, color=INK_2)

    handles = [
        Line2D([], [], linestyle="", marker="o", markersize=5, color=C_NORMAL, alpha=0.6, label="더미 정상 (계정별)"),
        Line2D([], [], linestyle="", marker="o", markersize=5, color=C_MACRO, alpha=0.6, label="더미 매크로 (계정별)"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor=INK, linewidth=1.3, label="더미 IQR (Q1~Q3)"),
        Line2D([], [], color=INK, linewidth=2.2, label="더미 중앙값"),
    ] + [Line2D([], [], linestyle="", marker=m, markersize=9, color=c, markeredgecolor=SURFACE, label=lab)
         for _, lab, c, m in ACCOUNTS.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.115, right=0.975, top=0.855, bottom=0.215)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def ecdf(values):
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, len(x) + 1) / len(x)


def interval_figure(series, long_gap, overhead, out_path, rng):
    """series: {행 이름: (값 배열, 색, 가상 여부)} — 이벤트 단위 완료 시각 간격."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 6.0), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    style(ax2)
    ax2.grid(axis="y", color=GRID, linewidth=0.8)

    labels = list(series)
    for i, label in enumerate(labels):
        values, color, hypothetical = series[label]
        y0 = len(labels) - 1 - i
        ys = y0 + rng.uniform(-0.22, 0.22, len(values))
        if hypothetical:
            ax.scatter(values, ys, s=34, facecolor=SURFACE, edgecolor=color, linewidth=1.4, zorder=3)
        else:
            ax.scatter(values, ys, s=34, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        if len(values) >= 4:
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            ax.add_patch(Rectangle((q1, y0 - 0.34), q3 - q1, 0.68, fill=False, edgecolor=INK, linewidth=1.2,
                                   linestyle="--" if hypothetical else "-", zorder=2))
            ax.plot([med, med], [y0 - 0.34, y0 + 0.34], color=INK, linewidth=2, zorder=4)
        if label.startswith("Fast-02 원본") and long_gap is not None:
            y_long = ys[int(np.argmax(values))]
            ax.scatter([long_gap], [y_long], s=230, facecolor="none", edgecolor=C_ALERT, linewidth=2, zorder=5)
            ax.annotate(f"긴 간격 {long_gap:.1f}초\n(브라우저 종료·재기동 {overhead:.1f}초 포함)", (long_gap, y_long),
                        xytext=(0, 30), textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=INK,
                        arrowprops=dict(arrowstyle="-", color=C_ALERT, linewidth=1))
        style_line = (0, (5, 3)) if hypothetical else "-"
        if len(values) >= 4:
            x, y = ecdf(values)
            ax2.step(np.concatenate([[x[0]], x]), np.concatenate([[0], y]), where="post", color=color, linewidth=2,
                     linestyle=style_line, label=label.replace("\n", " "))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"{lab}\n간격 {len(series[lab][0])}개" for lab in reversed(labels)], color=INK, fontsize=9.5)
    ax.set_ylim(-0.6, len(labels) - 0.2)
    ax.set_xlim(0, None)
    ax.set_xlabel("연속한 예매 완료 시각 사이의 간격 · 초 · 선형 축", color=INK_2, fontsize=10)
    ax.set_title("개별 간격 (점 하나 = 간격 하나, 상자 = Q1~Q3, 굵은 선 = 중앙값)", loc="left", fontsize=10.5, color=INK)
    ax2.set_xlim(0, None)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("연속한 예매 완료 시각 사이의 간격 · 초 · 선형 축", color=INK_2, fontsize=10)
    ax2.set_ylabel("누적 비율 (ECDF)", color=INK_2, fontsize=10)
    ax2.set_title("누적 분포 (간격 4개 이상인 계정)", loc="left", fontsize=10.5, color=INK)
    ax2.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)

    fig.suptitle("매크로 모드별 개별 예매 간격 (이벤트 단위)", x=0.012, y=0.985, ha="left", fontsize=14, color=INK,
                 fontweight="bold")
    fig.text(0.012, 0.935, "채운 점·실선 = 관측값. 빈 점·점선 = 가상 시나리오(긴 간격 1개를 나머지 간격의 중앙값으로 대체). "
             "세로 흩뿌림은 겹침 방지용이며 가로축 값은 원본.\n이 값은 이벤트 단위 간격이다. 더미의 avg_booking_interval 은 계정 단위 평균이므로 "
             "이 그래프와 직접 비교하지 않는다.", ha="left", va="top", fontsize=9, color=INK_2)
    fig.subplots_adjust(left=0.115, right=0.98, top=0.80, bottom=0.10, wspace=0.16)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-tag", default="distribution_comparison")
    args = parser.parse_args()
    out = EXP / "output" / args.out_tag
    if out.exists():
        raise SystemExit(f"{out} 가 이미 있습니다. 덮어쓰지 않습니다. --out-tag 로 다른 이름을 지정하세요.")

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    rng = np.random.default_rng(14)  # 그래프의 세로 jitter 전용

    # ---------- 1. 입력 확인 ----------
    record = json.loads(BUILD_RECORD.read_text(encoding="utf-8"))
    dummy = pd.read_csv(DUMMY_PATH)
    data_dir = EXP / "data" / REAL_TAG
    real = pd.read_csv(data_dir / "real_features.csv").set_index("external_user_id")
    missing = [a for a in ACCOUNTS if a not in real.index]
    if missing:
        raise SystemExit(f"실측 Feature 에 계정이 없습니다: {missing}")
    real = real.loc[list(ACCOUNTS)]
    events = pd.read_csv(data_dir / "events.csv", dtype=str, keep_default_na=False)

    dummy_hash = sha256(DUMMY_PATH)
    verification = {
        "dummy_file": str(DUMMY_PATH.relative_to(EXP)),
        "dummy_sha256": dummy_hash,
        "matches_build_record_training_data": dummy_hash == record["training_data"]["sha256"],
        "matches_ai_repo_file": REPO_DUMMY_PATH.exists() and sha256(REPO_DUMMY_PATH) == dummy_hash,
        "build_record_training_file": record["training_data"]["file"],
        "build_record_note": record["purpose"],
        "dummy_rows": int(len(dummy)),
        "dummy_is_anomaly_counts": {str(k): int(v) for k, v in dummy["is_anomaly"].value_counts().items()},
        "dummy_scenario_counts": {k: int(v) for k, v in dummy["scenario"].value_counts().items()},
        "model_features": record["features"],
        "real_features_file": f"data/{REAL_TAG}/real_features.csv",
        "real_features_sha256": sha256(data_dir / "real_features.csv"),
        "real_events_sha256": sha256(data_dir / "events.csv"),
        "real_accounts": {a: {"booking_attempt_events": int(((events["external_user_id"] == a) & (events["event_type"] == "BOOKING_ATTEMPT")).sum()),
                              "booking_events": int(((events["external_user_id"] == a) & (events["event_type"] == "BOOKING")).sum()),
                              "booking_count_feature": int(real.loc[a, "booking_count"])} for a in ACCOUNTS},
        "card_env_distorted": bool(real["card_env_distorted"].all()),
        "ip_env_distorted": bool(real["ip_env_distorted"].all()),
    }
    if not verification["matches_build_record_training_data"]:
        raise SystemExit("더미 파일 해시가 BUILD_RECORD 의 학습 데이터 해시와 다릅니다. 분석을 멈춥니다.")

    normal = dummy[dummy["is_anomaly"] == 0]
    macro = dummy[dummy["scenario"] == "macro"]
    other = dummy[dummy["scenario"].isin(["hold_cancel", "multi_account"])]
    real_human = real[[ACCOUNTS[a][0] == "human" for a in real.index]]
    real_macro = real[[ACCOUNTS[a][0] == "macro" for a in real.index]]

    # ---------- 4-a. 개별 간격 (이벤트 단위) ----------
    interval_rows = []
    for account in ACCOUNTS:
        own = events[events["external_user_id"] == account]
        done = pd.to_datetime(own[own["event_type"] == "BOOKING"]["occurred_at_utc"]).sort_values().reset_index(drop=True)
        attempts = own[own["event_type"] == "BOOKING_ATTEMPT"].copy()
        attempts["attempt_id"] = attempts["metadata"].map(lambda s: json.loads(s).get("attemptId"))
        starts = pd.to_datetime(attempts.groupby("attempt_id")["occurred_at_utc"].min()).sort_values().reset_index(drop=True)
        breakdown_path = data_dir / f"macro_gap_breakdown__{account}.csv"
        breakdown = pd.read_csv(breakdown_path).set_index("interval_index") if breakdown_path.exists() else None
        for k in range(len(done) - 1):
            row = {"account": account, "group": ACCOUNTS[account][0], "interval_index": k + 1,
                   "completion_gap_s": round((done[k + 1] - done[k]).total_seconds(), 3),
                   "attempt_start_gap_s": round((starts[k + 1] - starts[k]).total_seconds(), 3) if len(starts) == len(done) else np.nan,
                   "launch_overhead_s": breakdown.loc[k + 1, "launch_overhead_s"] if breakdown is not None and k + 1 in breakdown.index else np.nan}
            interval_rows.append(row)
    intervals = pd.DataFrame(interval_rows)
    intervals["is_long_gap"] = (intervals["completion_gap_s"] > LONG_GAP_SECONDS) & intervals["account"].isin(LONG_GAP_ACCOUNTS)

    def scenarios(values, long_mask):
        """원본 / 가상 제외 / 가상 대체(나머지 중앙값)."""
        values = np.asarray(values, dtype=float)
        result = {"원본 (관측)": values}
        if long_mask.any():
            rest = values[~long_mask]
            replaced = values.copy()
            replaced[long_mask] = np.median(rest)
            result["가상: 긴 간격 제외"] = rest
            result["가상: 긴 간격을 나머지 중앙값으로 대체"] = replaced
        return result

    stat_rows, corrected_feature = [], {}
    for account in ACCOUNTS:
        part = intervals[intervals["account"] == account]
        for basis, column in (("예매 완료 시각 간격", "completion_gap_s"), ("시도 시작 시각 간격 (Feature 계산 기준)", "attempt_start_gap_s")):
            values = part[column].to_numpy(dtype=float)
            long_mask = (values > LONG_GAP_SECONDS) & (account in LONG_GAP_ACCOUNTS)
            for scenario, vals in scenarios(values, long_mask).items():
                d = describe(vals)
                stat_rows.append({"account": account, "group": ACCOUNTS[account][0], "basis": basis, "scenario": scenario,
                                  "observed": scenario.startswith("원본"), "n_intervals": d["n"], "mean_s": d["mean"],
                                  "median_s": d["median"], "std_s_sample": d["std"], "q1_s": d["q1"], "q3_s": d["q3"],
                                  "iqr_s": d["iqr"], "min_s": d["min"], "max_s": d["max"],
                                  "population_variance_s2": float(np.var(vals))})
                if column == "attempt_start_gap_s" and scenario.startswith("가상: 긴 간격을"):
                    corrected_feature[account] = (float(np.mean(vals)), float(np.var(vals)))
    interval_stats = pd.DataFrame(stat_rows)

    # ---------- 3. 통계표 ----------
    stats_rows, dist_rows, pos_rows = [], [], []
    groups = {"더미 정상": normal, "더미 매크로": macro, "더미 기타 이상 (hold_cancel·multi_account, 참고)": other,
              "실측 정상": real_human, "실측 매크로": real_macro}
    for feature, (unit, _, note) in FEATURES.items():
        for label, frame in groups.items():
            d = describe(frame[feature])
            d.pop("std")
            stats_rows.append({"feature": feature, "unit": unit, "group": label, **d, "note": note})
        pairs = [("정상끼리", normal, real_human[feature].to_numpy(dtype=float), "관측"),
                 ("매크로끼리", macro, real_macro[feature].to_numpy(dtype=float), "관측")]
        if feature in ("avg_booking_interval", "booking_interval_variance") and corrected_feature:
            fixed = real_macro[feature].astype(float).copy()
            for account, (mean, var) in corrected_feature.items():
                fixed[account] = mean if feature == "avg_booking_interval" else var
            pairs.append(("매크로끼리", macro, fixed.to_numpy(), "가상: Fast-02 긴 간격을 나머지 중앙값으로 대체"))
        for comparison, dummy_frame, real_values, kind in pairs:
            dist_rows.append({"feature": feature, "unit": unit, "comparison": comparison, "real_values": kind,
                              "n_dummy": len(dummy_frame), "n_real": len(real_values),
                              "ks_statistic": ks_2samp(dummy_frame[feature], real_values).statistic,
                              "wasserstein_distance": wasserstein_distance(dummy_frame[feature], real_values),
                              "note": ("실측 표본이 매우 작아 탐색적 지표로만 본다. Wasserstein 값은 Feature 단위를 따르므로 "
                                       "Feature 사이에 크기를 비교하지 않는다." + (f" {note}." if note else ""))})
        for account, (group, *_rest) in ACCOUNTS.items():
            ref_label, ref = ("더미 정상", normal) if group == "human" else ("더미 매크로", macro)
            value = float(real.loc[account, feature])
            q1, q3 = ref[feature].quantile([0.25, 0.75])
            pos_rows.append({"feature": feature, "account": account, "value": value, "reference": ref_label,
                             "pct_of_reference_at_or_below": round(float((ref[feature] <= value).mean()) * 100, 1),
                             "ref_min": ref[feature].min(), "ref_q1": q1, "ref_median": ref[feature].median(),
                             "ref_q3": q3, "ref_max": ref[feature].max(),
                             "within_min_max": bool(ref[feature].min() <= value <= ref[feature].max()),
                             "within_iqr": bool(q1 <= value <= q3),
                             "pct_of_other_dummy_group_at_or_below": round(float(
                                 ((macro if group == "human" else normal)[feature] <= value).mean()) * 100, 1)})
    feature_stats, distances, positions = pd.DataFrame(stats_rows), pd.DataFrame(dist_rows), pd.DataFrame(pos_rows)

    # ---------- 더미 생성식과 실제 생성값 ----------
    gen_rows = []
    for scenario, (interval_dist, var_dist, count_dist) in GENERATION_SPEC.items():
        part = dummy[dummy["scenario"] == scenario]
        gen_rows.append({"scenario": scenario, "is_anomaly": int(part["is_anomaly"].iloc[0]), "n": len(part),
                         "avg_booking_interval_distribution": interval_dist,
                         "avg_interval_min": part["avg_booking_interval"].min(), "avg_interval_median": part["avg_booking_interval"].median(),
                         "avg_interval_max": part["avg_booking_interval"].max(),
                         "booking_interval_variance_distribution": var_dist,
                         "variance_min": part["booking_interval_variance"].min(), "variance_max": part["booking_interval_variance"].max(),
                         "booking_count_distribution": count_dist,
                         "corr_avg_interval_vs_variance": round(float(part["avg_booking_interval"].corr(part["booking_interval_variance"])), 3),
                         "individual_interval_distribution": "없음 (계정 단위 값을 바로 뽑는다)"})
    generation = pd.DataFrame(gen_rows)

    # ---------- 현실성 평가표 ----------
    assess_rows = []
    for feature, (verdict, difference, cause, improvement) in ASSESSMENT.items():
        p = positions[positions["feature"] == feature]
        evidence = "; ".join(f"{r.account}={r.value:.4g} ({r.reference} 범위 {'안' if r.within_min_max else '밖'}, "
                             f"IQR {'안' if r.within_iqr else '밖'})" for r in p.itertuples())
        assess_rows.append({"feature": feature, "평가": verdict, "주요 차이": difference, "추정 원인": cause,
                            "더미 생성 로직 개선점": improvement, "근거 (실측값과 더미 분포 내 위치)": evidence})
    assessment = pd.DataFrame(assess_rows)

    # ---------- 저장 ----------
    out.mkdir(parents=True)
    (out / "data_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    feature_stats.to_csv(out / "feature_stats.csv", index=False, encoding="utf-8-sig")
    distances.to_csv(out / "distance_metrics.csv", index=False, encoding="utf-8-sig")
    positions.to_csv(out / "real_positions.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(out / "individual_intervals.csv", index=False, encoding="utf-8-sig")
    interval_stats.to_csv(out / "interval_stats.csv", index=False, encoding="utf-8-sig")
    generation.to_csv(out / "dummy_generation_distributions.csv", index=False, encoding="utf-8-sig")
    assessment.to_csv(out / "realism_assessment.csv", index=False, encoding="utf-8-sig")

    for feature in FEATURES:
        feature_figure(feature, {"더미 정상": normal[feature].to_numpy(dtype=float), "더미 매크로": macro[feature].to_numpy(dtype=float)},
                       real, out / f"fig_feature_{feature}.png", rng)

    def gaps(account):
        return intervals[intervals["account"] == account]["completion_gap_s"].to_numpy(dtype=float)

    fast = gaps("macro-fast-02")
    long_mask = fast > LONG_GAP_SECONDS
    series = {"Fast-01": (gaps("macro-fast-01"), ACCOUNTS["macro-fast-01"][2], False),
              "Fast-02 원본": (fast, ACCOUNTS["macro-fast-02"][2], False)}
    long_gap = overhead = None
    if long_mask.any():
        replaced = fast.copy()
        replaced[long_mask] = np.median(fast[~long_mask])
        series["Fast-02 가상\n(긴 간격 대체)"] = (replaced, ACCOUNTS["macro-fast-02"][2], True)
        long_row = intervals[(intervals["account"] == "macro-fast-02") & intervals["is_long_gap"]].iloc[0]
        long_gap, overhead = float(long_row["completion_gap_s"]), float(long_row["launch_overhead_s"])
    series["Jitter-01"] = (gaps("macro-jitter-01"), ACCOUNTS["macro-jitter-01"][2], False)
    series["Human-like-01"] = (gaps("macro-human-01"), ACCOUNTS["macro-human-01"][2], False)
    interval_figure(series, long_gap, overhead, out / "fig_macro_intervals.png", rng)

    # ---------- 화면 요약 ----------
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    print(json.dumps({k: verification[k] for k in ("dummy_sha256", "matches_build_record_training_data", "matches_ai_repo_file", "dummy_rows")}, ensure_ascii=False))
    print("\n[Feature 통계]\n", feature_stats.drop(columns=["note", "unit"]).round(4).to_string(index=False))
    print("\n[KS / Wasserstein]\n", distances.drop(columns=["note"]).round(4).to_string(index=False))
    print("\n[실측 위치]\n", positions.round(4).to_string(index=False))
    print("\n[개별 간격 통계]\n", interval_stats.round(3).to_string(index=False))
    print("\n[더미 생성식]\n", generation.to_string(index=False))
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
