"""Fast 매크로 누적 예매 실험의 Feature 민감도 분석 (오프라인, 이번 실험 한정).

기존 모델·점수 계산은 그대로 쓰고(inference.load_bundle / assess_batch), 입력 Feature 값만
가상으로 바꿔 점수가 어떻게 달라지는지 본다. 모델, 학습 데이터, Feature 추출 로직, 정규화,
임계값, DB, 기존 결과 파일은 읽기만 한다.

실행 (실험 폴더에서):
    .venv/Scripts/python scripts/sensitivity_fast_macro.py

입력 (읽기 전용)
    data/count__macro-fast-02__n03|n10|n20/real_features.csv, attempt_intervals.csv
    data/count/real_features.csv   (비교군 macro-jitter-01, macro-human-01 의 20회 시점 값)
    model_build/models/behavior_model.joblib, 더미 학습 데이터

출력 (output/<out-tag>/, 기본 sensitivity_fast02 — 폴더가 이미 있으면 덮어쓰지 않고 멈춘다)
    A_sensitivity.csv            Feature 하나씩 바꿨을 때의 점수 (가상 입력)
    B_long_gap_scenarios.csv     긴 간격 제외·대체 시나리오 (가상 입력) 대 관측값
    C_snapshot_comparison.csv    3·10·20회 스냅샷의 실제 Feature·점수 (관측)
    C_feature_swap.csv           스냅샷 사이 Feature 교체 분해 (가상 입력)
    reference_accounts.csv       비교군의 실제 Feature·점수 (관측)
    fig_A_sensitivity.png, fig_B_long_gap.png, fig_C_swap.png

모든 점수는 card_reuse_count=1, ip_booking_density=1 로 고정한 값이다 (로컬 환경 왜곡 통제).
"observed" 는 실제 계산된 Feature 벡터(카드·IP만 1로 고정), "hypothetical" 은 값을 바꾼 가상 입력이다.
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP.parent.parent))  # AI 레포 루트 (기존 코드 import 전용)

from inference import assess_batch, load_bundle  # noqa: E402
from preprocessing import CAMEL_TO_SNAKE, FEATURE_SPECS  # noqa: E402

ACCOUNT = "macro-fast-02"
SNAPSHOTS = {3: "count__macro-fast-02__n03", 10: "count__macro-fast-02__n10", 20: "count__macro-fast-02__n20"}
REFERENCE_TAG = "count"
REFERENCES = ["macro-jitter-01", "macro-human-01", "macro-fast-01", "human-01", "human-02"]
VARIED = ["booking_count", "avg_booking_interval", "booking_interval_variance"]
FIXED = {"card_reuse_count": 1, "ip_booking_density": 1}
LONG_GAP_SECONDS = 20.0  # 이 값을 넘는 간격을 "긴 간격"으로 본다 (fast 모드의 나머지 간격은 5~8초)

C_BLUE, C_ORANGE = "#2a78d6", "#eb6834"  # 검증된 2색 (dataviz validate_palette 통과)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FEATURE_LABEL = {"booking_count": "booking_count (누적 예매 시도 수)",
                 "avg_booking_interval": "avg_booking_interval (초)",
                 "booking_interval_variance": "booking_interval_variance (초²)"}


class Scorer:
    """기존 서빙 경로(assess_batch)로 점수를 낸다. 모델과 정규화기는 번들 그대로."""

    def __init__(self):
        self.bundle = load_bundle(EXP / "model_build" / "models" / "behavior_model.joblib")
        self.columns = self.bundle["feature_columns"]
        self.predict_boundary = float(self.bundle["normalizer"].transform([-self.bundle["model"].offset_])[0])

    def score(self, vectors):
        payloads = []
        for i, vector in enumerate(vectors):
            features = {}
            for camel, snake in CAMEL_TO_SNAKE.items():
                value = vector[snake]
                features[camel] = int(round(value)) if FEATURE_SPECS[snake]["integer"] else float(value)
            payloads.append({"userId": f"v{i}", "calculatedAt": "2026-11-19T12:00:00+00:00",
                             "windowDays": 30, "features": features})
        results = assess_batch(payloads, self.bundle)
        bad = [r for r in results if r["status"] != "OK"]
        if bad:
            raise RuntimeError(f"점수 산출 실패: {bad[0]}")
        return [r["anomalyScore"] for r in results]


def controlled(row):
    """실제 Feature 행에서 모델 입력 6개를 꺼내고 카드·IP만 1로 고정한다."""
    vector = {c: float(row[c]) for c in ["avg_booking_interval", "booking_interval_variance", "cancel_rate", "booking_count"]}
    vector.update(FIXED)
    return vector


def bts_interval_features(gaps):
    """BTS FeatureAggregationService 와 같은 식: 평균과 모분산."""
    gaps = np.asarray(gaps, dtype=float)
    return float(gaps.mean()), float(((gaps - gaps.mean()) ** 2).mean())


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-tag", default="sensitivity_fast02")
    args = parser.parse_args()
    out = EXP / "output" / args.out_tag
    if out.exists():
        raise SystemExit(f"{out} 가 이미 있습니다. 덮어쓰지 않습니다. --out-tag 로 다른 이름을 지정하세요.")

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    scorer = Scorer()
    dummy = pd.read_csv(EXP / "model_build" / "ml_experiment" / "data" / "fairticket_synthetic_features.csv")
    normal, macro = dummy[dummy["is_anomaly"] == 0], dummy[dummy["scenario"] == "macro"]

    # ---------- 관측값 읽기 ----------
    snap_rows, snap_vec, snap_gaps = [], {}, {}
    for n, tag in SNAPSHOTS.items():
        features = pd.read_csv(EXP / "data" / tag / "real_features.csv")
        scores = pd.read_csv(EXP / "data" / tag / "real_scores.csv")
        row = features[features["external_user_id"] == ACCOUNT].iloc[0]
        recorded = scores[scores["external_user_id"] == ACCOUNT].iloc[0]
        intervals = pd.read_csv(EXP / "data" / tag / "attempt_intervals.csv")
        snap_gaps[n] = intervals.loc[intervals["external_user_id"] == ACCOUNT, "gap_seconds"].tolist()
        snap_vec[n] = controlled(row)
        rescored = scorer.score([snap_vec[n]])[0]
        if abs(rescored - float(recorded["anomaly_score_card_ip_1"])) > 1e-4:
            raise RuntimeError(f"n={n}: 재계산 점수 {rescored} 가 기록된 값 {recorded['anomaly_score_card_ip_1']} 과 다릅니다.")
        snap_rows.append({
            "kind": "observed", "n_attempts": n, "snapshot_tag": tag,
            "booking_count": int(row["booking_count"]), "avg_booking_interval": row["avg_booking_interval"],
            "booking_interval_variance": row["booking_interval_variance"], "cancel_rate": row["cancel_rate"],
            "n_intervals": len(snap_gaps[n]), "max_gap_seconds": max(snap_gaps[n]),
            "median_gap_seconds": float(np.median(snap_gaps[n])),
            "card_reuse_count_actual": int(row["card_reuse_count"]), "ip_booking_density_actual": int(row["ip_booking_density"]),
            "score_actual_card_ip": float(recorded["anomaly_score"]),
            "score_card_ip_1": rescored, "model_predict_anomaly_card_ip_1": rescored >= scorer.predict_boundary,
        })
    snapshots = pd.DataFrame(snap_rows)
    base = snap_vec[20]

    ref_features = pd.read_csv(EXP / "data" / REFERENCE_TAG / "real_features.csv").set_index("external_user_id")
    ref_rows, ref_vec = [], {}
    for account in REFERENCES:
        if account not in ref_features.index:
            continue
        ref_vec[account] = controlled(ref_features.loc[account])
        ref_rows.append({"kind": "observed", "account": account, "group": ref_features.loc[account, "group"],
                         **{c: ref_vec[account][c] for c in VARIED}, "cancel_rate": ref_vec[account]["cancel_rate"],
                         "score_card_ip_1": scorer.score([ref_vec[account]])[0]})
    references = pd.DataFrame(ref_rows)

    # ---------- A. Feature별 민감도 (가상 입력) ----------
    grids = {
        # 더미 정상 1~13, 더미 이상 10~49 를 모두 덮는다
        "booking_count": sorted(set(range(1, 51))),
        # 더미 전체 0.1~218초 + 이번 실험 관측값
        "avg_booking_interval": sorted(set(np.round(np.logspace(-1, np.log10(220), 60), 3))
                                       | {4.924, 5.7, 7.506, 8.565, 9.177, 21.901, 23.647, 58.865}),
        # 더미 전체 0.012~49.9 + 관측값(최대 264)
        "booking_interval_variance": sorted(set(np.round(np.logspace(-2, np.log10(300), 60), 4))
                                            | {0.071, 0.276, 3.684, 49.9, 52.599, 105.131, 264.388}),
    }
    a_vectors, a_meta = [], []
    for feature, values in grids.items():
        for value in values:
            vector = dict(base)
            vector[feature] = float(value)
            a_vectors.append(vector)
            a_meta.append({"kind": "hypothetical", "base": "fast n=20 (card=1, ip=1)", "varied_feature": feature,
                           "value": float(value), "base_value": base[feature]})
    a = pd.DataFrame(a_meta)
    a["score_card_ip_1"] = scorer.score(a_vectors)
    a["delta_vs_base"] = a["score_card_ip_1"] - snapshots.loc[snapshots["n_attempts"] == 20, "score_card_ip_1"].iloc[0]
    a["model_predict_anomaly"] = a["score_card_ip_1"] >= scorer.predict_boundary
    for feature in VARIED:
        mask = a["varied_feature"] == feature
        a.loc[mask, "in_dummy_normal_range"] = a.loc[mask, "value"].between(normal[feature].min(), normal[feature].max())
        a.loc[mask, "in_dummy_macro_range"] = a.loc[mask, "value"].between(macro[feature].min(), macro[feature].max())

    # ---------- B. 긴 간격 시나리오 ----------
    b_rows, b_vectors = [], []
    for n in SNAPSHOTS:  # n = 3, 10, 20
        gaps = snap_gaps[n]
        long_gaps = [g for g in gaps if g > LONG_GAP_SECONDS]
        normal_gaps = [g for g in gaps if g <= LONG_GAP_SECONDS]
        scenarios = [("observed", "관측값 그대로", gaps)]
        if long_gaps and normal_gaps:
            typical = float(np.median(normal_gaps))
            scenarios.append(("hypothetical", "긴 간격 제외", normal_gaps))
            scenarios.append(("hypothetical", f"긴 간격을 나머지 간격의 중앙값({typical:.2f}s)으로 대체",
                              [g if g <= LONG_GAP_SECONDS else typical for g in gaps]))
        for kind, name, used in scenarios:
            vector = dict(snap_vec[n])
            if len(used) >= 1:
                mean, variance = bts_interval_features(used)
                vector["avg_booking_interval"], vector["booking_interval_variance"] = mean, variance
            b_vectors.append(vector)
            b_rows.append({"kind": kind, "n_attempts": n, "scenario": name, "n_intervals_used": len(used),
                           "long_gaps_seconds": "|".join(f"{g:.3f}" for g in long_gaps),
                           "booking_count": int(vector["booking_count"]),
                           "avg_booking_interval": vector["avg_booking_interval"],
                           "booking_interval_variance": vector["booking_interval_variance"]})
    b = pd.DataFrame(b_rows)
    b["score_card_ip_1"] = scorer.score(b_vectors)
    b["delta_vs_observed"] = b["score_card_ip_1"] - b.groupby("n_attempts")["score_card_ip_1"].transform("first")
    b["model_predict_anomaly"] = b["score_card_ip_1"] >= scorer.predict_boundary

    # ---------- C. 스냅샷 사이 Feature 교체 분해 ----------
    def swap_score(start, end, subset):
        vector = dict(start)
        for feature in subset:
            vector[feature] = end[feature]
        return scorer.score([vector])[0]

    c_rows = []
    for a_n, b_n in ((3, 10), (10, 20), (3, 20)):
        start, end = snap_vec[a_n], snap_vec[b_n]
        cache = {subset: swap_score(start, end, subset)
                 for r in range(len(VARIED) + 1) for subset in itertools.combinations(VARIED, r)}
        s_start, s_end = cache[()], cache[tuple(VARIED)]
        for feature in VARIED:
            others = tuple(f for f in VARIED if f != feature)
            # Shapley: 세 Feature를 바꾸는 순서 6가지에 대한 평균 한계 변화
            marginals = []
            for order in itertools.permutations(VARIED):
                before = tuple(f for f in VARIED if f in order[:order.index(feature)])
                marginals.append(cache[tuple(f for f in VARIED if f in before or f == feature)] - cache[before])
            c_rows.append({
                "kind": "hypothetical", "from_n": a_n, "to_n": b_n, "feature": feature,
                "value_from": start[feature], "value_to": end[feature],
                "score_from": s_start, "score_to": s_end, "total_change": s_end - s_start,
                "only_this_changed": cache[(feature,)], "delta_only_this_changed": cache[(feature,)] - s_start,
                "all_but_this_changed": cache[others], "delta_when_this_added_last": s_end - cache[others],
                "shapley_share_of_change": float(np.mean(marginals)),
            })
    c = pd.DataFrame(c_rows)

    # ---------- 저장 ----------
    out.mkdir(parents=True)
    a.round(5).to_csv(out / "A_sensitivity.csv", index=False, encoding="utf-8-sig")
    b.round(5).to_csv(out / "B_long_gap_scenarios.csv", index=False, encoding="utf-8-sig")
    snapshots.round(5).to_csv(out / "C_snapshot_comparison.csv", index=False, encoding="utf-8-sig")
    c.round(5).to_csv(out / "C_feature_swap.csv", index=False, encoding="utf-8-sig")
    references.round(5).to_csv(out / "reference_accounts.csv", index=False, encoding="utf-8-sig")

    # ---------- 그림 A: 민감도 (Feature별 패널) ----------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    observed_points = {f"fast {n}회": snap_vec[n] for n in SNAPSHOTS}
    observed_points.update({"jitter 20회": ref_vec.get("macro-jitter-01"), "human모드 20회": ref_vec.get("macro-human-01")})
    for ax, feature in zip(axes, VARIED):
        part = a[a["varied_feature"] == feature]
        log = feature != "booking_count"
        ax.axvspan(normal[feature].min(), normal[feature].max(), color=C_BLUE, alpha=0.08, linewidth=0)
        ax.axvspan(macro[feature].min(), macro[feature].max(), color=C_ORANGE, alpha=0.12, linewidth=0)
        ax.axhline(scorer.predict_boundary, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
        ax.plot(part["value"], part["score_card_ip_1"], color=INK, linewidth=2)
        ax.scatter([base[feature]], [snapshots.loc[snapshots["n_attempts"] == 20, "score_card_ip_1"].iloc[0]],
                   s=70, color=INK, edgecolor=SURFACE, linewidth=2, zorder=4)
        # 관측값 표시: 값이 같은 계정은 한 라벨로 묶고, 가까운 라벨은 높이를 번갈아 겹치지 않게 한다
        merged = {}
        for label, vector in observed_points.items():
            if vector is not None:
                merged.setdefault(round(vector[feature], 3), []).append(label)
        for position, (value, labels) in enumerate(sorted(merged.items())):
            ax.plot([value] * 2, [0.0, 0.035], color=INK, linewidth=1.5)
            ax.text(value, 0.05 + 0.2 * (position % 3), " · ".join(labels), rotation=90, ha="center", va="bottom",
                    fontsize=7.5, color=INK_2)
        if log:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(FEATURE_LABEL[feature] + (" · 로그 축" if log else ""), color=INK_2, fontsize=9)
        ax.set_ylim(0, 1.02)
        style(ax)
    axes[0].set_ylabel("이상 점수 (카드·IP=1 고정)", color=INK_2, fontsize=9)
    axes[0].text(0.02, scorer.predict_boundary + 0.012, f"모델 predict 경계 {scorer.predict_boundary:.3f}",
                 transform=axes[0].get_yaxis_transform(), fontsize=8, color=INK_2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_BLUE, alpha=0.25), plt.Rectangle((0, 0), 1, 1, color=C_ORANGE, alpha=0.35),
               plt.Line2D([], [], color=INK, linewidth=2), plt.Line2D([], [], color=INK, marker="o", linewidth=0, markersize=7)]
    fig.legend(handles, ["더미 정상 범위", "더미 매크로 범위", "가상 입력 점수 (한 Feature만 변경)", "기준점: fast 20회 관측값"],
               loc="upper right", ncol=4, frameon=False, fontsize=8.5, labelcolor=INK_2, bbox_to_anchor=(0.99, 0.93))
    fig.suptitle("A. Feature 하나만 바꿨을 때의 이상 점수 (기준: fast 20회, 가상 입력)", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.14, wspace=0.06)
    fig.savefig(out / "fig_A_sensitivity.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---------- 그림 B: 누적 횟수별 점수, 관측 대 긴 간격 대체 ----------
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    obs = b[b["scenario"] == "관측값 그대로"].sort_values("n_attempts")
    rep = b[b["scenario"].str.startswith("긴 간격을")].sort_values("n_attempts")
    ax.axhline(scorer.predict_boundary, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(3, scorer.predict_boundary + 0.012, f"모델 predict 경계 {scorer.predict_boundary:.3f}", fontsize=8, color=INK_2)
    for part, color, label in ((obs, C_BLUE, "관측값"), (rep, C_ORANGE, "가상: 긴 간격(38초)을 일반 간격으로 대체")):
        ax.plot(part["n_attempts"], part["score_card_ip_1"], color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        for _, row in part.iterrows():
            ax.annotate(f"{row['score_card_ip_1']:.3f}", (row["n_attempts"], row["score_card_ip_1"]),
                        textcoords="offset points", xytext=(0, 9 if color == C_BLUE else -15), ha="center",
                        fontsize=9, color=INK)
    ax.set_xticks(list(SNAPSHOTS))
    ax.set_xlabel("누적 예매 시도 수 (fast 매크로)", color=INK_2, fontsize=9)
    ax.set_ylabel("이상 점수 (카드·IP=1 고정)", color=INK_2, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    style(ax)
    fig.suptitle("B. 누적 횟수별 점수: 관측값 대 긴 간격 대체", x=0.02, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.12)
    fig.savefig(out / "fig_B_long_gap.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---------- 그림 C: 스냅샷 사이 변화의 Feature별 몫 ----------
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    short = {"booking_count": "booking_count", "avg_booking_interval": "avg_interval", "booking_interval_variance": "variance"}
    for ax, (a_n, b_n) in zip(axes, ((3, 10), (10, 20), (3, 20))):
        part = c[(c["from_n"] == a_n) & (c["to_n"] == b_n)]
        y = np.arange(len(part))
        ax.barh(y, part["shapley_share_of_change"], height=0.5, color=INK_2, edgecolor=SURFACE, linewidth=1)
        for yi, value in zip(y, part["shapley_share_of_change"]):
            ax.annotate(f"{value:+.3f}", (value, yi), textcoords="offset points",
                        xytext=(6 if value >= 0 else -6, 0), va="center",
                        ha="left" if value >= 0 else "right", fontsize=9, color=INK)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([short[f] for f in part["feature"]], fontsize=9, color=INK)
        ax.invert_yaxis()
        total = part["total_change"].iloc[0]
        ax.set_title(f"{a_n}회 → {b_n}회  (점수 {part['score_from'].iloc[0]:.3f} → {part['score_to'].iloc[0]:.3f}, {total:+.3f})",
                     fontsize=9.5, color=INK, loc="left")
        style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.25)
        ax.set_xlabel("점수 변화 중 이 Feature의 몫 (순서 평균)", color=INK_2, fontsize=8.5)
    fig.suptitle("C. 스냅샷 사이 점수 변화를 Feature별로 나눈 값 (가상 교체, 인과 기여도 아님)", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.17, wspace=0.08)
    fig.savefig(out / "fig_C_swap.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # ---------- 화면 요약 ----------
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(f"모델 predict 경계(정규화 점수) {scorer.predict_boundary:.4f}, 번들 임계값 {scorer.bundle['anomaly_threshold']}")
    print("\n[C] 스냅샷 (관측)")
    print(snapshots.drop(columns=["kind", "snapshot_tag"]).round(4).to_string(index=False))
    print("\n[비교군] (관측, 카드·IP=1)")
    print(references.round(4).to_string(index=False))
    print("\n[B] 긴 간격 시나리오")
    print(b.drop(columns=["long_gaps_seconds"]).round(4).to_string(index=False))
    print("\n[C] Feature 교체 분해 (가상)")
    print(c.round(4).to_string(index=False))
    print("\n[A] 민감도 요약 (가상)")
    for feature in VARIED:
        part = a[a["varied_feature"] == feature]
        lo, hi = part.loc[part["score_card_ip_1"].idxmin()], part.loc[part["score_card_ip_1"].idxmax()]
        print(f"  {feature}: 점수 범위 {lo['score_card_ip_1']:.3f} (값 {lo['value']:g}) ~ {hi['score_card_ip_1']:.3f} (값 {hi['value']:g})")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
