"""Isolation Forest 단건/배치 추론 성능을 측정한다.

모델 학습과 첫 호출 비용은 측정에서 제외한다. 따라서 결과는 이미 로드된
서빙 프로세스에서 ``assess`` 또는 ``assess_batch``를 호출하는 비용을 나타낸다.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow execution from the repository root with `python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train_model as tm
from inference import assess, assess_batch
from preprocessing import FEATURE_SPECS


DEFAULT_BATCH_SIZES = (1, 10, 50, 100)


def make_benchmark_features(count: int, seed: int = 42) -> pd.DataFrame:
    """서빙 Feature와 같은 분포의 결정론적 측정용 데이터를 만든다."""
    rng = np.random.default_rng(seed)
    booking_count = rng.integers(2, 12, count)
    cancel_count = rng.binomial(booking_count, 0.15)
    return pd.DataFrame(
        {
            "user_id": [f"benchmark_{index:06d}" for index in range(count)],
            "avg_booking_interval": rng.uniform(20_000, 150_000, count),
            "booking_interval_variance": rng.uniform(100, 10_000, count),
            "cancel_rate": cancel_count / booking_count,
            "card_reuse_count": rng.integers(1, 4, count),
            "ip_booking_density": rng.integers(1, 8, count),
            "booking_count": booking_count,
        }
    )


def make_payloads(count: int, seed: int = 42) -> list[dict]:
    """실제 /predict 요청과 동일한 구조의 유효 payload를 만든다."""
    features = make_benchmark_features(count, seed)
    payloads = []
    for _, row in features.iterrows():
        payloads.append(
            {
                "userId": row["user_id"],
                "calculatedAt": "2026-01-01T00:00:00+00:00",
                "windowDays": 30,
                "features": {
                    "bookingCount": int(row["booking_count"]),
                    "cancelRate": float(row["cancel_rate"]),
                    "avgBookingInterval": float(row["avg_booking_interval"]),
                    "bookingIntervalVariance": float(
                        row["booking_interval_variance"]
                    ),
                    "cardReuseCount": int(row["card_reuse_count"]),
                    "ipBookingDensity": int(row["ip_booking_density"]),
                },
            }
        )
    return payloads


def build_benchmark_bundle():
    features = make_benchmark_features(2_000)
    feature_columns = list(FEATURE_SPECS)
    model, normalizer, _ = tm.train_and_score(features, feature_columns)
    return tm.make_bundle(model, normalizer, feature_columns)


def percentile(samples: list[float], value: float) -> float:
    if len(samples) == 1:
        return samples[0] * 1000
    return statistics.quantiles(samples, n=100, method="inclusive")[value - 1] * 1000


def measure(
    payloads: list[dict],
    bundle: dict,
    batch_size: int,
    iterations: int,
    warmup: int,
) -> dict:
    """한 작업 단위의 latency와 처리한 item 기준 throughput을 측정한다."""
    batch = payloads[:batch_size]
    operation = assess if batch_size == 1 else assess_batch

    for _ in range(warmup):
        if batch_size == 1:
            operation(batch[0], bundle)
        else:
            operation(batch, bundle)

    samples = []
    total_items = 0
    started = time.perf_counter()
    for _ in range(iterations):
        operation_started = time.perf_counter()
        if batch_size == 1:
            operation(batch[0], bundle)
        else:
            operation(batch, bundle)
        samples.append(time.perf_counter() - operation_started)
        total_items += batch_size
    elapsed = time.perf_counter() - started

    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "latency_ms": {
            "p50": round(percentile(samples, 50), 3),
            "p95": round(percentile(samples, 95), 3),
            "p99": round(percentile(samples, 99), 3),
        },
        "throughput_items_per_second": round(total_items / elapsed, 2),
    }


def run_benchmark(
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    iterations: int = 100,
    warmup: int = 10,
) -> dict:
    """모델을 준비하고 지정한 추론 방식의 성능을 반환한다."""
    if iterations < 1:
        raise ValueError("iterations는 1 이상이어야 합니다.")
    if warmup < 0:
        raise ValueError("warmup은 0 이상이어야 합니다.")
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("batch_sizes에는 1 이상의 값이 하나 이상 필요합니다.")

    bundle = build_benchmark_bundle()
    payloads = make_payloads(max(batch_sizes))
    return {
        "model_version": bundle["model_version"],
        "iterations": iterations,
        "warmup": warmup,
        "results": [
            measure(payloads, bundle, size, iterations, warmup)
            for size in batch_sizes
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="1은 단건 순차 추론을 의미합니다.",
    )
    parser.add_argument("--output", type=Path, help="JSON 결과 저장 경로")
    args = parser.parse_args()

    result = run_benchmark(tuple(args.batch_sizes), args.iterations, args.warmup)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
