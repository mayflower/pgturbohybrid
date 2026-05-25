#!/usr/bin/env python3
"""Validate benchmark result artifacts against committed acceptance thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = Path("benchmarks/config/acceptance_thresholds.json")


METRIC_ALIASES = {
    "ndcg@10": ["nDCG@10", "ndcg@10", "ndcg_at_10", "ndcg_10"],
    "mrr@10": ["MRR@10", "mrr@10", "mrr_at_10", "mrr_10"],
    "recall@10": ["Recall@10", "recall@10", "recall_at_10", "recall_10"],
    "overlap@10": ["overlap@10", "Overlap@10", "overlap_at_10", "overlap_10"],
    "p95_ms": ["p95", "p95_ms", "latency_p95_ms", "p95_latency_ms"],
}


def canonical_metric(metrics: dict[str, Any], canonical: str) -> float | None:
    for key in METRIC_ALIASES[canonical]:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return None


def method_name(entry: dict[str, Any]) -> str | None:
    for key in ("method", "name", "id"):
        value = entry.get(key)
        if isinstance(value, str):
            return value
    return None


def entry_metrics(entry: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(entry.get("metrics") or {})
    latency = entry.get("latency") or {}
    if isinstance(latency, dict):
        for key in ("p50_ms", "p95_ms", "p99_ms"):
            if key in latency:
                metrics[key] = latency[key]
    for key in (
        "nDCG@10",
        "ndcg_at_10",
        "MRR@10",
        "mrr_at_10",
        "Recall@10",
        "recall_at_10",
        "overlap@10",
        "overlap_at_10",
        "p95",
        "p95_ms",
        "latency_p95_ms",
    ):
        if key in entry:
            metrics[key] = entry[key]
    return metrics


def find_methods(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    methods: dict[str, dict[str, Any]] = {}

    if isinstance(result.get("methods"), dict):
        for name, entry in result["methods"].items():
            if isinstance(entry, dict):
                methods[name] = entry_metrics(entry)

    for entry in result.get("results") or []:
        if not isinstance(entry, dict):
            continue
        name = method_name(entry)
        if name is not None:
            methods[name] = entry_metrics(entry)

    name = method_name(result)
    if name is not None:
        methods[name] = entry_metrics(result)

    return methods


def require_metric(metrics: dict[str, Any], canonical: str, method: str) -> float:
    value = canonical_metric(metrics, canonical)
    if value is None:
        aliases = ", ".join(METRIC_ALIASES[canonical])
        raise ValueError(f"{method} is missing required metric {canonical} ({aliases})")
    return value


def recommendations(config: dict[str, Any]) -> str:
    items = config.get("on_failure_recommendations") or []
    if not items:
        return ""
    return "\nRecommended quality fallbacks:\n- " + "\n- ".join(items)


def validate_fiqa_fast_defaults(result: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    config = thresholds["fiqa_openai_fast_defaults"]
    method = config["method"]
    baseline = config["baseline_method"]
    methods = find_methods(result)
    failures: list[str] = []

    if method not in methods:
        raise ValueError(f"artifact is missing method {method!r}")
    if baseline not in methods:
        raise ValueError(f"artifact is missing baseline method {baseline!r}")

    actual = methods[method]
    reference = methods[baseline]
    quality = config["quality"]

    actual_ndcg = require_metric(actual, "ndcg@10", method)
    baseline_ndcg = require_metric(reference, "ndcg@10", baseline)
    ndcg_delta = baseline_ndcg - actual_ndcg
    if ndcg_delta > quality["ndcg_at_10_max_delta_vs_baseline"]:
        failures.append(
            f"nDCG@10 delta {ndcg_delta:.6f} exceeds "
            f"{quality['ndcg_at_10_max_delta_vs_baseline']:.6f}"
        )

    actual_mrr = require_metric(actual, "mrr@10", method)
    baseline_mrr = require_metric(reference, "mrr@10", baseline)
    mrr_delta = baseline_mrr - actual_mrr
    if mrr_delta > quality["mrr_at_10_max_delta_vs_baseline"]:
        failures.append(
            f"MRR@10 delta {mrr_delta:.6f} exceeds "
            f"{quality['mrr_at_10_max_delta_vs_baseline']:.6f}"
        )

    actual_recall = canonical_metric(actual, "recall@10")
    baseline_recall = canonical_metric(reference, "recall@10")
    overlap = canonical_metric(actual, "overlap@10")
    if actual_recall is None or baseline_recall is None:
        if overlap is None:
            raise ValueError(f"{method} must report Recall@10 or overlap@10")
        if overlap < quality["overlap_at_10_min_vs_baseline"]:
            failures.append(
                f"overlap@10 {overlap:.6f} is below "
                f"{quality['overlap_at_10_min_vs_baseline']:.6f}"
            )
    else:
        recall_delta = baseline_recall - actual_recall
        if recall_delta > quality["recall_at_10_max_delta_vs_baseline"]:
            failures.append(
                f"Recall@10 delta {recall_delta:.6f} exceeds "
                f"{quality['recall_at_10_max_delta_vs_baseline']:.6f}"
            )
        if overlap is not None and overlap < quality["overlap_at_10_min_vs_baseline"]:
            failures.append(
                f"overlap@10 {overlap:.6f} is below "
                f"{quality['overlap_at_10_min_vs_baseline']:.6f}"
            )

    actual_p95 = require_metric(actual, "p95_ms", method)
    baseline_p95 = require_metric(reference, "p95_ms", baseline)
    speedup = baseline_p95 / actual_p95 if actual_p95 > 0 else float("inf")
    min_speedup = config["latency"]["p95_speedup_vs_baseline_min"]
    if speedup < min_speedup:
        failures.append(f"p95 speedup {speedup:.3f}x is below {min_speedup:.3f}x")

    return failures


def validate_fiqa_profile_matrix(result: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    config = thresholds["fiqa_openai_profile_matrix"]
    method_config = config["methods"]
    latency_method = method_config["latency"]
    quality_method = method_config["quality"]
    baseline_method = method_config["baseline"]
    methods = find_methods(result)
    failures: list[str] = []

    for method in (latency_method, quality_method, baseline_method):
        if method not in methods:
            raise ValueError(f"artifact is missing method {method!r}")

    latency = methods[latency_method]
    quality = methods[quality_method]

    latency_ndcg = require_metric(latency, "ndcg@10", latency_method)
    quality_ndcg = require_metric(quality, "ndcg@10", quality_method)
    ndcg_delta = quality_ndcg - latency_ndcg
    min_ndcg_delta = config["quality_vs_latency"]["ndcg_at_10_min_delta"]
    if ndcg_delta < min_ndcg_delta:
        failures.append(
            f"quality nDCG@10 delta vs latency {ndcg_delta:.6f} is below "
            f"{min_ndcg_delta:.6f}"
        )

    latency_mrr = require_metric(latency, "mrr@10", latency_method)
    quality_mrr = require_metric(quality, "mrr@10", quality_method)
    mrr_delta = quality_mrr - latency_mrr
    min_mrr_delta = config["quality_vs_latency"]["mrr_at_10_min_delta"]
    if mrr_delta < min_mrr_delta:
        failures.append(
            f"quality MRR@10 delta vs latency {mrr_delta:.6f} is below "
            f"{min_mrr_delta:.6f}"
        )

    if config["quality_vs_latency"].get("require_latency_metrics", False):
        require_metric(latency, "p95_ms", latency_method)
        require_metric(quality, "p95_ms", quality_method)
        require_metric(methods[baseline_method], "p95_ms", baseline_method)

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path, help="benchmark result JSON artifact")
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=DEFAULT_THRESHOLDS,
        help="acceptance threshold JSON file",
    )
    parser.add_argument(
        "--suite",
        default="fiqa_openai_fast_defaults",
        choices=["fiqa_openai_fast_defaults", "fiqa_openai_profile_matrix"],
    )
    args = parser.parse_args()

    result = json.loads(args.artifact.read_text())
    thresholds = json.loads(args.thresholds.read_text())

    if args.suite == "fiqa_openai_fast_defaults":
        failures = validate_fiqa_fast_defaults(result, thresholds)
    elif args.suite == "fiqa_openai_profile_matrix":
        failures = validate_fiqa_profile_matrix(result, thresholds)
    else:
        raise AssertionError(args.suite)

    if failures:
        config = thresholds[args.suite]
        print(f"{args.suite} acceptance failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        print(recommendations(config), file=sys.stderr)
        return 1

    print(f"{args.suite} acceptance passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
