#!/usr/bin/env python3
"""Validate non-flaky structural properties of a FIQA perf-smoke run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def walk_plan(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        found.append(node)
        for value in node.values():
            found.extend(walk_plan(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(walk_plan(item))
    return found


def plan_uses_turbohybrid(plan: Any) -> bool:
    for node in walk_plan(plan):
        if node.get("Node Type") in {"Index Scan", "Index Only Scan"}:
            if node.get("Index Name") == "fiqa_turbohybrid_idx":
                return True
    return False


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def simd_available(simd: dict[str, Any]) -> bool:
    if simd.get("simd") is True:
        return True
    if simd.get("simd_build_disabled") is True:
        return False

    architecture = str(simd.get("architecture", "")).lower()
    if architecture in {"arm64", "aarch64"}:
        return bool(simd.get("compile_arm_dotprod") or simd.get("compile_arm_i8mm"))
    if any(token in architecture for token in ("x86_64", "amd64", "x64", "i386")):
        return bool(simd.get("compile_avx2") or
                    simd.get("compile_avx512_weighted") or
                    simd.get("compile_avx512vnni") or
                    simd.get("compile_avx512vpopcntdq") or
                    simd.get("compile_avxvnni"))
    return False


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_p95_regression(payload: dict[str, Any], baseline: dict[str, Any],
                         max_factor: float, failures: list[str]) -> None:
    current = {row["method"]: row for row in payload.get("results", [])}
    previous = {row["method"]: row for row in baseline.get("results", [])}
    for method, row in current.items():
        if method not in previous:
            continue
        cur_p95 = row.get("latency", {}).get("p95_ms")
        old_p95 = previous[method].get("latency", {}).get("p95_ms")
        if isinstance(cur_p95, (int, float)) and isinstance(old_p95, (int, float)) and old_p95 > 0:
            require(cur_p95 <= old_p95 * max_factor,
                    f"{method} p95 regressed from {old_p95} ms to {cur_p95} ms",
                    failures)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json")
    parser.add_argument("--baseline-json", default="")
    parser.add_argument("--max-p95-factor", type=float, default=2.0)
    parser.add_argument("--allow-simd-disabled", action="store_true")
    args = parser.parse_args()

    payload = load_json(args.result_json)
    failures: list[str] = []

    require(payload.get("suite") == "pgturbohybrid_real_rag_fiqa",
            "unexpected or missing suite name", failures)
    require(payload.get("dimensions") == 1536,
            "benchmark did not use 1536-dimensional OpenAI-compatible embeddings",
            failures)
    require(payload.get("rows", 0) > 0 and payload.get("queries", 0) > 0 and payload.get("qrels", 0) > 0,
            "benchmark did not load real corpus, queries, and qrels", failures)

    simd = payload.get("environment", {}).get("pgturbohybrid_simd_capabilities")
    require(isinstance(simd, dict), "missing turbohybrid_simd_capabilities()", failures)
    if isinstance(simd, dict) and not args.allow_simd_disabled:
        require(simd_available(simd), "pgturbohybrid SIMD is disabled unexpectedly", failures)

    plan_checks = payload.get("plan_checks", {})
    turbo_methods = [row for row in payload.get("results", [])
                     if str(row.get("base_method", "")).startswith("pgturbohybrid")]
    require(bool(turbo_methods), "no pgturbohybrid result rows found", failures)
    for row in turbo_methods:
        method = row["method"]
        require(method in plan_checks, f"missing EXPLAIN plan for {method}", failures)
        if method in plan_checks:
            require(plan_uses_turbohybrid(plan_checks[method]),
                    f"{method} did not use fiqa_turbohybrid_idx", failures)

        scan_summary = row.get("scan_summary", {})
        require(scan_summary.get("index_used") is True,
                f"{method} did not report index_used=true", failures)
        if method == "pgturbohybrid":
            require(scan_summary.get("profile") == "latency",
                    "default pgturbohybrid run did not use latency profile", failures)
            require(scan_summary.get("dense_k_effective_max") == 100,
                    "default pgturbohybrid run did not use dense_k=100", failures)
            require(scan_summary.get("bm25_k_effective_max") == 100,
                    "default pgturbohybrid run did not use bm25_k=100", failures)
            require(scan_summary.get("final_k_inferred_count", 0) > 0,
                    "default pgturbohybrid run did not infer final_k from SQL LIMIT", failures)

        probe = row.get("bm25_cache_probe")
        require(isinstance(probe, list) and probe,
                f"{method} is missing BM25 cache probe diagnostics", failures)
        if isinstance(probe, list):
            warm_rows = [item for item in probe
                         if item.get("phase") in {"warm_after_one", "warm_after_ten"}]
            require(bool(warm_rows), f"{method} has no warm BM25 cache probe rows", failures)
            for item in warm_rows:
                label = f"{method}/{item.get('category')}/{item.get('phase')}"
                if isinstance(item.get("bm25_first_query"), bool):
                    require(item.get("bm25_first_query") is False,
                            f"{label} still reports bm25_first_query", failures)
                require(item.get("bm25_cache_hit") is True,
                        f"{label} did not hit the BM25 cache after warmup", failures)

    if args.baseline_json:
        check_p95_regression(payload, load_json(args.baseline_json),
                             args.max_p95_factor, failures)

    if failures:
        for failure in failures:
            print(f"perf-smoke failure: {failure}", file=sys.stderr)
        return 1

    print("perf-smoke structural checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
