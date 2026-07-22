#!/usr/bin/env python3
"""Deterministic PR quality gate; intentionally has no latency threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--min-dense-recall", type=float, default=0.90)
    parser.add_argument("--min-high-recall", type=float, default=0.97)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text())
    failures: list[str] = []
    if payload.get("seed") is None:
        failures.append("missing fixed seed")
    profiles = payload.get("profiles", {})
    for name, threshold in (("dense", args.min_dense_recall), ("high_recall", args.min_high_recall)):
        row = profiles.get(name, {})
        if row.get("recall_at_k", -1) < threshold:
            failures.append(f"{name} recall_at_k below {threshold}")
        if row.get("index_used") is not True:
            failures.append(f"{name} did not use turbohybrid index")
        if row.get("dead_results", 1) != 0:
            failures.append(f"{name} returned dead tuples")
        if row.get("linear_fallback") is True:
            failures.append(f"{name} used linear fallback")
        if row.get("candidate_count", 0) > row.get("candidate_bound", -1):
            failures.append(f"{name} exceeded candidate bound")
        if str(row.get("kernel", "")) in {"", "none", "scalar_linear"}:
            failures.append(f"{name} did not report an allowed native kernel")
    if failures:
        for failure in failures:
            print(f"quality-gate failure: {failure}")
        return 1
    print("deterministic recall quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
