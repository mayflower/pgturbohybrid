#!/usr/bin/env python3
"""Compare pg_colbert_llama vectors with PyLate for a text file."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Comparison:
    pg_dim: int
    pylate_dim: int
    pg_count: int
    pylate_count: int
    compared_count: int
    max_abs_diff: float
    mean_abs_diff: float
    max_norm_deviation: float
    max_cosine_delta: float


@dataclass(frozen=True)
class RankingSmoke:
    query: str
    expected_index: int
    pg_top_index: int
    pylate_top_index: int
    pg_scores: list[float]
    pylate_scores: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pg_colbert_llama JSON output against PyLate.",
    )
    parser.add_argument("--pg-dsn", "--dsn", dest="pg_dsn", default=os.environ.get("DATABASE_URL", ""))
    parser.add_argument(
        "--pg-model-alias",
        "--pg-model",
        dest="pg_model_alias",
        required=True,
        help="PostgreSQL model spec, for example alias:query",
    )
    parser.add_argument(
        "--model-name-or-path",
        "--pylate-model",
        dest="model_name_or_path",
        required=True,
        help="PyLate/Hugging Face model id or path",
    )
    parser.add_argument("--texts-file", required=True, help="UTF-8 file with one text per line")
    parser.add_argument("--role", choices=("query", "doc"), required=True)
    parser.add_argument("--model-dir", default=None, help="SET pg_colbert_llama.model_dir before encoding")
    parser.add_argument("--expected-dim", type=int, default=None)
    parser.add_argument("--count-tolerance", type=int, default=2)
    parser.add_argument("--max-abs-diff", type=float, default=5e-3)
    parser.add_argument("--max-norm-deviation", type=float, default=5e-3)
    parser.add_argument("--max-cosine-delta", type=float, default=5e-3)
    parser.add_argument(
        "--ranking-query",
        default=None,
        help="Optional query for a MaxSim ranking smoke test.",
    )
    parser.add_argument(
        "--ranking-docs-file",
        default=None,
        help="UTF-8 file with one candidate passage per line for the ranking smoke test.",
    )
    parser.add_argument(
        "--ranking-expected-index",
        type=int,
        default=0,
        help="Zero-based candidate index expected to rank first in the ranking smoke test.",
    )
    parser.add_argument(
        "--pg-query-model-alias",
        default=None,
        help="PostgreSQL query model spec for ranking smoke, for example alias:query.",
    )
    parser.add_argument(
        "--pg-doc-model-alias",
        default=None,
        help="PostgreSQL doc model spec for ranking smoke, for example alias:doc.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable result")
    return parser.parse_args()


def read_texts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as file:
        texts = [line.rstrip("\n") for line in file]
    texts = [text for text in texts if text]
    if not texts:
        raise SystemExit("--texts-file did not contain any non-empty lines")
    return texts


def pg_vectors(cur: Any, pg_model: str, text: str) -> list[list[float]]:
    cur.execute("SELECT colbert(%s, %s)::text", (pg_model, text))
    payload = json.loads(cur.fetchone()[0])
    return [[float(value) for value in row] for row in payload["vectors"]]


def pylate_vectors(model: Any, text: str, role: str) -> list[list[float]]:
    encoded: Any = model.encode([text], is_query=role == "query")
    first = encoded[0] if isinstance(encoded, (list, tuple)) else encoded
    if hasattr(first, "tolist"):
        first = first.tolist()
    return [[float(value) for value in row] for row in first]


def load_pylate_model(model_name: str) -> Any:
    try:
        from pylate import models
    except ImportError as exc:
        raise SystemExit("PyLate is not installed; install pylate before running parity checks") from exc

    return models.ColBERT(model_name)


def l2_norm(row: list[float]) -> float:
    return math.sqrt(sum(value * value for value in row))


def l2_normalize(row: list[float]) -> list[float]:
    norm = l2_norm(row)
    if norm == 0.0:
        return row
    return [value / norm for value in row]


def compare(pg: list[list[float]], pylate: list[list[float]]) -> Comparison:
    if not pg or not pylate:
        raise SystemExit("both PostgreSQL and PyLate must return at least one token vector")
    pg_dim = len(pg[0])
    pylate_dim = len(pylate[0])
    if any(len(row) != pg_dim for row in pg):
        raise SystemExit("PostgreSQL returned ragged vectors")
    if any(len(row) != pylate_dim for row in pylate):
        raise SystemExit("PyLate returned ragged vectors")

    compared = min(len(pg), len(pylate))
    if pg_dim != pylate_dim:
        return Comparison(pg_dim, pylate_dim, len(pg), len(pylate), compared, math.inf, math.inf, math.inf, math.inf)

    max_abs = 0.0
    total_abs = 0.0
    total_values = 0
    max_norm_deviation = 0.0
    max_cos_delta = 0.0
    for pg_row, pylate_row in zip(pg[:compared], pylate[:compared]):
        pg_norm = l2_normalize(pg_row)
        pylate_norm = l2_normalize(pylate_row)
        cosine = sum(a * b for a, b in zip(pg_norm, pylate_norm))
        max_cos_delta = max(max_cos_delta, abs(1.0 - cosine))
        max_norm_deviation = max(max_norm_deviation, abs(1.0 - l2_norm(pg_row)))
        for pg_value, pylate_value in zip(pg_row, pylate_row):
            delta = abs(pg_value - pylate_value)
            max_abs = max(max_abs, delta)
            total_abs += delta
            total_values += 1

    return Comparison(
        pg_dim=pg_dim,
        pylate_dim=pylate_dim,
        pg_count=len(pg),
        pylate_count=len(pylate),
        compared_count=compared,
        max_abs_diff=max_abs,
        mean_abs_diff=total_abs / total_values,
        max_norm_deviation=max_norm_deviation,
        max_cosine_delta=max_cos_delta,
    )


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def maxsim(query: list[list[float]], document: list[list[float]]) -> float:
    if not query or not document:
        raise SystemExit("ranking smoke vectors must be non-empty")
    q_dim = len(query[0])
    d_dim = len(document[0])
    if q_dim != d_dim:
        raise SystemExit(f"ranking smoke dim mismatch: query dim {q_dim}, doc dim {d_dim}")
    return sum(max(dot(query_row, doc_row) for doc_row in document) for query_row in query)


def top_index(scores: list[float]) -> int:
    if not scores:
        raise SystemExit("ranking smoke produced no scores")
    return max(range(len(scores)), key=lambda idx: scores[idx])


def infer_role_alias(pg_model_alias: str, role: str) -> str:
    if ":" not in pg_model_alias:
        raise SystemExit("PostgreSQL model spec must use alias:role syntax")
    alias, _old_role = pg_model_alias.rsplit(":", 1)
    return f"{alias}:{role}"


def run_ranking_smoke(
    cur: Any,
    model: Any,
    pg_query_model: str,
    pg_doc_model: str,
    query: str,
    docs: list[str],
    expected_index: int,
) -> RankingSmoke:
    if expected_index < 0 or expected_index >= len(docs):
        raise SystemExit("--ranking-expected-index is outside --ranking-docs-file")

    pg_query = pg_vectors(cur, pg_query_model, query)
    pylate_query = pylate_vectors(model, query, "query")
    pg_doc_vectors = [pg_vectors(cur, pg_doc_model, doc) for doc in docs]
    pylate_doc_vectors = [pylate_vectors(model, doc, "doc") for doc in docs]

    pg_scores = [maxsim(pg_query, doc_vectors) for doc_vectors in pg_doc_vectors]
    pylate_scores = [maxsim(pylate_query, doc_vectors) for doc_vectors in pylate_doc_vectors]

    return RankingSmoke(
        query=query,
        expected_index=expected_index,
        pg_top_index=top_index(pg_scores),
        pylate_top_index=top_index(pylate_scores),
        pg_scores=pg_scores,
        pylate_scores=pylate_scores,
    )


def main() -> int:
    args = parse_args()
    texts = read_texts(args.texts_file)
    model = load_pylate_model(args.model_name_or_path)
    results: list[Comparison] = []
    ranking: RankingSmoke | None = None

    try:
        import psycopg
    except ImportError as exc:
        raise SystemExit("psycopg is not installed; install psycopg before running parity checks") from exc

    with psycopg.connect(args.pg_dsn) as conn:
        with conn.cursor() as cur:
            if args.model_dir:
                cur.execute("SET pg_colbert_llama.model_dir = %s", (args.model_dir,))
            if args.expected_dim:
                cur.execute("SET pg_colbert_llama.expected_dim = %s", (args.expected_dim,))
            for text in texts:
                results.append(
                    compare(
                        pg_vectors(cur, args.pg_model_alias, text),
                        pylate_vectors(model, text, args.role),
                    )
                )
            if args.ranking_query or args.ranking_docs_file:
                if not args.ranking_query or not args.ranking_docs_file:
                    raise SystemExit(
                        "--ranking-query and --ranking-docs-file must be provided together"
                    )
                ranking_docs = read_texts(args.ranking_docs_file)
                pg_query_model = args.pg_query_model_alias or infer_role_alias(
                    args.pg_model_alias,
                    "query",
                )
                pg_doc_model = args.pg_doc_model_alias or infer_role_alias(
                    args.pg_model_alias,
                    "doc",
                )
                ranking = run_ranking_smoke(
                    cur,
                    model,
                    pg_query_model,
                    pg_doc_model,
                    args.ranking_query,
                    ranking_docs,
                    args.ranking_expected_index,
                )

    vector_parity_passed = all(
        result.pg_dim == result.pylate_dim
        and abs(result.pg_count - result.pylate_count) <= args.count_tolerance
        and result.max_abs_diff <= args.max_abs_diff
        and result.max_norm_deviation <= args.max_norm_deviation
        and result.max_cosine_delta <= args.max_cosine_delta
        for result in results
    )
    ranking_passed = ranking is None or (
        ranking.pg_top_index == ranking.expected_index
        and ranking.pylate_top_index == ranking.expected_index
    )
    passed = vector_parity_passed and ranking_passed

    payload: dict[str, Any] = {
        "passed": passed,
        "vector_parity_passed": vector_parity_passed,
        "ranking_passed": ranking_passed,
        "texts": len(results),
        "max_abs_diff": max(result.max_abs_diff for result in results),
        "mean_abs_diff": sum(result.mean_abs_diff for result in results) / len(results),
        "max_norm_deviation": max(result.max_norm_deviation for result in results),
        "max_cosine_delta": max(result.max_cosine_delta for result in results),
        "comparisons": [result.__dict__ for result in results],
    }
    if ranking is not None:
        payload["ranking"] = ranking.__dict__
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
