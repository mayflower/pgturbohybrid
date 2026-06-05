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


@dataclass(frozen=True)
class TokenPlanComparison:
    input_text: str
    token_plan_parity_passed: bool
    retain_parity_passed: bool
    vector_count_passed: bool
    first_token_mismatch: dict[str, Any] | None
    first_retain_mismatch: dict[str, Any] | None
    pg_final_token_ids: list[int]
    golden_final_token_ids: list[int]


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
    parser.add_argument(
        "--golden-token-plan",
        default=None,
        help="Optional converter token-plan golden JSON to compare before vectors",
    )
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


def pg_debug(cur: Any, pg_model: str, text: str) -> dict[str, Any]:
    cur.execute("SELECT colbert_debug(%s, %s)::text", (pg_model, text))
    return json.loads(cur.fetchone()[0])


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


def normalize_piece(piece: Any) -> str | None:
    if piece is None:
        return None
    return str(piece)


def first_sequence_mismatch(
    left: list[Any],
    right: list[Any],
    left_name: str,
    right_name: str,
) -> dict[str, Any] | None:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return {
                "index": index,
                left_name: left_value,
                right_name: right_value,
            }
    if len(left) != len(right):
        return {
            "index": min(len(left), len(right)),
            f"{left_name}_length": len(left),
            f"{right_name}_length": len(right),
        }
    return None


def load_golden_token_plans(path: str, role: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)
    plans = payload.get("plans")
    if not isinstance(plans, list):
        raise SystemExit("--golden-token-plan must contain a plans array")
    filtered = [plan for plan in plans if plan.get("role", role) == role]
    if not filtered:
        raise SystemExit(f"--golden-token-plan contains no plans for role {role!r}")
    return filtered


def select_golden_plan(
    plans: list[dict[str, Any]],
    text: str,
    index: int,
) -> dict[str, Any]:
    for plan in plans:
        if plan.get("input_text") == text:
            return plan
    if index < len(plans):
        return plans[index]
    raise SystemExit(
        f"--golden-token-plan has no plan for input {text!r} at position {index}"
    )


def retained_token_ids(token_ids: list[int], retain_mask: list[int]) -> list[int]:
    return [token_id for token_id, retained in zip(token_ids, retain_mask) if retained]


def compare_token_plan(
    debug_payload: dict[str, Any],
    golden_plan: dict[str, Any],
    input_text: str,
) -> TokenPlanComparison:
    tokens = debug_payload.get("token_plan", {}).get("tokens", [])
    if not isinstance(tokens, list):
        raise SystemExit("colbert_debug payload did not contain token_plan.tokens")

    pg_token_ids = [int(token["id"]) for token in tokens]
    pg_pieces = [normalize_piece(token.get("piece")) for token in tokens]
    pg_retain_mask = [1 if bool(token.get("retained")) else 0 for token in tokens]
    pg_final = retained_token_ids(pg_token_ids, pg_retain_mask)

    golden_token_ids = golden_plan.get("token_ids_after_padding_truncation")
    if not isinstance(golden_token_ids, list):
        raise SystemExit("golden token plan lacks token_ids_after_padding_truncation")
    golden_token_ids = [int(token_id) for token_id in golden_token_ids]

    golden_pieces = golden_plan.get("token_pieces")
    if isinstance(golden_pieces, list) and any(piece is not None for piece in pg_pieces):
        token_mismatch = first_sequence_mismatch(
            list(zip(golden_token_ids, [normalize_piece(piece) for piece in golden_pieces])),
            list(zip(pg_token_ids, pg_pieces)),
            "golden",
            "pg",
        )
    else:
        token_mismatch = first_sequence_mismatch(
            golden_token_ids,
            pg_token_ids,
            "golden",
            "pg",
        )

    golden_retain_mask = golden_plan.get("retain_mask")
    if not isinstance(golden_retain_mask, list):
        raise SystemExit("golden token plan lacks retain_mask")
    golden_retain_mask = [int(value) for value in golden_retain_mask]
    retain_mismatch = first_sequence_mismatch(
        golden_retain_mask,
        pg_retain_mask,
        "golden",
        "pg",
    )
    golden_final = retained_token_ids(golden_token_ids, golden_retain_mask)

    expected_count = golden_plan.get("final_vector_count")
    if expected_count is None:
        expected_count = len(golden_final)
    vector_count = int(debug_payload.get("vector_count", -1))

    return TokenPlanComparison(
        input_text=input_text,
        token_plan_parity_passed=token_mismatch is None,
        retain_parity_passed=retain_mismatch is None and pg_final == golden_final,
        vector_count_passed=vector_count == int(expected_count),
        first_token_mismatch=token_mismatch,
        first_retain_mismatch=retain_mismatch,
        pg_final_token_ids=pg_final,
        golden_final_token_ids=golden_final,
    )


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
    golden_plans = (
        load_golden_token_plans(args.golden_token_plan, args.role)
        if args.golden_token_plan
        else None
    )
    model = load_pylate_model(args.model_name_or_path)
    results: list[Comparison] = []
    token_plan_results: list[TokenPlanComparison] = []
    ranking: RankingSmoke | None = None

    try:
        import psycopg
    except ImportError as exc:
        raise SystemExit("psycopg is not installed; install psycopg before running parity checks") from exc

    with psycopg.connect(args.pg_dsn) as conn:
        with conn.cursor() as cur:
            if args.model_dir:
                cur.execute("SELECT set_config(%s, %s, false)", ("pg_colbert_llama.model_dir", args.model_dir))
            if args.expected_dim:
                cur.execute("SELECT set_config(%s, %s, false)", ("pg_colbert_llama.expected_dim", str(args.expected_dim)))
            for index, text in enumerate(texts):
                if golden_plans is not None:
                    token_plan_results.append(
                        compare_token_plan(
                            pg_debug(cur, args.pg_model_alias, text),
                            select_golden_plan(golden_plans, text, index),
                            text,
                        )
                    )
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

    token_plan_parity_passed = all(
        result.token_plan_parity_passed for result in token_plan_results
    ) if token_plan_results else True
    retain_parity_passed = all(
        result.retain_parity_passed for result in token_plan_results
    ) if token_plan_results else True
    vector_count_passed = all(
        result.vector_count_passed for result in token_plan_results
    ) if token_plan_results else True
    vector_parity_passed = all(
        result.pg_dim == result.pylate_dim
        and abs(result.pg_count - result.pylate_count) <= args.count_tolerance
        and vector_count_passed
        and result.max_abs_diff <= args.max_abs_diff
        and result.max_norm_deviation <= args.max_norm_deviation
        and result.max_cosine_delta <= args.max_cosine_delta
        for result in results
    )
    ranking_passed = ranking is None or (
        ranking.pg_top_index == ranking.expected_index
        and ranking.pylate_top_index == ranking.expected_index
    )
    first_token_mismatch = next(
        (
            result.first_token_mismatch
            for result in token_plan_results
            if result.first_token_mismatch is not None
        ),
        None,
    )
    first_retain_mismatch = next(
        (
            result.first_retain_mismatch
            for result in token_plan_results
            if result.first_retain_mismatch is not None
        ),
        None,
    )
    passed = (
        token_plan_parity_passed
        and retain_parity_passed
        and vector_parity_passed
        and ranking_passed
    )

    payload: dict[str, Any] = {
        "passed": passed,
        "token_plan_parity_passed": token_plan_parity_passed,
        "retain_parity_passed": retain_parity_passed,
        "vector_parity_passed": vector_parity_passed,
        "ranking_passed": ranking_passed,
        "first_token_mismatch": first_token_mismatch,
        "first_retain_mismatch": first_retain_mismatch,
        "texts": len(results),
        "max_abs_diff": max(result.max_abs_diff for result in results),
        "mean_abs_diff": sum(result.mean_abs_diff for result in results) / len(results),
        "max_norm_deviation": max(result.max_norm_deviation for result in results),
        "max_cosine_delta": max(result.max_cosine_delta for result in results),
        "comparisons": [result.__dict__ for result in results],
    }
    if token_plan_results:
        payload["token_plan_comparisons"] = [
            result.__dict__ for result in token_plan_results
        ]
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
