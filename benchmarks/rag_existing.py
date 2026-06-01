#!/usr/bin/env python3
"""Benchmark TurboHybrid against an existing RAG retrieval query."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


FINAL_K = 10
RRF_K = 60
DEFAULT_INDEX_NAME = "pgturbohybrid_rag_bench_idx"


def run_psql(database: str, sql: str, *, quiet: bool = True) -> str:
    cmd = ["psql", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-d", database, "-c", sql]
    if quiet:
        cmd.insert(1, "-q")
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()
    return result.stdout.strip()


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_ident(value: str) -> str:
    if not value:
        raise ValueError("empty SQL identifier")
    return '"' + value.replace('"', '""') + '"'


def quote_qualified_name(value: str) -> str:
    parts = value.split(".")
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"invalid qualified SQL name: {value!r}")
    return ".".join(quote_ident(part) for part in parts)


def normalize_sql_template(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    while text.endswith(";"):
        text = text[:-1].rstrip()
    if not text:
        raise ValueError(f"empty SQL template: {path}")
    return text


def vector_literal(value: Any) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped.startswith("[") or not stripped.endswith("]"):
            raise ValueError("vector_query string must be a pgvector literal like '[0.1,0.2]'")
        return stripped
    if isinstance(value, list):
        return "[" + ",".join(format(float(item), ".9g") for item in value) + "]"
    raise ValueError("vector_query must be a pgvector literal string or an array of numbers")


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "runs": len(values),
        "mean_ms": round(statistics.mean(values), 3) if values else 0.0,
        "p50_ms": round(percentile(values, 50), 3),
        "p95_ms": round(percentile(values, 95), 3),
        "p99_ms": round(percentile(values, 99), 3),
        "qps": round(1000.0 / statistics.mean(values), 3) if values else 0.0,
    }


def query_record(query_id: Any, text_query: Any, vector_query: Any) -> dict[str, str]:
    if query_id is None:
        raise ValueError("query row is missing query_id")
    if text_query is None:
        raise ValueError(f"query {query_id!r} is missing text_query")
    if vector_query is None:
        raise ValueError(f"query {query_id!r} is missing vector_query")
    return {
        "query_id": str(query_id),
        "text_query": str(text_query),
        "vector_query": vector_literal(vector_query),
    }


def load_queries_jsonl(path: Path, max_queries: int) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            queries.append(query_record(
                row.get("query_id") or row.get("id"),
                row.get("text_query") or row.get("query") or row.get("text"),
                row.get("vector_query") or row.get("embedding") or row.get("vector"),
            ))
            if max_queries > 0 and len(queries) >= max_queries:
                break
    if not queries:
        raise ValueError(f"no queries loaded from {path}")
    return queries


def load_queries_table(args: argparse.Namespace) -> list[dict[str, str]]:
    if not (args.query_table and args.query_id_column and
            args.query_text_column and args.query_vector_column):
        raise ValueError(
            "table query mode requires --query-table, --query-id-column, "
            "--query-text-column, and --query-vector-column"
        )
    table = quote_qualified_name(args.query_table)
    query_id_column = quote_ident(args.query_id_column)
    text_column = quote_ident(args.query_text_column)
    vector_column = quote_ident(args.query_vector_column)
    where_clause = f"WHERE {args.query_where}" if args.query_where else ""
    limit_clause = f"LIMIT {args.max_queries}" if args.max_queries > 0 else ""
    order_clause = f"ORDER BY {query_id_column}"
    rows_text = run_psql(args.database, f"""
SELECT COALESCE(json_agg(row_to_json(q)), '[]'::json)::text
FROM (
    SELECT
        {query_id_column}::text AS query_id,
        {text_column}::text AS text_query,
        {vector_column}::text AS vector_query
    FROM {table}
    {where_clause}
    {order_clause}
    {limit_clause}
) q;
""")
    rows = json.loads(rows_text)
    queries = [
        query_record(row.get("query_id"), row.get("text_query"), row.get("vector_query"))
        for row in rows
    ]
    if not queries:
        raise ValueError(f"no queries loaded from table {args.query_table}")
    return queries


def load_queries(args: argparse.Namespace) -> list[dict[str, str]]:
    sources = int(bool(args.queries_jsonl)) + int(bool(args.query_table))
    if sources != 1:
        raise ValueError("provide exactly one query source: --queries-jsonl or --query-table")
    if args.queries_jsonl:
        return load_queries_jsonl(Path(args.queries_jsonl), args.max_queries)
    return load_queries_table(args)


def check_jsonl_query_source(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_record(
                row.get("query_id") or row.get("id"),
                row.get("text_query") or row.get("query") or row.get("text"),
                row.get("vector_query") or row.get("embedding") or row.get("vector"),
            )
            return f"first query row is valid at line {line_no}"
    raise ValueError(f"no queries found in {path}")


def render_baseline_sql(template: str, query: dict[str, str],
                        args: argparse.Namespace) -> str:
    replacements = {
        "vector_query": sql_literal(query["vector_query"]),
        "text_query": sql_literal(query["text_query"]),
        "final_k": str(args.final_k),
        "dense_k": str(args.dense_k),
        "bm25_k": str(args.bm25_k),
        "rrf_k": str(args.rrf_k),
    }
    sql = template
    for key, value in replacements.items():
        sql = sql.replace("{" + key + "}", value)
    return sql


def render_turbohybrid_sql(query: dict[str, str], args: argparse.Namespace) -> str:
    table = quote_qualified_name(args.table)
    id_column = quote_ident(args.id_column)
    vector_column = quote_ident(args.vector_column)
    query_args = [
        f"vector_query => {sql_literal(query['vector_query'])}::vector",
        (
            "text_query => websearch_to_tsquery("
            f"{sql_literal(args.text_search_config)}, {sql_literal(query['text_query'])})"
        ),
    ]
    if args.explicit_budgets:
        query_args.extend([
            f"dense_k => {args.dense_k}",
            f"bm25_k => {args.bm25_k}",
            f"rrf_k => {args.rrf_k}",
            f"final_k => {args.final_k}",
        ])
    return f"""
SELECT {id_column}::text AS doc_id
FROM {table}
ORDER BY {vector_column} {args.vector_operator} turbohybrid_query(
    {", ".join(query_args)}
)
LIMIT {args.final_k}
""".strip()


def query_settings(profile: str | None, *, force_index: bool, local: bool = False) -> str:
    set_keyword = "SET LOCAL" if local else "SET"
    settings = []
    if profile:
        settings.append(f"{set_keyword} turbohybrid.profile = {sql_literal(profile)};")
    if force_index:
        settings.append(f"{set_keyword} enable_seqscan = off;")
    return "\n".join(settings) + ("\n" if settings else "")


def execute_timed_query(database: str, query_sql: str, *,
                        include_scan_stats: bool, profile: str | None,
                        force_index: bool = False) -> dict[str, Any]:
    marker = "__pgturbohybrid_rag_benchmark_stats__"
    stats_expr = (
        "COALESCE(turbohybrid_last_scan_stats()::jsonb, '{}'::jsonb)"
        if include_scan_stats else
        "NULL::jsonb"
    )
    settings = query_settings(profile, force_index=force_index, local=True)
    sql = f"""
BEGIN;
{settings}DECLARE pgturbohybrid_rag_benchmark_cursor NO SCROLL CURSOR FOR
{query_sql};
FETCH ALL FROM pgturbohybrid_rag_benchmark_cursor;
SELECT jsonb_build_object(
    'marker', {sql_literal(marker)},
    'last_scan_stats', {stats_expr}
)::text;
COMMIT;
"""
    started = time.monotonic()
    text = run_psql(database, sql)
    elapsed_ms = round((time.monotonic() - started) * 1000.0, 6)
    ids: list[str] = []
    scan_stats = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            ids.append(line)
            continue
        if isinstance(row, dict) and row.get("marker") == marker:
            scan_stats = row.get("last_scan_stats")
        else:
            ids.append(line)
    if scan_stats is None and include_scan_stats:
        raise ValueError("timed query did not return TurboHybrid scan stats")
    return {
        "ids": ids,
        "elapsed_ms": elapsed_ms,
        "row_count": len(ids),
        "last_scan_stats": scan_stats,
        "ok": True,
    }


def explain_query(database: str, query_sql: str, profile: str | None, *,
                  force_index: bool = False) -> Any:
    settings = query_settings(profile, force_index=force_index)
    text = run_psql(database, settings + "EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS, SETTINGS) " + query_sql)
    return json.loads(text)


def overlap_at_k(left: list[str], right: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    left_set = set(left[:k])
    right_set = set(right[:k])
    denom = min(k, len(left_set), len(right_set))
    if denom == 0:
        return 0.0
    return len(left_set & right_set) / float(denom)


def create_turbohybrid_index(args: argparse.Namespace) -> dict[str, Any]:
    table = quote_qualified_name(args.table)
    index_name = quote_qualified_name(args.index_name)
    vector_column = quote_ident(args.vector_column)
    tsvector_column = quote_ident(args.tsvector_column)
    with_clause = " WITH (exact_storage = on)" if args.exact_storage else ""
    concurrently = "" if args.no_concurrently else " CONCURRENTLY"
    before_lsn = run_psql(args.database, "SELECT pg_current_wal_lsn();")
    started = time.monotonic()
    run_psql(args.database, f"""
CREATE INDEX{concurrently} IF NOT EXISTS {index_name} ON {table}
USING turbohybrid (
    {vector_column} {args.vector_opclass},
    {tsvector_column} bm25_tsvector_turbohybrid_ops
){with_clause};
""", quiet=False)
    elapsed = time.monotonic() - started
    after_lsn = run_psql(args.database, "SELECT pg_current_wal_lsn();")
    wal_bytes = float(run_psql(args.database, f"SELECT pg_wal_lsn_diff('{after_lsn}', '{before_lsn}');"))
    index_size = int(run_psql(args.database, f"SELECT pg_relation_size({sql_literal(args.index_name)}::regclass);"))
    return {
        "created_or_reused": args.index_name,
        "elapsed_seconds": round(elapsed, 3),
        "wal_bytes": wal_bytes,
        "index_bytes": index_size,
    }


def analyze_table(args: argparse.Namespace) -> None:
    if args.no_analyze:
        return
    run_psql(args.database, f"ANALYZE {quote_qualified_name(args.table)};", quiet=False)


def extension_version(database: str, name: str) -> str:
    return run_psql(
        database,
        f"SELECT COALESCE((SELECT extversion FROM pg_extension WHERE extname = {sql_literal(name)}), 'not-installed');",
    )


def require_extension(database: str, name: str) -> str:
    version = extension_version(database, name)
    if version == "not-installed":
        raise ValueError(f"{name} extension is not installed in {database}")
    return version


def table_count(args: argparse.Namespace) -> int:
    return int(run_psql(args.database, f"SELECT count(*) FROM {quote_qualified_name(args.table)};"))


def turbohybrid_index_stats(args: argparse.Namespace) -> Any:
    try:
        text = run_psql(
            args.database,
            f"SELECT turbohybrid_index_stats({sql_literal(args.index_name)}::regclass)::text;",
        )
        return json.loads(text)
    except subprocess.CalledProcessError:
        return None


def preflight_checks(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, fn: Any) -> None:
        try:
            detail = fn()
            checks.append({"name": name, "ok": True, "detail": detail})
        except Exception as exc:  # noqa: BLE001 - preflight reports all failures.
            checks.append({"name": name, "ok": False, "detail": str(exc)})

    check("psql database connection", lambda: run_psql(args.database, "SELECT current_database();"))
    check("pgvector extension", lambda: require_extension(args.database, "vector"))
    check("pgturbohybrid extension", lambda: require_extension(args.database, "pgturbohybrid"))
    check(
        "TurboHybrid diagnostics function",
        lambda: run_psql(args.database, "SELECT turbohybrid_simd_capabilities()::text;"),
    )
    check(
        "document table and columns",
        lambda: run_psql(
            args.database,
            "SELECT "
            f"{quote_ident(args.id_column)}::text, "
            f"{quote_ident(args.vector_column)}::text, "
            f"{quote_ident(args.tsvector_column)}::text "
            f"FROM {quote_qualified_name(args.table)} LIMIT 0;",
        ) or "ok",
    )
    check("baseline SQL file", lambda: f"{Path(args.baseline_sql)}: {len(normalize_sql_template(Path(args.baseline_sql)))} bytes")
    if args.queries_jsonl:
        check("JSONL query source", lambda: check_jsonl_query_source(Path(args.queries_jsonl)))
    else:
        check(
            "query table and columns",
            lambda: run_psql(
                args.database,
                "SELECT "
                f"{quote_ident(args.query_id_column)}::text, "
                f"{quote_ident(args.query_text_column)}::text, "
                f"{quote_ident(args.query_vector_column)}::text "
                f"FROM {quote_qualified_name(args.query_table)} LIMIT 0;",
            ) or "ok",
        )
    ok = all(item["ok"] for item in checks)
    return {
        "benchmark": "rag_existing",
        "preflight": True,
        "ok": ok,
        "checks": checks,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    queries = load_queries(args)
    baseline_template = normalize_sql_template(Path(args.baseline_sql))
    index_build = create_turbohybrid_index(args) if args.create_turbohybrid_index else None
    analyze_table(args)

    for _ in range(args.warmup):
        for query in queries:
            execute_timed_query(
                args.database,
                render_baseline_sql(baseline_template, query, args),
                include_scan_stats=False,
                profile=None,
            )
            execute_timed_query(
                args.database,
                render_turbohybrid_sql(query, args),
                include_scan_stats=True,
                profile=args.profile,
                force_index=args.force_index,
            )

    per_query: list[dict[str, Any]] = []
    baseline_times: list[float] = []
    turbohybrid_times: list[float] = []
    overlaps: list[float] = []

    for measured_run in range(1, args.measured_runs + 1):
        for query in queries:
            baseline = execute_timed_query(
                args.database,
                render_baseline_sql(baseline_template, query, args),
                include_scan_stats=False,
                profile=None,
            )
            turbohybrid = execute_timed_query(
                args.database,
                render_turbohybrid_sql(query, args),
                include_scan_stats=True,
                profile=args.profile,
                force_index=args.force_index,
            )
            baseline_ids = baseline["ids"]
            turbohybrid_ids = turbohybrid["ids"]
            overlap = overlap_at_k(turbohybrid_ids, baseline_ids, args.final_k)
            baseline_times.append(float(baseline["elapsed_ms"]))
            turbohybrid_times.append(float(turbohybrid["elapsed_ms"]))
            overlaps.append(overlap)
            row: dict[str, Any] = {
                "run": measured_run,
                "query_id": query["query_id"],
                "baseline_ms": baseline["elapsed_ms"],
                "turbohybrid_ms": turbohybrid["elapsed_ms"],
                "overlap_at_k_vs_baseline": round(overlap, 6),
                "baseline_ids": baseline_ids,
                "turbohybrid_ids": turbohybrid_ids,
                "turbohybrid_last_scan_stats": turbohybrid.get("last_scan_stats"),
            }
            if args.include_query_text:
                row["text_query"] = query["text_query"]
            per_query.append(row)

    payload: dict[str, Any] = {
        "benchmark": "rag_existing",
        "description": "Bring-your-own PostgreSQL RAG benchmark",
        "database": args.database,
        "table": args.table,
        "id_column": args.id_column,
        "vector_column": args.vector_column,
        "tsvector_column": args.tsvector_column,
        "query_count": len(queries),
        "document_count": table_count(args),
        "final_k": args.final_k,
        "dense_k": args.dense_k,
        "bm25_k": args.bm25_k,
        "rrf_k": args.rrf_k,
        "profile": args.profile,
        "force_index": args.force_index,
        "timing_scope": "psql_client_wall_clock_ms",
        "explicit_budgets": args.explicit_budgets,
        "warmup": args.warmup,
        "measured_runs": args.measured_runs,
        "index_build": index_build,
        "results": [
            {
                "method": "baseline_sql",
                "summary": summarize_ms(baseline_times),
            },
            {
                "method": "pgturbohybrid",
                "summary": summarize_ms(turbohybrid_times),
                "overlap_at_k_vs_baseline_mean": round(statistics.mean(overlaps), 6) if overlaps else 0.0,
                "index_stats": turbohybrid_index_stats(args),
            },
        ],
        "per_query": [] if args.summary_only else per_query,
        "privacy": {
            "query_text_included": args.include_query_text,
            "note": "query text is omitted by default; result ids are included for overlap debugging",
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "postgres_version": run_psql(args.database, "SHOW server_version;"),
            "pgvector_version": extension_version(args.database, "vector"),
            "pgturbohybrid_version": extension_version(args.database, "pgturbohybrid"),
            "pgturbohybrid_simd_capabilities": json.loads(
                run_psql(args.database, "SELECT turbohybrid_simd_capabilities()::text;")
            ),
            "host_cpu_count": os.cpu_count(),
            "host_load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        },
    }

    if args.explain:
        first_query = queries[0]
        payload["plan_checks"] = {
            "baseline_sql": explain_query(
                args.database,
                render_baseline_sql(baseline_template, first_query, args),
                None,
            ),
            "pgturbohybrid": explain_query(
                args.database,
                render_turbohybrid_sql(first_query, args),
                args.profile,
                force_index=args.force_index,
            ),
        }

    return payload


def default_output_path() -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return str(Path("benchmarks/results") / f"rag_existing_{stamp}.json")


def write_output(path: str, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path == "-":
        print(text)
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", ""))
    parser.add_argument("--table", required=True, help="document table, optionally schema-qualified")
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--vector-column", required=True)
    parser.add_argument("--tsvector-column", required=True)
    parser.add_argument("--queries-jsonl", default="")
    parser.add_argument("--query-table", default="")
    parser.add_argument("--query-id-column", default="")
    parser.add_argument("--query-text-column", default="")
    parser.add_argument("--query-vector-column", default="")
    parser.add_argument("--query-where", default="")
    parser.add_argument("--baseline-sql", required=True)
    parser.add_argument("--create-turbohybrid-index", action="store_true")
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--vector-opclass", default="vector_cosine_turbohybrid_ops")
    parser.add_argument("--vector-operator", default="<~>")
    parser.add_argument("--exact-storage", action="store_true")
    parser.add_argument("--no-concurrently", action="store_true",
                        help="use CREATE INDEX instead of CREATE INDEX CONCURRENTLY")
    parser.add_argument("--no-analyze", action="store_true")
    parser.add_argument("--text-search-config", default="english")
    parser.add_argument("--profile", default="latency", choices=("balanced", "latency", "quality", "matched_recall", "debug"))
    parser.add_argument("--dense-k", type=int, default=100)
    parser.add_argument("--bm25-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--final-k", type=int, default=FINAL_K)
    parser.add_argument("--explicit-budgets", action="store_true",
                        help="pass dense_k, bm25_k, rrf_k, and final_k to turbohybrid_query")
    parser.add_argument("--no-force-index", dest="force_index", action="store_false",
                        help="allow the planner to choose a non-index TurboHybrid path")
    parser.set_defaults(force_index=True)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="run preflight checks and exit before creating indexes or running queries")
    parser.add_argument("--include-query-text", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output", default=os.environ.get("OUTPUT", default_output_path()))


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.database:
        parser.error("--database is required unless PGDATABASE is set")
    if args.final_k <= 0:
        parser.error("--final-k must be positive")
    if args.dense_k < 0 or args.bm25_k < 0:
        parser.error("--dense-k and --bm25-k must be non-negative")
    if args.warmup < 0 or args.measured_runs <= 0:
        parser.error("--warmup must be non-negative and --measured-runs must be positive")
    if bool(args.queries_jsonl) == bool(args.query_table):
        parser.error("provide exactly one query source: --queries-jsonl or --query-table")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    args = parser.parse_args()
    validate_args(args, parser)
    if args.check:
        payload = preflight_checks(args)
        write_output("-", payload)
        raise SystemExit(0 if payload["ok"] else 2)
    try:
        payload = run_benchmark(args)
    except (json.JSONDecodeError, ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    write_output(args.output, payload)


if __name__ == "__main__":
    main()
