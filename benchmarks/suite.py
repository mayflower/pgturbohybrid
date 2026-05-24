#!/usr/bin/env python3
"""Small reproducible benchmark harness for standalone pgturbohybrid."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"


def run_psql(database: str, sql: str) -> str:
    cmd = ["psql", "-q", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-d", database, "-c", sql]
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout.strip()


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


def summarize_ms(samples: list[float]) -> dict[str, float]:
    return {
        "avg_ms": statistics.mean(samples) if samples else 0.0,
        "p50_ms": percentile(samples, 50),
        "p95_ms": percentile(samples, 95),
        "p99_ms": percentile(samples, 99),
        "min_ms": min(samples) if samples else 0.0,
        "max_ms": max(samples) if samples else 0.0,
    }


def timed_psql(database: str, sql: str) -> float:
    query = sql.strip().removesuffix(";")
    explained = run_psql(
        database,
        "SET enable_seqscan = off;\n"
        f"EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) {query};",
    )
    return float(json.loads(explained)[0]["Execution Time"])


def write_output(path: str, payload: dict[str, Any]) -> None:
    if not path:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


def vector_expr(dimensions: int) -> str:
    return (
        "('[' || (SELECT string_agg((((g + d) % 17)::float8 / 17)::text, ',') "
        f"FROM generate_series(1, {dimensions}) AS d) || ']'"
        f")::vector({dimensions})"
    )


def query_vector(dimensions: int) -> str:
    values = ["1" if i == 0 else "0" for i in range(dimensions)]
    return "'[" + ",".join(values) + f"]'::vector({dimensions})"


def setup_fixture(database: str, rows: int, dimensions: int) -> None:
    sql = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
DROP TABLE IF EXISTS pgturbohybrid_bench_docs;
CREATE TABLE pgturbohybrid_bench_docs (
    id int PRIMARY KEY,
    embedding vector({dimensions}) NOT NULL,
    body text NOT NULL,
    body_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
);
INSERT INTO pgturbohybrid_bench_docs (id, embedding, body)
SELECT g,
       ({vector_expr(dimensions)}),
       CASE
           WHEN g % 10 = 0 THEN 'postgres hybrid search'
           WHEN g % 10 = 1 THEN 'vector database search'
           WHEN g % 10 = 2 THEN 'bm25 lexical retrieval'
           ELSE 'ordinary document text'
       END
FROM generate_series(1, {rows}) AS g;
ANALYZE pgturbohybrid_bench_docs;
"""
    run_psql(database, sql)


def build_index(database: str, method: str) -> dict[str, Any]:
    if method == "postgres_sql_rrf":
        sql = """
DROP INDEX IF EXISTS pgturbohybrid_bench_hnsw_idx;
DROP INDEX IF EXISTS pgturbohybrid_bench_fts_idx;
CREATE INDEX pgturbohybrid_bench_hnsw_idx ON pgturbohybrid_bench_docs
USING hnsw (embedding vector_cosine_ops);
CREATE INDEX pgturbohybrid_bench_fts_idx ON pgturbohybrid_bench_docs
USING gin (body_tsv);
"""
        indexes = ["pgturbohybrid_bench_hnsw_idx", "pgturbohybrid_bench_fts_idx"]
    elif method == "pgturbohybrid":
        sql = """
DROP INDEX IF EXISTS pgturbohybrid_bench_idx;
CREATE INDEX pgturbohybrid_bench_idx ON pgturbohybrid_bench_docs
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = on);
"""
        indexes = ["pgturbohybrid_bench_idx"]
    elif method == "pgturbohybrid_exact_storage_off":
        sql = """
DROP INDEX IF EXISTS pgturbohybrid_bench_idx;
CREATE INDEX pgturbohybrid_bench_idx ON pgturbohybrid_bench_docs
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = off);
"""
        indexes = ["pgturbohybrid_bench_idx"]
    else:
        raise ValueError(f"unsupported method: {method}")

    start = time.perf_counter()
    run_psql(database, sql)
    build_ms = (time.perf_counter() - start) * 1000.0
    index_size = int(
        run_psql(
            database,
            "SELECT COALESCE(sum(pg_relation_size(indexrelid)), 0) "
            "FROM pg_index WHERE indexrelid::regclass::text IN ("
            + ",".join("'" + item + "'" for item in indexes)
            + ");",
        )
        or "0"
    )
    return {"build_ms": build_ms, "index_bytes": index_size, "indexes": indexes}


def query_sql(method: str, dimensions: int, dense_k: int, bm25_k: int, final_k: int) -> str:
    qv = query_vector(dimensions)
    if method == "postgres_sql_rrf":
        return f"""
WITH dense AS MATERIALIZED (
    SELECT id, row_number() OVER () AS rank
    FROM (
        SELECT d.id
        FROM pgturbohybrid_bench_docs d
        ORDER BY d.embedding <=> {qv}
        LIMIT {dense_k}
    ) s
),
lexical AS MATERIALIZED (
    SELECT id, row_number() OVER () AS rank
    FROM (
        SELECT d.id
        FROM pgturbohybrid_bench_docs d
        WHERE d.body_tsv @@ websearch_to_tsquery('english', 'postgres hybrid search')
        ORDER BY ts_rank_cd(d.body_tsv, websearch_to_tsquery('english', 'postgres hybrid search')) DESC, d.id
        LIMIT {bm25_k}
    ) s
)
SELECT COALESCE(dense.id, lexical.id) AS id
FROM dense
FULL OUTER JOIN lexical USING (id)
ORDER BY COALESCE(1.0 / (60 + dense.rank), 0.0) +
         COALESCE(1.0 / (60 + lexical.rank), 0.0) DESC,
         COALESCE(dense.id, lexical.id)
LIMIT {final_k};
"""
    return f"""
SELECT id
FROM pgturbohybrid_bench_docs
ORDER BY embedding <~> turbohybrid_query(
    vector_query => {qv},
    text_query => websearch_to_tsquery('english', 'postgres hybrid search'),
    dense_k => {dense_k},
    bm25_k => {bm25_k},
    final_k => {final_k}
)
LIMIT {final_k};
"""


def run_method(database: str, method: str, dimensions: int, dense_k: int, bm25_k: int, final_k: int,
               warmup: int, runs: int) -> dict[str, Any]:
    build = build_index(database, method)
    sql = query_sql(method, dimensions, dense_k, bm25_k, final_k)
    for _ in range(warmup):
        run_psql(database, "SET enable_seqscan = off;\n" + sql)
    samples = [timed_psql(database, sql) for _ in range(runs)]
    return {
        "method": method,
        "build": build,
        "latency": summarize_ms(samples),
        "runs": runs,
        "warmup": warmup,
    }


def command_run_system(args: argparse.Namespace) -> None:
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    setup_fixture(args.database, args.rows, args.dimensions)
    results = [
        run_method(args.database, method, args.dimensions, args.dense_k, args.bm25_k,
                   args.final_k, args.warmup, args.runs)
        for method in methods
    ]
    payload = {
        "suite": "pgturbohybrid_system_smoke",
        "dataset": "synthetic-postgres",
        "rows": args.rows,
        "dimensions": args.dimensions,
        "dense_k": args.dense_k,
        "bm25_k": args.bm25_k,
        "final_k": args.final_k,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "postgres_version": run_psql(args.database, "SHOW server_version;"),
            "pgvector_version": run_psql(args.database, "SELECT extversion FROM pg_extension WHERE extname = 'vector';"),
            "pgturbohybrid_version": run_psql(args.database, "SELECT extversion FROM pg_extension WHERE extname = 'pgturbohybrid';"),
        },
        "results": results,
    }
    write_output(args.output, payload)


def command_list(_: argparse.Namespace) -> None:
    methods = json.loads((CONFIG_DIR / "methods.json").read_text(encoding="utf-8"))
    datasets = json.loads((CONFIG_DIR / "datasets.json").read_text(encoding="utf-8"))
    print(json.dumps({"methods": methods["methods"], "datasets": datasets["datasets"]}, indent=2, sort_keys=True))


def command_plan(_: argparse.Namespace) -> None:
    print(json.dumps({
        "baselines": ["postgres_sql_rrf", "pgturbohybrid"],
        "smoke": {
            "command": "python3 benchmarks/suite.py run-system-synthetic --rows 1000 --runs 3",
            "requires": ["CREATE EXTENSION vector", "CREATE EXTENSION pgturbohybrid"],
        },
        "publishable_result_fields": [
            "hardware",
            "os",
            "postgres_version",
            "pgvector_ref",
            "pgturbohybrid_ref",
            "dataset",
            "embedding_model",
            "warmup",
            "runs",
            "p50_ms",
            "p95_ms",
            "p99_ms",
            "index_bytes",
            "build_ms",
            "wal_bytes",
            "recall",
            "nDCG",
            "MRR",
            "MAP",
        ],
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)

    list_parser = sub.add_parser("list", help="List configured benchmark datasets and methods")
    list_parser.set_defaults(func=command_list)

    plan_parser = sub.add_parser("plan", help="Show the reproducibility checklist")
    plan_parser.set_defaults(func=command_plan)

    run_parser = sub.add_parser("run-system-synthetic", help="Run a small synthetic systems benchmark")
    run_parser.add_argument("--database", default=os.environ.get("PGDATABASE", "contrib_regression"))
    run_parser.add_argument("--rows", type=int, default=1000)
    run_parser.add_argument("--dimensions", type=int, default=16)
    run_parser.add_argument("--methods", default="postgres_sql_rrf,pgturbohybrid")
    run_parser.add_argument("--dense-k", type=int, default=100)
    run_parser.add_argument("--bm25-k", type=int, default=100)
    run_parser.add_argument("--final-k", type=int, default=10)
    run_parser.add_argument("--warmup", type=int, default=1)
    run_parser.add_argument("--runs", type=int, default=3)
    run_parser.add_argument("--output", default="")
    run_parser.set_defaults(func=command_run_system)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
