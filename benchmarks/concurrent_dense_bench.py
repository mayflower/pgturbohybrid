#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psycopg[binary]>=3.2",
# ]
# ///
"""Concurrent dense pgturbohybrid benchmark with per-connection warmup.

This harness is intentionally client-side.  It opens N persistent PostgreSQL
connections, warms every connection, then starts the timed phase across those
same already-warmed connections.  That makes cold native-cache build artifacts
visible without letting them contaminate steady-state concurrency percentiles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


try:
    import psycopg  # type: ignore[import-not-found]

    DB_DRIVER = "psycopg"
except ImportError:  # pragma: no cover - depends on local benchmark env
    try:
        import psycopg2  # type: ignore[import-not-found]

        DB_DRIVER = "psycopg2"
    except ImportError:
        psycopg = None  # type: ignore[assignment]
        psycopg2 = None  # type: ignore[assignment]
        DB_DRIVER = ""


IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
STAT_FIELDS = (
    "native_cache_built_this_scan",
    "native_cache_build_us",
    "native_cache_used",
    "native_cache_scope",
    "native_cache_mode",
    "native_cache_reason",
    "native_cache_bytes",
    "native_cache_code_bytes",
    "native_cache_adj_bytes",
    "native_cache_exact_bytes",
    "graph_batch_us",
    "graph_base_us",
    "graph_traverse_us",
    "graph_code_pages_read",
    "graph_adj_pages_read",
)

NESTED_STAT_PATHS = {
    "graph_base_us": ("dense", "timing_us", "base"),
    "graph_batch_us": ("dense", "timing_us", "batch"),
    "graph_traverse_us": ("dense", "timing_us", "traverse"),
    "graph_code_pages_read": ("dense", "cache", "code_pages_read"),
    "graph_adj_pages_read": ("dense", "cache", "adj_pages_read"),
}


def require_driver() -> None:
    if DB_DRIVER:
        return
    raise SystemExit(
        "benchmarks/concurrent_dense_bench.py requires psycopg or psycopg2. "
        "Run it with uv so the inline script dependency is installed: "
        "uv run benchmarks/concurrent_dense_bench.py"
    )


def connect(dsn: str):
    require_driver()
    if DB_DRIVER == "psycopg":
        return psycopg.connect(dsn, autocommit=True)  # type: ignore[union-attr]
    conn = psycopg2.connect(dsn)  # type: ignore[union-attr]
    conn.autocommit = True
    return conn


def exec_sql(conn, sql: str, params: Iterable[Any] = ()) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))


def fetch_one(conn, sql: str, params: Iterable[Any] = ()) -> tuple[Any, ...] | None:
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        return cur.fetchone()


def fetch_all(conn, sql: str, params: Iterable[Any] = ()) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))
        return list(cur.fetchall())


def quote_ident(name: str) -> str:
    if not IDENT_RE.match(name):
        raise ValueError(f"unsupported SQL identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def quote_qualified(name: str) -> str:
    return ".".join(quote_ident(part) for part in name.split("."))


def parse_csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_csv_strings(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def jsonb_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def stat_bool(stats: dict[str, Any], key: str) -> bool:
    value = stat_value(stats, key)
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"true", "t", "1", "on", "yes"}


def stat_value(stats: dict[str, Any], key: str) -> Any:
    if key in stats:
        return stats.get(key)
    value: Any = stats
    for part in NESTED_STAT_PATHS.get(key, ()):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def stat_float(stats: dict[str, Any], key: str) -> float:
    value = stat_value(stats, key)
    if value is None or value == "":
        return 0.0
    return float(value)


def mean_stat(stats: list[dict[str, Any]], key: str) -> float | None:
    if not stats:
        return None
    return statistics.fmean(stat_float(st, key) for st in stats)


def max_stat(stats: list[dict[str, Any]], key: str) -> float | None:
    if not stats:
        return None
    return max(stat_float(st, key) for st in stats)


def compact_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: stat_value(stats, key) for key in STAT_FIELDS}


@dataclass
class Query:
    seq: int
    vector_text: str
    exact_ids: list[int] | None


@dataclass
class WorkerResult:
    client_id: int
    pid: int | None = None
    warm_durations_ms: list[float] = field(default_factory=list)
    timed_durations_ms: list[float] = field(default_factory=list)
    precision_values: list[float] = field(default_factory=list)
    first_warm_stats: dict[str, Any] = field(default_factory=dict)
    last_warm_stats: dict[str, Any] = field(default_factory=dict)
    first_timed_stats: dict[str, Any] = field(default_factory=dict)
    last_timed_stats: dict[str, Any] = field(default_factory=dict)
    timed_wall_ms: float = 0.0
    error: str | None = None


def create_synthetic_dataset(conn, table: str, rows: int, dimensions: int) -> None:
    qtable = quote_qualified(table)
    exec_sql(conn, f"DROP TABLE IF EXISTS {qtable}")
    exec_sql(
        conn,
        f"""
        CREATE TABLE {qtable} (
            id int PRIMARY KEY,
            embedding vector({dimensions})
        )
        """,
    )
    exec_sql(
        conn,
        f"""
        INSERT INTO {qtable}(id, embedding)
        SELECT g,
               (
                 SELECT array_agg(
                            (sin(g * 0.001 + d * 0.017)
                             + 0.05 * cos(g * 0.013 + d * 0.031))::real
                            ORDER BY d)
                 FROM generate_series(1, %s) AS d
               )::real[]::vector
        FROM generate_series(1, %s) AS g
        """,
        (dimensions, rows),
    )


def copy_external_dataset(
    conn,
    source_table: str,
    source_vector_column: str,
    work_table: str,
    rows: int,
    dimensions: int,
) -> None:
    source = quote_qualified(source_table)
    source_col = quote_ident(source_vector_column)
    target = quote_qualified(work_table)
    exec_sql(conn, f"DROP TABLE IF EXISTS {target}")
    exec_sql(
        conn,
        f"""
        CREATE TABLE {target} AS
        SELECT row_number() OVER ()::int AS id,
               {source_col}::vector({dimensions}) AS embedding
        FROM {source}
        WHERE {source_col} IS NOT NULL
        LIMIT %s
        """,
        (rows,),
    )
    exec_sql(conn, f"ALTER TABLE {target} ADD PRIMARY KEY (id)")


def build_index(
    conn,
    table: str,
    index_name: str,
    index_options: str,
    initial_ef_search: int,
    native_segments: int,
    rebuild: bool,
) -> tuple[float | None, int | None]:
    qtable = quote_qualified(table)
    qindex = quote_qualified(index_name)
    if rebuild:
        exec_sql(conn, f"DROP INDEX IF EXISTS {qindex}")
        options = index_options.strip()
        options_lower = options.lower()
        if "native_segments" not in options_lower:
            options = f"{options}, native_segments = {native_segments}" if options else f"native_segments = {native_segments}"
            options_lower = options.lower()
        if "graph_ef_search" not in options_lower:
            options = f"{options}, graph_ef_search = {initial_ef_search}" if options else f"graph_ef_search = {initial_ef_search}"
        started = time.perf_counter()
        exec_sql(
            conn,
            f"""
            CREATE INDEX {qindex} ON {qtable}
            USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
            WITH ({options})
            """,
        )
        build_ms: float | None = (time.perf_counter() - started) * 1000.0
    else:
        build_ms = None

    exec_sql(conn, f"ANALYZE {qtable}")
    row = fetch_one(conn, "SELECT pg_relation_size(%s::regclass)", (index_name,))
    size = int(row[0]) if row else None
    return build_ms, size


def set_index_ef_search(conn, index_name: str, ef_search: int) -> None:
    exec_sql(conn, f"ALTER INDEX {quote_qualified(index_name)} SET (graph_ef_search = {ef_search})")


def create_queries(
    conn,
    table: str,
    query_table: str,
    query_count: int,
    dimensions: int,
) -> None:
    qtable = quote_qualified(table)
    qq = quote_qualified(query_table)
    exec_sql(conn, f"DROP TABLE IF EXISTS {qq}")
    exec_sql(
        conn,
        f"""
        CREATE TABLE {qq} (
            seq int PRIMARY KEY,
            qvec vector({dimensions}) NOT NULL,
            exact_ids int[]
        )
        """,
    )
    exec_sql(
        conn,
        f"""
        INSERT INTO {qq}(seq, qvec)
        SELECT row_number() OVER (ORDER BY id)::int - 1 AS seq, embedding
        FROM {qtable}
        ORDER BY id
        LIMIT %s
        """,
        (query_count,),
    )


def load_ground_truth_from_table(conn, query_table: str, ground_truth_table: str) -> None:
    qq = quote_qualified(query_table)
    qgt = quote_qualified(ground_truth_table)
    exec_sql(
        conn,
        f"""
        UPDATE {qq} q
        SET exact_ids = gt.exact_ids
        FROM {qgt} gt
        WHERE gt.seq = q.seq
        """,
    )


def compute_ground_truth(conn, table: str, query_table: str, final_k: int) -> None:
    qtable = quote_qualified(table)
    qq = quote_qualified(query_table)
    queries = fetch_all(conn, f"SELECT seq, qvec::text FROM {qq} ORDER BY seq")
    for seq, qvec in queries:
        row = fetch_one(
            conn,
            f"""
            SELECT array_agg(id ORDER BY dist)
            FROM (
                SELECT id, embedding <=> %s::vector AS dist
                FROM {qtable}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            ) s
            """,
            (qvec, qvec, final_k),
        )
        exec_sql(conn, f"UPDATE {qq} SET exact_ids = %s WHERE seq = %s", (row[0] if row else None, seq))


def load_queries(conn, query_table: str) -> list[Query]:
    qq = quote_qualified(query_table)
    rows = fetch_all(conn, f"SELECT seq, qvec::text, exact_ids FROM {qq} ORDER BY seq")
    queries: list[Query] = []
    for seq, vector_text, exact_ids in rows:
        queries.append(Query(int(seq), str(vector_text), list(exact_ids) if exact_ids is not None else None))
    if not queries:
        raise RuntimeError("query set is empty")
    return queries


def get_last_scan_stats(conn) -> dict[str, Any]:
    row = fetch_one(conn, "SELECT turbohybrid_last_scan_stats()")
    return jsonb_value(row[0] if row else None)


def run_dense_query(
    conn,
    table: str,
    query: Query,
    dense_k: int,
    final_k: int,
) -> list[int]:
    qtable = quote_qualified(table)
    rows = fetch_all(
        conn,
        f"""
        SELECT id
        FROM {qtable}
        ORDER BY embedding <~> turbohybrid_query(
            vector_query => %s::vector,
            dense_k => %s,
            final_k => %s
        )
        LIMIT %s
        """,
        (query.vector_text, dense_k, final_k, final_k),
    )
    return [int(row[0]) for row in rows]


def set_worker_gucs(conn, scope: str, cache_mb: int, app_name: str) -> None:
    settings = {
        "application_name": app_name,
        "enable_seqscan": "off",
        "jit": "off",
        "max_parallel_workers_per_gather": "0",
        "turbohybrid.native_cache_scope": scope,
        "turbohybrid.native_cache_max_mb": str(cache_mb),
    }
    for key, value in settings.items():
        exec_sql(conn, "SELECT set_config(%s, %s, false)", (key, value))


def run_worker(
    *,
    dsn: str,
    table: str,
    queries: list[Query],
    client_id: int,
    clients: int,
    scope: str,
    cache_mb: int,
    warm_queries: int,
    timed_queries: int,
    dense_k: int,
    final_k: int,
    ready_barrier: threading.Barrier,
    timed_barrier: threading.Barrier,
    result: WorkerResult,
) -> None:
    try:
        conn = connect(dsn)
        try:
            set_worker_gucs(conn, scope, cache_mb, f"concurrent_dense_bench_{client_id}")
            row = fetch_one(conn, "SELECT pg_backend_pid()")
            result.pid = int(row[0]) if row else None
            ready_barrier.wait()

            for i in range(warm_queries):
                query = queries[(client_id + i * clients) % len(queries)]
                started = time.perf_counter()
                run_dense_query(conn, table, query, dense_k, final_k)
                result.warm_durations_ms.append((time.perf_counter() - started) * 1000.0)
                if i == 0:
                    result.first_warm_stats = get_last_scan_stats(conn)
            if warm_queries > 0:
                result.last_warm_stats = get_last_scan_stats(conn)

            timed_barrier.wait()
            timed_started = time.perf_counter()
            for i in range(timed_queries):
                query = queries[(client_id + i * clients) % len(queries)]
                started = time.perf_counter()
                ids = run_dense_query(conn, table, query, dense_k, final_k)
                result.timed_durations_ms.append((time.perf_counter() - started) * 1000.0)
                if query.exact_ids:
                    overlap = len(set(ids) & set(query.exact_ids))
                    result.precision_values.append(overlap / float(final_k))
                if i == 0:
                    result.first_timed_stats = get_last_scan_stats(conn)
            result.timed_wall_ms = (time.perf_counter() - timed_started) * 1000.0
            if timed_queries > 0:
                result.last_timed_stats = get_last_scan_stats(conn)
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover - benchmark error path
        result.error = str(exc)
        for barrier in (ready_barrier, timed_barrier):
            try:
                barrier.abort()
            except Exception:
                pass


def run_variant(
    *,
    dsn: str,
    table: str,
    queries: list[Query],
    clients: int,
    scope: str,
    cache_mb: int,
    ef_search: int,
    warm_queries: int,
    timed_queries: int,
    dense_k: int,
    final_k: int,
) -> dict[str, Any]:
    ready_barrier = threading.Barrier(clients)
    timed_barrier = threading.Barrier(clients)
    results = [WorkerResult(client_id=i) for i in range(clients)]
    threads = [
        threading.Thread(
            target=run_worker,
            kwargs={
                "dsn": dsn,
                "table": table,
                "queries": queries,
                "client_id": i,
                "clients": clients,
                "scope": scope,
                "cache_mb": cache_mb,
                "warm_queries": warm_queries,
                "timed_queries": timed_queries,
                "dense_k": dense_k,
                "final_k": final_k,
                "ready_barrier": ready_barrier,
                "timed_barrier": timed_barrier,
                "result": results[i],
            },
            daemon=True,
        )
        for i in range(clients)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    errors = [r.error for r in results if r.error]
    timed_durations = [d for r in results for d in r.timed_durations_ms]
    warm_durations = [d for r in results for d in r.warm_durations_ms]
    timed_wall_ms = max((r.timed_wall_ms for r in results), default=0.0)
    timed_count = sum(len(r.timed_durations_ms) for r in results)
    rps = timed_count / (timed_wall_ms / 1000.0) if timed_wall_ms > 0 else None

    warm_first_stats = [r.first_warm_stats for r in results if r.first_warm_stats]
    timed_first_stats = [r.first_timed_stats for r in results if r.first_timed_stats]
    timed_last_stats = [r.last_timed_stats for r in results if r.last_timed_stats]
    precisions = [p for r in results for p in r.precision_values]

    row: dict[str, Any] = {
        "status": "error" if errors else "ok",
        "error": "; ".join(errors) if errors else "",
        "clients": clients,
        "native_cache_scope_requested": scope,
        "native_cache_max_mb": cache_mb,
        "graph_ef_search": ef_search,
        "warm_queries_per_client": warm_queries,
        "timed_queries_per_client": timed_queries,
        "timed_queries_total": timed_count,
        "rps": rps,
        "p50_ms": percentile(timed_durations, 0.50),
        "p95_ms": percentile(timed_durations, 0.95),
        "p99_ms": percentile(timed_durations, 0.99),
        "warm_p95_ms": percentile(warm_durations, 0.95),
        "timed_wall_ms": timed_wall_ms,
        "precision_at_k_avg": statistics.fmean(precisions) if precisions else None,
        "precision_at_k_min": min(precisions) if precisions else None,
        "warmup_native_cache_built_this_scan": any(stat_bool(st, "native_cache_built_this_scan") for st in warm_first_stats),
        "warmup_native_cache_build_us": max_stat(warm_first_stats, "native_cache_build_us"),
        "native_cache_built_this_scan": any(stat_bool(st, "native_cache_built_this_scan") for st in timed_first_stats),
        "native_cache_build_us": max_stat(timed_first_stats, "native_cache_build_us"),
        "warm_native_cache_built_any": any(stat_bool(st, "native_cache_built_this_scan") for st in warm_first_stats),
        "warm_native_cache_build_us_max": max_stat(warm_first_stats, "native_cache_build_us"),
        "timed_native_cache_built_any": any(stat_bool(st, "native_cache_built_this_scan") for st in timed_first_stats),
        "timed_native_cache_build_us_max": max_stat(timed_first_stats, "native_cache_build_us"),
        "native_cache_used_any": any(stat_bool(st, "native_cache_used") for st in timed_last_stats),
        "native_cache_scope_observed": next((str(st.get("native_cache_scope")) for st in timed_last_stats if st.get("native_cache_scope") is not None), None),
        "native_cache_mode_observed": next((str(st.get("native_cache_mode")) for st in timed_last_stats if st.get("native_cache_mode") is not None), None),
        "native_cache_reason": next((str(st.get("native_cache_reason")) for st in timed_last_stats if st.get("native_cache_reason") is not None), None),
        "native_cache_bytes_max": max_stat(timed_last_stats, "native_cache_bytes"),
        "native_cache_code_bytes_max": max_stat(timed_last_stats, "native_cache_code_bytes"),
        "native_cache_adj_bytes_max": max_stat(timed_last_stats, "native_cache_adj_bytes"),
        "native_cache_exact_bytes_max": max_stat(timed_last_stats, "native_cache_exact_bytes"),
        "graph_batch_us": mean_stat(timed_last_stats, "graph_batch_us"),
        "graph_batch_us_avg": mean_stat(timed_last_stats, "graph_batch_us"),
        "graph_batch_us_max": max_stat(timed_last_stats, "graph_batch_us"),
        "graph_base_us": mean_stat(timed_last_stats, "graph_base_us"),
        "graph_base_us_avg": mean_stat(timed_last_stats, "graph_base_us"),
        "graph_base_us_max": max_stat(timed_last_stats, "graph_base_us"),
        "graph_traverse_us": mean_stat(timed_last_stats, "graph_traverse_us"),
        "graph_traverse_us_avg": mean_stat(timed_last_stats, "graph_traverse_us"),
        "graph_traverse_us_max": max_stat(timed_last_stats, "graph_traverse_us"),
        "graph_code_pages_read": mean_stat(timed_last_stats, "graph_code_pages_read"),
        "graph_code_pages_read_avg": mean_stat(timed_last_stats, "graph_code_pages_read"),
        "graph_code_pages_read_max": max_stat(timed_last_stats, "graph_code_pages_read"),
        "graph_adj_pages_read": mean_stat(timed_last_stats, "graph_adj_pages_read"),
        "graph_adj_pages_read_avg": mean_stat(timed_last_stats, "graph_adj_pages_read"),
        "graph_adj_pages_read_max": max_stat(timed_last_stats, "graph_adj_pages_read"),
        "client_pids": [r.pid for r in results],
        "client_results": [
            {
                "client_id": r.client_id,
                "pid": r.pid,
                "warm_durations_ms": r.warm_durations_ms,
                "timed_durations_ms": r.timed_durations_ms,
                "first_warm_stats": compact_stats(r.first_warm_stats),
                "first_timed_stats": compact_stats(r.first_timed_stats),
                "last_timed_stats": compact_stats(r.last_timed_stats),
                "precision_at_k_avg": statistics.fmean(r.precision_values) if r.precision_values else None,
                "error": r.error,
            }
            for r in results
        ],
    }
    return row


def supports_scope(dsn: str, scope: str) -> tuple[bool, str | None]:
    conn = connect(dsn)
    try:
        exec_sql(conn, "SELECT set_config('turbohybrid.native_cache_scope', %s, false)", (scope,))
        return True, None
    except Exception as exc:  # pragma: no cover - depends on extension build
        return False, str(exc)
    finally:
        conn.close()


def write_outputs(rows: list[dict[str, Any]], out_csv: Path, out_json: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scalar_rows = [{k: v for k, v in row.items() if k != "client_results"} for row in rows]
    fieldnames = sorted({key for row in scalar_rows for key in row.keys()})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)
    with out_json.open("w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("")
    print("segments,scope,cache_mb,ef,clients,rps,p50_ms,p95_ms,p99_ms,warm_built,timed_built,code_pages,adj_pages,batch_us,traverse_us,precision")
    for row in rows:
        print(
            ",".join(
                str(value if value is not None else "")
                for value in (
                    row.get("native_segments"),
                    row.get("native_cache_scope_requested"),
                    row.get("native_cache_max_mb"),
                    row.get("graph_ef_search"),
                    row.get("clients"),
                    round(row["rps"], 2) if row.get("rps") is not None else "",
                    round(row["p50_ms"], 3) if row.get("p50_ms") is not None else "",
                    round(row["p95_ms"], 3) if row.get("p95_ms") is not None else "",
                    round(row["p99_ms"], 3) if row.get("p99_ms") is not None else "",
                    row.get("warm_native_cache_built_any"),
                    row.get("timed_native_cache_built_any"),
                    round(row["graph_code_pages_read_avg"], 1) if row.get("graph_code_pages_read_avg") is not None else "",
                    round(row["graph_adj_pages_read_avg"], 1) if row.get("graph_adj_pages_read_avg") is not None else "",
                    round(row["graph_batch_us_avg"], 1) if row.get("graph_batch_us_avg") is not None else "",
                    round(row["graph_traverse_us_avg"], 1) if row.get("graph_traverse_us_avg") is not None else "",
                    round(row["precision_at_k_avg"], 4) if row.get("precision_at_k_avg") is not None else "",
                )
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pgturbohybrid dense concurrency with persistent warmed connections."
    )
    parser.add_argument("--dsn", default=os.environ.get("PGDATABASE", "pgturbohybrid_benchmark"),
                        help="libpq DSN or database name (default: PGDATABASE or pgturbohybrid_benchmark)")
    parser.add_argument("--source-table", help="external table to copy embeddings from; synthetic data is used when omitted")
    parser.add_argument("--source-vector-column", default="embedding")
    parser.add_argument("--work-table", default="concurrent_dense_bench_docs")
    parser.add_argument("--index-name", default="concurrent_dense_bench_idx")
    parser.add_argument("--query-table", default="concurrent_dense_bench_queries")
    parser.add_argument("--rows", type=int, default=1_183_514,
                        help="rows to copy/build; default matches glove-100-angular scale")
    parser.add_argument("--dimensions", type=int, default=100)
    parser.add_argument("--query-count", type=int, default=64)
    parser.add_argument("--clients", type=parse_csv_ints, default=parse_csv_ints("1,8"))
    parser.add_argument("--native-cache-scopes", type=parse_csv_strings, default=parse_csv_strings("off,per_backend,shared"))
    parser.add_argument("--native-cache-max-mb", type=parse_csv_ints, default=parse_csv_ints("0,64,512"))
    parser.add_argument("--native-segments", type=parse_csv_ints, default=parse_csv_ints("1"),
                        help="comma-separated native_segments reloption values; rebuilds the index for each value unless --reuse-index is set")
    parser.add_argument("--graph-ef-search", type=parse_csv_ints, default=parse_csv_ints("64,96,128"))
    parser.add_argument("--warm-queries", type=int, default=20)
    parser.add_argument("--timed-queries", type=int, default=100)
    parser.add_argument("--dense-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--index-options", default="quantization_bits = 4")
    parser.add_argument("--reuse-dataset", action="store_true", help="reuse --work-table instead of rebuilding/copying data")
    parser.add_argument("--reuse-index", action="store_true", help="reuse --index-name instead of rebuilding it")
    parser.add_argument("--compute-ground-truth", action="store_true",
                        help="compute exact top-k for the query set before running variants")
    parser.add_argument("--ground-truth-table",
                        help="table with columns (seq int, exact_ids int[]) to use instead of computing ground truth")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/output"))
    parser.add_argument("--output-prefix", default="concurrent_dense_bench")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_driver()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_csv = args.output_dir / f"{args.output_prefix}-{timestamp}.csv"
    out_json = args.output_dir / f"{args.output_prefix}-{timestamp}.json"

    conn = connect(args.dsn)
    try:
        exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE")
        if not args.reuse_dataset:
            if args.source_table:
                print(f"copying {args.rows} rows from {args.source_table}.{args.source_vector_column} -> {args.work_table}")
                copy_external_dataset(
                    conn,
                    args.source_table,
                    args.source_vector_column,
                    args.work_table,
                    args.rows,
                    args.dimensions,
                )
            else:
                print(f"building synthetic fallback dataset {args.work_table} ({args.rows} x {args.dimensions})")
                create_synthetic_dataset(conn, args.work_table, args.rows, args.dimensions)
        else:
            print(f"reusing dataset {args.work_table}")

        create_queries(conn, args.work_table, args.query_table, args.query_count, args.dimensions)
        if args.ground_truth_table:
            print(f"loading ground truth from {args.ground_truth_table}")
            load_ground_truth_from_table(conn, args.query_table, args.ground_truth_table)
        elif args.compute_ground_truth:
            print("computing exact ground truth for query set")
            compute_ground_truth(conn, args.work_table, args.query_table, args.final_k)
        queries = load_queries(conn, args.query_table)
    finally:
        conn.close()

    rows: list[dict[str, Any]] = []
    supported_scope_cache: dict[str, tuple[bool, str | None]] = {}
    for scope in args.native_cache_scopes:
        supported_scope_cache[scope] = supports_scope(args.dsn, scope)

    for native_segments in args.native_segments:
        admin = connect(args.dsn)
        try:
            print(
                f"building index {args.index_name} native_segments={native_segments}"
                if not args.reuse_index
                else f"reusing index {args.index_name}"
            )
            build_ms, index_size_bytes = build_index(
                admin,
                args.work_table,
                args.index_name,
                args.index_options,
                args.graph_ef_search[0],
                native_segments,
                not args.reuse_index,
            )
            print(f"native_segments={native_segments} index_size_bytes={index_size_bytes} build_ms={build_ms}")
        finally:
            admin.close()

        for ef_search in args.graph_ef_search:
            admin = connect(args.dsn)
            try:
                set_index_ef_search(admin, args.index_name, ef_search)
            finally:
                admin.close()

            for scope in args.native_cache_scopes:
                supported, error = supported_scope_cache[scope]
                if not supported:
                    for cache_mb in args.native_cache_max_mb:
                        for clients in args.clients:
                            rows.append(
                                {
                                    "status": "skipped",
                                    "error": error,
                                    "clients": clients,
                                    "native_segments": native_segments,
                                    "native_cache_scope_requested": scope,
                                    "native_cache_max_mb": cache_mb,
                                    "graph_ef_search": ef_search,
                                    "build_ms": build_ms,
                                    "index_size_bytes": index_size_bytes,
                                    "rows": args.rows,
                                    "dimensions": args.dimensions,
                                }
                            )
                    continue

                for cache_mb in args.native_cache_max_mb:
                    for clients in args.clients:
                        print(f"running segments={native_segments} scope={scope} cache_mb={cache_mb} ef={ef_search} clients={clients}")
                        row = run_variant(
                            dsn=args.dsn,
                            table=args.work_table,
                            queries=queries,
                            clients=clients,
                            scope=scope,
                            cache_mb=cache_mb,
                            ef_search=ef_search,
                            warm_queries=args.warm_queries,
                            timed_queries=args.timed_queries,
                            dense_k=args.dense_k,
                            final_k=args.final_k,
                        )
                        row.update(
                            {
                                "timestamp_utc": timestamp,
                                "driver": DB_DRIVER,
                                "dsn": args.dsn,
                                "table": args.work_table,
                                "index": args.index_name,
                                "rows": args.rows,
                                "dimensions": args.dimensions,
                                "query_count": len(queries),
                                "dense_k": args.dense_k,
                                "final_k": args.final_k,
                                "native_segments": native_segments,
                                "build_ms": build_ms,
                                "index_size_bytes": index_size_bytes,
                                "index_options": args.index_options,
                                "ground_truth_available": any(q.exact_ids for q in queries),
                            }
                        )
                        rows.append(row)
                        write_outputs(rows, out_csv, out_json)
    write_outputs(rows, out_csv, out_json)
    print_summary(rows)
    print(f"\nwrote {out_csv}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
