#!/usr/bin/env python3
"""Run DBpedia ColBERT multivector generation, insert, retrieval, and quality benchmarks."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "psycopg[binary]>=3.2",
#   "pyarrow>=16",
# ]
# ///

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg


DEFAULT_METHODS = "pgturbohybrid_colbert_multivector_query_only"
QUERY_ONLY_METHOD = "pgturbohybrid_colbert_multivector_query_only"
RRF_METHOD = "pgturbohybrid_colbert_multivector_rrf"
DEFAULT_MODEL_PATH = ".nix-dev/models/colbert-15m/sauerkraut-modern.gguf"


@dataclass(frozen=True)
class QueryItem:
    query_id: str
    query_text: str
    multivector_text: str


@dataclass
class WorkerResult:
    client_id: int
    warm_durations_ms: list[float] = field(default_factory=list)
    timed_durations_ms: list[float] = field(default_factory=list)
    timed_wall_ms: float = 0.0
    first_timed_stats: dict[str, Any] | None = None
    last_timed_stats: dict[str, Any] | None = None
    error: str | None = None


def require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to read DBpedia parquet shards. "
            "Use the Nix bench shell or run through th-bench-dbpedia-colbert."
        ) from exc
    return pq


def portable_path(path: Path) -> str:
    try:
        return str(path.absolute().relative_to(Path.cwd().absolute()))
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def portable_argv(argv: list[str]) -> list[str]:
    result: list[str] = []
    for item in argv:
        if item.startswith(("/", "./", "../")):
            result.append(portable_path(Path(item)))
        else:
            result.append(item)
    return result


def command_metadata() -> dict[str, Any]:
    executable = portable_path(Path(sys.executable))
    argv = portable_argv(sys.argv)
    full_argv = [executable, *argv]
    return {
        "executable": executable,
        "argv": argv,
        "full_argv": full_argv,
        "shell": shlex.join(full_argv),
        "cwd": portable_path(Path.cwd()),
        "env": {
            "PGDATABASE": os.environ.get("PGDATABASE", ""),
            "PGHOST": os.environ.get("PGHOST", ""),
            "PGPORT": os.environ.get("PGPORT", ""),
            "PGUSER": os.environ.get("PGUSER", ""),
            "DBPEDIA_DATASET": os.environ.get("DBPEDIA_DATASET", ""),
            "BEIR_DBPEDIA_DATASET": os.environ.get("BEIR_DBPEDIA_DATASET", ""),
            "BEIR_DBPEDIA_QRELS": os.environ.get("BEIR_DBPEDIA_QRELS", ""),
            "PG_COLBERT_LLAMA_TEST_MODEL": os.environ.get("PG_COLBERT_LLAMA_TEST_MODEL", ""),
        },
    }


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * pct / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(values), 3) if values else 0.0,
        "p50_ms": round(percentile(values, 50), 3),
        "p95_ms": round(percentile(values, 95), 3),
        "p99_ms": round(percentile(values, 99), 3),
        "qps": round(1000.0 / statistics.mean(values), 3) if values else 0.0,
        "runs": len(values),
    }


def summarize_ints(values: list[int]) -> dict[str, float]:
    as_float = [float(value) for value in values]
    return {
        "mean": round(statistics.mean(as_float), 3) if values else 0.0,
        "min": min(values) if values else 0,
        "p50": round(percentile(as_float, 50), 3),
        "p95": round(percentile(as_float, 95), 3),
        "max": max(values) if values else 0,
        "count": len(values),
    }


def dcg(rels: list[int]) -> float:
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))


def metrics_for_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int) -> dict[str, float]:
    ndcg_total = 0.0
    recall_total = 0.0
    mrr_total = 0.0
    map_total = 0.0
    count = 0
    for qid, relevant in qrels.items():
        docs = run.get(qid, [])[:k]
        rels = [relevant.get(doc_id, 0) for doc_id in docs]
        ideal = sorted(relevant.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal)
        ndcg_total += dcg(rels) / ideal_dcg if ideal_dcg > 0 else 0.0
        hits = sum(1 for rel in rels if rel > 0)
        recall_total += hits / min(len(relevant), k)

        rr = 0.0
        precision_sum = 0.0
        seen_hits = 0
        for idx, rel in enumerate(rels, start=1):
            if rel > 0:
                if rr == 0.0:
                    rr = 1.0 / idx
                seen_hits += 1
                precision_sum += seen_hits / idx
        mrr_total += rr
        map_total += precision_sum / min(len(relevant), k)
        count += 1

    if count == 0:
        return {f"ndcg@{k}": 0.0, f"recall@{k}": 0.0, f"mrr@{k}": 0.0, f"map@{k}": 0.0}
    return {
        f"ndcg@{k}": round(ndcg_total / count, 6),
        f"recall@{k}": round(recall_total / count, 6),
        f"mrr@{k}": round(mrr_total / count, 6),
        f"map@{k}": round(map_total / count, 6),
    }


def find_parquet_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    return sorted({path.resolve() for path in files if path.is_file()})


def find_corpus_parquet(dataset: Path) -> list[Path]:
    files = find_parquet_files(dataset, ("data/*.parquet", "train/*.parquet", "*.parquet", "default/train/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no DBpedia corpus parquet shards found under {dataset}")
    return files


def find_query_parquet(beir_dataset: Path) -> list[Path]:
    files = find_parquet_files(beir_dataset, ("queries/*.parquet", "*queries*.parquet", "*.parquet"))
    if not files:
        raise FileNotFoundError(f"no BEIR DBpedia query parquet file found under {beir_dataset}")
    return files


def resolve_qrels_path(path_arg: str | None, beir_dataset: Path) -> Path:
    candidates: list[Path] = []
    if path_arg:
        candidates.append(Path(path_arg))
    env_path = os.environ.get("BEIR_DBPEDIA_QRELS")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        beir_dataset / "qrels" / "test.tsv",
        beir_dataset / "test.tsv",
        beir_dataset.parent / "beir-dbpedia-entity-qrels" / "test.tsv",
    ])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("could not find BEIR DBpedia qrels; pass --qrels or set BEIR_DBPEDIA_QRELS")


def load_queries(beir_dataset: Path) -> dict[str, str]:
    pq = require_pyarrow()
    queries: dict[str, str] = {}
    for path in find_query_parquet(beir_dataset):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["_id", "title", "text"]):
            for row in batch.to_pylist():
                query_id = str(row["_id"])
                parts = [row.get("title") or "", row.get("text") or ""]
                queries[query_id] = " ".join(part for part in parts if part).strip()
    if not queries:
        raise ValueError(f"no BEIR queries loaded from {beir_dataset}")
    return queries


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        has_header = "query-id" in header or "query_id" in header
        if not has_header:
            f.seek(0)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"unexpected qrels row in {path}: {line!r}")
            qid, docid, score = parts
            qrels.setdefault(qid, {})[docid] = int(score)
    if not qrels:
        raise ValueError(f"no qrels loaded from {path}")
    return qrels


def choose_query_ids(qrels: dict[str, dict[str, int]], queries: dict[str, str], max_queries: int) -> list[str]:
    qids = [qid for qid in sorted(qrels) if qid in queries]
    if max_queries > 0:
        qids = qids[:max_queries]
    if not qids:
        raise ValueError("no benchmark queries have both text and qrels")
    return qids


def corpus_rows(dataset: Path, max_docs: int) -> Iterable[tuple[str, str, str]]:
    pq = require_pyarrow()
    emitted = 0
    for path in find_corpus_parquet(dataset):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024, columns=["_id", "title", "text"]):
            for row in batch.to_pylist():
                if max_docs > 0 and emitted >= max_docs:
                    return
                emitted += 1
                yield str(row["_id"]), row.get("title") or "", row.get("text") or ""


def query_rows(qids: list[str], queries: dict[str, str]) -> Iterable[tuple[str, str]]:
    for qid in qids:
        yield qid, queries[qid]


def qrel_rows(qids: list[str], qrels: dict[str, dict[str, int]]) -> Iterable[tuple[str, str, int]]:
    for qid in qids:
        for doc_id, score in sorted(qrels.get(qid, {}).items()):
            yield qid, doc_id, score


def connect(args: argparse.Namespace) -> psycopg.Connection[Any]:
    return psycopg.connect(
        dbname=args.database,
        autocommit=True,
        application_name="dbpedia_colbert_multivector",
    )


def exec_sql(conn: psycopg.Connection[Any], sql: str, params: tuple[Any, ...] | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params)


def fetch_one(conn: psycopg.Connection[Any], sql: str, params: tuple[Any, ...] | None = None) -> tuple[Any, ...] | None:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def fetch_all(conn: psycopg.Connection[Any], sql: str, params: tuple[Any, ...] | None = None) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def jsonb_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"value": value}
    return {"value": value}


def copy_rows(conn: psycopg.Connection[Any], sql: str, rows: Iterable[tuple[Any, ...]]) -> int:
    count = 0
    with conn.cursor() as cur:
        with cur.copy(sql) as copy:
            for row in rows:
                copy.write_row(row)
                count += 1
    return count


def set_colbert_gucs(conn: psycopg.Connection[Any], args: argparse.Namespace) -> None:
    settings = {
        "jit": "off",
        "max_parallel_workers_per_gather": "0",
        "pg_colbert_llama.model_dir": str(args.model_path.parent),
        "pg_colbert_llama.expected_dim": str(args.expected_dim),
        "pg_colbert_llama.max_doc_vectors": str(args.max_doc_vectors),
        "pg_colbert_llama.max_query_vectors": str(args.max_query_vectors),
    }
    for key, value in settings.items():
        exec_sql(conn, "SELECT set_config(%s, %s, false)", (key, value))


def set_retrieval_gucs(conn: psycopg.Connection[Any], args: argparse.Namespace, app_name: str) -> None:
    settings = {
        "application_name": app_name,
        "enable_seqscan": "off",
        "jit": "off",
        "max_parallel_workers_per_gather": "0",
        "pg_colbert_llama.model_dir": str(args.model_path.parent),
        "pg_colbert_llama.expected_dim": str(args.expected_dim),
        "pg_colbert_llama.max_doc_vectors": str(args.max_doc_vectors),
        "pg_colbert_llama.max_query_vectors": str(args.max_query_vectors),
    }
    for key, value in settings.items():
        exec_sql(conn, "SELECT set_config(%s, %s, false)", (key, value))


def setup_schema(conn: psycopg.Connection[Any]) -> None:
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS vector")
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS pgturbohybrid")
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS pg_colbert_llama")
    exec_sql(
        conn,
        """
        CREATE TABLE IF NOT EXISTS dbpedia_colbert_docs (
            doc_id text PRIMARY KEY,
            title text NOT NULL,
            body text NOT NULL,
            body_tsv tsvector GENERATED ALWAYS AS (
                to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(body, ''))
            ) STORED,
            colbert turbohybrid_multivector
        )
        """,
    )
    exec_sql(
        conn,
        """
        CREATE TABLE IF NOT EXISTS dbpedia_colbert_queries (
            query_id text PRIMARY KEY,
            query_text text NOT NULL,
            colbert turbohybrid_multivector
        )
        """,
    )
    exec_sql(
        conn,
        """
        CREATE TABLE IF NOT EXISTS dbpedia_colbert_qrels (
            query_id text NOT NULL,
            doc_id text NOT NULL,
            relevance int NOT NULL,
            PRIMARY KEY (query_id, doc_id)
        )
        """,
    )


def load_data(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    qids: list[str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    if args.reuse_data and not args.force_reload:
        row = fetch_one(
            conn,
            """
            SELECT
              coalesce((SELECT count(*) FROM dbpedia_colbert_docs), 0),
              coalesce((SELECT count(*) FROM dbpedia_colbert_queries), 0),
              coalesce((SELECT count(*) FROM dbpedia_colbert_qrels), 0)
            """,
        )
        if row and int(row[0]) > 0 and int(row[1]) > 0 and int(row[2]) > 0:
            return {"reused": True, "documents": int(row[0]), "queries": int(row[1]), "qrels": int(row[2])}

    started = time.perf_counter()
    exec_sql(conn, "DROP INDEX IF EXISTS dbpedia_colbert_docs_colbert_idx")
    exec_sql(conn, "TRUNCATE dbpedia_colbert_qrels, dbpedia_colbert_queries, dbpedia_colbert_docs")
    doc_count = copy_rows(
        conn,
        "COPY dbpedia_colbert_docs (doc_id, title, body) FROM STDIN",
        corpus_rows(args.dataset, args.max_docs),
    )
    query_count = copy_rows(
        conn,
        "COPY dbpedia_colbert_queries (query_id, query_text) FROM STDIN",
        query_rows(qids, queries),
    )
    qrel_count = copy_rows(
        conn,
        "COPY dbpedia_colbert_qrels (query_id, doc_id, relevance) FROM STDIN",
        qrel_rows(qids, qrels),
    )
    return {
        "reused": False,
        "documents": doc_count,
        "queries": query_count,
        "qrels": qrel_count,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def selected_doc_ids(conn: psycopg.Connection[Any], limit: int) -> list[str]:
    if limit > 0:
        rows = fetch_all(conn, "SELECT doc_id FROM dbpedia_colbert_docs ORDER BY doc_id LIMIT %s", (limit,))
    else:
        rows = fetch_all(conn, "SELECT doc_id FROM dbpedia_colbert_docs ORDER BY doc_id")
    return [str(row[0]) for row in rows]


def selected_query_ids(conn: psycopg.Connection[Any]) -> list[str]:
    rows = fetch_all(conn, "SELECT query_id FROM dbpedia_colbert_queries ORDER BY query_id")
    return [str(row[0]) for row in rows]


def measure_generation_sample(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    doc_ids: list[str],
    query_ids: list[str],
) -> dict[str, Any]:
    set_colbert_gucs(conn, args)
    doc_latencies: list[float] = []
    doc_counts: list[int] = []
    for doc_id in doc_ids[: args.generation_sample_docs]:
        started = time.perf_counter()
        row = fetch_one(
            conn,
            """
            WITH encoded AS MATERIALIZED (
              SELECT colbert_mv(%s, concat_ws(' ', nullif(title, ''), body)) AS mv
              FROM dbpedia_colbert_docs
              WHERE doc_id = %s
            )
            SELECT turbohybrid_multivector_dims(mv), turbohybrid_multivector_count(mv)
            FROM encoded
            """,
            (f"{args.model_alias}:doc", doc_id),
        )
        doc_latencies.append((time.perf_counter() - started) * 1000.0)
        if row:
            doc_counts.append(int(row[1]))

    query_latencies: list[float] = []
    query_counts: list[int] = []
    for query_id in query_ids:
        started = time.perf_counter()
        row = fetch_one(
            conn,
            """
            WITH encoded AS MATERIALIZED (
              SELECT colbert_mv(%s, query_text) AS mv
              FROM dbpedia_colbert_queries
              WHERE query_id = %s
            )
            SELECT turbohybrid_multivector_dims(mv), turbohybrid_multivector_count(mv)
            FROM encoded
            """,
            (f"{args.model_alias}:query", query_id),
        )
        query_latencies.append((time.perf_counter() - started) * 1000.0)
        if row:
            query_counts.append(int(row[1]))

    return {
        "documents": summarize_ms(doc_latencies),
        "document_token_vectors": summarize_ints(doc_counts),
        "queries": summarize_ms(query_latencies),
        "query_token_vectors": summarize_ints(query_counts),
        "note": "generation-only sample; persisted insert/update phase below includes generation plus storage",
    }


def persist_multivectors(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    doc_ids: list[str],
    query_ids: list[str],
) -> dict[str, Any]:
    if args.reuse_embeddings and not args.force_reload:
        row = fetch_one(
            conn,
            """
            SELECT
              count(*) FILTER (WHERE colbert IS NOT NULL),
              (SELECT count(*) FROM dbpedia_colbert_docs),
              (SELECT count(*) FILTER (WHERE colbert IS NOT NULL) FROM dbpedia_colbert_queries),
              (SELECT count(*) FROM dbpedia_colbert_queries)
            FROM dbpedia_colbert_docs
            """,
        )
        if row and int(row[0]) == int(row[1]) and int(row[2]) == int(row[3]) and int(row[1]) > 0:
            return {"reused": True, "documents": int(row[1]), "queries": int(row[3])}

    set_colbert_gucs(conn, args)
    doc_latencies: list[float] = []
    doc_counts: list[int] = []
    for offset in range(0, len(doc_ids), args.insert_batch_size):
        batch = doc_ids[offset : offset + args.insert_batch_size]
        started = time.perf_counter()
        rows = fetch_all(
            conn,
            """
            UPDATE dbpedia_colbert_docs
            SET colbert = colbert_mv(%s, concat_ws(' ', nullif(title, ''), body))
            WHERE doc_id = ANY(%s::text[])
            RETURNING turbohybrid_multivector_count(colbert)
            """,
            (f"{args.model_alias}:doc", batch),
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        per_doc = elapsed / max(len(rows), 1)
        doc_latencies.extend([per_doc] * len(rows))
        doc_counts.extend(int(row[0]) for row in rows)
        if len(doc_latencies) % max(args.progress_every, 1) == 0:
            print(f"persisted {len(doc_latencies)}/{len(doc_ids)} document multivectors", file=sys.stderr)

    query_latencies: list[float] = []
    query_counts: list[int] = []
    for query_id in query_ids:
        started = time.perf_counter()
        row = fetch_one(
            conn,
            """
            UPDATE dbpedia_colbert_queries
            SET colbert = colbert_mv(%s, query_text)
            WHERE query_id = %s
            RETURNING turbohybrid_multivector_count(colbert)
            """,
            (f"{args.model_alias}:query", query_id),
        )
        query_latencies.append((time.perf_counter() - started) * 1000.0)
        if row:
            query_counts.append(int(row[0]))

    return {
        "reused": False,
        "documents": summarize_ms(doc_latencies),
        "document_token_vectors": summarize_ints(doc_counts),
        "queries": summarize_ms(query_latencies),
        "query_token_vectors": summarize_ints(query_counts),
        "note": "persisted phase measures PostgreSQL generation plus row storage/WAL",
    }


def build_index(conn: psycopg.Connection[Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.reuse_index:
        row = fetch_one(conn, "SELECT to_regclass('dbpedia_colbert_docs_colbert_idx') IS NOT NULL")
        if row and bool(row[0]):
            stats = fetch_one(conn, "SELECT turbohybrid_index_stats('dbpedia_colbert_docs_colbert_idx'::regclass)")
            size = fetch_one(conn, "SELECT pg_relation_size('dbpedia_colbert_docs_colbert_idx'::regclass)")
            return {
                "reused": True,
                "index_bytes": int(size[0]) if size else 0,
                "index_stats": jsonb_value(stats[0]) if stats else {},
            }

    exec_sql(conn, "DROP INDEX IF EXISTS dbpedia_colbert_docs_colbert_idx")
    started = time.perf_counter()
    exec_sql(
        conn,
        """
        CREATE INDEX dbpedia_colbert_docs_colbert_idx
        ON dbpedia_colbert_docs USING turbohybrid (
          colbert multivector_maxsim_ip_turbohybrid_ops,
          body_tsv bm25_tsvector_turbohybrid_ops
        )
        WITH (quantization_bits = 4, exact_storage = off)
        """,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    stats = fetch_one(conn, "SELECT turbohybrid_index_stats('dbpedia_colbert_docs_colbert_idx'::regclass)")
    size = fetch_one(conn, "SELECT pg_relation_size('dbpedia_colbert_docs_colbert_idx'::regclass)")
    return {
        "reused": False,
        "elapsed_ms": round(elapsed_ms, 3),
        "index_bytes": int(size[0]) if size else 0,
        "index_stats": jsonb_value(stats[0]) if stats else {},
    }


def load_encoded_queries(conn: psycopg.Connection[Any]) -> list[QueryItem]:
    rows = fetch_all(
        conn,
        """
        SELECT query_id, query_text, colbert::text
        FROM dbpedia_colbert_queries
        WHERE colbert IS NOT NULL
        ORDER BY query_id
        """,
    )
    queries = [QueryItem(str(row[0]), str(row[1]), str(row[2])) for row in rows]
    if not queries:
        raise RuntimeError("no encoded query multivectors are available")
    return queries


def run_retrieval_query(
    conn: psycopg.Connection[Any],
    method: str,
    query: QueryItem,
    args: argparse.Namespace,
    final_k: int,
) -> list[str]:
    if method == QUERY_ONLY_METHOD:
        rows = fetch_all(
            conn,
            """
            SELECT doc_id
            FROM dbpedia_colbert_docs
            WHERE colbert IS NOT NULL
            ORDER BY colbert <~> turbohybrid_query(
              multivector_query => %s::turbohybrid_multivector,
              dense_k => %s,
              final_k => %s
            )
            LIMIT %s
            """,
            (query.multivector_text, args.dense_k, final_k, final_k),
        )
    elif method == RRF_METHOD:
        rows = fetch_all(
            conn,
            """
            SELECT doc_id
            FROM dbpedia_colbert_docs
            WHERE colbert IS NOT NULL
            ORDER BY colbert <~> turbohybrid_query(
              multivector_query => %s::turbohybrid_multivector,
              text_query => websearch_to_tsquery('simple', %s),
              fusion => 'rrf',
              dense_k => %s,
              bm25_k => %s,
              rrf_k => %s,
              final_k => %s
            )
            LIMIT %s
            """,
            (query.multivector_text, query.query_text, args.dense_k, args.bm25_k, args.rrf_k, final_k, final_k),
        )
    else:
        raise ValueError(f"unknown method: {method}")
    return [str(row[0]) for row in rows]


def last_scan_stats(conn: psycopg.Connection[Any]) -> dict[str, Any]:
    row = fetch_one(conn, "SELECT turbohybrid_last_scan_stats()")
    return jsonb_value(row[0] if row else None)


def run_serial_retrieval(
    conn: psycopg.Connection[Any],
    method: str,
    queries: list[QueryItem],
    args: argparse.Namespace,
    final_k: int,
) -> tuple[dict[str, list[str]], list[float], list[dict[str, Any]]]:
    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    scan_stats: list[dict[str, Any]] = []
    for query in queries:
        started = time.perf_counter()
        run[query.query_id] = run_retrieval_query(conn, method, query, args, final_k)
        latencies.append((time.perf_counter() - started) * 1000.0)
        scan_stats.append(last_scan_stats(conn))
    return run, latencies, scan_stats


def run_worker(
    *,
    args: argparse.Namespace,
    method: str,
    queries: list[QueryItem],
    client_id: int,
    ready_barrier: threading.Barrier,
    timed_barrier: threading.Barrier,
    result: WorkerResult,
) -> None:
    try:
        conn = connect(args)
        try:
            set_retrieval_gucs(conn, args, f"dbpedia_colbert_{method}_{client_id}")
            ready_barrier.wait()
            for i in range(args.warm_queries):
                query = queries[(client_id + i * args.clients) % len(queries)]
                started = time.perf_counter()
                run_retrieval_query(conn, method, query, args, args.final_k)
                result.warm_durations_ms.append((time.perf_counter() - started) * 1000.0)

            timed_barrier.wait()
            timed_started = time.perf_counter()
            for i in range(args.timed_queries):
                query = queries[(client_id + i * args.clients) % len(queries)]
                started = time.perf_counter()
                run_retrieval_query(conn, method, query, args, args.final_k)
                result.timed_durations_ms.append((time.perf_counter() - started) * 1000.0)
                if i == 0:
                    result.first_timed_stats = last_scan_stats(conn)
            result.timed_wall_ms = (time.perf_counter() - timed_started) * 1000.0
            if args.timed_queries > 0:
                result.last_timed_stats = last_scan_stats(conn)
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover - benchmark error path
        result.error = str(exc)
        for barrier in (ready_barrier, timed_barrier):
            try:
                barrier.abort()
            except Exception:
                pass


def run_parallel_retrieval(args: argparse.Namespace, method: str, queries: list[QueryItem]) -> dict[str, Any]:
    ready_barrier = threading.Barrier(args.clients)
    timed_barrier = threading.Barrier(args.clients)
    results = [WorkerResult(client_id=i) for i in range(args.clients)]
    threads = [
        threading.Thread(
            target=run_worker,
            kwargs={
                "args": args,
                "method": method,
                "queries": queries,
                "client_id": i,
                "ready_barrier": ready_barrier,
                "timed_barrier": timed_barrier,
                "result": results[i],
            },
            daemon=True,
        )
        for i in range(args.clients)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    errors = [result.error for result in results if result.error]
    if errors:
        raise RuntimeError("; ".join(errors))
    timed = [value for result in results for value in result.timed_durations_ms]
    warm = [value for result in results for value in result.warm_durations_ms]
    wall_ms = max((result.timed_wall_ms for result in results), default=0.0)
    total_queries = len(timed)
    return {
        "clients": args.clients,
        "warm_queries_per_client": args.warm_queries,
        "timed_queries_per_client": args.timed_queries,
        "warm_latency": summarize_ms(warm),
        "timed_latency": summarize_ms(timed),
        "timed_wall_ms": round(wall_ms, 3),
        "throughput_qps": round((total_queries * 1000.0 / wall_ms), 3) if wall_ms > 0 else 0.0,
        "first_timed_scan_stats": results[0].first_timed_stats if results else None,
        "last_timed_scan_stats": results[0].last_timed_stats if results else None,
    }


def method_metrics(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    final_k: int,
    quality_k: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in sorted({10, final_k, quality_k}):
        if k <= 0 or k > final_k:
            continue
        metrics.update(metrics_for_run(run, qrels, k))
    return metrics


def scan_summary(scan_stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not scan_stats:
        return {}
    first = scan_stats[0]
    last = scan_stats[-1]
    return {
        "first": first,
        "last": last,
        "runs": len(scan_stats),
        "index_used_runs": sum(1 for stats in scan_stats if stats.get("index_used") is True),
        "multivector_enabled_runs": sum(1 for stats in scan_stats if stats.get("multivector_enabled") is True),
    }


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    env_dataset = os.environ.get("DBPEDIA_DATASET")
    env_beir = os.environ.get("BEIR_DBPEDIA_DATASET")
    env_model = os.environ.get("PG_COLBERT_LLAMA_TEST_MODEL")
    if args.dataset is None and env_dataset:
        args.dataset = Path(env_dataset)
    if args.beir_dataset is None and env_beir:
        args.beir_dataset = Path(env_beir)
    if args.model_path is None:
        args.model_path = Path(env_model or DEFAULT_MODEL_PATH)

    if not args.reuse_data and args.dataset is None:
        raise SystemExit("pass --dataset or set DBPEDIA_DATASET")
    if args.dataset is not None:
        args.dataset = args.dataset.resolve()
        if not args.dataset.exists():
            raise SystemExit(f"dataset path does not exist: {args.dataset}")
    if args.beir_dataset is None:
        raise SystemExit("pass --beir-dataset or set BEIR_DBPEDIA_DATASET")
    args.beir_dataset = args.beir_dataset.resolve()
    if not args.beir_dataset.exists():
        raise SystemExit(f"BEIR dataset path does not exist: {args.beir_dataset}")

    args.model_path = args.model_path.resolve()
    if not args.model_path.is_file():
        raise SystemExit(
            f"model file does not exist: {args.model_path}\n"
            "Use johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF and pass --model-path."
        )
    if args.model_alias is None:
        args.model_alias = args.model_path.stem
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = sorted(set(methods) - {QUERY_ONLY_METHOD, RRF_METHOD})
    if unknown:
        raise SystemExit(f"unknown method(s): {', '.join(unknown)}")
    args.methods = methods
    args.output = args.output or Path(
        "benchmarks/results/"
        + datetime.now(timezone.utc).strftime("dbpedia-colbert-multivector-%Y%m%dT%H%M%SZ.json")
    )
    args.output = args.output.resolve()
    if args.quality_k > args.final_k:
        print(
            f"--quality-k {args.quality_k} is greater than --final-k {args.final_k}; "
            "quality metrics will be capped by final_k",
            file=sys.stderr,
        )
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", "pgturbohybrid_dbpedia_colbert"))
    parser.add_argument("--dataset", type=Path, default=None, help="Qdrant DBpedia parquet dataset root")
    parser.add_argument("--beir-dataset", type=Path, default=None, help="BEIR DBpedia dataset root containing queries")
    parser.add_argument("--qrels", default=None, help="BEIR DBpedia qrels TSV path")
    parser.add_argument("--model-path", type=Path, default=None, help="GGUF ColBERT model path")
    parser.add_argument("--model-alias", default=None, help="pg_colbert_llama model alias, defaults to model stem")
    parser.add_argument("--expected-dim", type=int, default=128)
    parser.add_argument("--max-doc-vectors", type=int, default=512)
    parser.add_argument("--max-query-vectors", type=int, default=64)
    parser.add_argument("--max-docs", type=int, default=1000, help="documents to load; 0 means all available")
    parser.add_argument("--max-queries", type=int, default=32, help="queries to load from qrels; 0 means all")
    parser.add_argument("--generation-sample-docs", type=int, default=100)
    parser.add_argument("--insert-batch-size", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--dense-k", type=int, default=100)
    parser.add_argument("--bm25-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--quality-k", type=int, default=10)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--warm-queries", type=int, default=2)
    parser.add_argument("--timed-queries", type=int, default=10)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--reuse-embeddings", action="store_true")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return validate_args(parser.parse_args())


def main() -> None:
    args = parse_args()
    queries = load_queries(args.beir_dataset)
    qrels_path = resolve_qrels_path(args.qrels, args.beir_dataset)
    all_qrels = read_qrels(qrels_path)
    qids = choose_query_ids(all_qrels, queries, args.max_queries)
    qrels = {qid: all_qrels[qid] for qid in qids}

    conn = connect(args)
    try:
        setup_schema(conn)
        load_phase = load_data(conn, args, qids, queries, qrels)
        doc_ids = selected_doc_ids(conn, args.max_docs)
        query_ids = selected_query_ids(conn)
        generation_phase = measure_generation_sample(conn, args, doc_ids, query_ids)
        insert_phase = persist_multivectors(conn, args, doc_ids, query_ids)
        index_phase = build_index(conn, args)
        set_retrieval_gucs(conn, args, "dbpedia_colbert_serial")
        encoded_queries = load_encoded_queries(conn)

        result_methods: list[dict[str, Any]] = []
        for method in args.methods:
            run, serial_latencies, serial_stats = run_serial_retrieval(conn, method, encoded_queries, args, args.final_k)
            parallel = run_parallel_retrieval(args, method, encoded_queries)
            result_methods.append({
                "method": method,
                "retrieval_mode": "multivector_query_only" if method == QUERY_ONLY_METHOD else "multivector_plus_bm25_rrf",
                "metrics": method_metrics(run, qrels, args.final_k, args.quality_k),
                "serial_latency": summarize_ms(serial_latencies),
                "parallel_8x": parallel,
                "top10_by_query": {qid: docs[:10] for qid, docs in run.items()},
                "scan_summary": scan_summary(serial_stats),
                "index_stats": index_phase.get("index_stats", {}),
            })

        output = {
            "suite": "dbpedia_colbert_multivector",
            "layer": "ir_quality_and_systems",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "command": command_metadata(),
            "dataset": {
                "name": "dbpedia-colbert-multivector",
                "corpus_path": portable_path(args.dataset) if args.dataset else None,
                "beir_dataset_path": portable_path(args.beir_dataset),
                "qrels_path": portable_path(qrels_path),
                "documents": load_phase["documents"],
                "queries": load_phase["queries"],
                "qrels": load_phase["qrels"],
                "max_docs": args.max_docs,
                "max_queries": args.max_queries,
            },
            "model": {
                "gguf": "johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF",
                "source_model": "VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m",
                "model_path": portable_path(args.model_path),
                "model_alias": args.model_alias,
                "expected_dim": args.expected_dim,
                "max_doc_vectors": args.max_doc_vectors,
                "max_query_vectors": args.max_query_vectors,
            },
            "settings": {
                "dense_k": args.dense_k,
                "bm25_k": args.bm25_k,
                "rrf_k": args.rrf_k,
                "final_k": args.final_k,
                "quality_k": args.quality_k,
                "clients": args.clients,
                "warm_queries": args.warm_queries,
                "timed_queries": args.timed_queries,
            },
            "phases": {
                "load_text": load_phase,
                "generation_in_postgres": generation_phase,
                "insert_generated_multivectors": insert_phase,
                "build_index": index_phase,
            },
            "results": result_methods,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(args.output)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
