#!/usr/bin/env python3
"""Run DBpedia ColBERT multivector generation, insert, retrieval, and quality benchmarks."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
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
EXACT_SCAN_METHOD = "pgturbohybrid_colbert_multivector_exact_scan"
DEFAULT_MODEL_PATH = ".nix-dev/models/colbert-15m/sauerkraut-modern.gguf"


@dataclass(frozen=True)
class QueryItem:
    query_id: str
    query_text: str


@dataclass
class WorkerResult:
    client_id: int
    warm_durations_ms: list[float] = field(default_factory=list)
    timed_durations_ms: list[float] = field(default_factory=list)
    timed_wall_ms: float = 0.0
    first_timed_stats: dict[str, Any] | None = None
    last_timed_stats: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class PersistWorkerResult:
    worker_id: int
    doc_latencies_ms: list[float] = field(default_factory=list)
    batch_latencies_ms: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    doc_vector_counts: list[int] = field(default_factory=list)
    batches: int = 0
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


def iter_corpus_rows(dataset: Path) -> Iterable[tuple[str, str, str]]:
    pq = require_pyarrow()
    for path in find_corpus_parquet(dataset):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024, columns=["_id", "title", "text"]):
            for row in batch.to_pylist():
                yield str(row["_id"]), row.get("title") or "", row.get("text") or ""


def corpus_rows(
    dataset: Path,
    max_docs: int,
    priority_doc_ids: set[str] | None = None,
) -> Iterable[tuple[str, str, str]]:
    if not priority_doc_ids or max_docs == 0:
        emitted = 0
        for row in iter_corpus_rows(dataset):
            if max_docs > 0 and emitted >= max_docs:
                return
            emitted += 1
            yield row
        return

    emitted: set[str] = set()
    for doc_id, title, text in iter_corpus_rows(dataset):
        if doc_id not in priority_doc_ids or doc_id in emitted:
            continue
        emitted.add(doc_id)
        yield doc_id, title, text
        if len(emitted) >= max_docs:
            return

    for doc_id, title, text in iter_corpus_rows(dataset):
        if doc_id in emitted:
            continue
        emitted.add(doc_id)
        yield doc_id, title, text
        if len(emitted) >= max_docs:
            return


def query_rows(qids: list[str], queries: dict[str, str]) -> Iterable[tuple[str, str]]:
    for qid in qids:
        yield qid, queries[qid]


def qrel_rows(qids: list[str], qrels: dict[str, dict[str, int]]) -> Iterable[tuple[str, str, int]]:
    for qid in qids:
        for doc_id, score in sorted(qrels.get(qid, {}).items()):
            yield qid, doc_id, score


def qrel_doc_ids(qids: list[str], qrels: dict[str, dict[str, int]]) -> set[str]:
    return {doc_id for qid in qids for doc_id in qrels.get(qid, {})}


def filter_qrels_to_loaded_docs(
    qids: list[str],
    qrels: dict[str, dict[str, int]],
    loaded_doc_ids: set[str],
) -> tuple[list[str], dict[str, dict[str, int]]]:
    filtered: dict[str, dict[str, int]] = {}
    filtered_qids: list[str] = []
    for qid in qids:
        qid_qrels = {
            doc_id: score
            for doc_id, score in qrels.get(qid, {}).items()
            if doc_id in loaded_doc_ids
        }
        if qid_qrels:
            filtered[qid] = qid_qrels
            filtered_qids.append(qid)
    return filtered_qids, filtered


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


def load_pgturbohybrid_library(conn: psycopg.Connection[Any]) -> None:
    # CREATE EXTENSION defines the SQL objects, but the C library and its GUCs
    # are loaded lazily when a C function is first called in this session.
    fetch_one(conn, "SELECT turbohybrid_last_scan_stats()")


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
        "pg_colbert_llama.threads": str(args.generation_threads),
        "pg_colbert_llama.n_batch": str(args.generation_n_batch),
        "pg_colbert_llama.batch_sequences": str(args.generation_batch_sequences),
        "pg_colbert_llama.n_gpu_layers": str(args.generation_n_gpu_layers),
        "pg_colbert_llama.expected_dim": str(args.expected_dim),
        "pg_colbert_llama.max_doc_vectors": str(args.max_doc_vectors),
        "pg_colbert_llama.max_query_vectors": str(args.max_query_vectors),
        "pg_colbert_llama.query_length": str(args.query_length),
        "turbohybrid.multivector_subvector_k": str(args.multivector_subvector_k),
        "turbohybrid.multivector_unique_docs_per_token": str(args.multivector_unique_docs_per_token),
        "turbohybrid.multivector_max_raw_hits_per_token": str(args.multivector_max_raw_hits_per_token),
        "turbohybrid.multivector_adaptive_widening": args.multivector_adaptive_widening,
        "turbohybrid.multivector_doc_candidate_k": str(args.multivector_doc_candidate_k),
        "turbohybrid.multivector_exact_rerank_k": str(args.multivector_exact_rerank_k),
        "turbohybrid.multivector_candidate_source": args.multivector_candidate_source,
        "turbohybrid.multivector_plain_fallback": args.multivector_plain_fallback,
        "turbohybrid.multivector_plain_fallback_max_docs": str(args.multivector_plain_fallback_max_docs),
        "turbohybrid.multivector_plain_fallback_candidate_fraction": str(args.multivector_plain_fallback_candidate_fraction),
        "turbohybrid.multivector_candidate_reservoirs": args.multivector_candidate_reservoirs,
        "turbohybrid.multivector_per_token_doc_reservoir_k": str(args.multivector_per_token_doc_reservoir_k),
        "turbohybrid.multivector_coverage_reservoir_k": str(args.multivector_coverage_reservoir_k),
        "turbohybrid.multivector_bm25_candidate_injection": args.multivector_bm25_candidate_injection,
        "turbohybrid.multivector_debug_skip_query_tokens": "",
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
        "pg_colbert_llama.query_length": str(args.query_length),
        "turbohybrid.multivector_subvector_k": str(args.multivector_subvector_k),
        "turbohybrid.multivector_unique_docs_per_token": str(args.multivector_unique_docs_per_token),
        "turbohybrid.multivector_max_raw_hits_per_token": str(args.multivector_max_raw_hits_per_token),
        "turbohybrid.multivector_adaptive_widening": args.multivector_adaptive_widening,
        "turbohybrid.multivector_doc_candidate_k": str(args.multivector_doc_candidate_k),
        "turbohybrid.multivector_exact_rerank_k": str(args.multivector_exact_rerank_k),
        "turbohybrid.multivector_candidate_source": args.multivector_candidate_source,
        "turbohybrid.multivector_plain_fallback": args.multivector_plain_fallback,
        "turbohybrid.multivector_plain_fallback_max_docs": str(args.multivector_plain_fallback_max_docs),
        "turbohybrid.multivector_plain_fallback_candidate_fraction": str(args.multivector_plain_fallback_candidate_fraction),
        "turbohybrid.multivector_candidate_reservoirs": args.multivector_candidate_reservoirs,
        "turbohybrid.multivector_per_token_doc_reservoir_k": str(args.multivector_per_token_doc_reservoir_k),
        "turbohybrid.multivector_coverage_reservoir_k": str(args.multivector_coverage_reservoir_k),
        "turbohybrid.multivector_bm25_candidate_injection": args.multivector_bm25_candidate_injection,
        "turbohybrid.multivector_debug_skip_query_tokens": "",
    }
    for key, value in settings.items():
        exec_sql(conn, "SELECT set_config(%s, %s, false)", (key, value))


def setup_schema(conn: psycopg.Connection[Any], include_colbert_llama: bool = True) -> None:
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS vector")
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS pgturbohybrid")
    load_pgturbohybrid_library(conn)
    if include_colbert_llama:
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


def validate_embedding_health(conn: psycopg.Connection[Any], args: argparse.Namespace) -> dict[str, Any]:
    set_colbert_gucs(conn, args)
    query = jsonb_value(fetch_one(conn, "SELECT colbert(%s, 'red planet')", (f"{args.model_alias}:query",))[0])
    doc = jsonb_value(fetch_one(conn, "SELECT colbert(%s, 'red planet')", (f"{args.model_alias}:doc",))[0])
    punct_doc = jsonb_value(
        fetch_one(conn, "SELECT colbert(%s, 'Mars is often called the red planet.')", (f"{args.model_alias}:doc",))[0]
    )

    query_tokens = [int(token) for token in query.get("token_ids", [])]
    doc_tokens = [int(token) for token in doc.get("token_ids", [])]
    punct_doc_tokens = [int(token) for token in punct_doc.get("token_ids", [])]
    expected_query_prefix = [101, 30522, 2417, 4774, 102]
    expected_doc_tokens = [101, 30523, 2417, 4774, 102]
    failures: list[str] = []

    if int(query.get("dim", 0)) != args.expected_dim or int(doc.get("dim", 0)) != args.expected_dim:
        failures.append("probe embeddings have the wrong dimension")
    if query_tokens[: len(expected_query_prefix)] != expected_query_prefix:
        failures.append(f"query tokenization probe mismatch: got {query_tokens[:len(expected_query_prefix)]}")
    if args.query_length <= args.max_query_vectors and len(query_tokens) != args.query_length:
        failures.append(f"query expansion length mismatch: got {len(query_tokens)}, expected {args.query_length}")
    if 100 in query_tokens[: len(expected_query_prefix)]:
        failures.append("query tokenization produced [UNK] for normal probe words")
    if doc_tokens != expected_doc_tokens:
        failures.append(f"document tokenization probe mismatch: got {doc_tokens}")
    if 1012 in punct_doc_tokens:
        failures.append("document tokenization retained '.' punctuation skiplist token")

    result = {
        "validated": not failures,
        "query_probe_token_ids": query_tokens,
        "document_probe_token_ids": doc_tokens,
        "punctuation_probe_token_ids": punct_doc_tokens,
        "expected_query_prefix": expected_query_prefix,
        "expected_document_tokens": expected_doc_tokens,
        "query_length": args.query_length,
        "failures": failures,
    }
    if failures and not args.allow_unvalidated_embeddings:
        raise RuntimeError("invalid ColBERT embedding preflight: " + "; ".join(failures))
    return result


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
              coalesce((SELECT count(*) FROM dbpedia_colbert_qrels), 0),
              coalesce((
                SELECT count(*)
                FROM dbpedia_colbert_qrels q
                JOIN dbpedia_colbert_docs d ON d.doc_id = q.doc_id
              ), 0)
            """,
        )
        if row and int(row[0]) > 0 and int(row[1]) > 0 and int(row[2]) > 0 and int(row[2]) == int(row[3]):
            return {
                "reused": True,
                "documents": int(row[0]),
                "queries": int(row[1]),
                "qrels": int(row[2]),
                "qrels_in_loaded_docs": int(row[3]),
            }

    started = time.perf_counter()
    exec_sql(conn, "DROP INDEX IF EXISTS dbpedia_colbert_docs_colbert_idx")
    exec_sql(conn, "TRUNCATE dbpedia_colbert_qrels, dbpedia_colbert_queries, dbpedia_colbert_docs")
    requested_qrel_doc_ids = qrel_doc_ids(qids, qrels)
    doc_count = copy_rows(
        conn,
        "COPY dbpedia_colbert_docs (doc_id, title, body) FROM STDIN",
        corpus_rows(args.dataset, args.max_docs, requested_qrel_doc_ids if args.prioritize_qrels else None),
    )
    loaded_doc_ids = set(selected_doc_ids(conn, 0))
    loaded_qids, loaded_qrels = filter_qrels_to_loaded_docs(qids, qrels, loaded_doc_ids)
    if not loaded_qids:
        raise RuntimeError(
            "none of the selected qrels are present in the loaded corpus; "
            "increase --max-docs or verify the corpus/qrels id space"
        )
    query_count = copy_rows(
        conn,
        "COPY dbpedia_colbert_queries (query_id, query_text) FROM STDIN",
        query_rows(loaded_qids, queries),
    )
    qrel_count = copy_rows(
        conn,
        "COPY dbpedia_colbert_qrels (query_id, doc_id, relevance) FROM STDIN",
        qrel_rows(loaded_qids, loaded_qrels),
    )
    return {
        "reused": False,
        "documents": doc_count,
        "queries": query_count,
        "qrels": qrel_count,
        "input_qrels": sum(len(qrels.get(qid, {})) for qid in qids),
        "requested_qrel_doc_ids": len(requested_qrel_doc_ids),
        "qrels_in_loaded_docs": qrel_count,
        "queries_with_loaded_qrels": len(loaded_qids),
        "prioritized_qrels": bool(args.prioritize_qrels),
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


def loaded_qrels(conn: psycopg.Connection[Any]) -> dict[str, dict[str, int]]:
    rows = fetch_all(
        conn,
        """
        SELECT query_id, doc_id, relevance
        FROM dbpedia_colbert_qrels
        ORDER BY query_id, doc_id
        """,
    )
    qrels: dict[str, dict[str, int]] = {}
    for query_id, doc_id, relevance in rows:
        qrels.setdefault(str(query_id), {})[str(doc_id)] = int(relevance)
    if not qrels:
        raise RuntimeError("no qrels are available for the loaded benchmark corpus")
    return qrels


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


def count_existing_embeddings(conn: psycopg.Connection[Any]) -> tuple[int, int]:
    row = fetch_one(
        conn,
        """
        SELECT
          count(*) FILTER (WHERE colbert IS NOT NULL),
          count(*)
        FROM dbpedia_colbert_docs
        """,
    )
    return (int(row[0]), int(row[1])) if row else (0, 0)


def ids_missing_embeddings(conn: psycopg.Connection[Any], table_name: str, id_column: str) -> list[str]:
    rows = fetch_all(
        conn,
        f"""
        SELECT {id_column}
        FROM {table_name}
        WHERE colbert IS NULL
        ORDER BY {id_column}
        """,
    )
    return [str(row[0]) for row in rows]


def run_persist_worker(
    *,
    args: argparse.Namespace,
    batches: list[list[str]],
    next_batch: list[int],
    batch_lock: threading.Lock,
    ready_barrier: threading.Barrier,
    start_event: threading.Event,
    progress: dict[str, int],
    progress_lock: threading.Lock,
    total_docs: int,
    result: PersistWorkerResult,
) -> None:
    conn = connect(args)
    try:
        set_colbert_gucs(conn, args)
        if args.generation_warmup:
            fetch_one(conn, "SELECT colbert_mv_batch(%s, %s::text[])", (f"{args.model_alias}:doc", ["warmup document"]))
        ready_barrier.wait()
        start_event.wait()
        while True:
            with batch_lock:
                if next_batch[0] >= len(batches):
                    break
                batch = batches[next_batch[0]]
                next_batch[0] += 1

            started = time.perf_counter()
            rows = fetch_all(
                conn,
                """
                WITH requested AS MATERIALIZED (
                  SELECT doc_id, ord
                  FROM unnest(%s::text[]) WITH ORDINALITY AS u(doc_id, ord)
                ),
                encoded AS MATERIALIZED (
                  SELECT
                    array_agg(d.doc_id ORDER BY r.ord) AS doc_ids,
                    colbert_mv_batch(
                      %s,
                      array_agg(concat_ws(' ', nullif(d.title, ''), d.body) ORDER BY r.ord)
                    ) AS multivectors
                  FROM requested r
                  JOIN dbpedia_colbert_docs d ON d.doc_id = r.doc_id
                )
                UPDATE dbpedia_colbert_docs d
                SET colbert = encoded.multivectors[i]
                FROM encoded,
                     generate_subscripts(encoded.doc_ids, 1) AS s(i)
                WHERE d.doc_id = encoded.doc_ids[i]
                RETURNING turbohybrid_multivector_count(d.colbert)
                """,
                (batch, f"{args.model_alias}:doc"),
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            per_doc = elapsed / max(len(rows), 1)
            result.doc_latencies_ms.extend([per_doc] * len(rows))
            result.batch_latencies_ms.append(elapsed)
            result.batch_sizes.append(len(rows))
            result.doc_vector_counts.extend(int(row[0]) for row in rows)
            result.batches += 1

            with progress_lock:
                progress["documents"] += len(rows)
                completed = progress["documents"]
                next_report = progress["next_report"]
                if completed >= next_report:
                    print(f"persisted {completed}/{total_docs} document multivectors", file=sys.stderr)
                    step = max(args.progress_every, 1)
                    progress["next_report"] = ((completed // step) + 1) * step
    except Exception as exc:  # pragma: no cover - benchmark error path
        result.error = str(exc)
        try:
            ready_barrier.abort()
        except Exception:
            pass
    finally:
        conn.close()


def persist_document_multivectors(
    args: argparse.Namespace,
    doc_ids: list[str],
) -> tuple[list[float], list[int], dict[str, Any]]:
    if not doc_ids:
        return [], [], {
            "workers": 0,
            "batches": 0,
            "elapsed_ms": 0.0,
        }

    batches = [doc_ids[offset : offset + args.insert_batch_size] for offset in range(0, len(doc_ids), args.insert_batch_size)]
    worker_count = max(1, min(args.generation_clients, len(batches)))
    next_batch = [0]
    batch_lock = threading.Lock()
    ready_barrier = threading.Barrier(worker_count + 1)
    start_event = threading.Event()
    progress_lock = threading.Lock()
    progress = {
        "documents": 0,
        "next_report": max(args.progress_every, 1),
    }
    results = [PersistWorkerResult(worker_id=i) for i in range(worker_count)]
    threads = [
        threading.Thread(
            target=run_persist_worker,
            kwargs={
                "args": args,
                "batches": batches,
                "next_batch": next_batch,
                "batch_lock": batch_lock,
                "ready_barrier": ready_barrier,
                "start_event": start_event,
                "progress": progress,
                "progress_lock": progress_lock,
                "total_docs": len(doc_ids),
                "result": result,
            },
        )
        for result in results
    ]
    for thread in threads:
        thread.start()
    try:
        ready_barrier.wait()
    except threading.BrokenBarrierError:
        pass
    errors = [result.error for result in results if result.error]
    if errors:
        for thread in threads:
            thread.join()
        raise RuntimeError("parallel document multivector persistence failed: " + "; ".join(errors))
    started = time.perf_counter()
    start_event.set()
    for thread in threads:
        thread.join()

    errors = [result.error for result in results if result.error]
    if errors:
        raise RuntimeError("parallel document multivector persistence failed: " + "; ".join(errors))

    doc_latencies: list[float] = []
    batch_latencies: list[float] = []
    batch_sizes: list[int] = []
    doc_counts: list[int] = []
    for result in results:
        doc_latencies.extend(result.doc_latencies_ms)
        batch_latencies.extend(result.batch_latencies_ms)
        batch_sizes.extend(result.batch_sizes)
        doc_counts.extend(result.doc_vector_counts)

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return doc_latencies, doc_counts, {
        "workers": worker_count,
        "threads_per_worker": args.generation_threads,
        "n_gpu_layers": args.generation_n_gpu_layers,
        "warmup_excluded": bool(args.generation_warmup),
        "batches": sum(result.batches for result in results),
        "elapsed_ms": elapsed_ms,
        "documents_per_second": round(len(doc_counts) / (elapsed_ms / 1000.0), 3) if elapsed_ms > 0 else 0.0,
        "batch_latency": summarize_ms(batch_latencies),
        "batch_size": summarize_ints(batch_sizes),
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
    existing_doc_embeddings, total_docs = count_existing_embeddings(conn)
    docs_to_encode = (
        ids_missing_embeddings(conn, "dbpedia_colbert_docs", "doc_id")
        if args.reuse_embeddings and not args.force_reload
        else doc_ids
    )
    doc_latencies, doc_counts, parallel_stats = persist_document_multivectors(args, docs_to_encode)

    query_latencies: list[float] = []
    query_counts: list[int] = []
    query_ids_to_encode = (
        ids_missing_embeddings(conn, "dbpedia_colbert_queries", "query_id")
        if args.reuse_embeddings and not args.force_reload
        else query_ids
    )
    for query_id in query_ids_to_encode:
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
        "parallel_document_workers": parallel_stats["workers"],
        "parallel_document_threads_per_worker": parallel_stats["threads_per_worker"],
        "parallel_document_warmup_excluded": parallel_stats["warmup_excluded"],
        "parallel_document_batches": parallel_stats["batches"],
        "parallel_document_elapsed_ms": parallel_stats["elapsed_ms"],
        "parallel_document_docs_per_second": parallel_stats["documents_per_second"],
        "document_batch_latency": parallel_stats["batch_latency"],
        "document_batch_size": parallel_stats["batch_size"],
        "existing_document_embeddings": existing_doc_embeddings,
        "total_documents": total_docs,
        "generated_document_embeddings": len(doc_counts),
        "remaining_document_embeddings": max(total_docs - existing_doc_embeddings - len(doc_counts), 0),
        "generated_query_embeddings": len(query_counts),
        "note": "persisted phase measures PostgreSQL generation plus row storage/WAL",
    }


def load_precomputed_multivectors(conn: psycopg.Connection[Any], args: argparse.Namespace) -> dict[str, Any]:
    from dbpedia_colbert_hf_dataset import import_precomputed_dataset_to_postgres

    if args.reuse_data and not args.force_reload:
        row = fetch_one(
            conn,
            """
            SELECT
              coalesce((SELECT count(*) FROM dbpedia_colbert_docs), 0),
              coalesce((SELECT count(*) FILTER (WHERE colbert IS NOT NULL) FROM dbpedia_colbert_docs), 0),
              coalesce((SELECT count(*) FROM dbpedia_colbert_queries), 0),
              coalesce((SELECT count(*) FILTER (WHERE colbert IS NOT NULL) FROM dbpedia_colbert_queries), 0),
              coalesce((SELECT count(*) FROM dbpedia_colbert_qrels), 0),
              coalesce((
                SELECT count(*)
                FROM dbpedia_colbert_qrels q
                JOIN dbpedia_colbert_docs d ON d.doc_id = q.doc_id
                JOIN dbpedia_colbert_queries bq ON bq.query_id = q.query_id
              ), 0)
            """,
        )
        if (
            row
            and int(row[0]) > 0
            and int(row[0]) == int(row[1])
            and int(row[2]) > 0
            and int(row[2]) == int(row[3])
            and int(row[4]) > 0
            and int(row[4]) == int(row[5])
        ):
            return {
                "reused": True,
                "source": args.precomputed_dataset,
                "documents": int(row[0]),
                "queries": int(row[2]),
                "qrels": int(row[4]),
                "qrels_in_loaded_docs": int(row[5]),
            }

    return import_precomputed_dataset_to_postgres(
        conn=conn,
        source=args.precomputed_dataset,
        batch_size=args.precomputed_batch_size,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        force_reload=True,
    )


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
        SELECT query_id, query_text
        FROM dbpedia_colbert_queries
        WHERE colbert IS NOT NULL
        ORDER BY query_id
        """,
    )
    queries = [QueryItem(str(row[0]), str(row[1])) for row in rows]
    if not queries:
        raise RuntimeError("no encoded query multivectors are available")
    return queries


def run_retrieval_query(
    conn: psycopg.Connection[Any],
    method: str,
    query: QueryItem,
    args: argparse.Namespace,
    final_k: int,
    dense_k: int | None = None,
) -> list[str]:
    if method == QUERY_ONLY_METHOD:
        rows = fetch_all(
            conn,
            """
            SELECT d.doc_id
            FROM dbpedia_colbert_docs d
            WHERE d.colbert IS NOT NULL
            ORDER BY d.colbert <~> (
              SELECT turbohybrid_query(
                multivector_query => q.colbert,
                dense_k => %s,
                final_k => %s
              )
              FROM dbpedia_colbert_queries q
              WHERE q.query_id = %s
            )
            LIMIT %s
            """,
            (dense_k or args.dense_k, final_k, query.query_id, final_k),
        )
    elif method == EXACT_SCAN_METHOD:
        rows = fetch_all(
            conn,
            """
            SELECT d.doc_id
            FROM dbpedia_colbert_docs d
            JOIN dbpedia_colbert_queries q ON q.query_id = %s
            WHERE d.colbert IS NOT NULL
            ORDER BY turbohybrid_multivector_maxsim_distance(
                       q.colbert,
                       d.colbert
                     ),
                     d.doc_id
            LIMIT %s
            """,
            (query.query_id, final_k),
        )
    elif method == RRF_METHOD:
        rows = fetch_all(
            conn,
            """
            SELECT d.doc_id
            FROM dbpedia_colbert_docs d
            WHERE d.colbert IS NOT NULL
            ORDER BY d.colbert <~> (
              SELECT turbohybrid_query(
                multivector_query => q.colbert,
                text_query => websearch_to_tsquery('simple', q.query_text),
                fusion => 'rrf',
                dense_k => %s,
                bm25_k => %s,
                rrf_k => %s,
                final_k => %s
              )
              FROM dbpedia_colbert_queries q
              WHERE q.query_id = %s
            )
            LIMIT %s
            """,
            (dense_k or args.dense_k, args.bm25_k, args.rrf_k, final_k, query.query_id, final_k),
        )
    else:
        raise ValueError(f"unknown method: {method}")
    return [str(row[0]) for row in rows]


def uses_turbohybrid_index(method: str) -> bool:
    return method in {QUERY_ONLY_METHOD, RRF_METHOD}


def last_scan_stats(conn: psycopg.Connection[Any]) -> dict[str, Any]:
    row = fetch_one(conn, "SELECT turbohybrid_last_scan_stats()")
    return jsonb_value(row[0] if row else None)


def tid_parts(tid_text: str) -> tuple[int, int]:
    block, offset = tid_text.strip("()").split(",", 1)
    return int(block), int(offset)


def exact_admission_top(
    conn: psycopg.Connection[Any],
    query: QueryItem,
    admission_k: int,
) -> list[dict[str, Any]]:
    rows = fetch_all(
        conn,
        """
        SELECT d.doc_id,
               d.ctid::text,
               turbohybrid_multivector_maxsim_distance(q.colbert, d.colbert) AS distance
        FROM dbpedia_colbert_docs d
        JOIN dbpedia_colbert_queries q ON q.query_id = %s
        WHERE d.colbert IS NOT NULL
        ORDER BY turbohybrid_multivector_maxsim_distance(q.colbert, d.colbert),
                 d.doc_id
        LIMIT %s
        """,
        (query.query_id, admission_k),
    )
    result: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        block, offset = tid_parts(str(row[1]))
        result.append({
            "rank": rank,
            "doc_id": str(row[0]),
            "heap_block": block,
            "heap_offset": offset,
            "distance": float(row[2]),
            "maxsim": -float(row[2]),
        })
    return result


def trace_key(entry: dict[str, Any]) -> tuple[int, int] | None:
    try:
        return int(entry["heap_block"]), int(entry["heap_offset"])
    except (KeyError, TypeError, ValueError):
        return None


def scan_stat_int(stats: dict[str, Any], key: str) -> int:
    value = stats.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def scan_stat_bool(stats: dict[str, Any], key: str) -> bool:
    return stats.get(key) is True or stats.get(key) == "true"


def memory_estimate_from_stats(stats: dict[str, Any]) -> int:
    return scan_stat_int(stats, "multivector_memory_bytes_estimate")


def run_admission_budget(
    conn: psycopg.Connection[Any],
    query: QueryItem,
    args: argparse.Namespace,
    budget: int,
    exact_top: list[dict[str, Any]],
) -> dict[str, Any]:
    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_doc_candidate_k', %s, false)", (str(budget),))
    exec_sql(
        conn,
        "SELECT set_config('turbohybrid.multivector_exact_rerank_k', %s, false)",
        (str(max(args.multivector_exact_rerank_k, budget)),),
    )
    started = time.perf_counter()
    docs = run_retrieval_query(
        conn,
        QUERY_ONLY_METHOD,
        query,
        args,
        max(args.final_k, args.admission_k),
        dense_k=budget,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    stats = last_scan_stats(conn)
    trace = stats.get("multivector_admission_trace")
    trace_entries = trace if isinstance(trace, list) else []
    trace_by_heap = {
        key: entry
        for entry in trace_entries
        if isinstance(entry, dict) and (key := trace_key(entry)) is not None
    }
    candidate_source = stats.get("multivector_candidate_source")
    fallback_used = bool(stats.get("multivector_plain_fallback_used"))
    infer_admission_from_results = (
        not trace_by_heap
        and fallback_used
        and candidate_source in {"exact_doc_scan", "doc_graph_prototype", "plain_fallback"}
    )

    admitted_ranks: list[int] = []
    exact_top_with_admission: list[dict[str, Any]] = []
    for exact in exact_top:
        key = (int(exact["heap_block"]), int(exact["heap_offset"]))
        entry = trace_by_heap.get(key)
        result_rank = docs.index(exact["doc_id"]) + 1 if exact["doc_id"] in docs else None
        inferred_admitted = infer_admission_from_results and result_rank is not None
        admitted = entry is not None or inferred_admitted
        if admitted:
            admitted_ranks.append(int(exact["rank"]))
        exact_top_with_admission.append({
            "rank": exact["rank"],
            "doc_id": exact["doc_id"],
            "admitted_before_rerank": admitted,
            "candidate_rank_before_rerank": (
                int(entry["candidate_rank_before_truncation"])
                if isinstance(entry, dict) and entry.get("candidate_rank_before_truncation") is not None
                else int(exact["rank"])
                if inferred_admitted
                else None
            ),
            "retained_for_exact_rerank": (
                bool(entry.get("retained_for_exact_rerank"))
                if isinstance(entry, dict) and "retained_for_exact_rerank" in entry
                else True
                if inferred_admitted
                else None
            ),
            "exact_rerank_score": (
                float(entry["exact_rerank_score"])
                if isinstance(entry, dict) and entry.get("exact_rerank_score") is not None
                else float(exact["maxsim"])
                if inferred_admitted
                else None
            ),
            "exact_rerank_rank": result_rank,
        })

    top10 = exact_top[: min(10, len(exact_top))]
    top10_admitted = sum(1 for item in exact_top_with_admission[: len(top10)] if item["admitted_before_rerank"])
    return {
        "budget": budget,
        "candidate_source": candidate_source,
        "exact_token_scan_nodes_scored": scan_stat_int(stats, "multivector_exact_token_scan_nodes_scored"),
        "plain_fallback_used": bool(stats.get("multivector_plain_fallback_used")),
        "plain_fallback_reason": stats.get("multivector_plain_fallback_reason"),
        "plain_fallback_docs_scored": scan_stat_int(stats, "multivector_plain_fallback_docs_scored"),
        "plain_fallback_pairs": scan_stat_int(stats, "multivector_plain_fallback_pairs"),
        "doc_graph_prototype_enabled": scan_stat_bool(stats, "multivector_doc_graph_prototype_enabled"),
        "doc_graph_docs_scored": scan_stat_int(stats, "multivector_doc_graph_docs_scored"),
        "doc_graph_edges_visited": scan_stat_int(stats, "multivector_doc_graph_edges_visited"),
        "doc_graph_candidates": scan_stat_int(stats, "multivector_doc_graph_candidates"),
        "doc_graph_heap_fetches": scan_stat_int(stats, "multivector_doc_graph_heap_fetches"),
        "doc_graph_warning": stats.get("multivector_doc_graph_warning"),
        "reservoirs_enabled": scan_stat_bool(stats, "multivector_reservoirs_enabled"),
        "reservoir_score_docs": scan_stat_int(stats, "multivector_reservoir_score_docs"),
        "reservoir_coverage_docs": scan_stat_int(stats, "multivector_reservoir_coverage_docs"),
        "reservoir_mean_docs": scan_stat_int(stats, "multivector_reservoir_mean_docs"),
        "reservoir_per_token_docs": scan_stat_int(stats, "multivector_reservoir_per_token_docs"),
        "reservoir_bm25_docs": scan_stat_int(stats, "multivector_reservoir_bm25_docs"),
        "reservoir_union_docs": scan_stat_int(stats, "multivector_reservoir_union_docs"),
        "reservoir_duplicates": scan_stat_int(stats, "multivector_reservoir_duplicates"),
        "bm25_injection_enabled": scan_stat_bool(stats, "multivector_bm25_injection_enabled"),
        "bm25_injection_candidates": scan_stat_int(stats, "multivector_bm25_injection_candidates"),
        "bm25_injection_retained": scan_stat_int(stats, "multivector_bm25_injection_retained"),
        "bm25_injection_exact_reranked": scan_stat_int(stats, "multivector_bm25_injection_exact_reranked"),
        "latency_ms": round(latency_ms, 3),
        "result_doc_ids": docs,
        "exact_top1_admitted_before_rerank": bool(exact_top_with_admission and exact_top_with_admission[0]["admitted_before_rerank"]),
        "exact_top10_admission_recall": round(top10_admitted / len(top10), 6) if top10 else 0.0,
        "exact_top1_candidate_rank_before_rerank": (
            exact_top_with_admission[0]["candidate_rank_before_rerank"] if exact_top_with_admission else None
        ),
        "exact_top1_exact_rerank_rank": (
            exact_top_with_admission[0]["exact_rerank_rank"] if exact_top_with_admission else None
        ),
        "raw_subvector_hits": scan_stat_int(stats, "multivector_raw_subvector_hits"),
        "unique_docs": scan_stat_int(stats, "multivector_unique_docs"),
        "maxsim_updates": scan_stat_int(stats, "multivector_maxsim_updates"),
        "doc_candidates": scan_stat_int(stats, "multivector_doc_candidates"),
        "exact_rerank_docs": scan_stat_int(stats, "multivector_exact_rerank_docs"),
        "memory_bytes_estimate": memory_estimate_from_stats(stats),
        "trace_available": scan_stat_bool(stats, "multivector_admission_trace_available"),
        "trace_entries": len(trace_entries),
        "admission_inferred_from_result_docs": infer_admission_from_results,
        "exact_top": exact_top_with_admission,
        "scan_stats": stats,
    }


def run_admission_debug(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    queries: list[QueryItem],
) -> dict[str, Any]:
    budgets = [int(item.strip()) for item in args.admission_budget_sweep.split(",") if item.strip()]
    if not budgets:
        raise ValueError("--admission-budget-sweep must contain at least one integer budget")
    budgets = sorted({budget for budget in budgets if budget > 0})
    if not budgets:
        raise ValueError("--admission-budget-sweep budgets must be positive")

    set_retrieval_gucs(conn, args, "dbpedia_colbert_admission_debug")
    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_debug_admission', 'trace', false)")
    exec_sql(
        conn,
        "SELECT set_config('turbohybrid.multivector_debug_trace_limit', %s, false)",
        (str(min(args.admission_trace_limit, 1000)),),
    )

    per_query: list[dict[str, Any]] = []
    latency_by_budget: dict[int, list[float]] = {budget: [] for budget in budgets}
    top1_first_budget_values: list[int] = []
    top1_admitted_queries = 0
    top10_recall_values: list[float] = []

    for query in queries:
        exact_top = exact_admission_top(conn, query, args.admission_k)
        budget_results: list[dict[str, Any]] = []
        first_budget: int | None = None
        top1_candidate_rank: int | None = None
        top1_exact_rank: int | None = None
        for budget in budgets:
            result = run_admission_budget(conn, query, args, budget, exact_top)
            budget_results.append(result)
            latency_by_budget[budget].append(float(result["latency_ms"]))
            if result["exact_top1_admitted_before_rerank"] and first_budget is None:
                first_budget = budget
                top1_candidate_rank = result["exact_top1_candidate_rank_before_rerank"]
                top1_exact_rank = result["exact_top1_exact_rerank_rank"]
        if first_budget is not None:
            top1_first_budget_values.append(first_budget)
            top1_admitted_queries += 1
        if budget_results:
            top10_recall_values.append(float(budget_results[-1]["exact_top10_admission_recall"]))

        per_query.append({
            "query_id": query.query_id,
            "query_text": query.query_text,
            "exact_top": exact_top,
            "exact_top1_admitted_before_rerank": first_budget is not None,
            "exact_top10_admission_recall": budget_results[-1]["exact_top10_admission_recall"] if budget_results else 0.0,
            "exact_top1_first_budget_admitted": first_budget,
            "exact_top1_candidate_rank_before_rerank": top1_candidate_rank,
            "exact_top1_exact_rerank_rank": top1_exact_rank,
            "budgets": budget_results,
        })

    aggregate = {
        "queries": len(per_query),
        "admission_k": args.admission_k,
        "budget_sweep": budgets,
        "exact_top1_admission_rate": round(top1_admitted_queries / len(per_query), 6) if per_query else 0.0,
        "exact_top10_admission_recall": round(statistics.mean(top10_recall_values), 6) if top10_recall_values else 0.0,
        "exact_top1_first_budget_admitted": summarize_ints(top1_first_budget_values),
        "latency_by_budget": {
            str(budget): summarize_ms(values)
            for budget, values in latency_by_budget.items()
        },
    }

    set_retrieval_gucs(conn, args, "dbpedia_colbert_serial")
    return {
        "enabled": True,
        "candidate_source": args.multivector_candidate_source,
        "plain_fallback": args.multivector_plain_fallback,
        "candidate_reservoirs": args.multivector_candidate_reservoirs,
        "bm25_candidate_injection": args.multivector_bm25_candidate_injection,
        "retrieval_method": QUERY_ONLY_METHOD,
        "aggregate": aggregate,
        "per_query": per_query,
    }


def run_token_ablation(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    queries: list[QueryItem],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any] | None:
    if not args.token_ablation_query_id:
        return None

    query = next((item for item in queries if item.query_id == args.token_ablation_query_id), None)
    if query is None:
        available = ", ".join(item.query_id for item in queries[:10])
        raise ValueError(
            f"--token-ablation-query-id {args.token_ablation_query_id!r} was not loaded; "
            f"first loaded query ids: {available}"
        )

    final_k = args.token_ablation_final_k or args.final_k
    dense_k = args.token_ablation_dense_k or args.dense_k
    relevant = qrels.get(query.query_id, {})

    set_retrieval_gucs(conn, args, "dbpedia_colbert_token_ablation")
    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_debug_admission', 'summary', false)")

    def run_variant(label: str, skip_tokens: str) -> dict[str, Any]:
        exec_sql(
            conn,
            "SELECT set_config('turbohybrid.multivector_debug_skip_query_tokens', %s, false)",
            (skip_tokens,),
        )
        started = time.perf_counter()
        docs = run_retrieval_query(
            conn,
            QUERY_ONLY_METHOD,
            query,
            args,
            final_k,
            dense_k=dense_k,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        stats = last_scan_stats(conn)
        hits = [doc_id for doc_id in docs[:final_k] if relevant.get(doc_id, 0) > 0]
        token_stats = stats.get("multivector_query_token_stats")
        return {
            "label": label,
            "skip_tokens": skip_tokens,
            "latency_ms": round(latency_ms, 3),
            "top_docs": docs,
            "relevant_hits": hits,
            "recall_at_k": round(len(hits) / len(relevant), 6) if relevant else None,
            "scan_stats": stats,
            "query_token_stats": token_stats if isinstance(token_stats, list) else [],
        }

    variants = [run_variant("baseline", "")]
    if args.token_ablation_skip_tokens:
        variants.append(run_variant("skip_tokens", args.token_ablation_skip_tokens))

    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_debug_skip_query_tokens', '', false)")
    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_debug_admission', 'off', false)")
    return {
        "enabled": True,
        "query_id": query.query_id,
        "query_text": query.query_text,
        "final_k": final_k,
        "dense_k": dense_k,
        "relevant_docs": len(relevant),
        "variants": variants,
    }


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
        scan_stats.append(last_scan_stats(conn) if uses_turbohybrid_index(method) else {})
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
                if i == 0 and uses_turbohybrid_index(method):
                    result.first_timed_stats = last_scan_stats(conn)
            result.timed_wall_ms = (time.perf_counter() - timed_started) * 1000.0
            if args.timed_queries > 0 and uses_turbohybrid_index(method):
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
        "total_timed_queries": total_queries,
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


SYNTHETIC_GATE_QUERY = """turbohybrid_multivector(ARRAY[
    '[1,0,0,0]'::vector,
    '[0,1,0,0]'::vector,
    '[0,0,1,0]'::vector,
    '[0,0,0,1]'::vector
])"""


def setup_multivector_recall_gate(conn: psycopg.Connection[Any]) -> None:
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS vector")
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS pgturbohybrid")
    load_pgturbohybrid_library(conn)
    exec_sql(conn, "DROP TABLE IF EXISTS mv_recall_gate_docs")
    exec_sql(
        conn,
        """
        CREATE TEMP TABLE mv_recall_gate_docs (
            id text PRIMARY KEY,
            colbert turbohybrid_multivector
        ) ON COMMIT PRESERVE ROWS
        """,
    )
    exec_sql(
        conn,
        """
        INSERT INTO mv_recall_gate_docs VALUES
          ('spike_1', turbohybrid_multivector(ARRAY['[1,0,0,0]'::vector])),
          ('spike_2', turbohybrid_multivector(ARRAY['[0,1,0,0]'::vector])),
          ('spike_3', turbohybrid_multivector(ARRAY['[0,0,1,0]'::vector])),
          ('spike_4', turbohybrid_multivector(ARRAY['[0,0,0,1]'::vector])),
          ('good', turbohybrid_multivector(ARRAY[
            '[0.8,0.6,0,0]'::vector,
            '[0.6,0.8,0,0]'::vector,
            '[0,0,0.8,0.6]'::vector,
            '[0,0,0.6,0.8]'::vector
          ]))
        """,
    )
    create_synthetic_gate_index(conn, "token_nodes")
    exec_sql(conn, "ANALYZE mv_recall_gate_docs")


def create_synthetic_gate_index(conn: psycopg.Connection[Any], graph_mode: str) -> None:
    if graph_mode not in {"token_nodes", "document_nodes"}:
        raise ValueError(f"unsupported synthetic gate graph mode: {graph_mode}")
    exec_sql(conn, "DROP INDEX IF EXISTS mv_recall_gate_docs_idx")
    exec_sql(
        conn,
        f"""
        CREATE INDEX mv_recall_gate_docs_idx ON mv_recall_gate_docs USING turbohybrid
          (colbert multivector_cosine_turbohybrid_ops)
          WITH (multivector_graph = {graph_mode},
                graph_m = 4,
                graph_ef_construction = 8,
                graph_ef_search = 8)
        """,
    )


def synthetic_exact_top(conn: psycopg.Connection[Any], limit: int) -> list[dict[str, Any]]:
    rows = fetch_all(
        conn,
        f"""
        WITH q AS (SELECT {SYNTHETIC_GATE_QUERY} AS mv)
        SELECT d.id,
               d.ctid::text,
               turbohybrid_multivector_maxsim_distance(q.mv, d.colbert) AS distance
        FROM mv_recall_gate_docs d, q
        ORDER BY turbohybrid_multivector_maxsim_distance(q.mv, d.colbert), d.id
        LIMIT %s
        """,
        (limit,),
    )
    exact: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        block, offset = tid_parts(str(row[1]))
        exact.append({
            "rank": rank,
            "doc_id": str(row[0]),
            "heap_block": block,
            "heap_offset": offset,
            "distance": float(row[2]),
            "maxsim": -float(row[2]),
        })
    return exact


def run_synthetic_exact_scan(conn: psycopg.Connection[Any], final_k: int) -> dict[str, Any]:
    started = time.perf_counter()
    rows = fetch_all(
        conn,
        f"""
        WITH q AS (SELECT {SYNTHETIC_GATE_QUERY} AS mv)
        SELECT d.id
        FROM mv_recall_gate_docs d, q
        ORDER BY turbohybrid_multivector_maxsim_distance(q.mv, d.colbert), d.id
        LIMIT %s
        """,
        (final_k,),
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    docs = [str(row[0]) for row in rows]
    return {
        "mode": "exact_scan",
        "candidate_source": "exact_scan",
        "result_doc_ids": docs,
        "top1": docs[0] if docs else None,
        "latency_ms": round(latency_ms, 3),
        "latency": summarize_ms([latency_ms]),
        "exact_top1_admitted_before_rerank": True,
        "exact_top10_admission_recall": 1.0,
        "doc_candidates": len(docs),
        "exact_rerank_docs": len(docs),
        "memory_bytes_estimate": 0,
        "scan_stats": {},
    }


def run_synthetic_gate_mode(
    conn: psycopg.Connection[Any],
    mode: dict[str, str],
    exact_top: list[dict[str, Any]],
    budget: int,
    final_k: int,
) -> dict[str, Any]:
    create_synthetic_gate_index(conn, mode.get("graph_mode", "token_nodes"))
    settings = {
        "enable_seqscan": "off",
        "jit": "off",
        "turbohybrid.multivector_subvector_k": "1",
        "turbohybrid.multivector_unique_docs_per_token": "1",
        "turbohybrid.multivector_max_raw_hits_per_token": "1",
        "turbohybrid.multivector_doc_candidate_k": str(budget),
        "turbohybrid.multivector_exact_rerank_k": str(max(budget, final_k, 10)),
        "turbohybrid.multivector_adaptive_widening": "off",
        "turbohybrid.multivector_candidate_source": mode["candidate_source"],
        "turbohybrid.multivector_plain_fallback": mode["plain_fallback"],
        "turbohybrid.multivector_candidate_reservoirs": mode["reservoirs"],
        "turbohybrid.multivector_debug_admission": "trace",
        "turbohybrid.multivector_debug_trace_limit": "1000",
    }
    for key, value in settings.items():
        exec_sql(conn, "SELECT set_config(%s, %s, false)", (key, value))

    started = time.perf_counter()
    rows = fetch_all(
        conn,
        f"""
        WITH q AS (SELECT {SYNTHETIC_GATE_QUERY} AS mv)
        SELECT d.id
        FROM mv_recall_gate_docs d, q
        ORDER BY d.colbert <~> turbohybrid_query(
            multivector_query => q.mv,
            dense_k => %s,
            final_k => %s
        )
        LIMIT %s
        """,
        (budget, final_k, final_k),
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    docs = [str(row[0]) for row in rows]
    stats = last_scan_stats(conn)
    trace = stats.get("multivector_admission_trace")
    trace_entries = trace if isinstance(trace, list) else []
    trace_by_heap = {
        key: entry
        for entry in trace_entries
        if isinstance(entry, dict) and (key := trace_key(entry)) is not None
    }
    inferred_exact_doc_admission = (
        mode["candidate_source"] in {"exact_doc_scan", "doc_graph_prototype"}
        or mode["plain_fallback"] == "force"
        or mode.get("graph_mode") == "document_nodes"
    )
    exact_with_admission: list[dict[str, Any]] = []
    for exact in exact_top:
        key = (int(exact["heap_block"]), int(exact["heap_offset"]))
        entry = trace_by_heap.get(key)
        result_rank = docs.index(exact["doc_id"]) + 1 if exact["doc_id"] in docs else None
        admitted = entry is not None or (inferred_exact_doc_admission and result_rank is not None)
        exact_with_admission.append({
            "rank": exact["rank"],
            "doc_id": exact["doc_id"],
            "admitted_before_rerank": admitted,
            "exact_rerank_rank": result_rank,
            "candidate_rank_before_rerank": (
                int(entry["candidate_rank_before_truncation"])
                if isinstance(entry, dict) and entry.get("candidate_rank_before_truncation") is not None
                else int(exact["rank"])
                if admitted and inferred_exact_doc_admission
                else None
            ),
        })
    top = exact_with_admission[: min(10, len(exact_with_admission))]
    admitted_count = sum(1 for item in top if item["admitted_before_rerank"])
    return {
        "mode": mode["name"],
        "candidate_source": stats.get("multivector_candidate_source"),
        "configured_candidate_source": mode["candidate_source"],
        "graph_mode": stats.get("multivector_graph_mode"),
        "configured_graph_mode": mode.get("graph_mode", "token_nodes"),
        "plain_fallback": mode["plain_fallback"],
        "reservoirs": mode["reservoirs"],
        "result_doc_ids": docs,
        "top1": docs[0] if docs else None,
        "latency_ms": round(latency_ms, 3),
        "latency": summarize_ms([latency_ms]),
        "exact_top1_admitted_before_rerank": bool(exact_with_admission and exact_with_admission[0]["admitted_before_rerank"]),
        "exact_top10_admission_recall": round(admitted_count / len(top), 6) if top else 0.0,
        "doc_candidates": max(
            scan_stat_int(stats, "multivector_doc_candidates"),
            scan_stat_int(stats, "multivector_doc_graph_candidates"),
        ),
        "exact_rerank_docs": max(
            scan_stat_int(stats, "multivector_exact_rerank_docs"),
            scan_stat_int(stats, "multivector_doc_graph_exact_rerank_docs"),
        ),
        "memory_bytes_estimate": memory_estimate_from_stats(stats),
        "raw_subvector_hits": scan_stat_int(stats, "multivector_raw_subvector_hits"),
        "unique_docs": scan_stat_int(stats, "multivector_unique_docs"),
        "trace_entries": len(trace_entries),
        "exact_top": exact_with_admission,
        "scan_stats": stats,
    }


def markdown_recall_gate_summary(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    lines = [
        "## Multivector Recall Gate",
        "",
        f"Status: **{status}**",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Exact top-1: `{report['exact_top1']}`",
        f"- Small candidate budget: `{report['candidate_budget']}`",
        f"- DBpedia data: `{report['dbpedia_status']}`",
        "",
        "| mode | top1 | top1 admitted | top10 admission recall | p50 ms | p95 ms | candidates | exact rerank docs | memory bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["results"]:
        latency = item.get("latency", {})
        lines.append(
            "| {mode} | {top1} | {admitted} | {recall:.3f} | {p50:.3f} | {p95:.3f} | {candidates} | {rerank} | {memory} |".format(
                mode=item["mode"],
                top1=item.get("top1"),
                admitted="yes" if item.get("exact_top1_admitted_before_rerank") else "no",
                recall=float(item.get("exact_top10_admission_recall", 0.0)),
                p50=float(latency.get("p50_ms", 0.0)),
                p95=float(latency.get("p95_ms", 0.0)),
                candidates=item.get("doc_candidates", 0),
                rerank=item.get("exact_rerank_docs", 0),
                memory=item.get("memory_bytes_estimate", 0),
            )
        )
    lines.extend([
        "",
        "Gate condition: `exact_scan`, `plain_fallback`, `exact_doc_scan`, "
        "`doc_graph_prototype`, and `document_nodes` must return and admit the "
        "synthetic exact top-1 document at the configured small candidate "
        "budget. DBpedia admission reporting remains optional and is run with "
        "`--admission-debug` when local data is available.",
        "",
    ])
    return "\n".join(lines)


def markdown_benchmark_summary(report: dict[str, Any]) -> str:
    dataset = report.get("dataset", {})
    settings = report.get("settings", {})
    lines = [
        "## DBpedia ColBERT Multivector Benchmark",
        "",
        f"- Documents: `{dataset.get('documents', 0)}`",
        f"- Queries: `{dataset.get('queries', 0)}`",
        f"- Qrels: `{dataset.get('qrels', 0)}`",
        f"- Final K: `{settings.get('final_k', 0)}`",
        f"- Clients: `{settings.get('clients', 0)}`",
        f"- Candidate source: `{settings.get('multivector_candidate_source', '')}`",
        f"- Plain fallback: `{settings.get('multivector_plain_fallback', '')}`",
        f"- Reservoirs: `{settings.get('multivector_candidate_reservoirs', '')}`",
        "",
        "| method | recall@10 | ndcg@10 | serial p50 ms | serial p95 ms | 8x qps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report.get("results", []):
        metrics = item.get("metrics", {})
        serial = item.get("serial_latency", {})
        parallel = item.get("parallel_8x", {})
        lines.append(
            "| {method} | {recall:.6f} | {ndcg:.6f} | {p50:.3f} | {p95:.3f} | {qps:.3f} |".format(
                method=item.get("method", ""),
                recall=float(metrics.get("recall@10", 0.0)),
                ndcg=float(metrics.get("ndcg@10", 0.0)),
                p50=float(serial.get("p50_ms", 0.0)),
                p95=float(serial.get("p95_ms", 0.0)),
                qps=float(parallel.get("throughput_qps", 0.0)),
            )
        )

    admission = report.get("admission_debug")
    if isinstance(admission, dict):
        aggregate = admission.get("aggregate", {})
        first_budget = aggregate.get("exact_top1_first_budget_admitted", {})
        lines.extend([
            "",
            "### Admission Debug",
            "",
            f"- Candidate source: `{admission.get('candidate_source', '')}`",
            f"- Plain fallback: `{admission.get('plain_fallback', '')}`",
            f"- Reservoirs: `{admission.get('candidate_reservoirs', '')}`",
            f"- Admission K: `{aggregate.get('admission_k', 0)}`",
            f"- Exact top-1 admission rate: `{float(aggregate.get('exact_top1_admission_rate', 0.0)):.6f}`",
            f"- Exact top-10 admission recall: `{float(aggregate.get('exact_top10_admission_recall', 0.0)):.6f}`",
            f"- First admitted budget p50: `{first_budget.get('p50', 0)}`",
            "",
            "| budget | latency p50 ms | latency p95 ms | runs |",
            "|---:|---:|---:|---:|",
        ])
        latency_by_budget = aggregate.get("latency_by_budget", {})
        if isinstance(latency_by_budget, dict):
            for budget, latency in sorted(latency_by_budget.items(), key=lambda item: int(item[0])):
                if not isinstance(latency, dict):
                    continue
                lines.append(
                    "| {budget} | {p50:.3f} | {p95:.3f} | {runs} |".format(
                        budget=budget,
                        p50=float(latency.get("p50_ms", 0.0)),
                        p95=float(latency.get("p95_ms", 0.0)),
                        runs=int(latency.get("runs", 0)),
                    )
                )
        lines.extend([
            "",
            "Admission debug compares exact document top-K with the candidate set "
            "before exact rerank. Improvements here mean better admission; unchanged "
            "admission with different final metrics should be treated as reranking or "
            "metric variance until proven otherwise.",
        ])

    lines.append("")
    return "\n".join(lines)


def run_multivector_recall_gate(conn: psycopg.Connection[Any], args: argparse.Namespace) -> dict[str, Any]:
    setup_multivector_recall_gate(conn)
    exact_top = synthetic_exact_top(conn, 10)
    if not exact_top or exact_top[0]["doc_id"] != "good":
        raise RuntimeError(f"synthetic exact MaxSim top-1 is not good: {exact_top[:1]}")

    modes = [
        {"name": "token_graph", "graph_mode": "token_nodes", "candidate_source": "graph", "plain_fallback": "off", "reservoirs": "off"},
        {"name": "exact_token_scan", "graph_mode": "token_nodes", "candidate_source": "exact_token_scan", "plain_fallback": "off", "reservoirs": "off"},
        {"name": "reservoir_balanced", "graph_mode": "token_nodes", "candidate_source": "graph", "plain_fallback": "off", "reservoirs": "balanced"},
        {"name": "plain_fallback", "graph_mode": "token_nodes", "candidate_source": "graph", "plain_fallback": "force", "reservoirs": "off"},
        {"name": "exact_doc_scan", "graph_mode": "token_nodes", "candidate_source": "exact_doc_scan", "plain_fallback": "off", "reservoirs": "off"},
        {"name": "doc_graph_prototype", "graph_mode": "token_nodes", "candidate_source": "doc_graph_prototype", "plain_fallback": "off", "reservoirs": "off"},
        {"name": "document_nodes", "graph_mode": "document_nodes", "candidate_source": "graph", "plain_fallback": "off", "reservoirs": "off"},
    ]
    results = [run_synthetic_exact_scan(conn, args.final_k)]
    for mode in modes:
        results.append(run_synthetic_gate_mode(conn, mode, exact_top, args.recall_gate_budget, args.final_k))

    required_modes = {"exact_scan", "plain_fallback", "exact_doc_scan", "doc_graph_prototype", "document_nodes"}
    passed = all(
        item["mode"] not in required_modes
        or (item.get("top1") == "good" and item.get("exact_top1_admitted_before_rerank") is True)
        for item in results
    )
    report = {
        "suite": "multivector_recall_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "synthetic_many_moderate",
        "dbpedia_status": "not_required",
        "candidate_budget": args.recall_gate_budget,
        "final_k": args.final_k,
        "exact_top1": exact_top[0]["doc_id"],
        "exact_top": exact_top,
        "passed": passed,
        "results": results,
    }
    report["markdown_summary"] = markdown_recall_gate_summary(report)
    return report


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    env_dataset = os.environ.get("DBPEDIA_DATASET")
    env_beir = os.environ.get("BEIR_DBPEDIA_DATASET")
    env_model = os.environ.get("PG_COLBERT_LLAMA_TEST_MODEL")
    env_precomputed = os.environ.get("DBPEDIA_COLBERT_PRECOMPUTED_DATASET")
    if args.dataset is None and env_dataset:
        args.dataset = Path(env_dataset)
    if args.beir_dataset is None and env_beir:
        args.beir_dataset = Path(env_beir)
    if args.precomputed_dataset is None and env_precomputed:
        args.precomputed_dataset = env_precomputed
    if args.model_path is None:
        args.model_path = Path(env_model or DEFAULT_MODEL_PATH)
    if args.token_ablation_skip_tokens and not args.token_ablation_query_id:
        raise SystemExit("--token-ablation-skip-tokens requires --token-ablation-query-id")
    if args.token_ablation_final_k is not None and args.token_ablation_final_k <= 0:
        raise SystemExit("--token-ablation-final-k must be positive")
    if args.token_ablation_dense_k is not None and args.token_ablation_dense_k <= 0:
        raise SystemExit("--token-ablation-dense-k must be positive")

    if args.precomputed_dataset is None and not args.reuse_data and args.dataset is None and not args.multivector_recall_gate:
        raise SystemExit("pass --dataset or set DBPEDIA_DATASET")
    if args.dataset is not None:
        args.dataset = args.dataset.resolve()
        if not args.dataset.exists():
            raise SystemExit(f"dataset path does not exist: {args.dataset}")
    if args.precomputed_dataset is None and args.beir_dataset is None and not args.multivector_recall_gate:
        raise SystemExit("pass --beir-dataset or set BEIR_DBPEDIA_DATASET")
    if args.beir_dataset is not None:
        args.beir_dataset = args.beir_dataset.resolve()
    if args.beir_dataset is not None and not args.beir_dataset.exists():
        raise SystemExit(f"BEIR dataset path does not exist: {args.beir_dataset}")

    args.model_path = args.model_path.resolve()
    if args.precomputed_dataset is None and not args.model_path.is_file() and not args.multivector_recall_gate:
        raise SystemExit(
            f"model file does not exist: {args.model_path}\n"
            "Use johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF and pass --model-path."
        )
    if args.model_alias is None:
        args.model_alias = args.model_path.stem
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = sorted(set(methods) - {QUERY_ONLY_METHOD, RRF_METHOD, EXACT_SCAN_METHOD})
    if unknown:
        raise SystemExit(f"unknown method(s): {', '.join(unknown)}")
    args.methods = methods
    if args.generation_clients is None:
        args.generation_clients = args.clients
    if args.generation_clients < 1:
        raise SystemExit("--generation-clients must be at least 1")
    if args.precomputed_batch_size < 1:
        raise SystemExit("--precomputed-batch-size must be at least 1")
    if args.generation_threads < 1:
        raise SystemExit("--generation-threads must be at least 1")
    if args.generation_n_gpu_layers < 0:
        raise SystemExit("--generation-n-gpu-layers must be at least 0")
    if args.generation_batch_sequences < 1:
        raise SystemExit("--generation-batch-sequences must be at least 1")
    if args.generation_n_batch is None:
        args.generation_n_batch = max(512, args.generation_batch_sequences * args.max_doc_vectors)
    if args.generation_n_batch < 1:
        raise SystemExit("--generation-n-batch must be at least 1")
    if args.output is None:
        output_stem = "multivector-recall-gate" if args.multivector_recall_gate else "dbpedia-colbert-multivector"
        args.output = Path("benchmarks/results/" + datetime.now(timezone.utc).strftime(f"{output_stem}-%Y%m%dT%H%M%SZ.json"))
    args.output = args.output.resolve()
    if args.markdown_output is not None:
        args.markdown_output = args.markdown_output.resolve()
    if args.recall_gate_markdown_output is not None:
        args.recall_gate_markdown_output = args.recall_gate_markdown_output.resolve()
    if args.quality_k > args.final_k:
        print(
            f"--quality-k {args.quality_k} is greater than --final-k {args.final_k}; "
            "quality metrics will be capped by final_k",
            file=sys.stderr,
        )
    if args.admission_k < 1:
        raise SystemExit("--admission-k must be at least 1")
    if args.admission_trace_limit < 0:
        raise SystemExit("--admission-trace-limit must be non-negative")
    if args.multivector_plain_fallback_max_docs < 0:
        raise SystemExit("--multivector-plain-fallback-max-docs must be non-negative")
    if not 0.0 <= args.multivector_plain_fallback_candidate_fraction <= 1.0:
        raise SystemExit("--multivector-plain-fallback-candidate-fraction must be between 0 and 1")
    if args.multivector_per_token_doc_reservoir_k < 0:
        raise SystemExit("--multivector-per-token-doc-reservoir-k must be non-negative")
    if args.multivector_coverage_reservoir_k < 0:
        raise SystemExit("--multivector-coverage-reservoir-k must be non-negative")
    try:
        admission_budgets = [
            int(item.strip())
            for item in args.admission_budget_sweep.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise SystemExit("--admission-budget-sweep must contain comma-separated integers") from exc
    if args.admission_debug and (not admission_budgets or any(budget < 1 for budget in admission_budgets)):
        raise SystemExit("--admission-budget-sweep must contain positive integer budgets")
    if args.recall_gate_budget < 1:
        raise SystemExit("--recall-gate-budget must be positive")
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", "pgturbohybrid_dbpedia_colbert"))
    parser.add_argument("--dataset", type=Path, default=None, help="Qdrant DBpedia parquet dataset root")
    parser.add_argument("--beir-dataset", type=Path, default=None, help="BEIR DBpedia dataset root containing queries")
    parser.add_argument("--qrels", default=None, help="BEIR DBpedia qrels TSV path")
    parser.add_argument(
        "--precomputed-dataset",
        default=None,
        help="local directory or Hugging Face dataset repo id with precomputed DBpedia ColBERT multivectors",
    )
    parser.add_argument(
        "--precomputed-batch-size",
        type=int,
        default=4096,
        help="Parquet/COPY batch size when loading --precomputed-dataset",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="GGUF ColBERT model path")
    parser.add_argument("--model-alias", default=None, help="pg_colbert_llama model alias, defaults to model stem")
    parser.add_argument("--expected-dim", type=int, default=128)
    parser.add_argument("--max-doc-vectors", type=int, default=256)
    parser.add_argument("--max-query-vectors", type=int, default=32)
    parser.add_argument("--query-length", type=int, default=32)
    parser.add_argument("--max-docs", type=int, default=1000, help="documents to load; 0 means all available")
    parser.add_argument("--max-queries", type=int, default=32, help="queries to load from qrels; 0 means all")
    parser.add_argument(
        "--no-prioritize-qrels",
        dest="prioritize_qrels",
        action="store_false",
        help="load the first corpus rows instead of filling the sample with judged relevant documents first",
    )
    parser.add_argument("--generation-sample-docs", type=int, default=100)
    parser.add_argument("--insert-batch-size", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument(
        "--generation-clients",
        type=int,
        default=None,
        help="parallel PostgreSQL backends for document multivector generation; defaults to --clients",
    )
    parser.add_argument(
        "--generation-threads",
        type=int,
        default=1,
        help="llama.cpp worker threads per generation backend",
    )
    parser.add_argument(
        "--generation-n-gpu-layers",
        type=int,
        default=0,
        help="llama.cpp model layers to offload to GPU per generation backend",
    )
    parser.add_argument(
        "--generation-batch-sequences",
        type=int,
        default=8,
        help="independent documents per pg_colbert_llama C batch",
    )
    parser.add_argument(
        "--generation-n-batch",
        type=int,
        default=None,
        help="llama.cpp token batch budget for C batched generation; defaults to batch sequences times max doc vectors",
    )
    parser.add_argument(
        "--no-generation-warmup",
        dest="generation_warmup",
        action="store_false",
        help="include backend model/context warmup in persisted generation timing",
    )
    parser.add_argument("--dense-k", type=int, default=100)
    parser.add_argument("--bm25-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--multivector-subvector-k", type=int, default=100)
    parser.add_argument("--multivector-unique-docs-per-token", type=int, default=100)
    parser.add_argument("--multivector-max-raw-hits-per-token", type=int, default=400)
    parser.add_argument("--multivector-adaptive-widening", choices=("off", "auto", "on"), default="auto")
    parser.add_argument("--multivector-doc-candidate-k", type=int, default=100)
    parser.add_argument("--multivector-exact-rerank-k", type=int, default=100)
    parser.add_argument(
        "--multivector-candidate-source",
        choices=("graph", "exact_token_scan", "exact_doc_scan", "doc_graph_prototype"),
        default="graph",
        help="multivector candidate source for query-only/RRF retrieval; exact_* and doc_graph_prototype are debug validation modes",
    )
    parser.add_argument(
        "--multivector-plain-fallback",
        choices=("auto", "off", "force"),
        default="auto",
        help="exact heap MaxSim fallback for small or near-exhaustive multivector scans",
    )
    parser.add_argument("--multivector-plain-fallback-max-docs", type=int, default=1000)
    parser.add_argument("--multivector-plain-fallback-candidate-fraction", type=float, default=0.5)
    parser.add_argument(
        "--multivector-candidate-reservoirs",
        choices=("off", "conservative", "balanced"),
        default="conservative",
        help="bounded multivector candidate reservoir selection before exact rerank",
    )
    parser.add_argument("--multivector-per-token-doc-reservoir-k", type=int, default=1)
    parser.add_argument("--multivector-coverage-reservoir-k", type=int, default=10)
    parser.add_argument(
        "--multivector-bm25-candidate-injection",
        choices=("off", "hybrid_only", "dense_with_text"),
        default="off",
        help="inject BM25 candidates into the multivector exact MaxSim rerank candidate set",
    )
    parser.add_argument(
        "--admission-debug",
        action="store_true",
        help="run exact-vs-candidate admission recall diagnostics after normal retrieval",
    )
    parser.add_argument(
        "--admission-k",
        type=int,
        default=10,
        help="exact top-K document set to compare against candidate admission",
    )
    parser.add_argument(
        "--admission-budget-sweep",
        default="100,200,400,800,1600,3200,6400,10000",
        help="comma-separated multivector document candidate budgets to sweep",
    )
    parser.add_argument(
        "--admission-trace-limit",
        type=int,
        default=1000,
        help="bounded trace entries requested from turbohybrid_last_scan_stats()",
    )
    parser.add_argument(
        "--multivector-recall-gate",
        action="store_true",
        help="run the deterministic synthetic many-moderate admission recall gate only",
    )
    parser.add_argument(
        "--recall-gate-budget",
        type=int,
        default=1,
        help="document candidate budget for the synthetic multivector recall gate",
    )
    parser.add_argument(
        "--recall-gate-markdown-output",
        type=Path,
        default=None,
        help="optional Markdown summary path for --multivector-recall-gate",
    )
    parser.add_argument(
        "--token-ablation-query-id",
        default=None,
        help="run one multivector query with per-token diagnostics for this query id",
    )
    parser.add_argument(
        "--token-ablation-skip-tokens",
        default="",
        help="comma-separated query token ordinals to skip in the token-ablation variant",
    )
    parser.add_argument(
        "--token-ablation-final-k",
        type=int,
        default=None,
        help="final_k for the optional token-ablation query; defaults to --final-k",
    )
    parser.add_argument(
        "--token-ablation-dense-k",
        type=int,
        default=None,
        help="dense_k for the optional token-ablation query; defaults to --dense-k",
    )
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--quality-k", type=int, default=10)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--warm-queries", type=int, default=2)
    parser.add_argument("--timed-queries", type=int, default=10)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--reuse-embeddings", action="store_true")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument(
        "--allow-unvalidated-embeddings",
        action="store_true",
        help="continue even if the Sauerkraut 15m tokenization preflight fails",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="optional Markdown summary path for benchmark results or --multivector-recall-gate",
    )
    return validate_args(parser.parse_args())


def main() -> None:
    args = parse_args()
    if args.multivector_recall_gate:
        conn = connect(args)
        try:
            report = run_multivector_recall_gate(conn, args)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            markdown_paths = {
                path
                for path in (args.markdown_output, args.recall_gate_markdown_output)
                if path is not None
            }
            for markdown_path in sorted(markdown_paths):
                markdown_path.parent.mkdir(parents=True, exist_ok=True)
                markdown_path.write_text(report["markdown_summary"], encoding="utf-8")
            print(args.output)
            for markdown_path in sorted(markdown_paths):
                print(markdown_path)
            if not report["passed"]:
                raise SystemExit(1)
        finally:
            conn.close()
        return

    if args.precomputed_dataset is None:
        queries = load_queries(args.beir_dataset)
        qrels_path = resolve_qrels_path(args.qrels, args.beir_dataset)
        all_qrels = read_qrels(qrels_path)
        qids = choose_query_ids(all_qrels, queries, args.max_queries)
        qrels = {qid: all_qrels[qid] for qid in qids}
    else:
        qrels_path = None
        qids = []
        queries = {}
        qrels = {}

    conn = connect(args)
    try:
        setup_schema(conn, include_colbert_llama=args.precomputed_dataset is None)
        if args.precomputed_dataset is None:
            embedding_health = validate_embedding_health(conn, args)
            load_phase = load_data(conn, args, qids, queries, qrels)
        else:
            embedding_health = {
                "validated": True,
                "skipped": True,
                "reason": "precomputed multivectors loaded from dataset",
            }
            load_phase = load_precomputed_multivectors(conn, args)
        doc_ids = selected_doc_ids(conn, args.max_docs)
        query_ids = selected_query_ids(conn)
        qrels = loaded_qrels(conn)
        if args.precomputed_dataset is None:
            generation_phase = measure_generation_sample(conn, args, doc_ids, query_ids)
            insert_phase = persist_multivectors(conn, args, doc_ids, query_ids)
        else:
            generation_phase = {
                "skipped": True,
                "reason": "precomputed multivectors loaded from dataset",
            }
            insert_phase = {
                "reused": True,
                "precomputed": True,
                "documents": len(doc_ids),
                "queries": len(query_ids),
                "source": args.precomputed_dataset,
                "note": "runtime generation skipped; rows were loaded from precomputed multivector dataset",
            }
        index_phase = build_index(conn, args)
        set_retrieval_gucs(conn, args, "dbpedia_colbert_serial")
        encoded_queries = load_encoded_queries(conn)

        result_methods: list[dict[str, Any]] = []
        for method in args.methods:
            run, serial_latencies, serial_stats = run_serial_retrieval(conn, method, encoded_queries, args, args.final_k)
            parallel = run_parallel_retrieval(args, method, encoded_queries)
            result_methods.append({
                "method": method,
                "retrieval_mode": {
                    QUERY_ONLY_METHOD: "multivector_query_only",
                    RRF_METHOD: "multivector_plus_bm25_rrf",
                    EXACT_SCAN_METHOD: "multivector_exact_scan",
                }[method],
                "metrics": method_metrics(run, qrels, args.final_k, args.quality_k),
                "serial_latency": summarize_ms(serial_latencies),
                "parallel_8x": parallel,
                "top10_by_query": {qid: docs[:10] for qid, docs in run.items()},
                "scan_summary": scan_summary(serial_stats),
                "index_stats": index_phase.get("index_stats", {}),
            })

        admission_debug = (
            run_admission_debug(conn, args, encoded_queries)
            if args.admission_debug
            else None
        )
        token_ablation = run_token_ablation(conn, args, encoded_queries, qrels)

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
                "beir_dataset_path": portable_path(args.beir_dataset) if args.beir_dataset else None,
                "qrels_path": portable_path(qrels_path) if qrels_path else None,
                "precomputed_dataset": args.precomputed_dataset,
                "documents": load_phase["documents"],
                "queries": load_phase["queries"],
                "qrels": load_phase["qrels"],
                "max_docs": args.max_docs,
                "max_queries": args.max_queries,
                "prioritize_qrels": args.prioritize_qrels,
            },
            "model": {
                "gguf": "johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF",
                "source_model": "VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m",
                "model_path": portable_path(args.model_path),
                "model_alias": args.model_alias,
                "expected_dim": args.expected_dim,
                "max_doc_vectors": args.max_doc_vectors,
                "max_query_vectors": args.max_query_vectors,
                "query_length": args.query_length,
                "embedding_health": embedding_health,
            },
            "settings": {
                "dense_k": args.dense_k,
                "bm25_k": args.bm25_k,
                "rrf_k": args.rrf_k,
                "multivector_subvector_k": args.multivector_subvector_k,
                "multivector_unique_docs_per_token": args.multivector_unique_docs_per_token,
                "multivector_max_raw_hits_per_token": args.multivector_max_raw_hits_per_token,
                "multivector_adaptive_widening": args.multivector_adaptive_widening,
                "multivector_doc_candidate_k": args.multivector_doc_candidate_k,
                "multivector_exact_rerank_k": args.multivector_exact_rerank_k,
                "multivector_candidate_source": args.multivector_candidate_source,
                "multivector_plain_fallback": args.multivector_plain_fallback,
                "multivector_plain_fallback_max_docs": args.multivector_plain_fallback_max_docs,
                "multivector_plain_fallback_candidate_fraction": args.multivector_plain_fallback_candidate_fraction,
                "multivector_candidate_reservoirs": args.multivector_candidate_reservoirs,
                "multivector_per_token_doc_reservoir_k": args.multivector_per_token_doc_reservoir_k,
                "multivector_coverage_reservoir_k": args.multivector_coverage_reservoir_k,
                "multivector_bm25_candidate_injection": args.multivector_bm25_candidate_injection,
                "token_ablation_query_id": args.token_ablation_query_id,
                "token_ablation_skip_tokens": args.token_ablation_skip_tokens,
                "token_ablation_final_k": args.token_ablation_final_k,
                "token_ablation_dense_k": args.token_ablation_dense_k,
                "final_k": args.final_k,
                "quality_k": args.quality_k,
                "clients": args.clients,
                "generation_clients": args.generation_clients,
                "generation_threads": args.generation_threads,
                "generation_n_gpu_layers": args.generation_n_gpu_layers,
                "generation_batch_sequences": args.generation_batch_sequences,
                "generation_n_batch": args.generation_n_batch,
                "generation_warmup": args.generation_warmup,
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
        if admission_debug is not None:
            output["admission_debug"] = admission_debug
        if token_ablation is not None:
            output["token_ablation"] = token_ablation
        output["markdown_summary"] = markdown_benchmark_summary(output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(output["markdown_summary"], encoding="utf-8")
        print(args.output)
        if args.markdown_output is not None:
            print(args.markdown_output)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
