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
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import psycopg
except ImportError:  # pragma: no cover - exercised by --help without bench deps
    psycopg = None  # type: ignore[assignment]


DEFAULT_METHODS = "pgturbohybrid_colbert_multivector_query_only"
QUERY_ONLY_METHOD = "pgturbohybrid_colbert_multivector_query_only"
RRF_METHOD = "pgturbohybrid_colbert_multivector_rrf"
EXACT_SCAN_METHOD = "pgturbohybrid_colbert_multivector_exact_scan"
DEFAULT_MODEL_PATH = ".nix-dev/models/colbert-15m/sauerkraut-modern.gguf"
DEFAULT_COLBERT_MODEL_NAME = "VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m"
COLBERT_MODEL_DIMENSIONS = {
    "colbert-ir/colbertv2.0": 128,
    "answerdotai/answerai-colbert-small-v1": 96,
    "jinaai/jina-colbert-v2": 128,
    "jinaai/jina-colbert-v2-96": 96,
    "jinaai/jina-colbert-v2-64": 64,
    "lightonai/gte-moderncolbert-v1": 128,
    "chadboyda/reason-moderncolbert": 128,
    "vagosolutions/sauerkrautlm-multi-colbert-15m": 128,
    "johannhartmann/sauerkrautlm-multi-colbert-15m-gguf": 128,
    "vidore/colpali-v1.2": 128,
    "colpali-like-visual": 128,
}
DOCUMENT_NODE_STORAGE_CHOICES = ("f32", "f16", "sq8")
DOCUMENT_NODE_STORAGE_CACHE_CHOICES = ("auto", "resident", "paged")
DOCUMENT_NODE_TOKEN_POOLING_CHOICES = ("off", "kmeans", "greedy_cosine")
DOCUMENT_NODE_CENTROIDS_CHOICES = ("off", "kmeans")
MULTIVECTOR_PROXY_ENCODER_CHOICES = (
    "normalized_mean",
    "first_token",
    "centroid_mean",
    "mean_pool",
    "max_pool",
    "random_projection_fde",
    "learned_projection_placeholder",
)
MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_CHOICES = ("off", "bm25", "learned_sparse")
MULTIVECTOR_DOC_BUILD_SCORER_CHOICES = ("proxy", "exact_symmetric")
DOCUMENT_NODE_BASELINE_MODES = (
    "token_nodes",
    "exact_token_scan",
    "plain_fallback",
    "exact_doc_scan",
    "document_nodes",
    "proxy_vector",
    "centroid_lite",
    "quantized_inverted_experimental",
)
HYBRID_EVALUATION_MODES = (
    "exact_scan",
    "document_nodes",
    "document_nodes_bm25_admission",
    "document_nodes_bm25_rrf",
    "document_nodes_bm25_dbsf",
    "proxy_vector_document_nodes",
    "learned_sparse_exact_maxsim",
)
OPTIONAL_HYBRID_EVALUATION_MODES = (
    "quantized_inverted_experimental",
)
SUPPORTED_HYBRID_EVALUATION_MODES = (
    *HYBRID_EVALUATION_MODES,
    *OPTIONAL_HYBRID_EVALUATION_MODES,
)
DEFAULT_PARALLEL_SAFETY_MAX_SERIAL_P95_MS = 5000.0
DEFAULT_PARALLEL_SAFETY_MAX_DOCS_SCORED = 5000
DEFAULT_PARALLEL_SAFETY_MAX_EXACT_RERANK_DOCS = 1000
DEFAULT_PARALLEL_SAFETY_MAX_EXACT_PAIRS = 5_000_000
DEFAULT_PARALLEL_SAFETY_MAX_SIDECAR_BYTES = 512 * 1024 * 1024
DEFAULT_ADMISSION_BUDGET_SWEEP = "100,200,400,800,1600,3200,6400,10000"
DOCUMENT_NODE_SERVING_GRID_BUDGETS = (50, 100, 200, 400, 800)
DOCUMENT_NODE_SERVING_GRID_BUDGET_SWEEP = ",".join(
    str(budget) for budget in DOCUMENT_NODE_SERVING_GRID_BUDGETS
)
DOCUMENT_NODE_SERVING_GRID_EF = (50, 100, 200)
DOCUMENT_NODE_SERVING_GRID_OVERSAMPLING = (1, 2)
SERVING_STATS_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "core": (
        "multivector_candidate_source",
        "multivector_doc_graph_warning",
        "multivector_doc_graph_nodes",
        "multivector_doc_graph_docs_scored",
        "multivector_doc_graph_edges_visited",
        "multivector_doc_graph_candidates",
        "multivector_doc_graph_exact_rerank_docs",
        "multivector_exact_rerank_docs",
        "multivector_exact_rerank_pairs",
        "multivector_exact_kernel",
    ),
    "proxy": (
        "proxy_encoder_kind",
        "proxy_candidates",
        "proxy_top1_admission",
        "proxy_exact_rerank_docs",
        "multivector_centroid_count",
        "multivector_centroid_prerank_docs",
        "multivector_full_maxsim_rerank_docs",
    ),
    "centroid_lite": (
        "centroid_lists_visited",
        "centroid_docs_touched",
        "centroid_pruned_docs",
        "centroid_candidates",
    ),
    "storage_cache": (
        "multivector_doc_sidecar_cache_mode",
        "multivector_doc_sidecar_pages_read",
        "multivector_doc_sidecar_cache_hits",
        "multivector_doc_sidecar_cache_misses",
        "multivector_doc_sidecar_bytes_touched",
        "multivector_doc_sidecar_vectors_loaded",
    ),
    "pooling": (
        "multivector_tokens_original",
        "multivector_tokens_pooled",
        "multivector_token_pooling_ratio",
    ),
    "sparse_bm25_rescue": (
        "multivector_bm25_injection_enabled",
        "multivector_bm25_injection_candidates",
        "multivector_bm25_injection_retained",
        "learned_sparse_candidates",
        "learned_sparse_retained_for_maxsim",
        "learned_sparse_branch_latency_us",
    ),
}


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


@dataclass(frozen=True)
class DocumentNodeServingProfile:
    name: str
    candidate_source: str
    proxy_encoder: str = "normalized_mean"
    centroids: str = "off"
    centroid_count: str = "auto"
    storage_kind: str = "f16"
    cache_mode: str = "auto"
    token_pooling: str = "off"
    token_pooling_target_ratio: float = 1.0
    plain_fallback: str = "off"


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


def git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


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
    if psycopg is None:
        raise SystemExit(
            "psycopg is required to run the DBpedia ColBERT benchmark. "
            "Use the Nix benchmark helper or install psycopg[binary]."
        )
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


def loaded_document_count(conn: psycopg.Connection[Any]) -> int:
    row = fetch_one(conn, "SELECT count(*) FROM dbpedia_colbert_docs WHERE colbert IS NOT NULL")
    return int(row[0]) if row else 0


def multivector_dataset_stats(conn: psycopg.Connection[Any]) -> dict[str, Any]:
    row = fetch_one(
        conn,
        """
        WITH doc_stats AS (
          SELECT
            turbohybrid_multivector_count(colbert)::double precision AS token_count,
            turbohybrid_multivector_dims(colbert)::int AS dim
          FROM dbpedia_colbert_docs
          WHERE colbert IS NOT NULL
        )
        SELECT
          count(*)::bigint,
          avg(token_count)::double precision,
          percentile_cont(0.5) WITHIN GROUP (ORDER BY token_count)::double precision,
          percentile_cont(0.9) WITHIN GROUP (ORDER BY token_count)::double precision,
          min(token_count)::double precision,
          max(token_count)::double precision,
          min(dim)::int,
          max(dim)::int
        FROM doc_stats
        """,
    )
    if not row or int(row[0] or 0) == 0:
        return {
            "docs": 0,
            "avg_tokens": 0.0,
            "p50_tokens": 0.0,
            "p90_tokens": 0.0,
            "min_tokens": 0,
            "max_tokens": 0,
            "dim": 0,
            "dim_min": 0,
            "dim_max": 0,
        }

    dim_min = int(row[6] or 0)
    dim_max = int(row[7] or 0)
    return {
        "docs": int(row[0]),
        "avg_tokens": round(float(row[1] or 0.0), 3),
        "p50_tokens": round(float(row[2] or 0.0), 3),
        "p90_tokens": round(float(row[3] or 0.0), 3),
        "min_tokens": int(float(row[4] or 0.0)),
        "max_tokens": int(float(row[5] or 0.0)),
        "dim": dim_min if dim_min == dim_max else None,
        "dim_min": dim_min,
        "dim_max": dim_max,
    }


def validate_document_node_proxy_build(
    *,
    args: argparse.Namespace,
    index_stats: dict[str, Any],
    build_stats: dict[str, Any],
    row_count: int,
) -> dict[str, Any]:
    graph_mode = index_stats.get("multivector_graph_mode")
    requested_scorer = getattr(args, "multivector_doc_build_scorer", "proxy")
    observed_scorer = index_stats.get("multivector_doc_build_scorer")
    exact_calls = int(build_stats.get("multivector_doc_exact_build_distance_calls", 0) or 0)
    node_count = int(index_stats.get("node_count", 0) or 0)
    build_fast_edges = bool(index_stats.get("build_fast_edges", False))

    checks = {
        "graph_mode": graph_mode,
        "requested_doc_build_scorer": requested_scorer,
        "observed_doc_build_scorer": observed_scorer,
        "doc_exact_build_distance_calls": exact_calls,
        "node_count": node_count,
        "row_count": row_count,
        "build_fast_edges": build_fast_edges,
    }
    if graph_mode != "document_nodes":
        return {**checks, "enforced": False, "reason": "not_document_nodes"}
    if requested_scorer == "exact_symmetric":
        return {**checks, "enforced": False, "reason": "exact_symmetric_explicitly_allowed"}

    if observed_scorer != "proxy":
        raise SystemExit(
            "refusing benchmark: document_nodes index used "
            f"multivector_doc_build_scorer={observed_scorer!r}, expected 'proxy'"
        )
    if exact_calls != 0:
        raise SystemExit(
            "refusing benchmark: document_nodes proxy build made "
            f"{exact_calls} exact document-document MaxSim build-distance calls"
        )
    if not build_fast_edges:
        raise SystemExit(
            "refusing benchmark: document_nodes proxy index did not use "
            "fast edge construction"
        )
    if node_count != row_count:
        raise SystemExit(
            "refusing benchmark: document_nodes proxy index node_count "
            f"{node_count} does not match loaded document count {row_count}"
        )
    return {**checks, "enforced": True, "reason": "ok"}


def build_index_reloptions(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    pooling_mode = getattr(args, "multivector_token_pooling", "off")
    pooling_ratio = float(getattr(args, "multivector_token_pooling_target_ratio", 0.5))
    pooling_min_tokens = int(getattr(args, "multivector_token_pooling_min_tokens", 16))
    centroids = getattr(args, "multivector_centroids", "off")
    centroid_count = getattr(args, "multivector_centroid_count", "auto")
    if isinstance(centroid_count, str) and centroid_count.lower() == "auto":
        centroid_count_sql = 0
    else:
        centroid_count_sql = int(centroid_count)
    proxy_encoder = getattr(args, "multivector_proxy_encoder", "normalized_mean")
    reloptions = [
        "quantization_bits = 4",
        "exact_storage = off",
        f"multivector_graph = {args.multivector_graph}",
        f"multivector_doc_build_scorer = {args.multivector_doc_build_scorer}",
        f"multivector_token_pooling = {pooling_mode}",
        f"multivector_token_pooling_target_ratio = {pooling_ratio}",
        f"multivector_token_pooling_min_tokens = {pooling_min_tokens}",
        f"multivector_centroids = {centroids}",
        f"multivector_centroid_count = {centroid_count_sql}",
        f"multivector_proxy_encoder = {proxy_encoder}",
    ]
    index_graph_m = int(getattr(args, "index_graph_m", 0))
    index_graph_ef_construction = int(getattr(args, "index_graph_ef_construction", 0))
    index_graph_ef_search = int(getattr(args, "index_graph_ef_search", 0))
    index_native_segments = int(getattr(args, "index_native_segments", 0))
    if index_graph_m > 0:
        reloptions.append(f"graph_m = {index_graph_m}")
    if index_graph_ef_construction > 0:
        reloptions.append(f"graph_ef_construction = {index_graph_ef_construction}")
    if index_graph_ef_search > 0:
        reloptions.append(f"graph_ef_search = {index_graph_ef_search}")
    if index_native_segments > 0:
        reloptions.append(f"native_segments = {index_native_segments}")
    return reloptions, {
        "index_graph_m": index_graph_m,
        "index_graph_ef_construction": index_graph_ef_construction,
        "index_graph_ef_search": index_graph_ef_search,
        "index_native_segments": index_native_segments,
    }


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


def parse_sparse_jsonl(path: Path, preferred_id_key: str) -> list[tuple[str, list[int], list[float]]]:
    rows: list[tuple[str, list[int], list[float]]] = []
    id_keys = (preferred_id_key, "id")
    term_keys = ("term_ids", "terms", "indices")
    weight_keys = ("weights", "scores", "values")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL row") from exc
            if not isinstance(item, dict):
                raise SystemExit(f"{path}:{line_no}: expected a JSON object")

            sparse_id = next((item[key] for key in id_keys if key in item), None)
            terms = next((item[key] for key in term_keys if key in item), None)
            weights = next((item[key] for key in weight_keys if key in item), None)
            if sparse_id is None or terms is None or weights is None:
                raise SystemExit(
                    f"{path}:{line_no}: expected {preferred_id_key!r}, term_ids, and weights fields"
                )
            if not isinstance(terms, list) or not isinstance(weights, list) or len(terms) != len(weights):
                raise SystemExit(f"{path}:{line_no}: term_ids and weights must be equal-length arrays")

            term_ids: list[int] = []
            sparse_weights: list[float] = []
            for term, weight in zip(terms, weights):
                term_id = int(term)
                sparse_weight = float(weight)
                if term_id < 0 or not math.isfinite(sparse_weight):
                    raise SystemExit(f"{path}:{line_no}: sparse terms must be non-negative and weights finite")
                term_ids.append(term_id)
                sparse_weights.append(sparse_weight)
            rows.append((str(sparse_id), term_ids, sparse_weights))
    return rows


def require_complete_learned_sparse_args(args: argparse.Namespace) -> None:
    if (args.learned_sparse_doc_jsonl is None) != (args.learned_sparse_query_jsonl is None):
        raise SystemExit(
            "--learned-sparse-doc-jsonl and --learned-sparse-query-jsonl must be supplied together"
        )


def has_learned_sparse_docs(conn: psycopg.Connection[Any]) -> bool:
    row = fetch_one(
        conn,
        "SELECT EXISTS (SELECT 1 FROM dbpedia_colbert_docs WHERE learned_sparse IS NOT NULL)",
    )
    return bool(row and row[0])


def learned_sparse_text_query_sql() -> str:
    return """
    CASE
      WHEN q.learned_sparse IS NOT NULL THEN turbohybrid_sparse_vector_to_tsquery(q.learned_sparse)
      ELSE websearch_to_tsquery('simple', q.query_text)
    END
    """


def load_learned_sparse_vectors(conn: psycopg.Connection[Any], args: argparse.Namespace) -> dict[str, Any]:
    require_complete_learned_sparse_args(args)
    if args.learned_sparse_doc_jsonl is None:
        return {"skipped": True, "reason": "no learned sparse JSONL files supplied"}

    started = time.perf_counter()
    doc_rows = parse_sparse_jsonl(args.learned_sparse_doc_jsonl, "doc_id")
    query_rows_sparse = parse_sparse_jsonl(args.learned_sparse_query_jsonl, "query_id")
    doc_updates = 0
    query_updates = 0
    with conn.cursor() as cur:
        for doc_id, term_ids, weights in doc_rows:
            cur.execute(
                """
                UPDATE dbpedia_colbert_docs
                SET learned_sparse = turbohybrid_sparse_vector_from_arrays(%s::int[], %s::real[])
                WHERE doc_id = %s
                """,
                (term_ids, weights, doc_id),
            )
            doc_updates += cur.rowcount
        for query_id, term_ids, weights in query_rows_sparse:
            cur.execute(
                """
                UPDATE dbpedia_colbert_queries
                SET learned_sparse = turbohybrid_sparse_vector_from_arrays(%s::int[], %s::real[])
                WHERE query_id = %s
                """,
                (term_ids, weights, query_id),
            )
            query_updates += cur.rowcount

    if doc_rows and doc_updates == 0:
        raise RuntimeError(f"no learned sparse document rows matched loaded docs from {args.learned_sparse_doc_jsonl}")
    if query_rows_sparse and query_updates == 0:
        raise RuntimeError(f"no learned sparse query rows matched loaded queries from {args.learned_sparse_query_jsonl}")

    return {
        "skipped": False,
        "doc_jsonl": portable_path(args.learned_sparse_doc_jsonl),
        "query_jsonl": portable_path(args.learned_sparse_query_jsonl),
        "doc_rows_read": len(doc_rows),
        "doc_rows_updated": doc_updates,
        "query_rows_read": len(query_rows_sparse),
        "query_rows_updated": query_updates,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


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
        "turbohybrid.multivector_model_name": args.colbert_model_name,
        "turbohybrid.multivector_subvector_k": str(args.multivector_subvector_k),
        "turbohybrid.multivector_unique_docs_per_token": str(args.multivector_unique_docs_per_token),
        "turbohybrid.multivector_max_raw_hits_per_token": str(args.multivector_max_raw_hits_per_token),
        "turbohybrid.multivector_adaptive_widening": args.multivector_adaptive_widening,
        "turbohybrid.multivector_doc_candidate_k": str(args.multivector_doc_candidate_k),
        "turbohybrid.multivector_exact_rerank": args.multivector_exact_rerank,
        "turbohybrid.multivector_exact_rerank_k": str(args.multivector_exact_rerank_k),
        "turbohybrid.multivector_doc_graph_search_ef": str(args.multivector_doc_graph_search_ef),
        "turbohybrid.multivector_doc_graph_oversampling": str(args.multivector_doc_graph_oversampling),
        "turbohybrid.multivector_doc_graph_rescore_k": str(args.multivector_doc_graph_rescore_k),
        "turbohybrid.multivector_doc_storage": args.multivector_doc_storage,
        "turbohybrid.multivector_doc_storage_cache": args.multivector_doc_storage_cache,
        "turbohybrid.multivector_proxy_encoder": args.multivector_proxy_encoder,
        "turbohybrid.multivector_candidate_source": args.multivector_candidate_source,
        "turbohybrid.multivector_plain_fallback": args.multivector_plain_fallback,
        "turbohybrid.multivector_plain_fallback_max_docs": str(args.multivector_plain_fallback_max_docs),
        "turbohybrid.multivector_plain_fallback_candidate_fraction": str(args.multivector_plain_fallback_candidate_fraction),
        "turbohybrid.multivector_candidate_reservoirs": args.multivector_candidate_reservoirs,
        "turbohybrid.multivector_per_token_doc_reservoir_k": str(args.multivector_per_token_doc_reservoir_k),
        "turbohybrid.multivector_coverage_reservoir_k": str(args.multivector_coverage_reservoir_k),
        "turbohybrid.multivector_bm25_candidate_injection": args.multivector_bm25_candidate_injection,
        "turbohybrid.multivector_sparse_candidate_source": args.multivector_sparse_candidate_source,
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
        "turbohybrid.multivector_model_name": args.colbert_model_name,
        "turbohybrid.multivector_subvector_k": str(args.multivector_subvector_k),
        "turbohybrid.multivector_unique_docs_per_token": str(args.multivector_unique_docs_per_token),
        "turbohybrid.multivector_max_raw_hits_per_token": str(args.multivector_max_raw_hits_per_token),
        "turbohybrid.multivector_adaptive_widening": args.multivector_adaptive_widening,
        "turbohybrid.multivector_doc_candidate_k": str(args.multivector_doc_candidate_k),
        "turbohybrid.multivector_exact_rerank": args.multivector_exact_rerank,
        "turbohybrid.multivector_exact_rerank_k": str(args.multivector_exact_rerank_k),
        "turbohybrid.multivector_doc_graph_search_ef": str(args.multivector_doc_graph_search_ef),
        "turbohybrid.multivector_doc_graph_oversampling": str(args.multivector_doc_graph_oversampling),
        "turbohybrid.multivector_doc_graph_rescore_k": str(args.multivector_doc_graph_rescore_k),
        "turbohybrid.multivector_doc_storage": args.multivector_doc_storage,
        "turbohybrid.multivector_doc_storage_cache": args.multivector_doc_storage_cache,
        "turbohybrid.multivector_proxy_encoder": args.multivector_proxy_encoder,
        "turbohybrid.multivector_candidate_source": args.multivector_candidate_source,
        "turbohybrid.multivector_plain_fallback": args.multivector_plain_fallback,
        "turbohybrid.multivector_plain_fallback_max_docs": str(args.multivector_plain_fallback_max_docs),
        "turbohybrid.multivector_plain_fallback_candidate_fraction": str(args.multivector_plain_fallback_candidate_fraction),
        "turbohybrid.multivector_candidate_reservoirs": args.multivector_candidate_reservoirs,
        "turbohybrid.multivector_per_token_doc_reservoir_k": str(args.multivector_per_token_doc_reservoir_k),
        "turbohybrid.multivector_coverage_reservoir_k": str(args.multivector_coverage_reservoir_k),
        "turbohybrid.multivector_bm25_candidate_injection": args.multivector_bm25_candidate_injection,
        "turbohybrid.multivector_sparse_candidate_source": args.multivector_sparse_candidate_source,
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
            learned_sparse turbohybrid_sparse_vector,
            learned_sparse_tsv tsvector GENERATED ALWAYS AS (
                CASE
                  WHEN learned_sparse IS NULL THEN NULL
                  ELSE turbohybrid_sparse_vector_to_tsvector(learned_sparse)
                END
            ) STORED,
            colbert turbohybrid_multivector
        )
        """,
    )
    exec_sql(conn, "ALTER TABLE dbpedia_colbert_docs ADD COLUMN IF NOT EXISTS learned_sparse turbohybrid_sparse_vector")
    exec_sql(
        conn,
        """
        ALTER TABLE dbpedia_colbert_docs
        ADD COLUMN IF NOT EXISTS learned_sparse_tsv tsvector GENERATED ALWAYS AS (
            CASE
              WHEN learned_sparse IS NULL THEN NULL
              ELSE turbohybrid_sparse_vector_to_tsvector(learned_sparse)
            END
        ) STORED
        """,
    )
    exec_sql(
        conn,
        """
        CREATE TABLE IF NOT EXISTS dbpedia_colbert_queries (
            query_id text PRIMARY KEY,
            query_text text NOT NULL,
            learned_sparse turbohybrid_sparse_vector,
            colbert turbohybrid_multivector
        )
        """,
    )
    exec_sql(conn, "ALTER TABLE dbpedia_colbert_queries ADD COLUMN IF NOT EXISTS learned_sparse turbohybrid_sparse_vector")
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


def resolve_expected_dim(value: str, model_name: str) -> tuple[int, str]:
    requested = value.strip().lower()
    if requested == "auto":
        dim = COLBERT_MODEL_DIMENSIONS.get(model_name.lower())
        if dim is None:
            raise SystemExit(
                f"--expected-dim auto does not know model {model_name!r}; "
                "pass --expected-dim explicitly or add the model to the registry."
            )
        return dim, "model_registry"

    try:
        dim = int(value)
    except ValueError as exc:
        raise SystemExit("--expected-dim must be a positive integer or 'auto'") from exc
    if dim <= 0:
        raise SystemExit("--expected-dim must be positive")
    return dim, "explicit"


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
        prioritize_qrels=args.prioritize_qrels,
    )


def build_index(conn: psycopg.Connection[Any], args: argparse.Namespace) -> dict[str, Any]:
    reloptions, index_option_values = build_index_reloptions(args)
    if args.reuse_index:
        row = fetch_one(conn, "SELECT to_regclass('dbpedia_colbert_docs_colbert_idx') IS NOT NULL")
        if row and bool(row[0]):
            stats = fetch_one(conn, "SELECT turbohybrid_index_stats('dbpedia_colbert_docs_colbert_idx'::regclass)")
            build_stats = fetch_one(conn, "SELECT turbohybrid_last_build_stats()")
            size = fetch_one(conn, "SELECT pg_relation_size('dbpedia_colbert_docs_colbert_idx'::regclass)")
            index_stats = jsonb_value(stats[0]) if stats else {}
            build_stats_value = jsonb_value(build_stats[0]) if build_stats else {}
            row_count = loaded_document_count(conn)
            return {
                "reused": True,
                "reloptions": reloptions,
                **index_option_values,
                "index_bytes": int(size[0]) if size else 0,
                "index_stats": index_stats,
                "build_stats": build_stats_value,
                "safety_checks": validate_document_node_proxy_build(
                    args=args,
                    index_stats=index_stats,
                    build_stats={},
                    row_count=row_count,
                ),
            }

    exec_sql(conn, "DROP INDEX IF EXISTS dbpedia_colbert_docs_colbert_idx")
    started = time.perf_counter()
    needs_lexical_index = (
        RRF_METHOD in set(getattr(args, "methods", []))
        or bool(getattr(args, "hybrid_evaluation_harness", False))
        or getattr(args, "multivector_bm25_candidate_injection", "off") != "off"
        or getattr(args, "multivector_sparse_candidate_source", "off") != "off"
        or getattr(args, "learned_sparse_doc_jsonl", None) is not None
    )
    lexical_column = None
    if needs_lexical_index:
        lexical_column = "learned_sparse_tsv" if has_learned_sparse_docs(conn) else "body_tsv"
    reloptions_sql = ",\n              ".join(reloptions)
    if lexical_column is None:
        exec_sql(
            conn,
            f"""
            CREATE INDEX dbpedia_colbert_docs_colbert_idx
            ON dbpedia_colbert_docs USING turbohybrid (
              colbert multivector_maxsim_ip_turbohybrid_ops
            )
            WITH ({reloptions_sql})
            """,
        )
    else:
        exec_sql(
            conn,
            f"""
            CREATE INDEX dbpedia_colbert_docs_colbert_idx
            ON dbpedia_colbert_docs USING turbohybrid (
              colbert multivector_maxsim_ip_turbohybrid_ops,
              {lexical_column} bm25_tsvector_turbohybrid_ops
            )
            WITH ({reloptions_sql})
            """,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    build_stats = fetch_one(conn, "SELECT turbohybrid_last_build_stats()")
    stats = fetch_one(conn, "SELECT turbohybrid_index_stats('dbpedia_colbert_docs_colbert_idx'::regclass)")
    size = fetch_one(conn, "SELECT pg_relation_size('dbpedia_colbert_docs_colbert_idx'::regclass)")
    index_stats = jsonb_value(stats[0]) if stats else {}
    build_stats_value = jsonb_value(build_stats[0]) if build_stats else {}
    row_count = loaded_document_count(conn)
    return {
        "reused": False,
        "elapsed_ms": round(elapsed_ms, 3),
        "lexical_indexed": lexical_column is not None,
        "lexical_column": lexical_column,
        "reloptions": reloptions,
        **index_option_values,
        "index_bytes": int(size[0]) if size else 0,
        "index_stats": index_stats,
        "build_stats": build_stats_value,
        "safety_checks": validate_document_node_proxy_build(
            args=args,
            index_stats=index_stats,
            build_stats=build_stats_value,
            row_count=row_count,
        ),
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


def can_infer_admission_from_result_docs(
    candidate_source: Any,
    *,
    fallback_used: bool = False,
    plain_fallback: Any = None,
    graph_mode: Any = None,
) -> bool:
    """Return whether final result membership proves exact-rerank admission.

    Some document-level paths exact-rerank candidate documents but do not emit a
    bounded per-document admission trace. In those cases, a returned exact-top
    document proves it was admitted and retained for exact rerank, but not its
    pre-rerank candidate rank.
    """
    if fallback_used or plain_fallback == "force":
        return True
    if graph_mode == "document_nodes":
        return True
    return candidate_source in {
        "exact_doc_scan",
        "doc_graph_prototype",
        "plain_fallback",
        "document_nodes",
        "proxy_vector",
        "quantized_inverted_experimental",
    }


def scan_stat_int(stats: dict[str, Any], key: str) -> int:
    value = stats.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def scan_stat_bool(stats: dict[str, Any], key: str) -> bool:
    return stats.get(key) is True or stats.get(key) == "true"


def scan_stat_str(stats: dict[str, Any], key: str) -> str | None:
    value = stats.get(key)
    return value if isinstance(value, str) else None


def memory_estimate_from_stats(stats: dict[str, Any]) -> int:
    return scan_stat_int(stats, "multivector_memory_bytes_estimate")


def scan_docs_scored(stats: dict[str, Any]) -> int:
    return max(
        scan_stat_int(stats, "multivector_doc_graph_docs_scored"),
        scan_stat_int(stats, "multivector_plain_fallback_docs_scored"),
        scan_stat_int(stats, "multivector_doc_candidates"),
        scan_stat_int(stats, "multivector_unique_docs"),
    )


def scan_graph_edges_visited(stats: dict[str, Any]) -> int:
    return scan_stat_int(stats, "multivector_doc_graph_edges_visited")


def scan_exact_rerank_docs(stats: dict[str, Any]) -> int:
    return max(
        scan_stat_int(stats, "multivector_exact_rerank_docs"),
        scan_stat_int(stats, "multivector_doc_graph_exact_rerank_docs"),
    )


def exact_rerank_work_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact_rerank_candidates": scan_stat_int(stats, "exact_rerank_candidates"),
        "exact_rerank_tokens_evaluated": scan_stat_int(stats, "exact_rerank_tokens_evaluated"),
        "exact_rerank_tokens_skipped": scan_stat_int(stats, "exact_rerank_tokens_skipped"),
        "exact_rerank_pairs_saved": scan_stat_int(stats, "exact_rerank_pairs_saved"),
        "adaptive_rerank_topk_changed_vs_full": bool(
            stats.get("adaptive_rerank_topk_changed_vs_full", False)
        ),
    }


def proxy_work_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "proxy_encoder_kind": scan_stat_str(stats, "proxy_encoder_kind"),
        "proxy_candidates": scan_stat_int(stats, "proxy_candidates"),
        "proxy_top1_admission": scan_stat_bool(stats, "proxy_top1_admission"),
        "proxy_exact_rerank_docs": scan_stat_int(stats, "proxy_exact_rerank_docs"),
    }


def centroid_work_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "centroid_lists_visited": scan_stat_int(stats, "centroid_lists_visited"),
        "centroid_docs_touched": scan_stat_int(stats, "centroid_docs_touched"),
        "centroid_pruned_docs": scan_stat_int(stats, "centroid_pruned_docs"),
        "centroid_candidates": scan_stat_int(stats, "centroid_candidates"),
    }


def quantized_inverted_work_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "quantized_inverted_lists_visited": scan_stat_int(
            stats, "quantized_inverted_lists_visited"
        ),
        "quantized_inverted_postings_touched": scan_stat_int(
            stats, "quantized_inverted_postings_touched"
        ),
        "quantized_inverted_docs_scored": scan_stat_int(
            stats, "quantized_inverted_docs_scored"
        ),
        "quantized_inverted_candidates": scan_stat_int(
            stats, "quantized_inverted_candidates"
        ),
        "quantized_inverted_exact_rerank_docs": scan_stat_int(
            stats, "quantized_inverted_exact_rerank_docs"
        ),
        "quantized_inverted_codebook_size": scan_stat_int(
            stats, "quantized_inverted_codebook_size"
        ),
        "quantized_inverted_list_offset_bytes": scan_stat_int(
            stats, "quantized_inverted_list_offset_bytes"
        ),
        "quantized_inverted_posting_bytes": scan_stat_int(
            stats, "quantized_inverted_posting_bytes"
        ),
        "quantized_inverted_sidecar_bytes": scan_stat_int(
            stats, "quantized_inverted_sidecar_bytes"
        ),
    }


def learned_sparse_work_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "learned_sparse_candidates": scan_stat_int(stats, "learned_sparse_candidates"),
        "learned_sparse_retained_for_maxsim": scan_stat_int(
            stats, "learned_sparse_retained_for_maxsim"
        ),
        "learned_sparse_branch_latency_us": scan_stat_int(
            stats, "learned_sparse_branch_latency_us"
        ),
    }


def is_experimental_quantized_stat_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered.startswith("quantized_")
        or lowered.startswith("quantized_inverted_")
        or "codeword" in lowered
        or "posting" in lowered
    )


def extract_document_node_serving_stats(stats: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(stats, dict):
        return {}
    fields = {
        key
        for group_fields in SERVING_STATS_FIELD_GROUPS.values()
        for key in group_fields
    }
    extracted = {key: stats[key] for key in fields if key in stats}
    for key, value in stats.items():
        if is_experimental_quantized_stat_key(str(key)):
            extracted[str(key)] = value
    return dict(sorted(extracted.items()))


def document_node_serving_stats_available(extracted: dict[str, Any]) -> dict[str, bool]:
    if not isinstance(extracted, dict):
        extracted = {}
    available = {
        group: any(key in extracted for key in fields)
        for group, fields in SERVING_STATS_FIELD_GROUPS.items()
    }
    available["experimental_quantized"] = any(
        is_experimental_quantized_stat_key(str(key))
        for key in extracted
    )
    return available


def merge_document_node_serving_stats_available(
    samples_by_budget: dict[str, dict[str, Any]]
) -> dict[str, bool]:
    merged = {
        group: False
        for group in (*SERVING_STATS_FIELD_GROUPS.keys(), "experimental_quantized")
    }
    for sample in samples_by_budget.values():
        if not isinstance(sample, dict):
            continue
        available = sample.get("stats_available", {})
        if not isinstance(available, dict):
            continue
        for group, value in available.items():
            merged[str(group)] = bool(merged.get(str(group), False) or value)
    return dict(sorted(merged.items()))


def scan_stat_int_list(stats: dict[str, Any], key: str) -> list[int]:
    value = stats.get(key)
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def clone_args(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def sidecar_stats_from_scan(stats: dict[str, Any]) -> dict[str, Any]:
    dedicated_keys = (
        "multivector_doc_sidecar_cache_mode",
        "multivector_doc_sidecar_pages_read",
        "multivector_doc_sidecar_cache_hits",
        "multivector_doc_sidecar_cache_misses",
        "multivector_doc_sidecar_bytes_touched",
        "multivector_doc_sidecar_vectors_loaded",
    )
    return {
        "sidecar_stats_available": any(key in stats for key in dedicated_keys),
        "sidecar_cache_mode": str(stats.get("multivector_doc_sidecar_cache_mode", "none")),
        "sidecar_pages_read": scan_stat_int(stats, "multivector_doc_sidecar_pages_read"),
        "sidecar_bytes_touched": scan_stat_int(stats, "multivector_doc_sidecar_bytes_touched"),
        "sidecar_cache_hits": scan_stat_int(stats, "multivector_doc_sidecar_cache_hits"),
        "sidecar_cache_misses": scan_stat_int(stats, "multivector_doc_sidecar_cache_misses"),
        "sidecar_vectors_loaded": scan_stat_int(stats, "multivector_doc_sidecar_vectors_loaded"),
        "native_cache_used": scan_stat_bool(stats, "native_cache_used"),
        "native_cache_reused": scan_stat_bool(stats, "native_cache_reused"),
        "native_cache_built_this_scan": scan_stat_bool(stats, "native_cache_built_this_scan"),
        "native_cache_bytes": scan_stat_int(stats, "native_cache_bytes"),
        "native_cache_exact_bytes": scan_stat_int(stats, "native_cache_exact_bytes"),
        "native_cache_code_bytes": scan_stat_int(stats, "native_cache_code_bytes"),
        "native_cache_adj_bytes": scan_stat_int(stats, "native_cache_adj_bytes"),
        "native_cache_mode": scan_stat_str(stats, "native_cache_mode"),
        "native_cache_scope": scan_stat_str(stats, "native_cache_scope"),
        "memory_bytes_estimate": memory_estimate_from_stats(stats),
    }


def parse_int_grid(value: str, option_name: str) -> list[int]:
    try:
        items = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{option_name} must contain comma-separated integers") from exc
    items = sorted({item for item in items if item > 0})
    if not items:
        raise SystemExit(f"{option_name} must contain at least one positive integer")
    return items


def parse_choice_grid(value: str, choices: tuple[str, ...], option_name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise SystemExit(f"{option_name} must contain at least one value")
    unknown = sorted(set(items) - set(choices))
    if unknown:
        raise SystemExit(f"{option_name} contains unsupported value(s): {', '.join(unknown)}")
    return list(dict.fromkeys(items))


def parse_document_node_pooling_grid(value: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for raw in value.split(","):
        entry = raw.strip()
        if not entry:
            continue
        if ":" in entry:
            mode, ratio_text = entry.split(":", 1)
        else:
            mode, ratio_text = ("off" if entry == "1.0" else "greedy_cosine", entry)
        mode = mode.strip()
        ratio_text = ratio_text.strip()
        if mode not in DOCUMENT_NODE_TOKEN_POOLING_CHOICES:
            raise SystemExit(f"--document-node-pooling-grid contains unsupported mode: {mode}")
        try:
            ratio = float(ratio_text)
        except ValueError as exc:
            raise SystemExit("--document-node-pooling-grid ratios must be numeric") from exc
        if ratio <= 0.0 or ratio > 1.0:
            raise SystemExit("--document-node-pooling-grid ratios must be in (0, 1]")
        if mode == "off" and ratio != 1.0:
            raise SystemExit("--document-node-pooling-grid uses off only with ratio 1.0")
        key = (mode, ratio)
        if key in seen:
            continue
        seen.add(key)
        items.append({"mode": mode, "ratio": ratio})
    if not items:
        raise SystemExit("--document-node-pooling-grid must contain at least one entry")
    return items


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
        and can_infer_admission_from_result_docs(candidate_source, fallback_used=fallback_used)
    )

    admitted_ranks: list[int] = []
    exact_top_with_admission: list[dict[str, Any]] = []
    for exact in exact_top:
        key = (int(exact["heap_block"]), int(exact["heap_offset"]))
        entry = trace_by_heap.get(key)
        result_rank = docs.index(exact["doc_id"]) + 1 if exact["doc_id"] in docs else None
        inferred_admitted = infer_admission_from_results and result_rank is not None
        admitted = entry is not None or inferred_admitted
        admission_evidence = "trace" if entry is not None else "result_doc" if inferred_admitted else "missing"
        if admitted:
            admitted_ranks.append(int(exact["rank"]))
        exact_top_with_admission.append({
            "rank": exact["rank"],
            "doc_id": exact["doc_id"],
            "admitted_before_rerank": admitted,
            "admission_evidence": admission_evidence,
            "candidate_rank_before_rerank": (
                int(entry["candidate_rank_before_truncation"])
                if isinstance(entry, dict) and entry.get("candidate_rank_before_truncation") is not None
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
    top1_admission = bool(exact_top_with_admission and exact_top_with_admission[0]["admitted_before_rerank"])
    latency_ms_rounded = round(latency_ms, 3)
    docs_scored = scan_docs_scored(stats)
    graph_edges_visited = scan_graph_edges_visited(stats)
    exact_rerank_docs = scan_exact_rerank_docs(stats)
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
        **learned_sparse_work_from_stats(stats),
        "latency_ms": latency_ms_rounded,
        "latency": {"ms": latency_ms_rounded},
        "result_doc_ids": docs,
        "exact_top1_admission": top1_admission,
        "exact_top1_admitted": top1_admission,
        "exact_top1_admitted_before_rerank": top1_admission,
        "exact_top10_admission_recall": round(top10_admitted / len(top10), 6) if top10 else 0.0,
        "exact_top1_candidate_rank_before_rerank": (
            exact_top_with_admission[0]["candidate_rank_before_rerank"] if exact_top_with_admission else None
        ),
        "exact_top1_rank": (
            exact_top_with_admission[0]["exact_rerank_rank"] if exact_top_with_admission else None
        ),
        "exact_top1_exact_rerank_rank": (
            exact_top_with_admission[0]["exact_rerank_rank"] if exact_top_with_admission else None
        ),
        "raw_subvector_hits": scan_stat_int(stats, "multivector_raw_subvector_hits"),
        "unique_docs": scan_stat_int(stats, "multivector_unique_docs"),
        "maxsim_updates": scan_stat_int(stats, "multivector_maxsim_updates"),
        "doc_candidates": scan_stat_int(stats, "multivector_doc_candidates"),
        "docs_scored": docs_scored,
        "graph_edges_visited": graph_edges_visited,
        "exact_rerank_docs": exact_rerank_docs,
        **exact_rerank_work_from_stats(stats),
        **proxy_work_from_stats(stats),
        **centroid_work_from_stats(stats),
        **quantized_inverted_work_from_stats(stats),
        "sidecar_stats": sidecar_stats_from_scan(stats),
        "doc_graph_search_ef": scan_stat_int(stats, "multivector_doc_graph_search_ef"),
        "doc_graph_oversampling": scan_stat_int(stats, "multivector_doc_graph_oversampling"),
        "doc_graph_rescore_k": scan_stat_int(stats, "multivector_doc_graph_rescore_k"),
        "doc_graph_storage_kind": scan_stat_str(stats, "multivector_doc_graph_storage_kind"),
        "doc_graph_rescore_source": scan_stat_str(stats, "multivector_doc_graph_rescore_source"),
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
    top1_by_budget: dict[int, list[bool]] = {budget: [] for budget in budgets}
    top10_recall_by_budget: dict[int, list[float]] = {budget: [] for budget in budgets}
    docs_scored_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    graph_edges_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    exact_rerank_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    exact_rerank_tokens_evaluated_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    exact_rerank_tokens_skipped_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    exact_rerank_pairs_saved_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    proxy_candidates_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    proxy_exact_rerank_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    centroid_lists_visited_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    centroid_docs_touched_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    centroid_pruned_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    centroid_candidates_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    learned_sparse_candidates_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    learned_sparse_retained_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    learned_sparse_latency_us_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    quantized_inverted_candidates_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    quantized_inverted_postings_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    quantized_inverted_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    exact_top1_rank_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    native_cache_bytes_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    native_cache_exact_bytes_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    sidecar_pages_read_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    sidecar_bytes_touched_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    sidecar_cache_hits_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    sidecar_vectors_loaded_by_budget: dict[int, list[int]] = {budget: [] for budget in budgets}
    top1_first_budget_values: list[int] = []
    top1_admitted_queries = 0
    top10_recall_values: list[float] = []
    run_at_largest_budget: dict[str, list[str]] = {}

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
            top1_by_budget[budget].append(bool(result["exact_top1_admission"]))
            top10_recall_by_budget[budget].append(float(result["exact_top10_admission_recall"]))
            docs_scored_by_budget[budget].append(int(result["docs_scored"]))
            graph_edges_by_budget[budget].append(int(result["graph_edges_visited"]))
            exact_rerank_docs_by_budget[budget].append(int(result["exact_rerank_docs"]))
            exact_rerank_tokens_evaluated_by_budget[budget].append(
                int(result.get("exact_rerank_tokens_evaluated", 0))
            )
            exact_rerank_tokens_skipped_by_budget[budget].append(
                int(result.get("exact_rerank_tokens_skipped", 0))
            )
            exact_rerank_pairs_saved_by_budget[budget].append(
                int(result.get("exact_rerank_pairs_saved", 0))
            )
            proxy_candidates_by_budget[budget].append(int(result.get("proxy_candidates", 0)))
            proxy_exact_rerank_docs_by_budget[budget].append(
                int(result.get("proxy_exact_rerank_docs", 0))
            )
            centroid_lists_visited_by_budget[budget].append(
                int(result.get("centroid_lists_visited", 0))
            )
            centroid_docs_touched_by_budget[budget].append(
                int(result.get("centroid_docs_touched", 0))
            )
            centroid_pruned_docs_by_budget[budget].append(
                int(result.get("centroid_pruned_docs", 0))
            )
            centroid_candidates_by_budget[budget].append(
                int(result.get("centroid_candidates", 0))
            )
            learned_sparse_candidates_by_budget[budget].append(
                int(result.get("learned_sparse_candidates", 0))
            )
            learned_sparse_retained_by_budget[budget].append(
                int(result.get("learned_sparse_retained_for_maxsim", 0))
            )
            learned_sparse_latency_us_by_budget[budget].append(
                int(result.get("learned_sparse_branch_latency_us", 0))
            )
            quantized_inverted_candidates_by_budget[budget].append(
                int(result.get("quantized_inverted_candidates", 0))
            )
            quantized_inverted_postings_by_budget[budget].append(
                int(result.get("quantized_inverted_postings_touched", 0))
            )
            quantized_inverted_docs_by_budget[budget].append(
                int(result.get("quantized_inverted_docs_scored", 0))
            )
            if result["exact_top1_exact_rerank_rank"] is not None:
                exact_top1_rank_by_budget[budget].append(int(result["exact_top1_exact_rerank_rank"]))
            sidecar_stats = result.get("sidecar_stats", {})
            if isinstance(sidecar_stats, dict):
                native_cache_bytes_by_budget[budget].append(int(sidecar_stats.get("native_cache_bytes", 0)))
                native_cache_exact_bytes_by_budget[budget].append(int(sidecar_stats.get("native_cache_exact_bytes", 0)))
                sidecar_pages_read_by_budget[budget].append(int(sidecar_stats.get("sidecar_pages_read", 0)))
                sidecar_bytes_touched_by_budget[budget].append(int(sidecar_stats.get("sidecar_bytes_touched", 0)))
                sidecar_cache_hits_by_budget[budget].append(int(sidecar_stats.get("sidecar_cache_hits", 0)))
                sidecar_vectors_loaded_by_budget[budget].append(int(sidecar_stats.get("sidecar_vectors_loaded", 0)))
            if result["exact_top1_admitted_before_rerank"] and first_budget is None:
                first_budget = budget
                top1_candidate_rank = result["exact_top1_candidate_rank_before_rerank"]
                top1_exact_rank = result["exact_top1_exact_rerank_rank"]
            if budget == budgets[-1]:
                run_at_largest_budget[query.query_id] = list(result["result_doc_ids"])
        if first_budget is not None:
            top1_first_budget_values.append(first_budget)
            top1_admitted_queries += 1
        if budget_results:
            top10_recall_values.append(float(budget_results[-1]["exact_top10_admission_recall"]))

        per_query.append({
            "query_id": query.query_id,
            "query_text": query.query_text,
            "exact_top": exact_top,
            "exact_top1_admitted": first_budget is not None,
            "exact_top1_admitted_before_rerank": first_budget is not None,
            "exact_top10_admission_recall": budget_results[-1]["exact_top10_admission_recall"] if budget_results else 0.0,
            "exact_top1_first_budget_admitted": first_budget,
            "first_budget_admitting_exact_top1": first_budget,
            "exact_top1_rank": top1_exact_rank,
            "exact_top1_candidate_rank_before_rerank": top1_candidate_rank,
            "exact_top1_exact_rerank_rank": top1_exact_rank,
            "budgets": budget_results,
        })

    aggregate = {
        "queries": len(per_query),
        "admission_k": args.admission_k,
        "budget_sweep": budgets,
        "exact_top1_admission": round(top1_admitted_queries / len(per_query), 6) if per_query else 0.0,
        "exact_top1_admitted": round(top1_admitted_queries / len(per_query), 6) if per_query else 0.0,
        "exact_top1_admission_rate": round(top1_admitted_queries / len(per_query), 6) if per_query else 0.0,
        "exact_top10_admission_recall": round(statistics.mean(top10_recall_values), 6) if top10_recall_values else 0.0,
        "exact_top1_first_budget_admitted": summarize_ints(top1_first_budget_values),
        "first_budget_admitting_exact_top1": summarize_ints(top1_first_budget_values),
        "admission_by_budget": {
            str(budget): {
                "exact_top1_admission": round(
                    sum(1 for value in top1_by_budget[budget] if value) / len(top1_by_budget[budget]),
                    6,
                ) if top1_by_budget[budget] else 0.0,
                "exact_top1_admitted": round(
                    sum(1 for value in top1_by_budget[budget] if value) / len(top1_by_budget[budget]),
                    6,
                ) if top1_by_budget[budget] else 0.0,
                "exact_top10_admission_recall": round(
                    statistics.mean(top10_recall_by_budget[budget]),
                    6,
                ) if top10_recall_by_budget[budget] else 0.0,
                "latency": summarize_ms(latency_by_budget[budget]),
                "docs_scored": summarize_ints(docs_scored_by_budget[budget]),
                "graph_edges_visited": summarize_ints(graph_edges_by_budget[budget]),
                "exact_rerank_docs": summarize_ints(exact_rerank_docs_by_budget[budget]),
                "exact_rerank_tokens_evaluated": summarize_ints(
                    exact_rerank_tokens_evaluated_by_budget[budget]
                ),
                "exact_rerank_tokens_skipped": summarize_ints(
                    exact_rerank_tokens_skipped_by_budget[budget]
                ),
                "exact_rerank_pairs_saved": summarize_ints(
                    exact_rerank_pairs_saved_by_budget[budget]
                ),
                "proxy_candidates": summarize_ints(proxy_candidates_by_budget[budget]),
                "proxy_exact_rerank_docs": summarize_ints(
                    proxy_exact_rerank_docs_by_budget[budget]
                ),
                "centroid_lists_visited": summarize_ints(
                    centroid_lists_visited_by_budget[budget]
                ),
                "centroid_docs_touched": summarize_ints(
                    centroid_docs_touched_by_budget[budget]
                ),
                "centroid_pruned_docs": summarize_ints(
                    centroid_pruned_docs_by_budget[budget]
                ),
                "centroid_candidates": summarize_ints(
                    centroid_candidates_by_budget[budget]
                ),
                "learned_sparse_candidates": summarize_ints(
                    learned_sparse_candidates_by_budget[budget]
                ),
                "learned_sparse_retained_for_maxsim": summarize_ints(
                    learned_sparse_retained_by_budget[budget]
                ),
                "learned_sparse_branch_latency_us": summarize_ints(
                    learned_sparse_latency_us_by_budget[budget]
                ),
                "quantized_inverted_candidates": summarize_ints(
                    quantized_inverted_candidates_by_budget[budget]
                ),
                "quantized_inverted_postings_touched": summarize_ints(
                    quantized_inverted_postings_by_budget[budget]
                ),
                "quantized_inverted_docs_scored": summarize_ints(
                    quantized_inverted_docs_by_budget[budget]
                ),
                "exact_top1_rank": summarize_ints(exact_top1_rank_by_budget[budget]),
                "native_cache_bytes": summarize_ints(native_cache_bytes_by_budget[budget]),
                "native_cache_exact_bytes": summarize_ints(native_cache_exact_bytes_by_budget[budget]),
                "sidecar_pages_read": summarize_ints(sidecar_pages_read_by_budget[budget]),
                "sidecar_bytes_touched": summarize_ints(sidecar_bytes_touched_by_budget[budget]),
                "sidecar_cache_hits": summarize_ints(sidecar_cache_hits_by_budget[budget]),
                "sidecar_vectors_loaded": summarize_ints(sidecar_vectors_loaded_by_budget[budget]),
            }
            for budget in budgets
        },
        "latency_by_budget": {
            str(budget): summarize_ms(values)
            for budget, values in latency_by_budget.items()
        },
        "run_at_largest_budget": run_at_largest_budget,
    }

    set_retrieval_gucs(conn, args, "dbpedia_colbert_serial")
    return {
        "enabled": True,
        "candidate_source": args.multivector_candidate_source,
        "graph_mode": args.multivector_graph,
        "storage_kind": args.multivector_doc_storage,
        "token_pooling": args.multivector_token_pooling,
        "token_pooling_target_ratio": args.multivector_token_pooling_target_ratio,
        "token_pooling_min_tokens": args.multivector_token_pooling_min_tokens,
        "centroids": args.multivector_centroids,
        "centroid_count": args.multivector_centroid_count,
        "doc_graph_search_ef": args.multivector_doc_graph_search_ef,
        "doc_graph_oversampling": args.multivector_doc_graph_oversampling,
        "doc_graph_rescore_k": args.multivector_doc_graph_rescore_k,
        "plain_fallback": args.multivector_plain_fallback,
        "candidate_reservoirs": args.multivector_candidate_reservoirs,
        "bm25_candidate_injection": args.multivector_bm25_candidate_injection,
        "sparse_candidate_source": args.multivector_sparse_candidate_source,
        "retrieval_method": QUERY_ONLY_METHOD,
        "aggregate": aggregate,
        "per_query": per_query,
    }


def grid_mode_args(args: argparse.Namespace, mode: str, graph_mode: str) -> argparse.Namespace:
    if mode == "token_nodes":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="graph",
            multivector_plain_fallback="off",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="auto",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="off",
            multivector_centroid_count="auto",
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    if mode == "exact_token_scan":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="exact_token_scan",
            multivector_plain_fallback="off",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="auto",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="off",
            multivector_centroid_count="auto",
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    if mode == "plain_fallback":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="graph",
            multivector_plain_fallback="force",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="auto",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="off",
            multivector_centroid_count="auto",
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    if mode == "exact_doc_scan":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="exact_doc_scan",
            multivector_plain_fallback="off",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="paged",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="off",
            multivector_centroid_count="auto",
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    if mode == "proxy_vector":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="proxy_vector",
            multivector_plain_fallback="off",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="auto",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="off",
            multivector_centroid_count="auto",
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    if mode == "centroid_lite":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="centroid_lite",
            multivector_plain_fallback="off",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="auto",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="kmeans",
            multivector_centroid_count=args.multivector_centroid_count,
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    if mode == "quantized_inverted_experimental":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="quantized_inverted_experimental",
            multivector_plain_fallback="off",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="auto",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="off",
            multivector_centroid_count="auto",
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    raise ValueError(f"unsupported grid mode: {mode}")


def grid_admission_summary(
    *,
    mode: str,
    args: argparse.Namespace,
    index_phase: dict[str, Any],
    admission: dict[str, Any],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    aggregate = admission.get("aggregate", {})
    run = aggregate.get("run_at_largest_budget", {})
    if not isinstance(run, dict):
        run = {}
    metrics = method_metrics(run, qrels, args.final_k, args.quality_k) if qrels else {}
    budget_sweep = aggregate.get("budget_sweep", [])
    largest_budget = int(budget_sweep[-1]) if isinstance(budget_sweep, list) and budget_sweep else None
    return {
        "mode": mode,
        "candidate_source": args.multivector_candidate_source,
        "graph_mode": args.multivector_graph,
        "storage_kind": args.multivector_doc_storage,
        "storage_cache_mode": args.multivector_doc_storage_cache,
        "token_pooling": args.multivector_token_pooling,
        "token_pooling_target_ratio": args.multivector_token_pooling_target_ratio,
        "token_pooling_min_tokens": args.multivector_token_pooling_min_tokens,
        "doc_graph_search_ef": args.multivector_doc_graph_search_ef,
        "doc_graph_oversampling": args.multivector_doc_graph_oversampling,
        "doc_graph_rescore_k": args.multivector_doc_graph_rescore_k,
        "index_graph_m": index_phase.get("index_graph_m", 0),
        "index_graph_ef_construction": index_phase.get("index_graph_ef_construction", 0),
        "index_graph_ef_search": index_phase.get("index_graph_ef_search", 0),
        "plain_fallback": args.multivector_plain_fallback,
        "largest_budget": largest_budget,
        "exact_top1_admission_rate": aggregate.get("exact_top1_admission_rate", 0.0),
        "exact_top1_admitted": aggregate.get("exact_top1_admitted", aggregate.get("exact_top1_admission", 0.0)),
        "exact_top10_admission_recall": aggregate.get("exact_top10_admission_recall", 0.0),
        "metrics": metrics,
        "admission_by_budget": aggregate.get("admission_by_budget", {}),
        "latency_by_budget": aggregate.get("latency_by_budget", {}),
        "index_stats": index_phase.get("index_stats", {}),
        "index_bytes": index_phase.get("index_bytes", 0),
    }


def document_node_serving_profiles(
    *,
    include_experimental: bool,
    centroid_count: str,
) -> list[DocumentNodeServingProfile]:
    profiles = [
        DocumentNodeServingProfile(
            name="proxy_normalized_mean_f16",
            candidate_source="proxy_vector",
            proxy_encoder="normalized_mean",
            storage_kind="f16",
        ),
        DocumentNodeServingProfile(
            name="docnodes_normalized_mean_f16",
            candidate_source="document_nodes",
            proxy_encoder="normalized_mean",
            storage_kind="f16",
        ),
        DocumentNodeServingProfile(
            name="centroid_mean_f16",
            candidate_source="proxy_vector",
            proxy_encoder="centroid_mean",
            centroids="kmeans",
            centroid_count=centroid_count,
            storage_kind="f16",
        ),
        DocumentNodeServingProfile(
            name="centroid_lite_f16",
            candidate_source="centroid_lite",
            centroids="kmeans",
            centroid_count=centroid_count,
            storage_kind="f16",
        ),
        DocumentNodeServingProfile(
            name="centroid_lite_f16_pool_050",
            candidate_source="centroid_lite",
            centroids="kmeans",
            centroid_count=centroid_count,
            storage_kind="f16",
            token_pooling="greedy_cosine",
            token_pooling_target_ratio=0.5,
        ),
        DocumentNodeServingProfile(
            name="proxy_normalized_mean_sq8",
            candidate_source="proxy_vector",
            proxy_encoder="normalized_mean",
            storage_kind="sq8",
        ),
    ]
    if include_experimental:
        profiles.append(
            DocumentNodeServingProfile(
                name="quantized_inverted_experimental_f32",
                candidate_source="quantized_inverted_experimental",
                storage_kind="f32",
                plain_fallback="off",
            )
        )
    return profiles


def _self_check_document_node_serving_profiles() -> None:
    base = document_node_serving_profiles(
        include_experimental=False,
        centroid_count="auto",
    )
    assert [profile.name for profile in base] == [
        "proxy_normalized_mean_f16",
        "docnodes_normalized_mean_f16",
        "centroid_mean_f16",
        "centroid_lite_f16",
        "centroid_lite_f16_pool_050",
        "proxy_normalized_mean_sq8",
    ]
    assert all(profile.candidate_source != "quantized_inverted_experimental" for profile in base)
    profiles_by_name = {profile.name: profile for profile in base}
    assert profiles_by_name["centroid_mean_f16"].proxy_encoder == "centroid_mean"
    assert profiles_by_name["centroid_mean_f16"].centroids == "kmeans"
    assert profiles_by_name["centroid_lite_f16_pool_050"].token_pooling == "greedy_cosine"
    assert profiles_by_name["centroid_lite_f16_pool_050"].token_pooling_target_ratio == 0.5
    experimental = document_node_serving_profiles(
        include_experimental=True,
        centroid_count="auto",
    )
    assert experimental[-1].name == "quantized_inverted_experimental_f32"
    assert experimental[-1].candidate_source == "quantized_inverted_experimental"
    assert experimental[-1].storage_kind == "f32"
    assert experimental[-1].plain_fallback == "off"


def document_node_serving_profile_args(
    args: argparse.Namespace,
    profile: DocumentNodeServingProfile,
    *,
    ef: int,
    oversampling: int,
) -> argparse.Namespace:
    return clone_args(
        args,
        multivector_graph="document_nodes",
        multivector_doc_build_scorer="proxy",
        multivector_candidate_source=profile.candidate_source,
        multivector_proxy_encoder=profile.proxy_encoder,
        multivector_centroids=profile.centroids,
        multivector_centroid_count=profile.centroid_count,
        multivector_doc_storage=profile.storage_kind,
        multivector_doc_storage_cache=profile.cache_mode,
        multivector_token_pooling=profile.token_pooling,
        multivector_token_pooling_target_ratio=profile.token_pooling_target_ratio,
        multivector_plain_fallback=profile.plain_fallback,
        multivector_candidate_reservoirs="off",
        multivector_doc_graph_search_ef=ef,
        multivector_doc_graph_oversampling=oversampling,
        multivector_doc_graph_rescore_k=0,
        admission_budget_sweep=args.admission_budget_sweep,
        reuse_index=False,
    )


def serving_grid_scan_stats_by_budget(admission: dict[str, Any]) -> dict[str, dict[str, Any]]:
    per_query = admission.get("per_query", [])
    if not isinstance(per_query, list) or not per_query:
        return {}
    samples: dict[str, dict[str, Any]] = {}
    for query_result in per_query:
        if not isinstance(query_result, dict):
            continue
        budgets = query_result.get("budgets", [])
        if not isinstance(budgets, list):
            continue
        for item in budgets:
            if not isinstance(item, dict):
                continue
            budget = item.get("budget")
            if budget is None:
                continue
            budget_key = str(budget)
            if budget_key in samples:
                continue
            stats = item.get("scan_stats", {})
            if not isinstance(stats, dict):
                continue
            extracted = extract_document_node_serving_stats(stats)
            samples[budget_key] = {
                "scan_stats_sample": extracted,
                "stats_available": document_node_serving_stats_available(extracted),
            }
    return {
        key: samples[key]
        for key in sorted(
            samples,
            key=lambda value: (
                0,
                int(value),
            )
            if value.isdigit()
            else (1, value),
        )
    }


def serving_grid_scan_stats_sample(admission: dict[str, Any], largest_budget: int | None) -> dict[str, Any]:
    if largest_budget is None:
        return {}
    by_budget = serving_grid_scan_stats_by_budget(admission)
    selected = by_budget.get(str(largest_budget), {})
    if not isinstance(selected, dict):
        return {}
    sample = selected.get("scan_stats_sample", {})
    return sample if isinstance(sample, dict) else {}


def document_node_serving_summary_row(
    *,
    profile: DocumentNodeServingProfile,
    args: argparse.Namespace,
    ef: int,
    oversampling: int,
    index_phase: dict[str, Any],
    admission: dict[str, Any],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    aggregate = admission.get("aggregate", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    budget_sweep = aggregate.get("budget_sweep", [])
    largest_budget = int(budget_sweep[-1]) if isinstance(budget_sweep, list) and budget_sweep else None
    latency_by_budget = aggregate.get("latency_by_budget", {})
    largest_latency = (
        latency_by_budget.get(str(largest_budget), {})
        if isinstance(latency_by_budget, dict) and largest_budget is not None
        else {}
    )
    if not isinstance(largest_latency, dict):
        largest_latency = {}
    admission_by_budget = aggregate.get("admission_by_budget", {})
    largest_work = (
        admission_by_budget.get(str(largest_budget), {})
        if isinstance(admission_by_budget, dict) and largest_budget is not None
        else {}
    )
    if not isinstance(largest_work, dict):
        largest_work = {}
    run = aggregate.get("run_at_largest_budget", {})
    if not isinstance(run, dict):
        run = {}
    metrics = method_metrics(run, qrels, args.final_k, args.quality_k) if qrels else {}
    stats_by_budget = serving_grid_scan_stats_by_budget(admission)
    stats_sample = serving_grid_scan_stats_sample(admission, largest_budget)
    return {
        "profile": profile.name,
        "candidate_source": profile.candidate_source,
        "graph_mode": "document_nodes",
        "proxy_encoder": profile.proxy_encoder,
        "centroids": profile.centroids,
        "centroid_count": profile.centroid_count,
        "storage_kind": profile.storage_kind,
        "cache_mode": profile.cache_mode,
        "storage_cache_mode": profile.cache_mode,
        "token_pooling": profile.token_pooling,
        "token_pooling_target_ratio": profile.token_pooling_target_ratio,
        "token_pooling_min_tokens": args.multivector_token_pooling_min_tokens,
        "ef": ef,
        "oversampling": oversampling,
        "largest_budget": largest_budget,
        "p50_ms": largest_latency.get("p50_ms"),
        "p95_ms": largest_latency.get("p95_ms"),
        "p50_latency_ms_at_largest_budget": largest_latency.get("p50_ms"),
        "p95_latency_ms_at_largest_budget": largest_latency.get("p95_ms"),
        "exact_top1_admission_rate": aggregate.get("exact_top1_admission_rate", 0.0),
        "exact_top10_admission_recall": aggregate.get("exact_top10_admission_recall", 0.0),
        "recall@10": metrics.get("recall@10") if metrics else None,
        "ndcg@10": metrics.get("ndcg@10") if metrics else None,
        "mrr@10": metrics.get("mrr@10") if metrics else None,
        "metrics": metrics,
        "index_bytes": index_phase.get("index_bytes", 0),
        "index_stats": index_phase.get("index_stats", {}),
        "largest_budget_work": largest_work,
        "serving_stats_by_budget": stats_by_budget,
        "stats_available": merge_document_node_serving_stats_available(stats_by_budget),
        "serving_stats_sample": stats_sample,
        "last_scan_stats_sample": stats_sample,
    }


def serving_row_id(row: dict[str, Any]) -> str:
    return "{profile}_ef{ef}_os{oversampling}".format(
        profile=row.get("profile", ""),
        ef=int(row.get("ef", 0) or 0),
        oversampling=int(row.get("oversampling", 0) or 0),
    )


def serving_row_p95(row: dict[str, Any]) -> float | None:
    value = row.get("p95_latency_ms_at_largest_budget")
    if value is None:
        value = row.get("p95_ms")
    if value is None:
        return None
    try:
        p95 = float(value)
    except (TypeError, ValueError):
        return None
    return p95 if math.isfinite(p95) else None


def serving_row_metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict):
            value = metrics.get(key)
    if value is None:
        return None
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    return metric if math.isfinite(metric) else None


def serving_storage_penalty(row: dict[str, Any]) -> int:
    return {
        "sq8": 0,
        "f16": 1,
        "f32": 2,
    }.get(str(row.get("storage_kind", "")), 3)


def serving_row_experimental(row: dict[str, Any]) -> bool:
    profile = str(row.get("profile", ""))
    return (
        str(row.get("candidate_source", "")) == "quantized_inverted_experimental"
        or profile == "quantized_inverted_experimental"
        or profile.startswith("quantized_inverted_experimental_")
    )


def serving_row_has_usable_metrics(row: dict[str, Any]) -> bool:
    return (
        serving_row_p95(row) is not None
        or serving_row_metric(row, "exact_top10_admission_recall") is not None
        or serving_row_metric(row, "ndcg@10") is not None
    )


def serving_recommendation_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": serving_row_id(row),
        "profile": row.get("profile"),
        "candidate_source": row.get("candidate_source"),
        "experimental": serving_row_experimental(row),
        "proxy_encoder": row.get("proxy_encoder"),
        "centroids": row.get("centroids"),
        "storage_kind": row.get("storage_kind"),
        "cache_mode": row.get("cache_mode"),
        "token_pooling": row.get("token_pooling"),
        "token_pooling_target_ratio": row.get("token_pooling_target_ratio"),
        "ef": row.get("ef"),
        "oversampling": row.get("oversampling"),
        "largest_budget": row.get("largest_budget"),
        "p50_ms": row.get("p50_ms"),
        "p95_ms": row.get("p95_ms"),
        "p50_latency_ms_at_largest_budget": row.get("p50_latency_ms_at_largest_budget"),
        "p95_latency_ms_at_largest_budget": row.get("p95_latency_ms_at_largest_budget"),
        "exact_top10_admission_recall": row.get("exact_top10_admission_recall"),
        "exact_top1_admission_rate": row.get("exact_top1_admission_rate"),
        "recall@10": row.get("recall@10"),
        "ndcg@10": row.get("ndcg@10"),
        "mrr@10": row.get("mrr@10"),
        "index_bytes": row.get("index_bytes"),
        "stats_available": row.get("stats_available", {}),
        "last_scan_stats_sample": row.get("last_scan_stats_sample", {}),
    }


def serving_profile_explanation(row: dict[str, Any]) -> str:
    if not isinstance(row, dict) or not row:
        return "No profile was selected."
    sample = row.get("last_scan_stats_sample", {})
    if not isinstance(sample, dict):
        sample = {}
    parts = [
        f"source={row.get('candidate_source', '')}",
        f"p95={float(serving_row_p95(row) or 0.0):.3f}ms",
        f"top10_admission={float(row.get('exact_top10_admission_recall') or 0.0):.6f}",
    ]
    if row.get("ndcg@10") is not None:
        parts.append(f"ndcg@10={float(row.get('ndcg@10') or 0.0):.6f}")
    for key, label in (
        ("multivector_doc_graph_warning", "warning"),
        ("proxy_encoder_kind", "proxy"),
        ("proxy_candidates", "proxy_candidates"),
        ("centroid_candidates", "centroid_candidates"),
        ("centroid_docs_touched", "centroid_docs_touched"),
        ("multivector_doc_graph_exact_rerank_docs", "doc_graph_rerank_docs"),
        ("multivector_exact_rerank_docs", "exact_rerank_docs"),
        ("multivector_exact_rerank_pairs", "exact_pairs"),
        ("multivector_exact_kernel", "kernel"),
        ("multivector_doc_sidecar_cache_mode", "sidecar_cache"),
        ("multivector_doc_sidecar_bytes_touched", "sidecar_bytes"),
        ("multivector_tokens_original", "tokens_original"),
        ("multivector_tokens_pooled", "tokens_pooled"),
        ("quantized_inverted_postings_touched", "quantized_postings"),
    ):
        if key in sample:
            parts.append(f"{label}={sample[key]}")
    return "; ".join(parts)


def serving_quality_available(rows: list[dict[str, Any]]) -> bool:
    return any(serving_row_metric(row, "ndcg@10") is not None for row in rows)


def serving_exact_baseline_from_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("method") == EXACT_SCAN_METHOD or item.get("retrieval_mode") == "multivector_exact_scan":
            metrics = item.get("metrics", {})
            serial_latency = item.get("serial_latency", {})
            return {
                "available": True,
                "method": item.get("method"),
                "metrics": metrics if isinstance(metrics, dict) else {},
                "serial_latency": serial_latency if isinstance(serial_latency, dict) else {},
            }
    return {
        "available": False,
        "reason": "exact scan method was not included in this benchmark run",
    }


def serving_threshold_failures(
    row: dict[str, Any],
    *,
    min_top10_admission: float,
    min_ndcg_ratio_vs_exact: float,
    max_p95_ms: float,
    exact_baseline_ndcg: float | None,
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    unavailable: list[str] = []
    p95 = serving_row_p95(row)
    if p95 is None:
        failures.append("missing_p95_latency")
    elif max_p95_ms > 0.0 and p95 > max_p95_ms:
        failures.append("p95_latency_above_cap")

    admission = serving_row_metric(row, "exact_top10_admission_recall")
    if admission is None:
        failures.append("missing_exact_top10_admission_recall")
    elif admission < min_top10_admission:
        failures.append("top10_admission_below_threshold")

    ndcg = serving_row_metric(row, "ndcg@10")
    if exact_baseline_ndcg is None:
        unavailable.append("ndcg_ratio_vs_exact")
    elif ndcg is None:
        failures.append("missing_ndcg@10")
    elif ndcg < exact_baseline_ndcg * min_ndcg_ratio_vs_exact:
        failures.append("ndcg_ratio_below_threshold")

    return failures, unavailable


def pareto_frontier_latency_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    use_ndcg = serving_quality_available(rows)
    frontier: list[dict[str, Any]] = []
    for row in rows:
        p95 = serving_row_p95(row)
        if p95 is None:
            continue
        admission = serving_row_metric(row, "exact_top10_admission_recall") or 0.0
        ndcg = serving_row_metric(row, "ndcg@10") or 0.0
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_p95 = serving_row_p95(other)
            other_comparable_p95 = other_p95 if other_p95 is not None else float("inf")
            other_admission = serving_row_metric(other, "exact_top10_admission_recall") or 0.0
            other_ndcg = serving_row_metric(other, "ndcg@10") or 0.0
            better_or_equal = (
                other_comparable_p95 <= p95
                and other_admission >= admission
                and (not use_ndcg or other_ndcg >= ndcg)
            )
            strictly_better = (
                other_comparable_p95 < p95
                or other_admission > admission
                or (use_ndcg and other_ndcg > ndcg)
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(serving_recommendation_row(row))
    return sorted(
        frontier,
        key=lambda item: (
            serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
            str(item.get("id", "")),
        ),
    )


def compute_document_node_serving_recommendation(
    serving_grid: dict[str, Any],
    *,
    exact_baseline: dict[str, Any],
    min_top10_admission: float,
    min_ndcg_ratio_vs_exact: float,
    max_p95_ms: float,
) -> dict[str, Any]:
    raw_rows = serving_grid.get("results", serving_grid.get("summary_rows", []))
    rows = [
        item
        for item in raw_rows
        if isinstance(item, dict)
    ]
    exact_metrics = exact_baseline.get("metrics", {}) if isinstance(exact_baseline, dict) else {}
    exact_baseline_ndcg = (
        serving_row_metric({"ndcg@10": exact_metrics.get("ndcg@10")}, "ndcg@10")
        if isinstance(exact_metrics, dict)
        else None
    )
    threshold_rows: list[dict[str, Any]] = []
    rejected_profiles: list[dict[str, Any]] = []
    for row in rows:
        failures, unavailable = serving_threshold_failures(
            row,
            min_top10_admission=min_top10_admission,
            min_ndcg_ratio_vs_exact=min_ndcg_ratio_vs_exact,
            max_p95_ms=max_p95_ms,
            exact_baseline_ndcg=exact_baseline_ndcg,
        )
        summary = serving_recommendation_row(row)
        summary["threshold_pass"] = not failures
        summary["failure_reasons"] = failures
        summary["unavailable_criteria"] = unavailable
        threshold_rows.append(summary)
        if failures:
            rejected_profiles.append(summary)

    eligible = [
        item for item in threshold_rows
        if (
            item["threshold_pass"]
            and not serving_row_experimental(item)
            and serving_row_p95(item) is not None
        )
    ]
    best_latency_safe = min(
        eligible,
        key=lambda item: (
            serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
            str(item.get("id", "")),
        ),
        default=None,
    )
    if best_latency_safe is not None:
        best_latency_safe = {
            **best_latency_safe,
            "criteria_unavailable": sorted({
                criterion
                for item in threshold_rows
                for criterion in item.get("unavailable_criteria", [])
            }),
        }

    use_ndcg = serving_quality_available(rows)
    best_quality_row = max(
        rows,
        key=lambda item: (
            (serving_row_metric(item, "ndcg@10") or 0.0)
            if use_ndcg
            else (serving_row_metric(item, "exact_top10_admission_recall") or 0.0),
            serving_row_metric(item, "exact_top10_admission_recall") or 0.0,
            0 if not serving_row_experimental(item) else -1,
            -(serving_row_p95(item) if serving_row_p95(item) is not None else float("inf")),
            str(item.get("profile", "")),
        ),
        default=None,
    )
    best_quality = serving_recommendation_row(best_quality_row) if best_quality_row else None
    if best_quality is not None:
        best_quality["selection_basis"] = "ndcg@10" if use_ndcg else "exact_top10_admission_recall"

    ordered_by_latency = sorted(
        rows,
        key=lambda item: (
            serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
            str(item.get("profile", "")),
            int(item.get("ef", 0) or 0),
            int(item.get("oversampling", 0) or 0),
        ),
    )
    latency_rank = {
        serving_row_id(item): rank
        for rank, item in enumerate(ordered_by_latency, start=1)
    }
    best_admission = max(
        (serving_row_metric(item, "exact_top10_admission_recall") or 0.0 for item in rows),
        default=0.0,
    )
    best_ndcg = max(
        (serving_row_metric(item, "ndcg@10") or 0.0 for item in rows),
        default=0.0,
    )
    non_experimental_has_usable_metrics = any(
        not serving_row_experimental(item) and serving_row_has_usable_metrics(item)
        for item in rows
    )
    balanced_candidates: list[dict[str, Any]] = []
    for row in rows:
        admission = serving_row_metric(row, "exact_top10_admission_recall") or 0.0
        ndcg = serving_row_metric(row, "ndcg@10") or 0.0
        admission_loss_penalty = round(max(0.0, best_admission - admission) * 100.0, 3)
        quality_loss_penalty = round(max(0.0, best_ndcg - ndcg) * 100.0, 3) if use_ndcg else 0.0
        storage_penalty = serving_storage_penalty(row)
        experimental_penalty = (
            1_000_000
            if serving_row_experimental(row) and non_experimental_has_usable_metrics
            else 0
        )
        score = (
            latency_rank.get(serving_row_id(row), len(rows) + 1)
            + admission_loss_penalty
            + quality_loss_penalty
            + storage_penalty
            + experimental_penalty
        )
        balanced_candidates.append({
            **serving_recommendation_row(row),
            "score": round(score, 3),
            "score_components": {
                "latency_rank": latency_rank.get(serving_row_id(row), len(rows) + 1),
                "admission_loss_penalty": admission_loss_penalty,
                "quality_loss_penalty": quality_loss_penalty,
                "storage_penalty": storage_penalty,
                "experimental_penalty": experimental_penalty,
            },
        })
    best_balanced = min(
        balanced_candidates,
        key=lambda item: (
            float(item["score"]),
            serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
            str(item.get("id", "")),
        ),
        default=None,
    )

    return {
        "thresholds": {
            "serving_min_top10_admission": min_top10_admission,
            "serving_min_ndcg_ratio_vs_exact": min_ndcg_ratio_vs_exact,
            "serving_max_p95_ms": max_p95_ms,
        },
        "exact_baseline": exact_baseline,
        "best_latency_safe": best_latency_safe,
        "best_quality": best_quality,
        "best_balanced": best_balanced,
        "balanced_formula": (
            "score = latency_rank + max(0,best_admission-admission)*100 "
            "+ max(0,best_ndcg-ndcg)*100 when qrels exist + storage_penalty; "
            "storage_penalty is sq8=0, f16=1, f32=2, other=3; "
            "experimental_penalty is 1000000 when any non-experimental profile has usable metrics"
        ),
        "pareto_frontier_latency_quality": pareto_frontier_latency_quality(rows),
        "pareto_frontier": pareto_frontier_latency_quality(rows),
        "profile_thresholds": sorted(
            threshold_rows,
            key=lambda item: (
                serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
                str(item.get("id", "")),
            ),
        ),
        "rejected_profiles": sorted(
            rejected_profiles,
            key=lambda item: (
                serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
                str(item.get("id", "")),
            ),
        ),
    }


def _synthetic_serving_row(
    profile: str,
    *,
    p95: float | None,
    admission: float,
    ndcg: float | None,
    storage: str = "f16",
    candidate_source: str = "proxy_vector",
    proxy_encoder: str = "normalized_mean",
    centroids: str = "off",
    ef: int = 100,
    oversampling: int = 1,
    stats_sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stats_sample is None:
        stats_sample = {}
    return {
        "profile": profile,
        "candidate_source": candidate_source,
        "proxy_encoder": proxy_encoder,
        "centroids": centroids,
        "storage_kind": storage,
        "cache_mode": "auto",
        "token_pooling": "off",
        "token_pooling_target_ratio": 1.0,
        "ef": ef,
        "oversampling": oversampling,
        "largest_budget": 800,
        "p50_latency_ms_at_largest_budget": None if p95 is None else round(p95 * 0.7, 3),
        "p95_latency_ms_at_largest_budget": p95,
        "exact_top1_admission_rate": admission,
        "exact_top10_admission_recall": admission,
        "recall@10": ndcg,
        "ndcg@10": ndcg,
        "mrr@10": ndcg,
        "index_bytes": 1000,
        "stats_available": document_node_serving_stats_available(stats_sample),
        "last_scan_stats_sample": stats_sample,
    }


def _self_check_document_node_serving_recommendation() -> None:
    rows = [
        _synthetic_serving_row("latency_winner_fails_admission", p95=8.0, admission=0.55, ndcg=0.70, storage="sq8"),
        _synthetic_serving_row("latency_safe_winner", p95=12.0, admission=0.86, ndcg=0.86, storage="f16"),
        _synthetic_serving_row(
            "balanced_winner",
            p95=20.0,
            admission=0.94,
            ndcg=0.88,
            storage="sq8",
            stats_sample={
                "multivector_candidate_source": "proxy_vector",
                "proxy_encoder_kind": "normalized_mean",
                "multivector_doc_graph_exact_rerank_docs": 400,
                "multivector_exact_rerank_pairs": 8192,
                "multivector_exact_kernel": "blocked_avx2",
                "multivector_doc_sidecar_cache_mode": "resident",
            },
        ),
        _synthetic_serving_row("quality_winner_slow", p95=80.0, admission=0.91, ndcg=0.90, storage="f32"),
        _synthetic_serving_row("missing_latency", p95=None, admission=0.88, ndcg=0.86, storage="f16"),
    ]
    grid = {"results": rows}
    exact = {"available": True, "metrics": {"ndcg@10": 0.90}}
    rec = compute_document_node_serving_recommendation(
        grid,
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert rec["best_latency_safe"]["profile"] == "latency_safe_winner"
    assert rec["best_quality"]["profile"] == "quality_winner_slow"
    assert rec["best_balanced"]["profile"] == "balanced_winner"
    assert rec["best_balanced"]["profile"] != rec["best_latency_safe"]["profile"]
    assert rec["best_balanced"]["profile"] != rec["best_quality"]["profile"]
    rejected = {item["profile"]: item["failure_reasons"] for item in rec["rejected_profiles"]}
    assert "top10_admission_below_threshold" in rejected["latency_winner_fails_admission"]
    assert "missing_p95_latency" in rejected["missing_latency"]
    rec_again = compute_document_node_serving_recommendation(
        grid,
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert rec == rec_again

    no_qrels_rows = [
        _synthetic_serving_row("fast_no_qrels", p95=10.0, admission=0.82, ndcg=None, storage="sq8"),
        _synthetic_serving_row("quality_no_qrels", p95=50.0, admission=0.95, ndcg=None, storage="f16"),
    ]
    no_qrels = compute_document_node_serving_recommendation(
        {"summary_rows": no_qrels_rows},
        exact_baseline={"available": False},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert no_qrels["best_latency_safe"]["profile"] == "fast_no_qrels"
    assert "ndcg_ratio_vs_exact" in no_qrels["best_latency_safe"]["criteria_unavailable"]
    assert no_qrels["best_quality"]["profile"] == "quality_no_qrels"
    assert no_qrels["best_quality"]["selection_basis"] == "exact_top10_admission_recall"

    exact_baseline_missing_rows = [
        _synthetic_serving_row("baseline_missing_fast", p95=7.0, admission=0.82, ndcg=0.50, storage="sq8"),
        _synthetic_serving_row("baseline_missing_quality", p95=40.0, admission=0.95, ndcg=0.90, storage="f16"),
    ]
    exact_baseline_missing = compute_document_node_serving_recommendation(
        {"results": exact_baseline_missing_rows},
        exact_baseline={"available": False, "reason": "synthetic exact baseline missing"},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert exact_baseline_missing["best_latency_safe"]["profile"] == "baseline_missing_fast"
    assert (
        "ndcg_ratio_vs_exact"
        in exact_baseline_missing["best_latency_safe"]["criteria_unavailable"]
    )
    assert exact_baseline_missing["best_quality"]["profile"] == "baseline_missing_quality"
    assert exact_baseline_missing["best_quality"]["selection_basis"] == "ndcg@10"

    none_pass = compute_document_node_serving_recommendation(
        {
            "results": [
                _synthetic_serving_row("too_low_admission", p95=9.0, admission=0.40, ndcg=0.60, storage="sq8"),
                _synthetic_serving_row("too_slow", p95=200.0, admission=0.90, ndcg=0.88, storage="f16"),
            ]
        },
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=100.0,
    )
    assert none_pass["best_latency_safe"] is None
    none_pass_rejected = {item["profile"]: item["failure_reasons"] for item in none_pass["rejected_profiles"]}
    assert "top10_admission_below_threshold" in none_pass_rejected["too_low_admission"]
    assert "p95_latency_above_cap" in none_pass_rejected["too_slow"]

    experimental_rows = [
        _synthetic_serving_row(
            "quantized_inverted_experimental_f32",
            p95=5.0,
            admission=0.99,
            ndcg=0.90,
            storage="f32",
            candidate_source="quantized_inverted_experimental",
        ),
        _synthetic_serving_row("valid_non_experimental", p95=30.0, admission=0.85, ndcg=0.87, storage="f16"),
    ]
    experimental_rec = compute_document_node_serving_recommendation(
        {"results": experimental_rows},
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert experimental_rec["best_latency_safe"]["profile"] == "valid_non_experimental"
    assert experimental_rec["best_balanced"]["profile"] == "valid_non_experimental"
    experimental_only = compute_document_node_serving_recommendation(
        {"results": [experimental_rows[0]]},
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert experimental_only["best_latency_safe"] is None
    assert experimental_only["best_balanced"]["profile"] == "quantized_inverted_experimental_f32"

    tie_rows = [
        _synthetic_serving_row("tie_a", p95=10.0, admission=0.90, ndcg=0.90, storage="f16"),
        _synthetic_serving_row("tie_b", p95=10.0, admission=0.90, ndcg=0.90, storage="f16"),
    ]
    tie = compute_document_node_serving_recommendation(
        {"results": tie_rows},
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert tie["best_latency_safe"]["profile"] == "tie_a"
    assert tie["best_balanced"]["profile"] == "tie_a"
    tie_again = compute_document_node_serving_recommendation(
        {"results": tie_rows},
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert tie == tie_again


def _self_check_document_node_serving_recommendation_schema_fixture() -> None:
    fixture_rows = [
        _synthetic_serving_row(
            "proxy_normalized_mean_f16",
            p95=8.0,
            admission=0.70,
            ndcg=0.91,
            storage="f16",
        ),
        _synthetic_serving_row(
            "centroid_mean_f16",
            p95=20.0,
            admission=0.86,
            ndcg=0.91,
            storage="f16",
            proxy_encoder="centroid_mean",
            centroids="kmeans",
        ),
        _synthetic_serving_row(
            "centroid_lite_f16",
            p95=55.0,
            admission=0.93,
            ndcg=0.94,
            storage="f16",
            candidate_source="centroid_lite",
            centroids="kmeans",
        ),
        _synthetic_serving_row(
            "quantized_inverted_experimental_f32",
            p95=6.0,
            admission=0.98,
            ndcg=0.93,
            storage="f32",
            candidate_source="quantized_inverted_experimental",
        ),
    ]
    report = {
        "document_node_serving_grid": {
            "enabled": True,
            "results": fixture_rows,
        }
    }
    report["document_node_serving_recommendation"] = compute_document_node_serving_recommendation(
        report["document_node_serving_grid"],
        exact_baseline={"available": True, "metrics": {"ndcg@10": 0.95}},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    rec = report["document_node_serving_recommendation"]
    assert rec
    assert rec["best_latency_safe"]["profile"] == "centroid_mean_f16"
    assert rec["best_quality"]["profile"] == "centroid_lite_f16"
    balanced_profile = rec["best_balanced"]["profile"]
    assert balanced_profile == "centroid_lite_f16"
    repeat = compute_document_node_serving_recommendation(
        report["document_node_serving_grid"],
        exact_baseline={"available": True, "metrics": {"ndcg@10": 0.95}},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert repeat["best_balanced"]["profile"] == balanced_profile
    thresholds = {item["profile"]: item for item in rec["profile_thresholds"]}
    assert thresholds["quantized_inverted_experimental_f32"]["experimental"] is True
    assert rec["best_latency_safe"]["profile"] != "quantized_inverted_experimental_f32"
    rejected = {item["profile"]: item["failure_reasons"] for item in rec["rejected_profiles"]}
    assert rejected["proxy_normalized_mean_f16"] == ["top10_admission_below_threshold"]
    markdown = markdown_benchmark_summary(report)
    assert "### Document-node serving recommendation" in markdown
    assert "centroid_mean_f16" in markdown
    assert "quantized_inverted_experimental_f32" in markdown


def _self_check_document_node_serving_stats_extraction() -> None:
    partial_stats = {
        "multivector_candidate_source": "proxy_vector",
        "multivector_doc_graph_warning": "document_node_proxy_vector_graph_traversal",
        "multivector_doc_graph_nodes": 1000,
        "multivector_doc_graph_exact_rerank_docs": 400,
        "multivector_exact_rerank_docs": 400,
        "multivector_exact_rerank_pairs": 123456,
        "multivector_exact_kernel": "blocked_neon",
        "proxy_encoder_kind": "normalized_mean",
        "proxy_candidates": 800,
        "proxy_top1_admission": True,
        "multivector_doc_sidecar_cache_mode": "resident",
        "multivector_doc_sidecar_cache_hits": 20,
        "multivector_tokens_original": 6400,
        "multivector_tokens_pooled": 3200,
        "multivector_bm25_injection_enabled": False,
        "learned_sparse_candidates": 0,
        "quantized_inverted_postings_touched": 0,
        "quantized_inverted_posting_bytes": 2048,
        "quantized_inverted_sidecar_bytes": 2304,
        "quantized_codeword_debug_counter": 7,
        "unrelated_large_field": "ignored",
    }
    extracted = extract_document_node_serving_stats(partial_stats)
    assert extracted["multivector_candidate_source"] == "proxy_vector"
    assert extracted["proxy_candidates"] == 800
    assert extracted["multivector_doc_sidecar_cache_hits"] == 20
    assert extracted["quantized_inverted_posting_bytes"] == 2048
    assert extracted["quantized_inverted_sidecar_bytes"] == 2304
    assert extracted["quantized_codeword_debug_counter"] == 7
    assert "unrelated_large_field" not in extracted
    available = document_node_serving_stats_available(extracted)
    assert available["core"] is True
    assert available["proxy"] is True
    assert available["storage_cache"] is True
    assert available["pooling"] is True
    assert available["sparse_bm25_rescue"] is True
    assert available["experimental_quantized"] is True
    assert document_node_serving_stats_available({})["core"] is False

    admission = {
        "per_query": [
            {
                "query_id": "q1",
                "budgets": [
                    {"budget": 50, "scan_stats": {"multivector_candidate_source": "proxy_vector"}},
                    {"budget": 800, "scan_stats": partial_stats},
                ],
            }
        ]
    }
    by_budget = serving_grid_scan_stats_by_budget(admission)
    assert by_budget["50"]["scan_stats_sample"]["multivector_candidate_source"] == "proxy_vector"
    assert by_budget["800"]["scan_stats_sample"]["multivector_exact_kernel"] == "blocked_neon"
    assert serving_grid_scan_stats_sample(admission, 800)["proxy_encoder_kind"] == "normalized_mean"
    assert serving_grid_scan_stats_sample({"per_query": []}, 800) == {}

    row = _synthetic_serving_row(
        "stats_roundtrip",
        p95=12.0,
        admission=0.91,
        ndcg=0.87,
        stats_sample=by_budget["800"]["scan_stats_sample"],
    )
    round_trip = json.loads(json.dumps(row))
    assert round_trip["last_scan_stats_sample"]["multivector_exact_kernel"] == "blocked_neon"
    rec = compute_document_node_serving_recommendation(
        {"summary_rows": [row]},
        exact_baseline={"available": True, "metrics": {"ndcg@10": 0.90}},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    markdown = markdown_benchmark_summary({
        "document_node_serving_recommendation": rec,
    })
    assert "Why latency-safe won" in markdown
    assert "blocked_neon" in markdown
    assert "document_node_proxy_vector_graph_traversal" in markdown


def _self_check_document_node_serving_grid_serialization() -> None:
    profile = DocumentNodeServingProfile(
        name="proxy_normalized_mean_f16",
        candidate_source="proxy_vector",
        proxy_encoder="normalized_mean",
        storage_kind="f16",
    )
    args = argparse.Namespace(
        final_k=10,
        quality_k=10,
        multivector_token_pooling_min_tokens=16,
        admission_budget_sweep="50,100,200,400,800",
    )
    admission = {
        "aggregate": {
            "budget_sweep": [50, 100, 200, 400, 800],
            "latency_by_budget": {
                "800": {"p50_ms": 12.5, "p95_ms": 21.0},
            },
            "admission_by_budget": {
                "800": {"exact_rerank_docs": {"p50": 400}},
            },
            "exact_top1_admission_rate": 0.75,
            "exact_top10_admission_recall": 0.92,
            "run_at_largest_budget": {},
        },
        "per_query": [
            {
                "query_id": "q1",
                "budgets": [
                    {
                        "budget": 800,
                        "scan_stats": {
                            "multivector_candidate_source": "proxy_vector",
                            "proxy_encoder_kind": "normalized_mean",
                            "multivector_exact_kernel": "blocked_scalar",
                        },
                    },
                ],
            }
        ],
    }
    row = document_node_serving_summary_row(
        profile=profile,
        args=args,
        ef=100,
        oversampling=2,
        index_phase={
            "index_bytes": 4096,
            "index_stats": {"multivector_proxy_encoder": "normalized_mean"},
        },
        admission=admission,
        qrels={},
    )
    assert row["graph_mode"] == "document_nodes"
    assert row["storage_cache_mode"] == "auto"
    assert row["p50_ms"] == 12.5
    assert row["p95_ms"] == 21.0
    assert row["last_scan_stats_sample"]["proxy_encoder_kind"] == "normalized_mean"

    grid = {
        "document_node_serving_grid": {
            "enabled": True,
            "profiles": [profile.__dict__],
            "budget_sweep": [50, 100, 200, 400, 800],
            "results": [row],
        }
    }
    serialized = json.loads(json.dumps(grid))
    assert serialized["document_node_serving_grid"]["results"][0]["profile"] == "proxy_normalized_mean_f16"
    markdown = markdown_benchmark_summary(serialized)
    assert "### Document-node serving grid" in markdown
    assert "proxy_normalized_mean_f16" in markdown
    assert "21.000" in markdown

    base_args = argparse.Namespace(admission_budget_sweep="25,50")
    profile_args = document_node_serving_profile_args(
        base_args,
        profile,
        ef=50,
        oversampling=1,
    )
    assert profile_args.admission_budget_sweep == "25,50"


def run_document_node_serving_grid(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    queries: list[QueryItem],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    profiles = document_node_serving_profiles(
        include_experimental=args.document_node_serving_grid_include_experimental,
        centroid_count=args.multivector_centroid_count,
    )
    summary_rows: list[dict[str, Any]] = []
    full_admission: list[dict[str, Any]] = []

    for profile in profiles:
        index_args = document_node_serving_profile_args(
            args,
            profile,
            ef=0,
            oversampling=1,
        )
        index_phase = build_index(conn, index_args)
        for ef in DOCUMENT_NODE_SERVING_GRID_EF:
            for oversampling in DOCUMENT_NODE_SERVING_GRID_OVERSAMPLING:
                mode_args = document_node_serving_profile_args(
                    args,
                    profile,
                    ef=ef,
                    oversampling=oversampling,
                )
                admission = run_admission_debug(conn, mode_args, queries)
                full_admission.append({
                    "profile": profile.name,
                    "ef": ef,
                    "oversampling": oversampling,
                    "admission_debug": admission,
                })
                summary_rows.append(
                    document_node_serving_summary_row(
                        profile=profile,
                        args=mode_args,
                        ef=ef,
                        oversampling=oversampling,
                        index_phase=index_phase,
                        admission=admission,
                        qrels=qrels,
                    )
                )

    return {
        "enabled": True,
        "production_oriented": True,
        "include_experimental": args.document_node_serving_grid_include_experimental,
        "budget_sweep": parse_int_grid(args.admission_budget_sweep, "--admission-budget-sweep"),
        "ef_grid": list(DOCUMENT_NODE_SERVING_GRID_EF),
        "oversampling_grid": list(DOCUMENT_NODE_SERVING_GRID_OVERSAMPLING),
        "cache_grid": ["auto"],
        "profiles": [profile.__dict__ for profile in profiles],
        "queries": len(queries),
        "results": summary_rows,
        "summary_rows": summary_rows,
        "admission_debug_runs": full_admission,
    }


def run_document_node_admission_grid(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    queries: list[QueryItem],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    storage_grid = args.document_node_storage_grid_values
    cache_grid = args.document_node_cache_grid_values
    pooling_grid = args.document_node_pooling_grid_values
    proxy_encoder_grid = args.document_node_proxy_encoder_grid_values
    ef_grid = args.document_node_ef_grid_values
    oversampling_grid = args.document_node_oversampling_grid_values
    budget_grid = parse_int_grid(args.admission_budget_sweep, "--admission-budget-sweep")
    results: list[dict[str, Any]] = []
    full_admission: list[dict[str, Any]] = []
    default_document_modes_done = False
    default_document_modes_reused_grid_index = False

    def run_default_document_modes(index_phase: dict[str, Any], *, reused_grid_index: bool) -> None:
        nonlocal default_document_modes_done, default_document_modes_reused_grid_index
        for mode in ("proxy_vector", "exact_doc_scan", "quantized_inverted_experimental"):
            mode_args = grid_mode_args(args, mode, "document_nodes")
            admission = run_admission_debug(conn, mode_args, queries)
            mode_name = f"{mode}_document_index" if mode == "exact_doc_scan" else mode
            full_admission.append({"mode": mode_name, "admission_debug": admission})
            results.append(grid_admission_summary(
                mode=mode_name,
                args=mode_args,
                index_phase=index_phase,
                admission=admission,
                qrels=qrels,
            ))
        default_document_modes_done = True
        default_document_modes_reused_grid_index = reused_grid_index

    token_args = clone_args(args, multivector_graph="token_nodes", reuse_index=False)
    token_index_phase = build_index(conn, token_args)
    for mode in ("token_nodes", "exact_token_scan", "plain_fallback", "exact_doc_scan"):
        mode_args = grid_mode_args(args, mode, "token_nodes")
        admission = run_admission_debug(conn, mode_args, queries)
        full_admission.append({"mode": mode, "admission_debug": admission})
        results.append(grid_admission_summary(
            mode=mode,
            args=mode_args,
            index_phase=token_index_phase,
            admission=admission,
            qrels=qrels,
        ))

    token_centroid_args = clone_args(
        args,
        multivector_graph="token_nodes",
        reuse_index=False,
        multivector_centroids="kmeans",
        multivector_centroid_count=args.multivector_centroid_count,
    )
    token_centroid_index_phase = build_index(conn, token_centroid_args)
    mode_args = grid_mode_args(token_centroid_args, "centroid_lite", "token_nodes")
    admission = run_admission_debug(conn, mode_args, queries)
    full_admission.append({"mode": "centroid_lite_token_index", "admission_debug": admission})
    results.append(grid_admission_summary(
        mode="centroid_lite_token_index",
        args=mode_args,
        index_phase=token_centroid_index_phase,
        admission=admission,
        qrels=qrels,
    ))

    for proxy_encoder in proxy_encoder_grid:
        proxy_centroids = "kmeans" if proxy_encoder == "centroid_mean" else args.multivector_centroids
        for pooling in pooling_grid:
            pooling_mode = str(pooling["mode"])
            pooling_ratio = float(pooling["ratio"])
            document_args = clone_args(
                args,
                multivector_graph="document_nodes",
                reuse_index=False,
                multivector_token_pooling=pooling_mode,
                multivector_token_pooling_target_ratio=pooling_ratio,
                multivector_proxy_encoder=proxy_encoder,
                multivector_centroids=proxy_centroids,
                multivector_centroid_count=args.multivector_centroid_count,
            )
            document_index_phase = build_index(conn, document_args)
            for storage in storage_grid:
                for cache_mode in cache_grid:
                    for ef in ef_grid:
                        for oversampling in oversampling_grid:
                            mode_args = clone_args(
                                args,
                                multivector_graph="document_nodes",
                                multivector_candidate_source="document_nodes",
                                multivector_plain_fallback="off",
                                multivector_candidate_reservoirs="off",
                                multivector_doc_storage=storage,
                                multivector_doc_storage_cache=cache_mode,
                                multivector_token_pooling=pooling_mode,
                                multivector_token_pooling_target_ratio=pooling_ratio,
                                multivector_proxy_encoder=proxy_encoder,
                                multivector_centroids=proxy_centroids,
                                multivector_centroid_count=args.multivector_centroid_count,
                                multivector_doc_graph_search_ef=ef,
                                multivector_doc_graph_oversampling=oversampling,
                                multivector_doc_graph_rescore_k=0,
                            )
                            admission = run_admission_debug(conn, mode_args, queries)
                            ratio_label = str(pooling_ratio).replace(".", "_")
                            mode_name = (
                                f"document_nodes_{storage}_{cache_mode}_{proxy_encoder}_"
                                f"{pooling_mode}_r{ratio_label}_ef{ef}_os{oversampling}"
                            )
                            full_admission.append({"mode": mode_name, "admission_debug": admission})
                            results.append(grid_admission_summary(
                                mode=mode_name,
                                args=mode_args,
                                index_phase=document_index_phase,
                                admission=admission,
                                qrels=qrels,
                            ))
            if (
                not default_document_modes_done
                and proxy_encoder == args.multivector_proxy_encoder
                and pooling_mode == "off"
                and pooling_ratio == 1.0
            ):
                run_default_document_modes(document_index_phase, reused_grid_index=True)

    if not default_document_modes_done:
        default_document_args = clone_args(
            args,
            multivector_graph="document_nodes",
            reuse_index=False,
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_proxy_encoder=args.multivector_proxy_encoder,
        )
        default_document_index_phase = build_index(conn, default_document_args)
        run_default_document_modes(default_document_index_phase, reused_grid_index=False)

    document_centroid_args = clone_args(
        args,
        multivector_graph="document_nodes",
        reuse_index=False,
        multivector_centroids="kmeans",
        multivector_centroid_count=args.multivector_centroid_count,
    )
    document_centroid_index_phase = build_index(conn, document_centroid_args)
    mode_args = grid_mode_args(document_centroid_args, "centroid_lite", "document_nodes")
    admission = run_admission_debug(conn, mode_args, queries)
    full_admission.append({"mode": "centroid_lite_document_index", "admission_debug": admission})
    results.append(grid_admission_summary(
        mode="centroid_lite_document_index",
        args=mode_args,
        index_phase=document_centroid_index_phase,
        admission=admission,
        qrels=qrels,
    ))

    return {
        "enabled": True,
        "rebuilds_index": True,
        "budget_sweep": budget_grid,
        "storage_grid": storage_grid,
        "cache_grid": cache_grid,
        "pooling_grid": pooling_grid,
        "proxy_encoder_grid": proxy_encoder_grid,
        "ef_grid": ef_grid,
        "oversampling_grid": oversampling_grid,
        "index_graph_m": args.index_graph_m,
        "index_graph_ef_construction": args.index_graph_ef_construction,
        "index_graph_ef_search": args.index_graph_ef_search,
        "index_native_segments": args.index_native_segments,
        "default_document_modes_reused_grid_index": default_document_modes_reused_grid_index,
        "modes": list(DOCUMENT_NODE_BASELINE_MODES),
        "queries": len(queries),
        "results": results,
        "admission_debug_runs": full_admission,
    }


def hybrid_evaluation_mode_args(args: argparse.Namespace, mode: str) -> argparse.Namespace:
    common = {
        "multivector_graph": "document_nodes",
        "multivector_plain_fallback": "off",
        "multivector_candidate_reservoirs": "off",
        "multivector_doc_storage_cache": args.multivector_doc_storage_cache,
        "multivector_doc_graph_search_ef": args.multivector_doc_graph_search_ef,
        "multivector_doc_graph_oversampling": args.multivector_doc_graph_oversampling,
        "multivector_doc_graph_rescore_k": args.multivector_doc_graph_rescore_k,
        "multivector_exact_rerank": args.multivector_exact_rerank,
        "multivector_exact_rerank_k": args.multivector_exact_rerank_k,
    }
    if mode == "exact_scan":
        return clone_args(args, **common, multivector_candidate_source="exact_doc_scan")
    if mode == "document_nodes":
        return clone_args(
            args,
            **common,
            multivector_candidate_source="document_nodes",
            multivector_bm25_candidate_injection="off",
            multivector_sparse_candidate_source="off",
        )
    if mode == "document_nodes_bm25_admission":
        return clone_args(
            args,
            **common,
            multivector_candidate_source="document_nodes",
            multivector_bm25_candidate_injection="dense_with_text",
            multivector_sparse_candidate_source="bm25",
        )
    if mode in {"document_nodes_bm25_rrf", "document_nodes_bm25_dbsf"}:
        return clone_args(
            args,
            **common,
            multivector_candidate_source="document_nodes",
            multivector_bm25_candidate_injection="hybrid_only",
            multivector_sparse_candidate_source="bm25",
        )
    if mode == "proxy_vector_document_nodes":
        return clone_args(
            args,
            **common,
            multivector_candidate_source="proxy_vector",
            multivector_bm25_candidate_injection="off",
            multivector_sparse_candidate_source="off",
        )
    if mode == "learned_sparse_exact_maxsim":
        return clone_args(
            args,
            **common,
            multivector_candidate_source="document_nodes",
            multivector_bm25_candidate_injection="off",
            multivector_sparse_candidate_source="learned_sparse",
        )
    if mode == "quantized_inverted_experimental":
        return clone_args(
            args,
            **common,
            multivector_candidate_source="quantized_inverted_experimental",
            multivector_bm25_candidate_injection="off",
            multivector_sparse_candidate_source="off",
        )
    raise ValueError(f"unsupported hybrid evaluation mode: {mode}")


def set_hybrid_mode_gucs(
    conn: psycopg.Connection[Any],
    mode: str,
    mode_args: argparse.Namespace,
) -> None:
    set_retrieval_gucs(conn, mode_args, f"dbpedia_colbert_hybrid_{mode}")
    branch_plan = "qdrant_like" if mode in {
        "document_nodes_bm25_rrf",
        "document_nodes_bm25_dbsf",
        "proxy_vector_document_nodes",
    } else "dense_only"
    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_branch_plan', %s, false)", (branch_plan,))
    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_debug_admission', 'trace', false)")
    exec_sql(
        conn,
        "SELECT set_config('turbohybrid.multivector_debug_trace_limit', %s, false)",
        (str(min(mode_args.admission_trace_limit, 1000)),),
    )
    if mode == "document_nodes_bm25_dbsf":
        exec_sql(
            conn,
            "SELECT set_config('turbohybrid.dbsf_min_branch_candidates', %s, false)",
            (str(max(1, getattr(mode_args, "hybrid_evaluation_dbsf_min_branch_candidates", 1))),),
        )


def run_hybrid_mode_query(
    conn: psycopg.Connection[Any],
    mode: str,
    query: QueryItem,
    args: argparse.Namespace,
    final_k: int,
) -> list[str]:
    if mode == "exact_scan":
        return run_retrieval_query(conn, EXACT_SCAN_METHOD, query, args, final_k)

    if mode in {"document_nodes", "proxy_vector_document_nodes", "quantized_inverted_experimental"}:
        return run_retrieval_query(conn, QUERY_ONLY_METHOD, query, args, final_k)

    if mode in {"document_nodes_bm25_admission", "learned_sparse_exact_maxsim"}:
        text_query_expr = (
            learned_sparse_text_query_sql()
            if mode == "learned_sparse_exact_maxsim"
            else "websearch_to_tsquery('simple', q.query_text)"
        )
        rows = fetch_all(
            conn,
            f"""
            SELECT d.doc_id
            FROM dbpedia_colbert_docs d
            WHERE d.colbert IS NOT NULL
            ORDER BY d.colbert <~> (
              SELECT turbohybrid_query(
                multivector_query => q.colbert,
                text_query => {text_query_expr},
                dense_k => %s,
                bm25_k => %s,
                final_k => %s,
                bm25_weight => 0
              )
              FROM dbpedia_colbert_queries q
              WHERE q.query_id = %s
            )
            LIMIT %s
            """,
            (args.dense_k, args.bm25_k, final_k, query.query_id, final_k),
        )
        return [str(row[0]) for row in rows]

    fusion = "rrf" if mode == "document_nodes_bm25_rrf" else "dbsf"
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
            fusion => %s,
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
        (fusion, args.dense_k, args.bm25_k, args.rrf_k, final_k, query.query_id, final_k),
    )
    return [str(row[0]) for row in rows]


def admission_from_trace_or_results(
    exact_top: list[dict[str, Any]],
    result_docs: list[str],
    stats: dict[str, Any],
    *,
    candidate_source: Any = None,
    graph_mode: Any = None,
    plain_fallback: Any = None,
    exact_scan: bool = False,
) -> dict[str, Any]:
    if exact_scan:
        return {
            "exact_top1_admission": True,
            "exact_top10_admission_recall": 1.0 if exact_top else 0.0,
            "candidate_failures": [],
        }

    trace = stats.get("multivector_admission_trace")
    trace_entries = trace if isinstance(trace, list) else []
    trace_by_heap = {
        key: entry
        for entry in trace_entries
        if isinstance(entry, dict) and (key := trace_key(entry)) is not None
    }
    infer_from_results = (
        not trace_by_heap
        and can_infer_admission_from_result_docs(
            candidate_source or stats.get("multivector_candidate_source"),
            fallback_used=bool(stats.get("multivector_plain_fallback_used")),
            plain_fallback=plain_fallback,
            graph_mode=graph_mode or stats.get("multivector_graph_mode"),
        )
    )
    candidate_failures: list[str] = []
    admitted = 0
    top = exact_top[: min(10, len(exact_top))]
    for item in top:
        key = (int(item["heap_block"]), int(item["heap_offset"]))
        is_admitted = key in trace_by_heap or (infer_from_results and item["doc_id"] in result_docs)
        if is_admitted:
            admitted += 1
        else:
            candidate_failures.append(item["doc_id"])
    return {
        "exact_top1_admission": bool(top and not candidate_failures[:1]),
        "exact_top10_admission_recall": round(admitted / len(top), 6) if top else 0.0,
        "admission_inferred_from_result_docs": infer_from_results,
        "candidate_failures": candidate_failures,
    }


def hybrid_scan_record(stats: dict[str, Any]) -> dict[str, Any]:
    branch_latency = scan_stat_int_list(stats, "branch_latency_us")
    branch_candidates = scan_stat_int_list(stats, "branch_candidate_counts")
    return {
        "branch_kinds": stats.get("branch_kinds", []),
        "branch_candidate_counts": branch_candidates,
        "branch_latency_us": branch_latency,
        "branch_latency_us_total": sum(branch_latency),
        "branch_candidates_total": sum(branch_candidates),
        "branch_fusion_mode": scan_stat_str(stats, "branch_fusion_mode"),
        "fusion_strategy": scan_stat_str(stats, "fusion_strategy"),
        "docs_scored": scan_docs_scored(stats),
        "exact_maxsim_pairs": scan_stat_int(stats, "multivector_exact_rerank_pairs"),
        "exact_rerank_docs": scan_exact_rerank_docs(stats),
        "sidecar_bytes": sidecar_stats_from_scan(stats).get("sidecar_bytes_touched", 0),
        "native_exact_bytes": sidecar_stats_from_scan(stats).get("native_cache_exact_bytes", 0),
        **exact_rerank_work_from_stats(stats),
        **proxy_work_from_stats(stats),
        **centroid_work_from_stats(stats),
        **quantized_inverted_work_from_stats(stats),
        **learned_sparse_work_from_stats(stats),
    }


def summarize_hybrid_scan_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "branch_latency_us_total": summarize_ints([int(item.get("branch_latency_us_total", 0)) for item in records]),
        "branch_candidates_total": summarize_ints([int(item.get("branch_candidates_total", 0)) for item in records]),
        "docs_scored": summarize_ints([int(item.get("docs_scored", 0)) for item in records]),
        "exact_maxsim_pairs": summarize_ints([int(item.get("exact_maxsim_pairs", 0)) for item in records]),
        "exact_rerank_docs": summarize_ints([int(item.get("exact_rerank_docs", 0)) for item in records]),
        "exact_rerank_pairs_saved": summarize_ints([int(item.get("exact_rerank_pairs_saved", 0)) for item in records]),
        "sidecar_bytes": summarize_ints([int(item.get("sidecar_bytes", 0)) for item in records]),
        "native_exact_bytes": summarize_ints([int(item.get("native_exact_bytes", 0)) for item in records]),
        "learned_sparse_candidates": summarize_ints([int(item.get("learned_sparse_candidates", 0)) for item in records]),
        "proxy_candidates": summarize_ints([int(item.get("proxy_candidates", 0)) for item in records]),
        "centroid_candidates": summarize_ints([int(item.get("centroid_candidates", 0)) for item in records]),
        "centroid_docs_touched": summarize_ints([int(item.get("centroid_docs_touched", 0)) for item in records]),
        "quantized_inverted_candidates": summarize_ints([int(item.get("quantized_inverted_candidates", 0)) for item in records]),
        "quantized_inverted_postings_touched": summarize_ints([int(item.get("quantized_inverted_postings_touched", 0)) for item in records]),
        "quantized_inverted_docs_scored": summarize_ints([int(item.get("quantized_inverted_docs_scored", 0)) for item in records]),
    }


def hybrid_mode_recommendation(mode: str | None, reason: str) -> dict[str, Any]:
    candidate_source = "document_nodes"
    branch_plan = "dense_only"
    sparse_source = "off"
    bm25_injection = "off"
    fusion = None
    if mode == "exact_scan":
        candidate_source = "exact_doc_scan"
    elif mode == "proxy_vector_document_nodes":
        candidate_source = "proxy_vector"
        branch_plan = "qdrant_like"
    elif mode == "document_nodes_bm25_admission":
        sparse_source = "bm25"
        bm25_injection = "dense_with_text"
    elif mode == "document_nodes_bm25_rrf":
        sparse_source = "bm25"
        branch_plan = "qdrant_like"
        fusion = "rrf"
    elif mode == "document_nodes_bm25_dbsf":
        sparse_source = "bm25"
        branch_plan = "qdrant_like"
        fusion = "dbsf"
    elif mode == "learned_sparse_exact_maxsim":
        sparse_source = "learned_sparse"
    elif mode == "quantized_inverted_experimental":
        candidate_source = "quantized_inverted_experimental"

    gucs: dict[str, str] = {
        "turbohybrid.multivector_candidate_source": candidate_source,
        "turbohybrid.multivector_branch_plan": branch_plan,
        "turbohybrid.multivector_sparse_candidate_source": sparse_source,
        "turbohybrid.multivector_bm25_candidate_injection": bm25_injection,
    }
    if fusion is not None:
        gucs["fusion"] = fusion
    return {
        "mode": mode,
        "reason": reason,
        "index_options": {
            "multivector_graph": "document_nodes",
        },
        "gucs": gucs,
    }


def hybrid_profile_recommendations(
    mode_results: list[dict[str, Any]],
    best_quality: dict[str, Any] | None,
    best_latency_under_floor: dict[str, Any] | None,
) -> dict[str, Any]:
    latency = min(
        mode_results,
        key=lambda item: float(item.get("latency", {}).get("p50_ms", float("inf"))),
        default=None,
    )
    high_recall = max(
        mode_results,
        key=lambda item: (
            float(item.get("exact_top10_admission_recall", 0.0)),
            float(item.get("exact_top1_admission_rate", 0.0)),
            float(item.get("metrics", {}).get("ndcg@10", 0.0)),
            -float(item.get("latency", {}).get("p50_ms", float("inf"))),
        ),
        default=None,
    )
    return {
        "latency": hybrid_mode_recommendation(
            latency.get("mode") if latency else None,
            "lowest p50 latency across evaluated modes",
        ),
        "balanced": hybrid_mode_recommendation(
            best_latency_under_floor.get("mode") if best_latency_under_floor else None,
            "lowest p50 latency among modes within the configured relative NDCG floor",
        ),
        "quality": hybrid_mode_recommendation(
            best_quality.get("mode") if best_quality else None,
            "highest NDCG@10 with recall and admission recall as tie breakers",
        ),
        "high_recall": hybrid_mode_recommendation(
            high_recall.get("mode") if high_recall else None,
            "highest exact top-10 admission recall with top-1 admission and NDCG tie breakers",
        ),
    }


def run_hybrid_evaluation_harness(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    queries: list[QueryItem],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    modes = [
        mode.strip()
        for mode in args.hybrid_evaluation_modes.split(",")
        if mode.strip()
    ]
    unknown = sorted(set(modes) - set(SUPPORTED_HYBRID_EVALUATION_MODES))
    if unknown:
        raise ValueError(f"unknown hybrid evaluation mode(s): {', '.join(unknown)}")

    document_index_args = clone_args(args, multivector_graph="document_nodes", reuse_index=False)
    index_phase = build_index(conn, document_index_args)
    exact_top_by_query = {
        query.query_id: exact_admission_top(conn, query, args.admission_k)
        for query in queries
    }

    mode_results: list[dict[str, Any]] = []
    for mode in modes:
        mode_args = hybrid_evaluation_mode_args(args, mode)
        set_hybrid_mode_gucs(conn, mode, mode_args)
        run: dict[str, list[str]] = {}
        latencies: list[float] = []
        scan_records: list[dict[str, Any]] = []
        top1_admissions: list[bool] = []
        top10_admissions: list[float] = []
        inferred_admissions = 0
        candidate_failures: dict[str, list[str]] = {}

        for query in queries:
            started = time.perf_counter()
            docs = run_hybrid_mode_query(conn, mode, query, mode_args, args.final_k)
            latency_ms = (time.perf_counter() - started) * 1000.0
            stats = last_scan_stats(conn) if mode != "exact_scan" else {}
            admission = admission_from_trace_or_results(
                exact_top_by_query.get(query.query_id, []),
                docs,
                stats,
                candidate_source=mode_args.multivector_candidate_source,
                graph_mode=mode_args.multivector_graph,
                plain_fallback=mode_args.multivector_plain_fallback,
                exact_scan=(mode == "exact_scan"),
            )
            run[query.query_id] = docs
            latencies.append(latency_ms)
            scan_records.append(hybrid_scan_record(stats) if stats else {})
            top1_admissions.append(bool(admission["exact_top1_admission"]))
            top10_admissions.append(float(admission["exact_top10_admission_recall"]))
            if admission.get("admission_inferred_from_result_docs"):
                inferred_admissions += 1
            failures = admission.get("candidate_failures", [])
            if isinstance(failures, list) and failures:
                candidate_failures[query.query_id] = [str(item) for item in failures]

        metrics = method_metrics(run, qrels, args.final_k, args.quality_k) if qrels else {}
        mode_results.append({
            "mode": mode,
            "metrics": metrics,
            "latency": summarize_ms(latencies),
            "exact_top1_admission_rate": round(sum(1 for item in top1_admissions if item) / len(top1_admissions), 6)
            if top1_admissions else 0.0,
            "exact_top10_admission_recall": round(statistics.mean(top10_admissions), 6)
            if top10_admissions else 0.0,
            "scan_work": summarize_hybrid_scan_records(scan_records),
            "admission_inferred_from_result_docs_queries": inferred_admissions,
            "candidate_admission_failures": candidate_failures,
            "failure_query_count": len(candidate_failures),
            "top10_by_query": {qid: docs[:10] for qid, docs in run.items()},
            "first_scan": scan_records[0] if scan_records else {},
            "index_stats": index_phase.get("index_stats", {}),
        })

    best_quality = max(
        mode_results,
        key=lambda item: (
            float(item.get("metrics", {}).get("ndcg@10", 0.0)),
            float(item.get("metrics", {}).get("recall@10", 0.0)),
            float(item.get("exact_top10_admission_recall", 0.0)),
        ),
        default=None,
    )
    best_ndcg = float(best_quality.get("metrics", {}).get("ndcg@10", 0.0)) if best_quality else 0.0
    floor = best_ndcg * float(args.hybrid_evaluation_quality_floor)
    eligible = [
        item for item in mode_results
        if float(item.get("metrics", {}).get("ndcg@10", 0.0)) >= floor
    ] or mode_results
    best_latency = min(
        eligible,
        key=lambda item: float(item.get("latency", {}).get("p50_ms", float("inf"))),
        default=None,
    )
    profile_recommendations = hybrid_profile_recommendations(
        mode_results,
        best_quality,
        best_latency,
    )
    return {
        "enabled": True,
        "rebuilds_index": True,
        "modes": modes,
        "queries": len(queries),
        "quality_floor_relative_to_best_ndcg": args.hybrid_evaluation_quality_floor,
        "index_phase": index_phase,
        "results": mode_results,
        "best_quality_mode": best_quality.get("mode") if best_quality else None,
        "best_latency_under_quality_floor_mode": best_latency.get("mode") if best_latency else None,
        "recommended_default_profile": profile_recommendations["balanced"],
        "profile_recommendations": profile_recommendations,
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


def scan_stat_samples(scan_stats: list[dict[str, Any]], key: str) -> list[int]:
    return [scan_stat_int(stats, key) for stats in scan_stats if key in stats]


def max_scan_work_value(scan_stats: list[dict[str, Any]], keys: tuple[str, ...]) -> int:
    values: list[int] = []
    for stats in scan_stats:
        values.append(max(scan_stat_int(stats, key) for key in keys))
    return max(values, default=0)


def parallel_safety_preflight(
    args: argparse.Namespace,
    method: str,
    serial_latencies: list[float],
    serial_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a skip decision when the serial probe is already pathological."""
    decision: dict[str, Any] = {
        "enabled": args.parallel_safety_preflight != "off",
        "forced": bool(args.force_parallel_retrieval),
        "method": method,
        "skipped": False,
        "reasons": [],
        "thresholds": {
            "max_serial_p95_ms": args.parallel_safety_max_serial_p95_ms,
            "max_docs_scored": args.parallel_safety_max_docs_scored,
            "max_exact_rerank_docs": args.parallel_safety_max_exact_rerank_docs,
            "max_exact_pairs": args.parallel_safety_max_exact_pairs,
            "max_sidecar_bytes": args.parallel_safety_max_sidecar_bytes,
        },
        "observed": {
            "serial_latency": summarize_ms(serial_latencies),
            "docs_scored_max": max(scan_docs_scored(stats) for stats in serial_stats) if serial_stats else 0,
            "exact_rerank_docs_max": max(scan_exact_rerank_docs(stats) for stats in serial_stats) if serial_stats else 0,
            "exact_pairs_max": max_scan_work_value(serial_stats, ("multivector_exact_rerank_pairs",)),
            "sidecar_bytes_max": max_scan_work_value(serial_stats, ("multivector_doc_sidecar_bytes_touched",)),
        },
    }
    if args.parallel_safety_preflight == "off" or args.force_parallel_retrieval:
        return decision

    reasons: list[str] = []
    observed = decision["observed"]
    latency = observed.get("serial_latency", {})
    p95_ms = float(latency.get("p95_ms", 0.0)) if isinstance(latency, dict) else 0.0
    if p95_ms > args.parallel_safety_max_serial_p95_ms:
        reasons.append(
            "serial p95 {observed:.3f} ms exceeds {limit:.3f} ms".format(
                observed=p95_ms,
                limit=args.parallel_safety_max_serial_p95_ms,
            )
        )
    if int(observed["docs_scored_max"]) > args.parallel_safety_max_docs_scored:
        reasons.append(
            "docs scored {observed} exceeds {limit}".format(
                observed=observed["docs_scored_max"],
                limit=args.parallel_safety_max_docs_scored,
            )
        )
    if int(observed["exact_rerank_docs_max"]) > args.parallel_safety_max_exact_rerank_docs:
        reasons.append(
            "exact rerank docs {observed} exceeds {limit}".format(
                observed=observed["exact_rerank_docs_max"],
                limit=args.parallel_safety_max_exact_rerank_docs,
            )
        )
    if int(observed["exact_pairs_max"]) > args.parallel_safety_max_exact_pairs:
        reasons.append(
            "exact MaxSim pairs {observed} exceeds {limit}".format(
                observed=observed["exact_pairs_max"],
                limit=args.parallel_safety_max_exact_pairs,
            )
        )
    if int(observed["sidecar_bytes_max"]) > args.parallel_safety_max_sidecar_bytes:
        reasons.append(
            "sidecar bytes {observed} exceeds {limit}".format(
                observed=observed["sidecar_bytes_max"],
                limit=args.parallel_safety_max_sidecar_bytes,
            )
        )

    if reasons:
        decision["skipped"] = True
        decision["reasons"] = reasons
    return decision


def run_parallel_retrieval_report(
    args: argparse.Namespace,
    method: str,
    queries: list[QueryItem],
    serial_latencies: list[float],
    serial_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    preflight = parallel_safety_preflight(args, method, serial_latencies, serial_stats)
    if args.skip_parallel_retrieval:
        return {
            "clients": args.clients,
            "warm_queries_per_client": args.warm_queries,
            "timed_queries_per_client": args.timed_queries,
            "total_timed_queries": 0,
            "failed": False,
            "skipped": True,
            "skip_reason": "disabled by --skip-parallel-retrieval",
            "safety_preflight": preflight,
        }
    if preflight["skipped"]:
        return {
            "clients": args.clients,
            "warm_queries_per_client": args.warm_queries,
            "timed_queries_per_client": args.timed_queries,
            "total_timed_queries": 0,
            "failed": False,
            "skipped": True,
            "skip_reason": "; ".join(preflight["reasons"]),
            "safety_preflight": preflight,
        }
    try:
        parallel = run_parallel_retrieval(args, method, queries)
    except Exception as exc:  # pragma: no cover - host benchmark failure path
        return {
            "clients": args.clients,
            "warm_queries_per_client": args.warm_queries,
            "timed_queries_per_client": args.timed_queries,
            "total_timed_queries": 0,
            "failed": True,
            "skipped": False,
            "error": str(exc),
            "safety_preflight": preflight,
        }

    parallel["failed"] = False
    parallel["skipped"] = False
    parallel["safety_preflight"] = preflight
    return parallel


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
                multivector_doc_build_scorer = proxy,
                exact_storage = off,
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
        not trace_by_heap
        and can_infer_admission_from_result_docs(
            mode["candidate_source"],
            plain_fallback=mode["plain_fallback"],
            graph_mode=mode.get("graph_mode"),
        )
    )
    exact_with_admission: list[dict[str, Any]] = []
    for exact in exact_top:
        key = (int(exact["heap_block"]), int(exact["heap_offset"]))
        entry = trace_by_heap.get(key)
        result_rank = docs.index(exact["doc_id"]) + 1 if exact["doc_id"] in docs else None
        admitted = entry is not None or (inferred_exact_doc_admission and result_rank is not None)
        admission_evidence = (
            "trace"
            if entry is not None
            else "result_doc"
            if inferred_exact_doc_admission and result_rank is not None
            else "missing"
        )
        exact_with_admission.append({
            "rank": exact["rank"],
            "doc_id": exact["doc_id"],
            "admitted_before_rerank": admitted,
            "admission_evidence": admission_evidence,
            "exact_rerank_rank": result_rank,
            "candidate_rank_before_rerank": (
                int(entry["candidate_rank_before_truncation"])
                if isinstance(entry, dict) and entry.get("candidate_rank_before_truncation") is not None
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
        **exact_rerank_work_from_stats(stats),
        **proxy_work_from_stats(stats),
        "memory_bytes_estimate": memory_estimate_from_stats(stats),
        "raw_subvector_hits": scan_stat_int(stats, "multivector_raw_subvector_hits"),
        "unique_docs": scan_stat_int(stats, "multivector_unique_docs"),
        "trace_entries": len(trace_entries),
        "admission_inferred_from_result_docs": inferred_exact_doc_admission,
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
        "`doc_graph_prototype`, `document_nodes`, and `proxy_vector` must return "
        "and admit the synthetic exact top-1 document at the configured small candidate "
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
        f"- Multivector graph: `{settings.get('multivector_graph', '')}`",
        f"- Candidate source: `{settings.get('multivector_candidate_source', '')}`",
        f"- Sparse candidate source: `{settings.get('multivector_sparse_candidate_source', '')}`",
        f"- Token pooling: `{settings.get('multivector_token_pooling', '')}`",
        f"- Token pooling target ratio: `{settings.get('multivector_token_pooling_target_ratio', '')}`",
        f"- Centroids: `{settings.get('multivector_centroids', '')}`",
        f"- Centroid count: `{settings.get('multivector_centroid_count', '')}`",
        f"- Proxy encoder: `{settings.get('multivector_proxy_encoder', '')}`",
        f"- Doc graph search EF: `{settings.get('multivector_doc_graph_search_ef', 0)}`",
        f"- Doc graph oversampling: `{settings.get('multivector_doc_graph_oversampling', 1)}`",
        f"- Doc graph rescore K: `{settings.get('multivector_doc_graph_rescore_k', 0)}`",
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
    skipped_parallel = [
        item
        for item in report.get("results", [])
        if isinstance(item, dict)
        and isinstance(item.get("parallel_8x"), dict)
        and item["parallel_8x"].get("skipped") is True
    ]
    failed_parallel = [
        item
        for item in report.get("results", [])
        if isinstance(item, dict)
        and isinstance(item.get("parallel_8x"), dict)
        and item["parallel_8x"].get("failed") is True
    ]
    if skipped_parallel or failed_parallel:
        lines.extend([
            "",
            "### Parallel Retrieval Notes",
            "",
        ])
        for item in skipped_parallel:
            parallel = item["parallel_8x"]
            lines.append(
                "- `{method}` skipped 8x retrieval: {reason}".format(
                    method=item.get("method", ""),
                    reason=parallel.get("skip_reason", ""),
                )
            )
        for item in failed_parallel:
            parallel = item["parallel_8x"]
            lines.append(
                "- `{method}` failed 8x retrieval: {error}".format(
                    method=item.get("method", ""),
                    error=parallel.get("error", ""),
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
            f"- Sparse candidate source: `{admission.get('sparse_candidate_source', '')}`",
            f"- Plain fallback: `{admission.get('plain_fallback', '')}`",
            f"- Reservoirs: `{admission.get('candidate_reservoirs', '')}`",
            f"- Admission K: `{aggregate.get('admission_k', 0)}`",
            f"- Exact top-1 admission rate: `{float(aggregate.get('exact_top1_admission_rate', 0.0)):.6f}`",
            f"- Exact top-10 admission recall: `{float(aggregate.get('exact_top10_admission_recall', 0.0)):.6f}`",
            f"- First admitted budget p50: `{first_budget.get('p50', 0)}`",
            "",
            "| budget | top1 admission | top10 admission recall | latency p50 ms | latency p95 ms | docs scored p50 | graph edges p50 | exact rerank docs p50 | pairs saved p50 | runs |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        admission_by_budget = aggregate.get("admission_by_budget", {})
        if isinstance(admission_by_budget, dict):
            for budget, budget_stats in sorted(admission_by_budget.items(), key=lambda item: int(item[0])):
                if not isinstance(budget_stats, dict):
                    continue
                latency = budget_stats.get("latency", {})
                docs_scored = budget_stats.get("docs_scored", {})
                graph_edges = budget_stats.get("graph_edges_visited", {})
                exact_rerank_docs = budget_stats.get("exact_rerank_docs", {})
                exact_rerank_pairs_saved = budget_stats.get("exact_rerank_pairs_saved", {})
                if not isinstance(latency, dict):
                    latency = {}
                if not isinstance(docs_scored, dict):
                    docs_scored = {}
                if not isinstance(graph_edges, dict):
                    graph_edges = {}
                if not isinstance(exact_rerank_docs, dict):
                    exact_rerank_docs = {}
                if not isinstance(exact_rerank_pairs_saved, dict):
                    exact_rerank_pairs_saved = {}
                lines.append(
                    "| {budget} | {top1:.6f} | {top10:.6f} | {p50:.3f} | {p95:.3f} | {docs_p50:.3f} | {edges_p50:.3f} | {rerank_p50:.3f} | {pairs_saved_p50:.3f} | {runs} |".format(
                        budget=budget,
                        top1=float(budget_stats.get("exact_top1_admission", 0.0)),
                        top10=float(budget_stats.get("exact_top10_admission_recall", 0.0)),
                        p50=float(latency.get("p50_ms", 0.0)),
                        p95=float(latency.get("p95_ms", 0.0)),
                        docs_p50=float(docs_scored.get("p50", 0.0)),
                        edges_p50=float(graph_edges.get("p50", 0.0)),
                        rerank_p50=float(exact_rerank_docs.get("p50", 0.0)),
                        pairs_saved_p50=float(exact_rerank_pairs_saved.get("p50", 0.0)),
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

    admission_grid = report.get("document_node_admission_grid")
    if isinstance(admission_grid, dict):
        pooling_grid_text = ",".join(
            "{mode}:{ratio}".format(mode=item.get("mode"), ratio=item.get("ratio"))
            for item in admission_grid.get("pooling_grid", [])
            if isinstance(item, dict)
        )
        lines.extend([
            "",
            "### Document-Node Admission Grid",
            "",
            f"- Rebuilds index: `{admission_grid.get('rebuilds_index', False)}`",
            f"- Budget sweep: `{','.join(str(item) for item in admission_grid.get('budget_sweep', []))}`",
            f"- Storage grid: `{','.join(str(item) for item in admission_grid.get('storage_grid', []))}`",
            f"- Pooling grid: `{pooling_grid_text}`",
            f"- EF grid: `{','.join(str(item) for item in admission_grid.get('ef_grid', []))}`",
            f"- Oversampling grid: `{','.join(str(item) for item in admission_grid.get('oversampling_grid', []))}`",
            f"- Index graph build: `m={admission_grid.get('index_graph_m', 0)}, ef_construction={admission_grid.get('index_graph_ef_construction', 0)}, ef_search={admission_grid.get('index_graph_ef_search', 0)}, native_segments={admission_grid.get('index_native_segments', 0)}`",
            "",
            "| mode | graph | storage | pooling | ratio | ef | oversampling | budget | top1 admission | top10 admission recall | recall@10 | ndcg@10 | mrr@10 | p50 ms | p95 ms | docs scored p50 | edges p50 | exact rerank p50 | native exact bytes p50 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        grid_results = admission_grid.get("results", [])
        if isinstance(grid_results, list):
            for item in grid_results:
                if not isinstance(item, dict):
                    continue
                budget = item.get("largest_budget")
                admission_by_budget = item.get("admission_by_budget", {})
                budget_stats = {}
                if isinstance(admission_by_budget, dict) and budget is not None:
                    budget_stats = admission_by_budget.get(str(budget), {})
                    if not isinstance(budget_stats, dict):
                        budget_stats = {}
                latency = budget_stats.get("latency", {}) if isinstance(budget_stats, dict) else {}
                docs_scored = budget_stats.get("docs_scored", {}) if isinstance(budget_stats, dict) else {}
                graph_edges = budget_stats.get("graph_edges_visited", {}) if isinstance(budget_stats, dict) else {}
                exact_rerank_docs = budget_stats.get("exact_rerank_docs", {}) if isinstance(budget_stats, dict) else {}
                native_cache_exact = budget_stats.get("native_cache_exact_bytes", {}) if isinstance(budget_stats, dict) else {}
                metrics = item.get("metrics", {})
                if not isinstance(latency, dict):
                    latency = {}
                if not isinstance(docs_scored, dict):
                    docs_scored = {}
                if not isinstance(graph_edges, dict):
                    graph_edges = {}
                if not isinstance(exact_rerank_docs, dict):
                    exact_rerank_docs = {}
                if not isinstance(native_cache_exact, dict):
                    native_cache_exact = {}
                if not isinstance(metrics, dict):
                    metrics = {}
                lines.append(
                    "| {mode} | {graph} | {storage} | {pooling} | {ratio:.2f} | {ef} | {oversampling} | {budget} | {top1:.6f} | {top10:.6f} | {recall:.6f} | {ndcg:.6f} | {mrr:.6f} | {p50:.3f} | {p95:.3f} | {docs:.3f} | {edges:.3f} | {rerank:.3f} | {native_exact:.3f} |".format(
                        mode=item.get("mode", ""),
                        graph=item.get("graph_mode", ""),
                        storage=item.get("storage_kind", ""),
                        pooling=item.get("token_pooling", ""),
                        ratio=float(item.get("token_pooling_target_ratio", 1.0) or 1.0),
                        ef=int(item.get("doc_graph_search_ef", 0) or 0),
                        oversampling=int(item.get("doc_graph_oversampling", 0) or 0),
                        budget=budget or "",
                        top1=float(item.get("exact_top1_admission_rate", 0.0)),
                        top10=float(item.get("exact_top10_admission_recall", 0.0)),
                        recall=float(metrics.get("recall@10", 0.0)),
                        ndcg=float(metrics.get("ndcg@10", 0.0)),
                        mrr=float(metrics.get("mrr@10", 0.0)),
                        p50=float(latency.get("p50_ms", 0.0)),
                        p95=float(latency.get("p95_ms", 0.0)),
                        docs=float(docs_scored.get("p50", 0.0)),
                        edges=float(graph_edges.get("p50", 0.0)),
                        rerank=float(exact_rerank_docs.get("p50", 0.0)),
                        native_exact=float(native_cache_exact.get("p50", 0.0)),
                    )
                )
        lines.extend([
            "",
            "The grid rebuilds token-node and document-node index layouts so the "
            "report can compare token baselines with document-node admission. "
            "Document-sidecar page/cache counters are emitted when the engine "
            "exposes them; until then the table includes native cache exact-byte "
            "counters as the available storage footprint signal.",
        ])

    serving_grid = report.get("document_node_serving_grid")
    if isinstance(serving_grid, dict):
        rows = serving_grid.get("results", serving_grid.get("summary_rows", []))
        if isinstance(rows, list):
            sorted_rows = sorted(
                [item for item in rows if isinstance(item, dict)],
                key=lambda item: (
                    serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
                    str(item.get("profile", "")),
                    int(item.get("ef", 0) or 0),
                    int(item.get("oversampling", 0) or 0),
                ),
            )
        else:
            sorted_rows = []
        lines.extend([
            "",
            "### Document-node serving grid",
            "",
            "| profile | source | graph | proxy | centroids | storage | cache | pooling | ef | oversampling | budget | top1 admission | top10 admission | recall@10 | ndcg@10 | mrr@10 | p50 ms | p95 ms | index bytes |",
            "|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for item in sorted_rows:
            lines.append(
                "| {profile} | {source} | {graph} | {proxy} | {centroids} | {storage} | {cache} | {pooling}:{ratio:.2f} | {ef} | {oversampling} | {budget} | {top1:.6f} | {top10:.6f} | {recall:.6f} | {ndcg:.6f} | {mrr:.6f} | {p50:.3f} | {p95:.3f} | {index_bytes} |".format(
                    profile=item.get("profile", ""),
                    source=item.get("candidate_source", ""),
                    graph=item.get("graph_mode", ""),
                    proxy=item.get("proxy_encoder", ""),
                    centroids=item.get("centroids", ""),
                    storage=item.get("storage_kind", ""),
                    cache=item.get("storage_cache_mode", item.get("cache_mode", "")),
                    pooling=item.get("token_pooling", ""),
                    ratio=float(item.get("token_pooling_target_ratio", 1.0) or 1.0),
                    ef=int(item.get("ef", 0) or 0),
                    oversampling=int(item.get("oversampling", 0) or 0),
                    budget=int(item.get("largest_budget", 0) or 0),
                    top1=float(item.get("exact_top1_admission_rate", 0.0) or 0.0),
                    top10=float(item.get("exact_top10_admission_recall", 0.0) or 0.0),
                    recall=float(item.get("recall@10", 0.0) or 0.0),
                    ndcg=float(item.get("ndcg@10", 0.0) or 0.0),
                    mrr=float(item.get("mrr@10", 0.0) or 0.0),
                    p50=float(item.get("p50_ms", item.get("p50_latency_ms_at_largest_budget", 0.0)) or 0.0),
                    p95=float(item.get("p95_ms", item.get("p95_latency_ms_at_largest_budget", 0.0)) or 0.0),
                    index_bytes=int(item.get("index_bytes", 0) or 0),
                )
            )

    serving_rec = report.get("document_node_serving_recommendation")
    if isinstance(serving_rec, dict):
        thresholds = serving_rec.get("thresholds", {})
        best_latency_safe = serving_rec.get("best_latency_safe") or {}
        best_quality = serving_rec.get("best_quality") or {}
        best_balanced = serving_rec.get("best_balanced") or {}
        if not isinstance(thresholds, dict):
            thresholds = {}
        if not isinstance(best_latency_safe, dict):
            best_latency_safe = {}
        if not isinstance(best_quality, dict):
            best_quality = {}
        if not isinstance(best_balanced, dict):
            best_balanced = {}
        lines.extend([
            "",
            "### Document-node serving recommendation",
            "",
            f"- Minimum top-10 admission: `{float(thresholds.get('serving_min_top10_admission', 0.0)):.3f}`",
            f"- Minimum NDCG ratio vs exact: `{float(thresholds.get('serving_min_ndcg_ratio_vs_exact', 0.0)):.3f}`",
            f"- Maximum p95 ms: `{float(thresholds.get('serving_max_p95_ms', 0.0)):.3f}` (`0` means no hard cap)",
            f"- Best latency-safe: `{best_latency_safe.get('id', '')}`",
            f"- Best quality: `{best_quality.get('id', '')}`",
            f"- Best balanced: `{best_balanced.get('id', '')}`",
            f"- Why latency-safe won: {serving_profile_explanation(best_latency_safe)}",
            f"- Why quality won: {serving_profile_explanation(best_quality)}",
            f"- Why balanced won: {serving_profile_explanation(best_balanced)}",
            "",
            "| pass | profile | source | proxy | storage | pooling | ef | oversampling | budget | top10 admission | ndcg@10 | p95 ms | reasons |",
            "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        rows = serving_rec.get("profile_thresholds", [])
        if isinstance(rows, list):
            for item in rows:
                if not isinstance(item, dict):
                    continue
                failures = item.get("failure_reasons", [])
                unavailable = item.get("unavailable_criteria", [])
                reasons = []
                if isinstance(failures, list):
                    reasons.extend(str(reason) for reason in failures)
                if isinstance(unavailable, list):
                    reasons.extend(f"unavailable:{criterion}" for criterion in unavailable)
                lines.append(
                    "| {passed} | {profile} | {source} | {proxy} | {storage} | {pooling}:{ratio:.2f} | {ef} | {oversampling} | {budget} | {admission:.6f} | {ndcg:.6f} | {p95:.3f} | {reasons} |".format(
                        passed="pass" if item.get("threshold_pass") else "fail",
                        profile=item.get("profile", ""),
                        source=item.get("candidate_source", ""),
                        proxy=item.get("proxy_encoder", ""),
                        storage=item.get("storage_kind", ""),
                        pooling=item.get("token_pooling", ""),
                        ratio=float(item.get("token_pooling_target_ratio", 1.0) or 1.0),
                        ef=int(item.get("ef", 0) or 0),
                        oversampling=int(item.get("oversampling", 0) or 0),
                        budget=int(item.get("largest_budget", 0) or 0),
                        admission=float(item.get("exact_top10_admission_recall", 0.0) or 0.0),
                        ndcg=float(item.get("ndcg@10", 0.0) or 0.0),
                        p95=float(serving_row_p95(item) or 0.0),
                        reasons=", ".join(reasons) if reasons else "",
                    )
                )
        lines.extend([
            "",
            "The balanced score is stable and simple: latency rank plus admission "
            "loss, optional NDCG loss when qrels exist, and a small storage "
            "penalty (`sq8=0`, `f16=1`, `f32=2`). Experimental profiles remain "
            "opt-in and should not be baked into serving defaults; they receive "
            "a large balanced-score penalty whenever a non-experimental profile "
            "has usable metrics. Final ranking remains exact heap MaxSim unless "
            "the candidate source is explicitly experimental.",
        ])

    hybrid_eval = report.get("hybrid_evaluation")
    if isinstance(hybrid_eval, dict):
        recommended = hybrid_eval.get("recommended_default_profile", {})
        if not isinstance(recommended, dict):
            recommended = {}
        profile_recommendations = hybrid_eval.get("profile_recommendations", {})
        if not isinstance(profile_recommendations, dict):
            profile_recommendations = {}
        lines.extend([
            "",
            "### Hybrid Evaluation Harness",
            "",
            f"- Rebuilds index: `{hybrid_eval.get('rebuilds_index', False)}`",
            f"- Modes: `{','.join(str(item) for item in hybrid_eval.get('modes', []))}`",
            f"- Best quality mode: `{hybrid_eval.get('best_quality_mode', '')}`",
            f"- Best latency under quality floor: `{hybrid_eval.get('best_latency_under_quality_floor_mode', '')}`",
            f"- Recommended default mode: `{recommended.get('mode', '')}`",
            "",
            "| mode | recall@10 | ndcg@10 | mrr@10 | top1 admission | top10 admission | p50 ms | p95 ms | branch us p50 | candidates p50 | exact pairs p50 | sidecar bytes p50 | failure queries |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        results = hybrid_eval.get("results", [])
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                metrics = item.get("metrics", {})
                latency = item.get("latency", {})
                work = item.get("scan_work", {})
                if not isinstance(metrics, dict):
                    metrics = {}
                if not isinstance(latency, dict):
                    latency = {}
                if not isinstance(work, dict):
                    work = {}
                branch_latency = work.get("branch_latency_us_total", {})
                branch_candidates = work.get("branch_candidates_total", {})
                exact_pairs = work.get("exact_maxsim_pairs", {})
                sidecar_bytes = work.get("sidecar_bytes", {})
                if not isinstance(branch_latency, dict):
                    branch_latency = {}
                if not isinstance(branch_candidates, dict):
                    branch_candidates = {}
                if not isinstance(exact_pairs, dict):
                    exact_pairs = {}
                if not isinstance(sidecar_bytes, dict):
                    sidecar_bytes = {}
                lines.append(
                    "| {mode} | {recall:.6f} | {ndcg:.6f} | {mrr:.6f} | {top1:.6f} | {top10:.6f} | {p50:.3f} | {p95:.3f} | {branch_p50:.3f} | {candidates_p50:.3f} | {pairs_p50:.3f} | {sidecar_p50:.3f} | {failures} |".format(
                        mode=item.get("mode", ""),
                        recall=float(metrics.get("recall@10", 0.0)),
                        ndcg=float(metrics.get("ndcg@10", 0.0)),
                        mrr=float(metrics.get("mrr@10", 0.0)),
                        top1=float(item.get("exact_top1_admission_rate", 0.0)),
                        top10=float(item.get("exact_top10_admission_recall", 0.0)),
                        p50=float(latency.get("p50_ms", 0.0)),
                        p95=float(latency.get("p95_ms", 0.0)),
                        branch_p50=float(branch_latency.get("p50", 0.0)),
                        candidates_p50=float(branch_candidates.get("p50", 0.0)),
                        pairs_p50=float(exact_pairs.get("p50", 0.0)),
                        sidecar_p50=float(sidecar_bytes.get("p50", 0.0)),
                        failures=int(item.get("failure_query_count", 0)),
                    )
                )
        if profile_recommendations:
            lines.extend([
                "",
                "| profile | mode | reason |",
                "|---|---|---|",
            ])
            for profile in ("latency", "balanced", "quality", "high_recall"):
                item = profile_recommendations.get(profile, {})
                if not isinstance(item, dict):
                    item = {}
                lines.append(
                    "| {profile} | {mode} | {reason} |".format(
                        profile=profile,
                        mode=item.get("mode", ""),
                        reason=item.get("reason", ""),
                    )
                )
        lines.extend([
            "",
            "The harness compares exact scan, document-node retrieval, BM25 "
            "admission-only, BM25 RRF, BM25 DBSF, proxy-vector document-node "
            "prefetch, and learned-sparse admission. Candidate admission failures "
            "list exact top-K documents not seen before final rerank, so a zero or "
            "low admission row is a candidate-generation failure, not a final-ranker "
            "success.",
        ])

    lines.append("")
    return "\n".join(lines)


def run_self_checks() -> None:
    _self_check_document_node_serving_profiles()
    _self_check_document_node_serving_recommendation()
    _self_check_document_node_serving_recommendation_schema_fixture()
    _self_check_document_node_serving_stats_extraction()
    _self_check_document_node_serving_grid_serialization()


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
        {"name": "document_nodes", "graph_mode": "document_nodes", "candidate_source": "document_nodes", "plain_fallback": "off", "reservoirs": "off"},
        {"name": "proxy_vector", "graph_mode": "document_nodes", "candidate_source": "proxy_vector", "plain_fallback": "off", "reservoirs": "off"},
    ]
    results = [run_synthetic_exact_scan(conn, args.final_k)]
    for mode in modes:
        results.append(run_synthetic_gate_mode(conn, mode, exact_top, args.recall_gate_budget, args.final_k))

    required_modes = {"exact_scan", "plain_fallback", "exact_doc_scan", "doc_graph_prototype", "document_nodes", "proxy_vector"}
    passed = all(
        item["mode"] not in required_modes
        or (item.get("top1") == "good" and item.get("exact_top1_admitted_before_rerank") is True)
        for item in results
    )
    report = {
        "suite": "multivector_recall_gate",
        "layer": "ir_quality_and_systems",
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
    require_complete_learned_sparse_args(args)
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
    if args.learned_sparse_doc_jsonl is not None:
        args.learned_sparse_doc_jsonl = args.learned_sparse_doc_jsonl.resolve()
        if not args.learned_sparse_doc_jsonl.is_file():
            raise SystemExit(f"learned sparse document JSONL does not exist: {args.learned_sparse_doc_jsonl}")
    if args.learned_sparse_query_jsonl is not None:
        args.learned_sparse_query_jsonl = args.learned_sparse_query_jsonl.resolve()
        if not args.learned_sparse_query_jsonl.is_file():
            raise SystemExit(f"learned sparse query JSONL does not exist: {args.learned_sparse_query_jsonl}")

    args.model_path = args.model_path.resolve()
    if args.precomputed_dataset is None and not args.model_path.is_file() and not args.multivector_recall_gate:
        raise SystemExit(
            f"model file does not exist: {args.model_path}\n"
            "Use johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF and pass --model-path."
        )
    if args.model_alias is None:
        args.model_alias = args.model_path.stem
    args.expected_dim_requested = str(args.expected_dim)
    args.expected_dim, args.expected_dim_source = resolve_expected_dim(
        args.expected_dim_requested,
        args.colbert_model_name,
    )
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
    if args.clients < 1:
        raise SystemExit("--clients must be at least 1")
    if args.warm_queries < 0:
        raise SystemExit("--warm-queries must be non-negative")
    if args.timed_queries < 0:
        raise SystemExit("--timed-queries must be non-negative")
    if args.parallel_safety_max_serial_p95_ms < 0:
        raise SystemExit("--parallel-safety-max-serial-p95-ms must be non-negative")
    if args.parallel_safety_max_docs_scored < 0:
        raise SystemExit("--parallel-safety-max-docs-scored must be non-negative")
    if args.parallel_safety_max_exact_rerank_docs < 0:
        raise SystemExit("--parallel-safety-max-exact-rerank-docs must be non-negative")
    if args.parallel_safety_max_exact_pairs < 0:
        raise SystemExit("--parallel-safety-max-exact-pairs must be non-negative")
    if args.parallel_safety_max_sidecar_bytes < 0:
        raise SystemExit("--parallel-safety-max-sidecar-bytes must be non-negative")
    if args.force_parallel_retrieval and args.skip_parallel_retrieval:
        raise SystemExit("--force-parallel-retrieval and --skip-parallel-retrieval are mutually exclusive")
    if args.admission_k < 1:
        raise SystemExit("--admission-k must be at least 1")
    if args.admission_trace_limit < 0:
        raise SystemExit("--admission-trace-limit must be non-negative")
    if args.multivector_plain_fallback_max_docs < 0:
        raise SystemExit("--multivector-plain-fallback-max-docs must be non-negative")
    if not 0.0 <= args.multivector_plain_fallback_candidate_fraction <= 1.0:
        raise SystemExit("--multivector-plain-fallback-candidate-fraction must be between 0 and 1")
    if args.multivector_doc_graph_search_ef < 0:
        raise SystemExit("--multivector-doc-graph-search-ef must be non-negative")
    if args.multivector_doc_graph_oversampling < 1:
        raise SystemExit("--multivector-doc-graph-oversampling must be at least 1")
    if args.multivector_doc_graph_rescore_k < 0:
        raise SystemExit("--multivector-doc-graph-rescore-k must be non-negative")
    if args.index_graph_m < 0:
        raise SystemExit("--index-graph-m must be non-negative")
    if args.index_graph_ef_construction < 0:
        raise SystemExit("--index-graph-ef-construction must be non-negative")
    if args.index_graph_ef_search < 0:
        raise SystemExit("--index-graph-ef-search must be non-negative")
    if args.index_native_segments < 0:
        raise SystemExit("--index-native-segments must be non-negative")
    if args.multivector_doc_build_scorer == "exact_symmetric":
        if not args.allow_exact_symmetric_build:
            raise SystemExit(
                "--multivector-doc-build-scorer exact_symmetric requires "
                "--allow-exact-symmetric-build"
            )
        print(
            "WARNING: exact_symmetric document-node graph builds compute exact "
            "document-document MaxSim inside topology construction; cost is "
            "O(distance_calls * La * Lb * d). Use proxy for DBpedia benchmark runs.",
            file=sys.stderr,
        )
    if not 0.0 < args.multivector_token_pooling_target_ratio <= 1.0:
        raise SystemExit("--multivector-token-pooling-target-ratio must be in (0, 1]")
    if args.multivector_token_pooling_min_tokens < 1:
        raise SystemExit("--multivector-token-pooling-min-tokens must be positive")
    if args.multivector_per_token_doc_reservoir_k < 0:
        raise SystemExit("--multivector-per-token-doc-reservoir-k must be non-negative")
    if args.multivector_coverage_reservoir_k < 0:
        raise SystemExit("--multivector-coverage-reservoir-k must be non-negative")
    if args.admission_budget_sweep is None:
        args.admission_budget_sweep = (
            DOCUMENT_NODE_SERVING_GRID_BUDGET_SWEEP
            if args.document_node_serving_grid
            else DEFAULT_ADMISSION_BUDGET_SWEEP
        )
    try:
        admission_budgets = [
            int(item.strip())
            for item in args.admission_budget_sweep.split(",")
            if item.strip()
        ]
    except ValueError as exc:
        raise SystemExit("--admission-budget-sweep must contain comma-separated integers") from exc
    if (
        args.admission_debug
        or args.document_node_admission_grid
        or args.document_node_serving_grid
        or args.hybrid_evaluation_harness
    ) and (
        not admission_budgets or any(budget < 1 for budget in admission_budgets)
    ):
        raise SystemExit("--admission-budget-sweep must contain positive integer budgets")
    if not 0.0 < args.hybrid_evaluation_quality_floor <= 1.0:
        raise SystemExit("--hybrid-evaluation-quality-floor must be in (0, 1]")
    if args.hybrid_evaluation_dbsf_min_branch_candidates < 1:
        raise SystemExit("--hybrid-evaluation-dbsf-min-branch-candidates must be positive")
    if not 0.0 <= args.serving_min_top10_admission <= 1.0:
        raise SystemExit("--serving-min-top10-admission must be in [0, 1]")
    if args.serving_min_ndcg_ratio_vs_exact < 0.0:
        raise SystemExit("--serving-min-ndcg-ratio-vs-exact must be non-negative")
    if args.serving_max_p95_ms < 0.0:
        raise SystemExit("--serving-max-p95-ms must be non-negative")
    hybrid_modes = [
        mode.strip()
        for mode in args.hybrid_evaluation_modes.split(",")
        if mode.strip()
    ]
    unknown_hybrid_modes = sorted(set(hybrid_modes) - set(SUPPORTED_HYBRID_EVALUATION_MODES))
    if unknown_hybrid_modes:
        raise SystemExit(
            "--hybrid-evaluation-modes contains unsupported value(s): "
            + ", ".join(unknown_hybrid_modes)
        )
    args.hybrid_evaluation_mode_values = hybrid_modes
    args.document_node_storage_grid_values = parse_choice_grid(
        args.document_node_storage_grid,
        DOCUMENT_NODE_STORAGE_CHOICES,
        "--document-node-storage-grid",
    )
    args.document_node_cache_grid_values = parse_choice_grid(
        args.document_node_cache_grid,
        DOCUMENT_NODE_STORAGE_CACHE_CHOICES,
        "--document-node-cache-grid",
    )
    args.document_node_ef_grid_values = parse_int_grid(
        args.document_node_ef_grid,
        "--document-node-ef-grid",
    )
    args.document_node_oversampling_grid_values = parse_int_grid(
        args.document_node_oversampling_grid,
        "--document-node-oversampling-grid",
    )
    args.document_node_pooling_grid_values = parse_document_node_pooling_grid(
        args.document_node_pooling_grid,
    )
    args.document_node_proxy_encoder_grid_values = parse_choice_grid(
        args.document_node_proxy_encoder_grid,
        tuple(choice for choice in MULTIVECTOR_PROXY_ENCODER_CHOICES if choice != "learned_projection_placeholder"),
        "--document-node-proxy-encoder-grid",
    )
    if args.recall_gate_budget < 1:
        raise SystemExit("--recall-gate-budget must be positive")
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run pure Python benchmark report construction self-checks and exit",
    )
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
    parser.add_argument(
        "--colbert-model-name",
        default=DEFAULT_COLBERT_MODEL_NAME,
        help="late-interaction model profile name for dimension auto-detection and benchmark provenance",
    )
    parser.add_argument("--expected-dim", default="auto", help="expected embedding dimension or 'auto'")
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
    parser.add_argument("--multivector-exact-rerank", choices=("off", "topk", "adaptive"), default="adaptive")
    parser.add_argument("--multivector-exact-rerank-k", type=int, default=100)
    parser.add_argument(
        "--multivector-graph",
        choices=("token_nodes", "document_nodes"),
        default="document_nodes",
        help="turbohybrid multivector graph layout used for the DBpedia ColBERT index",
    )
    parser.add_argument(
        "--multivector-doc-build-scorer",
        choices=MULTIVECTOR_DOC_BUILD_SCORER_CHOICES,
        default="proxy",
        help=(
            "document_nodes graph-build distance scorer; proxy is the production-safe "
            "default, exact_symmetric is a diagnostic-only O(distance_calls * La * Lb * d) path"
        ),
    )
    parser.add_argument(
        "--allow-exact-symmetric-build",
        action="store_true",
        help="allow diagnostic exact_symmetric document-document MaxSim graph topology builds",
    )
    parser.add_argument(
        "--multivector-doc-graph-search-ef",
        type=int,
        default=0,
        help="document-node graph traversal search_ef override; 0 uses the index graph_ef_search",
    )
    parser.add_argument(
        "--multivector-doc-graph-oversampling",
        type=int,
        default=1,
        help="candidate oversampling multiplier for document-node graph traversal",
    )
    parser.add_argument(
        "--multivector-doc-graph-rescore-k",
        type=int,
        default=0,
        help="document-node candidate rescore budget; 0 follows multivector_doc_candidate_k",
    )
    parser.add_argument(
        "--index-graph-m",
        type=int,
        default=0,
        help="turbohybrid index graph_m reloption; 0 uses the extension default",
    )
    parser.add_argument(
        "--index-graph-ef-construction",
        type=int,
        default=0,
        help="turbohybrid index graph_ef_construction reloption; 0 uses the extension default",
    )
    parser.add_argument(
        "--index-graph-ef-search",
        type=int,
        default=0,
        help="turbohybrid index graph_ef_search reloption; 0 uses the extension default",
    )
    parser.add_argument(
        "--index-native-segments",
        type=int,
        default=0,
        help="turbohybrid index native_segments reloption; 0 uses the extension default",
    )
    parser.add_argument(
        "--multivector-doc-storage",
        choices=("f32", "f16", "sq8"),
        default="f32",
        help="document-node MaxSim sidecar scoring storage for graph traversal",
    )
    parser.add_argument(
        "--multivector-doc-storage-cache",
        choices=("auto", "resident", "paged"),
        default="auto",
        help="document-node sidecar cache mode; paged keeps graph adjacency resident and reports sidecar page reads per scan",
    )
    parser.add_argument(
        "--multivector-token-pooling",
        choices=DOCUMENT_NODE_TOKEN_POOLING_CHOICES,
        default="off",
        help="document-node index-time token pooling mode",
    )
    parser.add_argument(
        "--multivector-token-pooling-target-ratio",
        type=float,
        default=0.5,
        help="target pooled/original document-token ratio when token pooling is enabled",
    )
    parser.add_argument(
        "--multivector-token-pooling-min-tokens",
        type=int,
        default=16,
        help="minimum document token count before index-time token pooling is applied",
    )
    parser.add_argument(
        "--multivector-proxy-encoder",
        choices=MULTIVECTOR_PROXY_ENCODER_CHOICES,
        default="normalized_mean",
        help="document-node proxy_vector fixed-dimensional encoder used at index build and query time",
    )
    parser.add_argument(
        "--multivector-centroids",
        choices=DOCUMENT_NODE_CENTROIDS_CHOICES,
        default="off",
        help="experimental PLAID-lite centroid mode persisted as an index reloption",
    )
    parser.add_argument(
        "--multivector-centroid-count",
        default="auto",
        help="per-document centroid count for centroid_lite; 'auto' maps to the extension default",
    )
    parser.add_argument(
        "--multivector-candidate-source",
        choices=("graph", "document_nodes", "exact_token_scan", "exact_doc_scan", "doc_graph_prototype", "proxy_vector", "centroid_lite", "quantized_inverted_experimental"),
        default="graph",
        help="multivector candidate source for query-only/RRF retrieval; document_nodes/proxy_vector require multivector_graph=document_nodes; centroid_lite also supports token_nodes compatibility indexes built with multivector_centroids=kmeans; quantized_inverted_experimental is a guarded research-only document-node branch using persisted experimental codeword postings before exact MaxSim rerank",
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
        "--multivector-sparse-candidate-source",
        choices=MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_CHOICES,
        default="off",
        help=(
            "explicit sparse candidate source for multivector admission; "
            "learned_sparse reuses exported sparse postings and exact-MaxSim "
            "reranks admitted documents"
        ),
    )
    parser.add_argument(
        "--learned-sparse-doc-jsonl",
        type=Path,
        default=None,
        help="optional JSONL with doc_id, term_ids, and weights for learned-sparse document admission",
    )
    parser.add_argument(
        "--learned-sparse-query-jsonl",
        type=Path,
        default=None,
        help="optional JSONL with query_id, term_ids, and weights for learned-sparse query admission",
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
        default=None,
        help=(
            "comma-separated multivector document candidate budgets to sweep; "
            f"--document-node-serving-grid defaults to {DOCUMENT_NODE_SERVING_GRID_BUDGET_SWEEP}, "
            f"other modes default to {DEFAULT_ADMISSION_BUDGET_SWEEP}"
        ),
    )
    parser.add_argument(
        "--admission-trace-limit",
        type=int,
        default=1000,
        help="bounded trace entries requested from turbohybrid_last_scan_stats()",
    )
    parser.add_argument(
        "--document-node-admission-grid",
        action="store_true",
        help="run DBpedia admission comparisons for token-node baselines and document-node storage/EF/oversampling grid",
    )
    parser.add_argument(
        "--document-node-serving-grid",
        action="store_true",
        help=(
            "run a compact production-oriented document-node serving grid over "
            "named f16/sq8 proxy and centroid profiles"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-include-experimental",
        action="store_true",
        help=(
            "include the guarded quantized_inverted_experimental_f32 profile in "
            "--document-node-serving-grid"
        ),
    )
    parser.add_argument(
        "--serving-min-top10-admission",
        type=float,
        default=0.80,
        help="minimum exact top-10 admission recall for --document-node-serving-grid best_latency_safe",
    )
    parser.add_argument(
        "--serving-min-ndcg-ratio-vs-exact",
        type=float,
        default=0.95,
        help=(
            "minimum ndcg@10 ratio versus the exact-scan baseline for "
            "--document-node-serving-grid best_latency_safe; ignored with an "
            "unavailable-criteria note when qrels or exact baseline metrics are missing"
        ),
    )
    parser.add_argument(
        "--serving-max-p95-ms",
        type=float,
        default=0.0,
        help="hard p95 latency cap for --document-node-serving-grid; 0 disables the cap",
    )
    parser.add_argument(
        "--document-node-storage-grid",
        default="f32,f16,sq8",
        help="comma-separated document-node storage kinds to sweep when --document-node-admission-grid is enabled",
    )
    parser.add_argument(
        "--document-node-cache-grid",
        default="auto,paged",
        help="comma-separated document-node sidecar cache modes to sweep when --document-node-admission-grid is enabled",
    )
    parser.add_argument(
        "--document-node-ef-grid",
        default="50,100,200,400,800",
        help="comma-separated document-node graph search EF values to sweep",
    )
    parser.add_argument(
        "--document-node-oversampling-grid",
        default="1,2,4,8",
        help="comma-separated document-node graph oversampling multipliers to sweep",
    )
    parser.add_argument(
        "--document-node-pooling-grid",
        default="off:1.0,greedy_cosine:0.75,greedy_cosine:0.5,greedy_cosine:0.33",
        help="comma-separated pooling mode:ratio entries to rebuild document-node indexes for",
    )
    parser.add_argument(
        "--document-node-proxy-encoder-grid",
        default="normalized_mean,centroid_mean,max_pool,random_projection_fde",
        help="comma-separated proxy encoders to rebuild document-node indexes for",
    )
    parser.add_argument(
        "--hybrid-evaluation-harness",
        action="store_true",
        help="run Prompt 15 end-to-end hybrid comparison modes over a document-node index",
    )
    parser.add_argument(
        "--hybrid-evaluation-modes",
        default=",".join(HYBRID_EVALUATION_MODES),
        help=(
            "comma-separated Prompt 15 hybrid modes to run; "
            "quantized_inverted_experimental is also supported when named explicitly"
        ),
    )
    parser.add_argument(
        "--hybrid-evaluation-quality-floor",
        type=float,
        default=0.95,
        help="relative NDCG@10 floor used to pick the fastest recommended hybrid profile",
    )
    parser.add_argument(
        "--hybrid-evaluation-dbsf-min-branch-candidates",
        type=int,
        default=1,
        help="DBSF minimum branch candidates for the Prompt 15 harness",
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
    parser.add_argument(
        "--parallel-safety-preflight",
        choices=("serial_probe", "off"),
        default="serial_probe",
        help="skip the 8x phase when the serial probe already shows pathological multivector scan work",
    )
    parser.add_argument(
        "--parallel-safety-max-serial-p95-ms",
        type=float,
        default=DEFAULT_PARALLEL_SAFETY_MAX_SERIAL_P95_MS,
        help="maximum serial p95 latency allowed before the 8x phase is skipped",
    )
    parser.add_argument(
        "--parallel-safety-max-docs-scored",
        type=int,
        default=DEFAULT_PARALLEL_SAFETY_MAX_DOCS_SCORED,
        help="maximum per-query document candidates/docs scored allowed before the 8x phase is skipped",
    )
    parser.add_argument(
        "--parallel-safety-max-exact-rerank-docs",
        type=int,
        default=DEFAULT_PARALLEL_SAFETY_MAX_EXACT_RERANK_DOCS,
        help="maximum exact MaxSim-reranked documents allowed before the 8x phase is skipped",
    )
    parser.add_argument(
        "--parallel-safety-max-exact-pairs",
        type=int,
        default=DEFAULT_PARALLEL_SAFETY_MAX_EXACT_PAIRS,
        help="maximum exact MaxSim token-pair evaluations allowed before the 8x phase is skipped",
    )
    parser.add_argument(
        "--parallel-safety-max-sidecar-bytes",
        type=int,
        default=DEFAULT_PARALLEL_SAFETY_MAX_SIDECAR_BYTES,
        help="maximum per-query document-sidecar bytes touched allowed before the 8x phase is skipped",
    )
    parser.add_argument(
        "--force-parallel-retrieval",
        action="store_true",
        help="run the 8x retrieval phase even when the serial safety preflight exceeds configured thresholds",
    )
    parser.add_argument(
        "--skip-parallel-retrieval",
        action="store_true",
        help="record serial retrieval metrics only and skip the 8x throughput phase",
    )
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
    args = parser.parse_args()
    if args.self_check:
        return args
    return validate_args(args)


def main() -> None:
    args = parse_args()
    if args.self_check:
        run_self_checks()
        print("self-check ok")
        return
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
        dataset_stats = multivector_dataset_stats(conn)
        learned_sparse_phase = load_learned_sparse_vectors(conn, args)
        index_phase = build_index(conn, args)
        set_retrieval_gucs(conn, args, "dbpedia_colbert_serial")
        encoded_queries = load_encoded_queries(conn)

        result_methods: list[dict[str, Any]] = []
        for method in args.methods:
            run, serial_latencies, serial_stats = run_serial_retrieval(conn, method, encoded_queries, args, args.final_k)
            parallel = run_parallel_retrieval_report(
                args,
                method,
                encoded_queries,
                serial_latencies,
                serial_stats,
            )
            result_methods.append({
                "method": method,
                "retrieval_mode": {
                    QUERY_ONLY_METHOD: "multivector_query_only",
                    RRF_METHOD: "multivector_plus_bm25_rrf",
                    EXACT_SCAN_METHOD: "multivector_exact_scan",
                }[method],
                "metrics": method_metrics(run, qrels, args.final_k, args.quality_k) if qrels else {},
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
        document_node_admission_grid = (
            run_document_node_admission_grid(conn, args, encoded_queries, qrels)
            if args.document_node_admission_grid
            else None
        )
        document_node_serving_grid = (
            run_document_node_serving_grid(conn, args, encoded_queries, qrels)
            if args.document_node_serving_grid
            else None
        )
        document_node_serving_recommendation = (
            compute_document_node_serving_recommendation(
                document_node_serving_grid,
                exact_baseline=serving_exact_baseline_from_results(result_methods),
                min_top10_admission=args.serving_min_top10_admission,
                min_ndcg_ratio_vs_exact=args.serving_min_ndcg_ratio_vs_exact,
                max_p95_ms=args.serving_max_p95_ms,
            )
            if document_node_serving_grid is not None
            else None
        )
        hybrid_evaluation = (
            run_hybrid_evaluation_harness(conn, args, encoded_queries, qrels)
            if args.hybrid_evaluation_harness
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
            "pgturbohybrid": {
                "git_sha": git_sha(),
            },
            "dataset": {
                "name": "dbpedia-colbert-multivector",
                "corpus_path": portable_path(args.dataset) if args.dataset else None,
                "beir_dataset_path": portable_path(args.beir_dataset) if args.beir_dataset else None,
                "qrels_path": portable_path(qrels_path) if qrels_path else None,
                "precomputed_dataset": args.precomputed_dataset,
                "stats": dataset_stats,
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
                "colbert_model_name": args.colbert_model_name,
                "model_path": portable_path(args.model_path),
                "model_alias": args.model_alias,
                "expected_dim": args.expected_dim,
                "expected_dim_requested": args.expected_dim_requested,
                "expected_dim_source": args.expected_dim_source,
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
                "multivector_exact_rerank": args.multivector_exact_rerank,
                "multivector_exact_rerank_k": args.multivector_exact_rerank_k,
                "multivector_graph": args.multivector_graph,
                "multivector_doc_build_scorer": args.multivector_doc_build_scorer,
                "allow_exact_symmetric_build": args.allow_exact_symmetric_build,
                "index_reloptions": index_phase.get("reloptions", []),
                "multivector_doc_graph_search_ef": args.multivector_doc_graph_search_ef,
                "multivector_doc_graph_oversampling": args.multivector_doc_graph_oversampling,
                "multivector_doc_graph_rescore_k": args.multivector_doc_graph_rescore_k,
                "index_graph_m": args.index_graph_m,
                "index_graph_ef_construction": args.index_graph_ef_construction,
                "index_graph_ef_search": args.index_graph_ef_search,
                "index_native_segments": args.index_native_segments,
                "multivector_doc_storage": args.multivector_doc_storage,
                "multivector_doc_storage_cache": args.multivector_doc_storage_cache,
                "multivector_token_pooling": args.multivector_token_pooling,
                "multivector_token_pooling_target_ratio": args.multivector_token_pooling_target_ratio,
                "multivector_token_pooling_min_tokens": args.multivector_token_pooling_min_tokens,
                "multivector_centroids": args.multivector_centroids,
                "multivector_centroid_count": args.multivector_centroid_count,
                "multivector_proxy_encoder": args.multivector_proxy_encoder,
                "multivector_candidate_source": args.multivector_candidate_source,
                "multivector_plain_fallback": args.multivector_plain_fallback,
                "multivector_plain_fallback_max_docs": args.multivector_plain_fallback_max_docs,
                "multivector_plain_fallback_candidate_fraction": args.multivector_plain_fallback_candidate_fraction,
                "multivector_candidate_reservoirs": args.multivector_candidate_reservoirs,
                "multivector_per_token_doc_reservoir_k": args.multivector_per_token_doc_reservoir_k,
                "multivector_coverage_reservoir_k": args.multivector_coverage_reservoir_k,
                "multivector_bm25_candidate_injection": args.multivector_bm25_candidate_injection,
                "multivector_sparse_candidate_source": args.multivector_sparse_candidate_source,
                "learned_sparse_doc_jsonl": portable_path(args.learned_sparse_doc_jsonl) if args.learned_sparse_doc_jsonl else None,
                "learned_sparse_query_jsonl": portable_path(args.learned_sparse_query_jsonl) if args.learned_sparse_query_jsonl else None,
                "document_node_admission_grid": args.document_node_admission_grid,
                "document_node_serving_grid": args.document_node_serving_grid,
                "document_node_serving_grid_include_experimental": args.document_node_serving_grid_include_experimental,
                "serving_min_top10_admission": args.serving_min_top10_admission,
                "serving_min_ndcg_ratio_vs_exact": args.serving_min_ndcg_ratio_vs_exact,
                "serving_max_p95_ms": args.serving_max_p95_ms,
                "document_node_storage_grid": args.document_node_storage_grid_values,
                "document_node_cache_grid": args.document_node_cache_grid_values,
                "document_node_pooling_grid": args.document_node_pooling_grid_values,
                "document_node_proxy_encoder_grid": args.document_node_proxy_encoder_grid_values,
                "document_node_ef_grid": args.document_node_ef_grid_values,
                "document_node_oversampling_grid": args.document_node_oversampling_grid_values,
                "hybrid_evaluation_harness": args.hybrid_evaluation_harness,
                "hybrid_evaluation_modes": args.hybrid_evaluation_mode_values,
                "hybrid_evaluation_quality_floor": args.hybrid_evaluation_quality_floor,
                "hybrid_evaluation_dbsf_min_branch_candidates": args.hybrid_evaluation_dbsf_min_branch_candidates,
                "token_ablation_query_id": args.token_ablation_query_id,
                "token_ablation_skip_tokens": args.token_ablation_skip_tokens,
                "token_ablation_final_k": args.token_ablation_final_k,
                "token_ablation_dense_k": args.token_ablation_dense_k,
                "final_k": args.final_k,
                "quality_k": args.quality_k,
                "clients": args.clients,
                "parallel_safety_preflight": args.parallel_safety_preflight,
                "parallel_safety_max_serial_p95_ms": args.parallel_safety_max_serial_p95_ms,
                "parallel_safety_max_docs_scored": args.parallel_safety_max_docs_scored,
                "parallel_safety_max_exact_rerank_docs": args.parallel_safety_max_exact_rerank_docs,
                "parallel_safety_max_exact_pairs": args.parallel_safety_max_exact_pairs,
                "parallel_safety_max_sidecar_bytes": args.parallel_safety_max_sidecar_bytes,
                "force_parallel_retrieval": args.force_parallel_retrieval,
                "skip_parallel_retrieval": args.skip_parallel_retrieval,
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
                "load_learned_sparse_vectors": learned_sparse_phase,
                "build_index": index_phase,
            },
            "results": result_methods,
        }
        if admission_debug is not None:
            output["admission_debug"] = admission_debug
        if document_node_admission_grid is not None:
            output["document_node_admission_grid"] = document_node_admission_grid
        if document_node_serving_grid is not None:
            output["document_node_serving_grid"] = document_node_serving_grid
        if document_node_serving_recommendation is not None:
            output["document_node_serving_recommendation"] = document_node_serving_recommendation
        if hybrid_evaluation is not None:
            output["hybrid_evaluation"] = hybrid_evaluation
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
