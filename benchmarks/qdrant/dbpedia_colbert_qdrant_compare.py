#!/usr/bin/env python3
"""Compare pgturbohybrid and Qdrant on the same DBpedia ColBERT multivectors."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub>=0.30",
#   "numpy>=1.26",
#   "psycopg[binary]>=3.2",
#   "pyarrow>=16",
#   "qdrant-client>=1.14",
# ]
# ///

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import psycopg
from psycopg import sql

BENCHMARKS_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from dbpedia_colbert_hf_dataset import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    import_precomputed_dataset_to_postgres,
    iter_matching_document_rows,
    iter_parquet_rows,
    parquet_files,
    resolve_source,
)
from dbpedia_colbert_multivector import (  # noqa: E402
    QUERY_ONLY_METHOD,
    QueryItem,
    exec_sql,
    fetch_all,
    fetch_one,
    git_sha,
    jsonb_value,
    last_scan_stats,
    loaded_document_count,
    metrics_for_run,
    multivector_dataset_stats,
    run_retrieval_query,
    scan_stat_int,
    summarize_ints,
    summarize_ms,
)


DEFAULT_DATABASE = os.environ.get("PGDATABASE", "pgturbohybrid_dbpedia_colbert_qdrant")
DEFAULT_DATASET_SOURCE = os.environ.get(
    "DBPEDIA_COLBERT_PRECOMPUTED_DATASET",
    ".nix-dev/hf-datasets/dbpedia-colbert-multivector-1m-f16",
)
LEGACY_DATASET_SOURCE = ".nix-dev/hf-datasets/dbpedia-colbert-multivector-1m"
DEFAULT_HF_DATASET_SOURCE = "johannhartmann/pgturbohybrid_dbpedia_colbert"
DEFAULT_RESULTS_DIR = Path("benchmarks/qdrant/results")
DEFAULT_COLLECTION = "dbpedia_colbert_pgturbohybrid_compare"


@dataclass(frozen=True)
class VectorRecord:
    item_id: str
    vector: np.ndarray


@dataclass
class QueryResult:
    run: dict[str, list[str]]
    latencies_ms: list[float]
    stats: list[dict[str, Any]]


def connect(database: str) -> psycopg.Connection[Any]:
    return psycopg.connect(
        dbname=database,
        autocommit=True,
        application_name="dbpedia_colbert_qdrant_compare",
    )


def admin_connect(database: str) -> psycopg.Connection[Any]:
    return psycopg.connect(dbname=database, autocommit=True)


def ensure_database(database: str, maintenance_database: str, recreate: bool) -> dict[str, Any]:
    with admin_connect(maintenance_database) as conn:
        existed = fetch_one(conn, "SELECT 1 FROM pg_database WHERE datname = %s", (database,)) is not None
        if existed and recreate:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s AND pid <> pg_backend_pid()
                    """,
                    (database,),
                )
                cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))
            existed = False
        if not existed:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        return {"database": database, "created": not existed, "recreated": recreate}


def size_human(value: int | None) -> str:
    if value is None:
        return "n/a"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024.0
    return f"{value} B"


def qdrant_imports() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise SystemExit(
            "qdrant-client is required unless --pg-only is set. "
            "Install benchmarks/qdrant/requirements.txt or run in the bench environment."
        ) from exc
    return QdrantClient, models


def qdrant_version_report(client: Any | None) -> dict[str, Any]:
    try:
        client_version = importlib.metadata.version("qdrant-client")
    except importlib.metadata.PackageNotFoundError:
        client_version = None

    server_version = None
    if client is not None:
        try:
            server_version = client.get_version()
        except Exception:  # noqa: BLE001 - older clients may not expose this endpoint
            server_version = None
    return {
        "client_package": client_version,
        "server": server_version,
    }


def require_qdrant(url: str, timeout: float) -> Any:
    QdrantClient, _models = qdrant_imports()
    client = QdrantClient(url=url, timeout=timeout)
    try:
        client.get_collections()
    except Exception as exc:  # noqa: BLE001 - surface client/network failure clearly
        raise SystemExit(
            f"Qdrant is not reachable at {url}. Start Qdrant or pass --pg-only "
            "for a PostgreSQL-only run."
        ) from exc
    return client


def resolve_dataset_source(source: str, cache_dir: Path) -> Path:
    path = Path(source)
    if path.exists():
        return path.resolve()
    if source == DEFAULT_DATASET_SOURCE:
        for candidate in (Path(DEFAULT_DATASET_SOURCE), Path(LEGACY_DATASET_SOURCE)):
            if candidate.exists():
                return candidate.resolve()
        return resolve_source(DEFAULT_HF_DATASET_SOURCE, cache_dir)
    return resolve_source(source, cache_dir)


def decode_multivector(row: dict[str, Any], expected_dim: int) -> np.ndarray:
    dim = int(row["colbert_dim"])
    count = int(row["colbert_count"])
    if dim != expected_dim:
        raise ValueError(f"{row}: dimension {dim} does not match expected {expected_dim}")
    values_f16 = row.get("colbert_values_f16")
    if values_f16 is None:
        raise ValueError("Qdrant comparison requires compact float16 parquet exports")
    values = np.frombuffer(bytes(values_f16), dtype="<f2").astype(np.float32)
    expected = dim * count
    if values.size != expected:
        raise ValueError(f"multivector has {values.size} values, expected {expected}")
    return values.reshape((count, dim))


def load_document_vectors(
    dataset_dir: Path,
    doc_ids: Sequence[str],
    *,
    batch_size: int,
    expected_dim: int,
) -> list[VectorRecord]:
    files = parquet_files(dataset_dir, "docs")
    wanted = set(doc_ids)
    records: dict[str, VectorRecord] = {}
    for row in iter_matching_document_rows(files, batch_size, wanted):
        doc_id = str(row["doc_id"])
        records[doc_id] = VectorRecord(doc_id, decode_multivector(row, expected_dim))
    missing = sorted(wanted - set(records))
    if missing:
        raise RuntimeError(f"{len(missing)} selected PostgreSQL docs are missing from the parquet source")
    return [records[doc_id] for doc_id in doc_ids]


def load_query_vectors(
    dataset_dir: Path,
    query_ids: Sequence[str],
    *,
    batch_size: int,
    expected_dim: int,
) -> dict[str, VectorRecord]:
    files = parquet_files(dataset_dir, "queries")
    wanted = set(query_ids)
    records: dict[str, VectorRecord] = {}
    for row in iter_parquet_rows(files, batch_size):
        query_id = str(row["query_id"])
        if query_id in wanted:
            records[query_id] = VectorRecord(query_id, decode_multivector(row, expected_dim))
        if len(records) == len(wanted):
            break
    missing = sorted(wanted - set(records))
    if missing:
        raise RuntimeError(f"{len(missing)} selected PostgreSQL queries are missing from the parquet source")
    return records


def selected_doc_ids(conn: psycopg.Connection[Any]) -> list[str]:
    rows = fetch_all(
        conn,
        """
        SELECT doc_id
        FROM dbpedia_colbert_docs
        WHERE colbert IS NOT NULL
        ORDER BY doc_id
        """,
    )
    return [str(row[0]) for row in rows]


def selected_queries(conn: psycopg.Connection[Any], max_queries: int) -> list[QueryItem]:
    rows = fetch_all(
        conn,
        """
        SELECT q.query_id, q.query_text
        FROM dbpedia_colbert_queries q
        WHERE q.colbert IS NOT NULL
          AND EXISTS (
            SELECT 1 FROM dbpedia_colbert_qrels r WHERE r.query_id = q.query_id
          )
        ORDER BY q.query_id
        LIMIT %s
        """,
        (max_queries if max_queries > 0 else 2_147_483_647,),
    )
    return [QueryItem(str(row[0]), str(row[1])) for row in rows]


def import_pg_dataset(conn: psycopg.Connection[Any], args: argparse.Namespace, dataset_dir: Path) -> dict[str, Any]:
    if args.reuse_pg_data:
        exists = fetch_one(conn, "SELECT to_regclass('dbpedia_colbert_docs') IS NOT NULL")
        if not exists or not bool(exists[0]):
            return import_precomputed_dataset_to_postgres(
                conn=conn,
                source=str(dataset_dir),
                cache_dir=args.cache_dir,
                batch_size=args.batch_size,
                max_docs=args.max_docs,
                max_queries=args.max_queries,
                force_reload=True,
                prioritize_qrels=not args.no_prioritize_qrels,
            )
        row = fetch_one(
            conn,
            """
            SELECT
              count(*) FILTER (WHERE colbert IS NOT NULL),
              (SELECT count(*) FROM dbpedia_colbert_queries WHERE colbert IS NOT NULL),
              (SELECT count(*) FROM dbpedia_colbert_qrels)
            FROM dbpedia_colbert_docs
            """,
        )
        if row and int(row[0]) >= args.max_docs and int(row[1]) > 0 and int(row[2]) > 0:
            return {
                "reused": True,
                "documents": int(row[0]),
                "queries": int(row[1]),
                "qrels": int(row[2]),
            }
    return import_precomputed_dataset_to_postgres(
        conn=conn,
        source=str(dataset_dir),
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        force_reload=not args.reuse_pg_data,
        prioritize_qrels=not args.no_prioritize_qrels,
    )


def set_pg_query_gucs(conn: psycopg.Connection[Any], args: argparse.Namespace) -> None:
    settings = {
        "jit": "off",
        "enable_seqscan": "off",
        "turbohybrid.multivector_candidate_source": "graph",
        "turbohybrid.multivector_exact_rerank": "topk",
        "turbohybrid.multivector_doc_candidate_k": str(args.multivector_doc_candidate_k),
        "turbohybrid.multivector_exact_rerank_k": str(args.exact_rerank_k),
        "turbohybrid.multivector_doc_graph_oversampling": str(args.graph_oversampling),
        "turbohybrid.multivector_doc_graph_search_ef": str(args.doc_graph_search_ef),
        "turbohybrid.multivector_doc_graph_rescore_k": str(args.doc_graph_rescore_k),
        "turbohybrid.multivector_plain_fallback": "off",
    }
    for key, value in settings.items():
        exec_sql(conn, "SELECT set_config(%s, %s, false)", (key, value))


def build_pg_index(conn: psycopg.Connection[Any], args: argparse.Namespace) -> dict[str, Any]:
    reloptions = [
        "quantization_bits = 4",
        "exact_storage = off",
        "multivector_graph = document_nodes",
        "multivector_doc_build_scorer = proxy",
        "multivector_proxy_encoder = normalized_mean",
    ]
    if args.graph_m > 0:
        reloptions.append(f"graph_m = {args.graph_m}")
    if args.graph_ef_construction > 0:
        reloptions.append(f"graph_ef_construction = {args.graph_ef_construction}")
    if args.graph_ef_search > 0:
        reloptions.append(f"graph_ef_search = {args.graph_ef_search}")
    if args.native_build_workers > 0:
        reloptions.append(f"native_build_workers = {args.native_build_workers}")

    if args.reuse_pg_index:
        exists = fetch_one(conn, "SELECT to_regclass('dbpedia_colbert_docs_colbert_idx') IS NOT NULL")
        if exists and bool(exists[0]):
            return pg_index_report(conn, reused=True, elapsed_ms=0.0, reloptions=reloptions)

    exec_sql(conn, "DROP INDEX IF EXISTS dbpedia_colbert_docs_colbert_idx")
    started = time.perf_counter()
    exec_sql(
        conn,
        f"""
        CREATE INDEX dbpedia_colbert_docs_colbert_idx
        ON dbpedia_colbert_docs USING turbohybrid (
          colbert multivector_maxsim_ip_turbohybrid_ops
        )
        WITH ({", ".join(reloptions)})
        """,
    )
    return pg_index_report(
        conn,
        reused=False,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
        reloptions=reloptions,
    )


def pg_index_report(
    conn: psycopg.Connection[Any],
    *,
    reused: bool,
    elapsed_ms: float,
    reloptions: Sequence[str],
) -> dict[str, Any]:
    index_stats = jsonb_value(
        fetch_one(conn, "SELECT turbohybrid_index_stats('dbpedia_colbert_docs_colbert_idx'::regclass)")[0]
    )
    build_stats_row = fetch_one(conn, "SELECT turbohybrid_last_build_stats()")
    build_stats = jsonb_value(build_stats_row[0]) if build_stats_row else {}
    safety_build_stats = (
        build_stats
        if not reused and build_stats.get("relation_name") == "dbpedia_colbert_docs_colbert_idx"
        else {}
    )
    index_size = fetch_one(conn, "SELECT pg_relation_size('dbpedia_colbert_docs_colbert_idx'::regclass)")
    report = {
        "engine": "pgturbohybrid",
        "reused": reused,
        "elapsed_ms": round(elapsed_ms, 3),
        "reloptions": list(reloptions),
        "index_bytes": int(index_size[0]) if index_size else None,
        "index_stats": index_stats,
        "build_stats": build_stats,
        "safety_checks": {
            "build_stats_current": bool(safety_build_stats),
            "row_count": loaded_document_count(conn),
        },
    }
    validate_pg_index_safety(
        {**report, "build_stats": safety_build_stats},
        row_count=int(report["safety_checks"]["row_count"]),
    )
    return report


def validate_pg_index_safety(index_report: dict[str, Any], *, row_count: int) -> None:
    index_stats = index_report.get("index_stats", {})
    build_stats = index_report.get("build_stats", {})
    scorer = index_stats.get("multivector_doc_build_scorer")
    calls = int(build_stats.get("multivector_doc_exact_build_distance_calls", 0) or 0)
    graph_mode = index_stats.get("multivector_graph_mode")
    node_count = int(index_stats.get("node_count", 0) or 0)
    build_fast_edges = bool(index_stats.get("build_fast_edges", False))
    if graph_mode != "document_nodes":
        raise SystemExit(f"refusing benchmark: pgturbohybrid graph mode is {graph_mode!r}, not document_nodes")
    if scorer != "proxy":
        raise SystemExit(f"refusing benchmark: multivector_doc_build_scorer is {scorer!r}, not proxy")
    if calls > 0:
        raise SystemExit(
            "refusing benchmark: pgturbohybrid index build made "
            f"{calls} exact document-document MaxSim build-distance calls"
        )
    if not build_fast_edges:
        raise SystemExit(
            "refusing benchmark: pgturbohybrid document_nodes proxy index "
            "did not use fast edge construction"
        )
    if node_count != row_count:
        raise SystemExit(
            "refusing benchmark: pgturbohybrid document_nodes index node_count "
            f"{node_count} does not match loaded document count {row_count}"
        )


def run_pg_queries(conn: psycopg.Connection[Any], queries: Sequence[QueryItem], args: argparse.Namespace) -> QueryResult:
    set_pg_query_gucs(conn, args)
    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    stats_rows: list[dict[str, Any]] = []
    query_args = argparse.Namespace(dense_k=args.dense_k)
    for query in queries:
        started = time.perf_counter()
        docs = run_retrieval_query(
            conn,
            QUERY_ONLY_METHOD,
            query,
            query_args,
            args.final_k,
            dense_k=args.dense_k,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        stats_rows.append(last_scan_stats(conn))
        run[query.query_id] = docs
    return QueryResult(run=run, latencies_ms=latencies, stats=stats_rows)


def create_qdrant_collection(client: Any, models: Any, args: argparse.Namespace) -> dict[str, Any]:
    if not args.reuse_qdrant:
        try:
            client.delete_collection(args.qdrant_collection)
        except Exception:  # noqa: BLE001 - absent collection is expected
            pass
        started = time.perf_counter()
        client.create_collection(
            collection_name=args.qdrant_collection,
            vectors_config=models.VectorParams(
                size=args.expected_dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
            ),
            hnsw_config=models.HnswConfigDiff(
                m=args.qdrant_m,
                ef_construct=args.qdrant_ef_construct,
            ),
        )
        return {"reused": False, "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3)}

    try:
        client.get_collection(args.qdrant_collection)
        return {"reused": True, "elapsed_ms": 0.0}
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"--reuse-qdrant was set, but collection {args.qdrant_collection!r} is missing") from exc


def insert_qdrant_documents(
    client: Any,
    models: Any,
    args: argparse.Namespace,
    docs: Sequence[VectorRecord],
) -> dict[str, Any]:
    if args.reuse_qdrant:
        return {"reused": True, "elapsed_ms": 0.0, "points": len(docs)}

    started = time.perf_counter()
    inserted = 0
    for offset in range(0, len(docs), args.qdrant_batch_size):
        batch = docs[offset : offset + args.qdrant_batch_size]
        points = [
            models.PointStruct(
                id=offset + idx,
                vector=record.vector.tolist(),
                payload={"doc_id": record.item_id},
            )
            for idx, record in enumerate(batch)
        ]
        client.upsert(collection_name=args.qdrant_collection, points=points, wait=True)
        inserted += len(points)
        if args.progress_every > 0 and inserted % args.progress_every == 0:
            print(f"qdrant inserted {inserted} documents", file=sys.stderr)
    return {
        "reused": False,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "points": inserted,
    }


def qdrant_query_points(client: Any, models: Any, args: argparse.Namespace, query_vector: np.ndarray) -> list[str]:
    search_params = models.SearchParams(hnsw_ef=args.qdrant_hnsw_ef) if args.qdrant_hnsw_ef > 0 else None
    try:
        response = client.query_points(
            collection_name=args.qdrant_collection,
            query=query_vector.tolist(),
            limit=args.final_k,
            search_params=search_params,
            with_payload=True,
        )
        points = response.points
    except AttributeError:
        points = client.search(
            collection_name=args.qdrant_collection,
            query_vector=query_vector.tolist(),
            limit=args.final_k,
            search_params=search_params,
            with_payload=True,
        )
    docs: list[str] = []
    for point in points:
        payload = getattr(point, "payload", None) or {}
        doc_id = payload.get("doc_id")
        if doc_id is not None:
            docs.append(str(doc_id))
    return docs


def run_qdrant_queries(
    client: Any,
    models: Any,
    args: argparse.Namespace,
    queries: Sequence[QueryItem],
    query_vectors: dict[str, VectorRecord],
) -> QueryResult:
    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    for query in queries:
        started = time.perf_counter()
        docs = qdrant_query_points(client, models, args, query_vectors[query.query_id].vector)
        latencies.append((time.perf_counter() - started) * 1000.0)
        run[query.query_id] = docs
    return QueryResult(run=run, latencies_ms=latencies, stats=[])


def qdrant_collection_report(client: Any, args: argparse.Namespace) -> dict[str, Any]:
    try:
        info = client.get_collection(args.qdrant_collection)
    except Exception:  # noqa: BLE001
        return {}
    result: dict[str, Any] = {}
    for key in ("status", "points_count", "vectors_count", "indexed_vectors_count", "segments_count"):
        value = getattr(info, key, None)
        if value is not None:
            result[key] = str(value) if key == "status" else value
    return result


def maxsim(query: np.ndarray, doc: np.ndarray) -> float:
    scores = query @ doc.T
    return float(np.max(scores, axis=1).sum())


def exact_topk(
    queries: Sequence[QueryItem],
    query_vectors: dict[str, VectorRecord],
    docs: Sequence[VectorRecord],
    k: int,
) -> dict[str, list[str]]:
    truth: dict[str, list[str]] = {}
    for query in queries:
        qv = query_vectors[query.query_id].vector
        scored = [(maxsim(qv, doc.vector), doc.item_id) for doc in docs]
        scored.sort(key=lambda item: (-item[0], item[1]))
        truth[query.query_id] = [doc_id for _score, doc_id in scored[:k]]
    return truth


def truth_as_qrels(truth: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    return {qid: {doc_id: 1 for doc_id in docs} for qid, docs in truth.items()}


def summarize_pg_stats(stats_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "exact_rerank_docs": summarize_ints([
            max(
                scan_stat_int(stats, "multivector_exact_rerank_docs"),
                scan_stat_int(stats, "multivector_doc_graph_exact_rerank_docs"),
                scan_stat_int(stats, "proxy_exact_rerank_docs"),
            )
            for stats in stats_rows
        ]),
        "exact_rerank_pairs": summarize_ints([
            scan_stat_int(stats, "multivector_exact_rerank_pairs") for stats in stats_rows
        ]),
        "proxy_graph_searches": summarize_ints([
            scan_stat_int(stats, "multivector_proxy_graph_searches") for stats in stats_rows
        ]),
        "subvector_searches": summarize_ints([
            scan_stat_int(stats, "multivector_subvector_searches") for stats in stats_rows
        ]),
        "last_scan_stats": stats_rows[-1] if stats_rows else {},
    }


def engine_summary(
    *,
    engine: str,
    build_ms: float,
    insert_ms: float,
    index_bytes: int | None,
    query_result: QueryResult,
    truth_qrels: dict[str, dict[str, int]],
    quality_k: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latency = summarize_ms(query_result.latencies_ms)
    metrics = metrics_for_run(query_result.run, truth_qrels, quality_k)
    summary = {
        "engine": engine,
        "build_ms": round(build_ms, 3),
        "insert_ms": round(insert_ms, 3),
        "index_bytes": index_bytes,
        "latency": latency,
        "quality": metrics,
    }
    if extra:
        summary.update(extra)
    return summary


def print_table(summaries: Sequence[dict[str, Any]], quality_k: int) -> None:
    headers = [
        "engine",
        "build_s",
        "insert_s",
        "storage",
        f"recall@{quality_k}",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "rerank_docs_p50",
        "rerank_pairs_p50",
    ]
    rows: list[list[str]] = []
    for item in summaries:
        latency = item.get("latency", {})
        quality = item.get("quality", {})
        pg_stats = item.get("pgturbohybrid_scan_summary", {})
        rerank_docs = pg_stats.get("exact_rerank_docs", {}) if isinstance(pg_stats, dict) else {}
        rerank_pairs = pg_stats.get("exact_rerank_pairs", {}) if isinstance(pg_stats, dict) else {}
        rows.append([
            str(item["engine"]),
            f"{float(item.get('build_ms', 0.0)) / 1000.0:.3f}",
            f"{float(item.get('insert_ms', 0.0)) / 1000.0:.3f}",
            size_human(item.get("index_bytes")),
            f"{float(quality.get(f'recall@{quality_k}', 0.0)):.4f}",
            f"{float(latency.get('p50_ms', 0.0)):.3f}",
            f"{float(latency.get('p95_ms', 0.0)):.3f}",
            f"{float(latency.get('p99_ms', 0.0)):.3f}",
            str(rerank_docs.get("p50", "n/a")) if isinstance(rerank_docs, dict) else "n/a",
            str(rerank_pairs.get("p50", "n/a")) if isinstance(rerank_pairs, dict) else "n/a",
        ])
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def write_result(args: argparse.Namespace, result: dict[str, Any]) -> Path | None:
    if args.output is None:
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = DEFAULT_RESULTS_DIR / f"dbpedia-colbert-qdrant-compare-{args.max_docs}-{timestamp}.json"
    elif str(args.output) == "-":
        print(json.dumps(result, indent=2, sort_keys=True))
        return None
    else:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.max_docs < 1:
        raise SystemExit("--max-docs must be at least 1")
    if args.max_queries < 1:
        raise SystemExit("--max-queries must be at least 1")
    if args.final_k < 1 or args.quality_k < 1:
        raise SystemExit("--final-k and --quality-k must be at least 1")
    if args.quality_k > args.final_k:
        raise SystemExit("--quality-k cannot exceed --final-k")
    if args.recall_subset_docs < 0:
        raise SystemExit("--recall-subset-docs must be non-negative")
    if args.expected_dim != 128:
        raise SystemExit("this Qdrant parity benchmark is fixed to 128-dimensional ColBERT vectors")
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--maintenance-database", default=os.environ.get("PGMAINTENANCE_DB", "postgres"))
    parser.add_argument("--create-database", action="store_true")
    parser.add_argument("--recreate-database", action="store_true")
    parser.add_argument("--precomputed-dataset", default=DEFAULT_DATASET_SOURCE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-docs", type=int, default=10_000)
    parser.add_argument("--max-queries", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--qdrant-batch-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--expected-dim", type=int, default=128)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--quality-k", type=int, default=10)
    parser.add_argument("--recall-subset-docs", type=int, default=0, help="0 means all selected docs")
    parser.add_argument("--dense-k", type=int, default=100)
    parser.add_argument("--graph-oversampling", type=int, default=1)
    parser.add_argument("--multivector-doc-candidate-k", type=int, default=100)
    parser.add_argument("--exact-rerank-k", type=int, default=100)
    parser.add_argument("--doc-graph-search-ef", type=int, default=100)
    parser.add_argument("--doc-graph-rescore-k", type=int, default=0)
    parser.add_argument("--graph-m", type=int, default=16)
    parser.add_argument("--graph-ef-construction", type=int, default=128)
    parser.add_argument("--graph-ef-search", type=int, default=0)
    parser.add_argument("--native-build-workers", type=int, default=0)
    parser.add_argument("--reuse-pg-data", action="store_true")
    parser.add_argument("--reuse-pg-index", action="store_true")
    parser.add_argument("--no-prioritize-qrels", action="store_true")
    parser.add_argument("--pg-only", action="store_true")
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-timeout", type=float, default=60.0)
    parser.add_argument("--qdrant-collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-m", type=int, default=16)
    parser.add_argument("--qdrant-ef-construct", type=int, default=128)
    parser.add_argument("--qdrant-hnsw-ef", type=int, default=0)
    parser.add_argument("--reuse-qdrant", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="JSON result path; default writes under benchmarks/qdrant/results")
    return validate_args(parser.parse_args())


def main() -> None:
    args = parse_args()
    if args.recreate_database:
        args.create_database = True
    if args.create_database:
        ensure_database(args.database, args.maintenance_database, args.recreate_database)

    qdrant_client = None
    qdrant_models = None
    if not args.pg_only:
        qdrant_client = require_qdrant(args.qdrant_url, args.qdrant_timeout)
        _QdrantClient, qdrant_models = qdrant_imports()

    dataset_dir = resolve_dataset_source(args.precomputed_dataset, args.cache_dir)
    conn = connect(args.database)
    try:
        load_report = import_pg_dataset(conn, args, dataset_dir)
        dataset_stats = multivector_dataset_stats(conn)
        doc_ids = selected_doc_ids(conn)
        queries = selected_queries(conn, args.max_queries)
        if not queries:
            raise SystemExit("no encoded queries with qrels are available after dataset import")
        query_ids = [query.query_id for query in queries]
        docs = load_document_vectors(dataset_dir, doc_ids, batch_size=args.batch_size, expected_dim=args.expected_dim)
        query_vectors = load_query_vectors(
            dataset_dir,
            query_ids,
            batch_size=args.batch_size,
            expected_dim=args.expected_dim,
        )

        recall_docs = docs
        if args.recall_subset_docs > 0:
            recall_docs = docs[: min(args.recall_subset_docs, len(docs))]
        truth = exact_topk(queries, query_vectors, recall_docs, args.quality_k)
        truth_qrels = truth_as_qrels(truth)

        pg_index = build_pg_index(conn, args)
        pg_query = run_pg_queries(conn, queries, args)
        pg_summary = engine_summary(
            engine="pgturbohybrid",
            build_ms=float(pg_index.get("elapsed_ms", 0.0)),
            insert_ms=float(load_report.get("elapsed_ms", 0.0)),
            index_bytes=pg_index.get("index_bytes"),
            query_result=pg_query,
            truth_qrels=truth_qrels,
            quality_k=args.quality_k,
            extra={
                "pgturbohybrid_index_stats": pg_index.get("index_stats", {}),
                "pgturbohybrid_build_stats": pg_index.get("build_stats", {}),
                "pgturbohybrid_scan_summary": summarize_pg_stats(pg_query.stats),
            },
        )

        summaries = [pg_summary]
        qdrant_report: dict[str, Any] = {}
        if not args.pg_only and qdrant_client is not None and qdrant_models is not None:
            create_report = create_qdrant_collection(qdrant_client, qdrant_models, args)
            insert_report = insert_qdrant_documents(qdrant_client, qdrant_models, args, docs)
            qdrant_query = run_qdrant_queries(qdrant_client, qdrant_models, args, queries, query_vectors)
            qdrant_info = qdrant_collection_report(qdrant_client, args)
            qdrant_report = {
                "collection_create": create_report,
                "insert": insert_report,
                "collection": qdrant_info,
            }
            summaries.append(
                engine_summary(
                    engine="qdrant",
                    build_ms=float(create_report.get("elapsed_ms", 0.0)),
                    insert_ms=float(insert_report.get("elapsed_ms", 0.0)),
                    index_bytes=None,
                    query_result=qdrant_query,
                    truth_qrels=truth_qrels,
                    quality_k=args.quality_k,
                    extra={"qdrant": qdrant_report},
                )
            )

        result = {
            "suite": "dbpedia_colbert_qdrant_compare",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "qdrant_version": qdrant_version_report(qdrant_client),
            "dataset": {
                "source": str(dataset_dir),
                "documents": len(doc_ids),
                "queries": len(queries),
                "recall_subset_docs": len(recall_docs),
                "stats": dataset_stats,
            },
            "settings": {
                "max_docs": args.max_docs,
                "max_queries": args.max_queries,
                "final_k": args.final_k,
                "quality_k": args.quality_k,
                "dense_k": args.dense_k,
                "graph_oversampling": args.graph_oversampling,
                "multivector_doc_candidate_k": args.multivector_doc_candidate_k,
                "exact_rerank_k": args.exact_rerank_k,
                "qdrant_url": None if args.pg_only else args.qdrant_url,
                "qdrant_collection": None if args.pg_only else args.qdrant_collection,
            },
            "load": load_report,
            "pgturbohybrid": {
                "git_sha": git_sha(),
                "index": pg_index,
                "query": {
                    "latency": summarize_ms(pg_query.latencies_ms),
                    "scan_summary": summarize_pg_stats(pg_query.stats),
                    "stats_json": pg_query.stats,
                },
            },
            "qdrant": qdrant_report,
            "truth": {
                "kind": "brute_force_exact_maxsim_subset",
                "topk": args.quality_k,
                "queries": truth,
            },
            "summary": summaries,
        }
    finally:
        conn.close()

    print_table(summaries, args.quality_k)
    output = write_result(args, result)
    if output is not None:
        print(f"\nwrote JSON result: {output}")


if __name__ == "__main__":
    main()
