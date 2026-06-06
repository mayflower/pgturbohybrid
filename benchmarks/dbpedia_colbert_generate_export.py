#!/usr/bin/env python3
"""Generate DBpedia ColBERT multivectors in PostgreSQL and export them."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub>=0.30",
#   "numpy>=1.26",
#   "psycopg[binary]>=3.2",
#   "pyarrow>=16",
# ]
# ///

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from dbpedia_colbert_hf_dataset import DEFAULT_OUTPUT_DIR, export_dataset
from dbpedia_colbert_multivector import (
    DEFAULT_METHODS,
    DEFAULT_MODEL_PATH,
    choose_query_ids,
    connect,
    load_data,
    load_queries,
    persist_multivectors,
    read_qrels,
    resolve_qrels_path,
    selected_doc_ids,
    selected_query_ids,
    setup_schema,
    validate_args as validate_benchmark_args,
    validate_embedding_health,
)


DEFAULT_DATABASE = (
    os.environ.get("DBPEDIA_COLBERT_PGDATABASE")
    or os.environ.get("PGDATABASE")
    or "pgturbohybrid_dbpedia_colbert"
)
DEFAULT_DATASET = ".deps/datasets/qdrant-dbpedia-openai3-large-1m"
DEFAULT_BEIR_DATASET = ".deps/datasets/beir-dbpedia-entity/queries"
DEFAULT_QRELS = ".deps/datasets/beir-dbpedia-entity-qrels/test-positive.tsv"


def admin_connect(maintenance_database: str) -> psycopg.Connection[Any]:
    return psycopg.connect(
        dbname=maintenance_database,
        autocommit=True,
        application_name="dbpedia_colbert_generate_export_admin",
    )


def database_exists(conn: psycopg.Connection[Any], database: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
        return cur.fetchone() is not None


def ensure_database(database: str, recreate: bool, maintenance_database: str) -> dict[str, Any]:
    with admin_connect(maintenance_database) as conn:
        existed = database_exists(conn, database)
        if existed and recreate:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = %s
                      AND pid <> pg_backend_pid()
                    """,
                    (database,),
                )
                cur.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))
            existed = False
        if not existed:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        return {
            "database": database,
            "maintenance_database": maintenance_database,
            "existed": existed,
            "recreated": recreate,
            "created": not existed,
        }


def build_benchmark_args(args: argparse.Namespace) -> argparse.Namespace:
    return validate_benchmark_args(
        argparse.Namespace(
            database=args.database,
            dataset=args.dataset,
            beir_dataset=args.beir_dataset,
            qrels=args.qrels,
            precomputed_dataset=None,
            precomputed_batch_size=4096,
            model_path=args.model_path,
            model_alias=args.model_alias,
            expected_dim=args.expected_dim,
            max_doc_vectors=args.max_doc_vectors,
            max_query_vectors=args.max_query_vectors,
            query_length=args.query_length,
            max_docs=args.max_docs,
            max_queries=args.max_queries,
            prioritize_qrels=args.prioritize_qrels,
            generation_sample_docs=0,
            insert_batch_size=args.insert_batch_size,
            progress_every=args.progress_every,
            methods=DEFAULT_METHODS,
            generation_clients=args.generation_clients,
            generation_threads=args.generation_threads,
            generation_n_gpu_layers=args.generation_n_gpu_layers,
            generation_batch_sequences=args.generation_batch_sequences,
            generation_n_batch=args.generation_n_batch,
            generation_warmup=args.generation_warmup,
            dense_k=100,
            bm25_k=100,
            rrf_k=60,
            multivector_subvector_k=100,
            multivector_unique_docs_per_token=100,
            multivector_max_raw_hits_per_token=400,
            multivector_adaptive_widening="auto",
            multivector_doc_candidate_k=100,
            multivector_exact_rerank_k=100,
            final_k=10,
            quality_k=10,
            clients=args.generation_clients,
            warm_queries=0,
            timed_queries=0,
            reuse_data=args.resume,
            reuse_embeddings=args.resume,
            reuse_index=False,
            force_reload=args.force_reload,
            allow_unvalidated_embeddings=args.allow_unvalidated_embeddings,
            output=None,
        )
    )


def export_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        database=args.database,
        output_dir=args.output_dir,
        force=args.force_export,
        batch_size=args.export_batch_size,
        progress_every=args.export_progress_every,
        repo_id=args.repo_id,
        upload=args.upload,
        private=args.private,
        expected_dim=args.expected_dim,
        max_doc_vectors=args.max_doc_vectors,
        max_query_vectors=args.max_query_vectors,
        embedding_format=args.embedding_format,
    )


def generate_database(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    bench_args = build_benchmark_args(args)

    queries = load_queries(bench_args.beir_dataset)
    qrels_path = resolve_qrels_path(bench_args.qrels, bench_args.beir_dataset)
    all_qrels = read_qrels(qrels_path)
    qids = choose_query_ids(all_qrels, queries, bench_args.max_queries)
    qrels = {qid: all_qrels[qid] for qid in qids}

    conn = connect(bench_args)
    try:
        setup_schema(conn)
        embedding_health = validate_embedding_health(conn, bench_args)
        load_phase = load_data(conn, bench_args, qids, queries, qrels)
        doc_ids = selected_doc_ids(conn, bench_args.max_docs)
        query_ids = selected_query_ids(conn)
        persist_phase = persist_multivectors(conn, bench_args, doc_ids, query_ids)
    finally:
        conn.close()

    return {
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "qrels_path": str(qrels_path),
        "embedding_health": embedding_health,
        "load_text": load_phase,
        "persist_multivectors": persist_phase,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--maintenance-database", default=os.environ.get("PGMAINTENANCE_DB", "postgres"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.environ.get("DBPEDIA_DATASET", DEFAULT_DATASET)),
        help=f"Qdrant DBpedia parquet dataset root (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--beir-dataset",
        type=Path,
        default=Path(os.environ.get("BEIR_DBPEDIA_DATASET", DEFAULT_BEIR_DATASET)),
        help=f"BEIR DBpedia dataset root containing queries (default: {DEFAULT_BEIR_DATASET})",
    )
    parser.add_argument(
        "--qrels",
        default=os.environ.get("BEIR_DBPEDIA_QRELS", DEFAULT_QRELS),
        help=f"BEIR DBpedia qrels TSV path (default: {DEFAULT_QRELS})",
    )
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("PG_COLBERT_LLAMA_TEST_MODEL", DEFAULT_MODEL_PATH)))
    parser.add_argument("--model-alias", default=None)
    parser.add_argument("--expected-dim", type=int, default=128)
    parser.add_argument("--max-doc-vectors", type=int, default=256)
    parser.add_argument("--max-query-vectors", type=int, default=32)
    parser.add_argument("--query-length", type=int, default=32)
    parser.add_argument("--max-docs", type=int, default=1_000_000, help="documents to load; 0 means all available")
    parser.add_argument("--max-queries", type=int, default=0, help="queries to load from qrels; 0 means all")
    parser.add_argument(
        "--no-prioritize-qrels",
        dest="prioritize_qrels",
        action="store_false",
        help="load corpus order directly instead of loading judged qrel documents first",
    )
    parser.add_argument("--generation-clients", type=int, default=8)
    parser.add_argument("--generation-threads", type=int, default=1)
    parser.add_argument("--generation-n-gpu-layers", type=int, default=0)
    parser.add_argument("--generation-batch-sequences", type=int, default=8)
    parser.add_argument("--generation-n-batch", type=int, default=None)
    parser.add_argument("--insert-batch-size", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--no-generation-warmup", dest="generation_warmup", action="store_false")
    parser.add_argument("--allow-unvalidated-embeddings", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse loaded text and already generated multivectors when the target database contains them",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="truncate benchmark tables before loading text and regenerating multivectors",
    )
    parser.add_argument(
        "--recreate-database",
        action="store_true",
        help="drop and recreate the target PostgreSQL database before generation",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-export", action="store_true", help="replace the output dataset directory if it exists")
    parser.add_argument("--export-batch-size", type=int, default=4096)
    parser.add_argument("--export-progress-every", type=int, default=10000)
    parser.add_argument(
        "--embedding-format",
        choices=["float16_le", "postgres_text"],
        default="float16_le",
        help="embedding storage format for docs/queries parquet files",
    )
    parser.add_argument("--repo-id", default=None, help="optional Hugging Face dataset repo id, e.g. owner/name")
    parser.add_argument("--upload", action="store_true", help="upload exported dataset to --repo-id")
    parser.add_argument("--private", action="store_true", help="create the Hugging Face dataset as private")
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.set_defaults(prioritize_qrels=True, generation_warmup=True)

    args = parser.parse_args()
    if args.generation_clients < 1:
        raise SystemExit("--generation-clients must be at least 1")
    if args.generation_threads < 1:
        raise SystemExit("--generation-threads must be at least 1")
    if args.insert_batch_size < 1:
        raise SystemExit("--insert-batch-size must be at least 1")
    if args.export_batch_size < 1:
        raise SystemExit("--export-batch-size must be at least 1")
    if args.upload and not args.repo_id:
        raise SystemExit("--upload requires --repo-id")
    if args.recreate_database and args.resume:
        raise SystemExit("--recreate-database cannot be combined with --resume")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    database_phase = ensure_database(args.database, args.recreate_database, args.maintenance_database)
    generation_phase = generate_database(args)
    export_phase = export_dataset(export_args(args))

    output = {
        "suite": "dbpedia_colbert_generate_export",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "command": {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "env": {
                "DBPEDIA_DATASET": os.environ.get("DBPEDIA_DATASET", ""),
                "BEIR_DBPEDIA_DATASET": os.environ.get("BEIR_DBPEDIA_DATASET", ""),
                "BEIR_DBPEDIA_QRELS": os.environ.get("BEIR_DBPEDIA_QRELS", ""),
                "PG_COLBERT_LLAMA_TEST_MODEL": os.environ.get("PG_COLBERT_LLAMA_TEST_MODEL", ""),
            },
        },
        "database": database_phase,
        "generation": generation_phase,
        "export": export_phase,
    }

    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
