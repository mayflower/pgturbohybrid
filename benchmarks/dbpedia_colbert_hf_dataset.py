#!/usr/bin/env python3
"""Export and import DBpedia ColBERT multivectors as a Hugging Face dataset."""

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
import shutil
import sys
import time
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
from psycopg.rows import dict_row


SCHEMA_VERSION = 1
COMPACT_SCHEMA_VERSION = 2
DEFAULT_DATABASE = "pgturbohybrid_dbpedia_colbert"
DEFAULT_OUTPUT_DIR = Path(".nix-dev/hf-datasets/dbpedia-colbert-multivector-1m")
DEFAULT_CACHE_DIR = Path(".nix-dev/hf-datasets/cache")
EMBEDDING_FORMAT_TEXT = "postgres_text"
EMBEDDING_FORMAT_F16 = "float16_le"


DOC_TEXT_SCHEMA = pa.schema([
    ("doc_id", pa.string()),
    ("title", pa.string()),
    ("body", pa.string()),
    ("colbert_multivector", pa.string()),
    ("colbert_dim", pa.int32()),
    ("colbert_count", pa.int32()),
])

QUERY_TEXT_SCHEMA = pa.schema([
    ("query_id", pa.string()),
    ("query_text", pa.string()),
    ("colbert_multivector", pa.string()),
    ("colbert_dim", pa.int32()),
    ("colbert_count", pa.int32()),
])

DOC_F16_SCHEMA = pa.schema([
    ("doc_id", pa.string()),
    ("title", pa.string()),
    ("body", pa.string()),
    ("colbert_values_f16", pa.binary()),
    ("colbert_dim", pa.int32()),
    ("colbert_count", pa.int32()),
])

QUERY_F16_SCHEMA = pa.schema([
    ("query_id", pa.string()),
    ("query_text", pa.string()),
    ("colbert_values_f16", pa.binary()),
    ("colbert_dim", pa.int32()),
    ("colbert_count", pa.int32()),
])

QREL_SCHEMA = pa.schema([
    ("query_id", pa.string()),
    ("doc_id", pa.string()),
    ("relevance", pa.int32()),
])


@dataclass
class WriteStats:
    rows: int = 0
    shards: int = 0


def connect(database: str) -> psycopg.Connection[Any]:
    return psycopg.connect(
        dbname=database,
        autocommit=True,
        application_name="dbpedia_colbert_hf_dataset",
    )


def exec_sql(conn: psycopg.Connection[Any], sql: str, params: tuple[Any, ...] | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params)


def fetch_one(conn: psycopg.Connection[Any], sql: str, params: tuple[Any, ...] | None = None) -> tuple[Any, ...] | None:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def fetch_many(
    conn: psycopg.Connection[Any],
    sql: str,
    params: tuple[Any, ...],
) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def copy_rows(conn: psycopg.Connection[Any], sql: str, rows: Iterable[tuple[Any, ...]]) -> int:
    count = 0
    with conn.cursor() as cur:
        with cur.copy(sql) as copy:
            for row in rows:
                copy.write_row(row)
                count += 1
    return count


def setup_schema(conn: psycopg.Connection[Any]) -> None:
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS vector")
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS pgturbohybrid")
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


def extension_versions(conn: psycopg.Connection[Any]) -> dict[str, str]:
    rows = fetch_many(
        conn,
        """
        SELECT extname, extversion
        FROM pg_extension
        WHERE extname IN ('vector', 'pgturbohybrid', 'pg_colbert_llama')
        ORDER BY extname
        """,
        (),
    )
    return {str(row["extname"]): str(row["extversion"]) for row in rows}


def count_rows(conn: psycopg.Connection[Any]) -> dict[str, int]:
    row = fetch_one(
        conn,
        """
        SELECT
          (SELECT count(*) FROM dbpedia_colbert_docs),
          (SELECT count(*) FILTER (WHERE colbert IS NOT NULL) FROM dbpedia_colbert_docs),
          (SELECT count(*) FROM dbpedia_colbert_queries),
          (SELECT count(*) FILTER (WHERE colbert IS NOT NULL) FROM dbpedia_colbert_queries),
          (SELECT count(*) FROM dbpedia_colbert_qrels)
        """,
    )
    if not row:
        return {}
    return {
        "documents": int(row[0]),
        "encoded_documents": int(row[1]),
        "queries": int(row[2]),
        "encoded_queries": int(row[3]),
        "qrels": int(row[4]),
    }


def require_complete_embeddings(counts: dict[str, int]) -> None:
    failures: list[str] = []
    if counts.get("documents", 0) == 0:
        failures.append("no documents are loaded")
    if counts.get("queries", 0) == 0:
        failures.append("no queries are loaded")
    if counts.get("qrels", 0) == 0:
        failures.append("no qrels are loaded")
    if counts.get("documents") != counts.get("encoded_documents"):
        failures.append(
            f"encoded documents {counts.get('encoded_documents', 0)} "
            f"do not match documents {counts.get('documents', 0)}"
        )
    if counts.get("queries") != counts.get("encoded_queries"):
        failures.append(
            f"encoded queries {counts.get('encoded_queries', 0)} "
            f"do not match queries {counts.get('queries', 0)}"
        )
    if failures:
        raise RuntimeError("; ".join(failures))


def reset_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"output directory already exists: {path}; pass --force to replace it")
        shutil.rmtree(path)
    (path / "docs").mkdir(parents=True)
    (path / "queries").mkdir(parents=True)
    (path / "qrels").mkdir(parents=True)


def postgres_binary_multivector_to_f16(row: dict[str, Any]) -> dict[str, Any]:
    binary = bytes(row.pop("colbert_binary"))
    dim = int(row["colbert_dim"])
    count = int(row["colbert_count"])
    if len(binary) < 16:
        raise ValueError("multivector binary payload is too small")
    version, binary_dim, binary_count, _flags = struct.unpack("!iiii", binary[:16])
    if version != 1:
        raise ValueError(f"unsupported multivector binary version {version}")
    if binary_dim != dim or binary_count != count:
        raise ValueError(
            f"multivector binary header dim/count {binary_dim}/{binary_count} "
            f"does not match SQL dim/count {dim}/{count}"
        )
    expected_bytes = 16 + dim * count * 4
    if len(binary) != expected_bytes:
        raise ValueError(f"multivector binary payload has {len(binary)} bytes, expected {expected_bytes}")

    values = np.frombuffer(binary, dtype=">f4", offset=16)
    row["colbert_values_f16"] = values.astype("<f2", copy=False).tobytes()
    return row


def write_parquet_shards(
    *,
    conn: psycopg.Connection[Any],
    output_dir: Path,
    rel_dir: str,
    name: str,
    schema: pa.Schema,
    sql: str,
    id_column: str,
    batch_size: int,
    progress_every: int,
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> WriteStats:
    stats = WriteStats()
    last_id = ""
    next_report = progress_every
    target_dir = output_dir / rel_dir
    while True:
        rows = fetch_many(conn, sql, (last_id, batch_size))
        if not rows:
            break
        if transform is not None:
            rows = [transform(row) for row in rows]
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(
            table,
            target_dir / f"{name}-{stats.shards:05d}.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        stats.rows += len(rows)
        stats.shards += 1
        last_id = str(rows[-1][id_column])
        if progress_every > 0 and stats.rows >= next_report:
            print(f"exported {stats.rows} {name} rows", file=sys.stderr)
            while next_report <= stats.rows:
                next_report += progress_every
    return stats


def write_qrel_shards(
    *,
    conn: psycopg.Connection[Any],
    output_dir: Path,
    batch_size: int,
    progress_every: int,
) -> WriteStats:
    stats = WriteStats()
    last_query_id = ""
    last_doc_id = ""
    next_report = progress_every
    while True:
        rows = fetch_many(
            conn,
            """
            SELECT query_id, doc_id, relevance::int4 AS relevance
            FROM dbpedia_colbert_qrels
            WHERE (query_id, doc_id) > (%s, %s)
            ORDER BY query_id, doc_id
            LIMIT %s
            """,
            (last_query_id, last_doc_id, batch_size),
        )
        if not rows:
            break
        table = pa.Table.from_pylist(rows, schema=QREL_SCHEMA)
        pq.write_table(
            table,
            output_dir / "qrels" / f"qrels-{stats.shards:05d}.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        stats.rows += len(rows)
        stats.shards += 1
        last_query_id = str(rows[-1]["query_id"])
        last_doc_id = str(rows[-1]["doc_id"])
        if progress_every > 0 and stats.rows >= next_report:
            print(f"exported {stats.rows} qrel rows", file=sys.stderr)
            while next_report <= stats.rows:
                next_report += progress_every
    return stats


def write_dataset_card(output_dir: Path, repo_id: str | None, metadata: dict[str, Any]) -> None:
    title = repo_id or output_dir.name
    embedding_format = metadata["embedding"]["format"]
    if embedding_format == EMBEDDING_FORMAT_F16:
        embedding_desc = "packed little-endian float16 values"
        doc_column = "`colbert_values_f16`,"
        precision_note = (
            "This compact export stores half-precision values and reconstructs "
            "`turbohybrid_multivector` values on import. Use it for benchmark "
            "loading where avoiding runtime model generation is more important "
            "than bit-exact float32 round-trips."
        )
    else:
        embedding_desc = "PostgreSQL `turbohybrid_multivector` text literals"
        doc_column = "`colbert_multivector`,"
        precision_note = "This export stores PostgreSQL text literals for exact database import."
    card = f"""---
license: other
pretty_name: DBpedia ColBERT multivectors for pgturbohybrid
task_categories:
- text-retrieval
tags:
- dbpedia
- colbert
- multivector
- pgturbohybrid
- postgresql
---

# {title}

Precomputed DBpedia ColBERT multivectors for pgturbohybrid benchmark runs.

The dataset stores {embedding_desc} for the document and query embeddings.
Importing these rows into PostgreSQL avoids llama.cpp embedding generation
during retrieval/index benchmarks.

{precision_note}

## Contents

- Documents: {metadata["dataset"]["documents"]:,}
- Queries: {metadata["dataset"]["queries"]:,}
- Qrels: {metadata["dataset"]["qrels"]:,}
- Embedding dimension: {metadata["model"]["expected_dim"]}
- Embedding format: `{embedding_format}`

## Files

- `docs/*.parquet`: `doc_id`, `title`, `body`, {doc_column}
  `colbert_dim`, `colbert_count`
- `queries/*.parquet`: `query_id`, `query_text`, {doc_column}
  `colbert_dim`, `colbert_count`
- `qrels/*.parquet`: `query_id`, `doc_id`, `relevance`
- `metadata.json`: source model, row counts, schema version, and export command

## Model

- GGUF: `johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF`
- Source model: `VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m`

## Import

```sh
hf download {repo_id or output_dir.as_posix()} --type dataset \\
  --local-dir .nix-dev/hf-datasets/dbpedia-colbert-multivector-1m

nix --extra-experimental-features 'nix-command flakes' develop .#bench --command \\
  th-dbpedia-colbert-hf-dataset import \\
  --source .nix-dev/hf-datasets/dbpedia-colbert-multivector-1m
```
"""
    (output_dir / "README.md").write_text(card, encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    conn = connect(args.database)
    try:
        counts = count_rows(conn)
        require_complete_embeddings(counts)
        reset_output_dir(output_dir, args.force)
        if args.embedding_format == EMBEDDING_FORMAT_F16:
            schema_version = COMPACT_SCHEMA_VERSION
            doc_schema = DOC_F16_SCHEMA
            query_schema = QUERY_F16_SCHEMA
            transform = postgres_binary_multivector_to_f16
            doc_embedding_select = "turbohybrid_multivector_send(colbert) AS colbert_binary"
            query_embedding_select = "turbohybrid_multivector_send(colbert) AS colbert_binary"
        else:
            schema_version = SCHEMA_VERSION
            doc_schema = DOC_TEXT_SCHEMA
            query_schema = QUERY_TEXT_SCHEMA
            transform = None
            doc_embedding_select = "colbert::text AS colbert_multivector"
            query_embedding_select = "colbert::text AS colbert_multivector"

        doc_stats = write_parquet_shards(
            conn=conn,
            output_dir=output_dir,
            rel_dir="docs",
            name="docs",
            schema=doc_schema,
            sql=f"""
                SELECT
                  doc_id,
                  title,
                  body,
                  {doc_embedding_select},
                  turbohybrid_multivector_dims(colbert)::int4 AS colbert_dim,
                  turbohybrid_multivector_count(colbert)::int4 AS colbert_count
                FROM dbpedia_colbert_docs
                WHERE colbert IS NOT NULL AND doc_id > %s
                ORDER BY doc_id
                LIMIT %s
            """,
            id_column="doc_id",
            batch_size=args.batch_size,
            progress_every=args.progress_every,
            transform=transform,
        )
        query_stats = write_parquet_shards(
            conn=conn,
            output_dir=output_dir,
            rel_dir="queries",
            name="queries",
            schema=query_schema,
            sql=f"""
                SELECT
                  query_id,
                  query_text,
                  {query_embedding_select},
                  turbohybrid_multivector_dims(colbert)::int4 AS colbert_dim,
                  turbohybrid_multivector_count(colbert)::int4 AS colbert_count
                FROM dbpedia_colbert_queries
                WHERE colbert IS NOT NULL AND query_id > %s
                ORDER BY query_id
                LIMIT %s
            """,
            id_column="query_id",
            batch_size=args.batch_size,
            progress_every=args.progress_every,
            transform=transform,
        )
        qrel_stats = write_qrel_shards(
            conn=conn,
            output_dir=output_dir,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
        )
        metadata = {
            "schema_version": schema_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "embedding": {
                "format": args.embedding_format,
                "lossless": args.embedding_format == EMBEDDING_FORMAT_TEXT,
            },
            "dataset": {
                "name": "dbpedia-colbert-multivector",
                "documents": doc_stats.rows,
                "document_shards": doc_stats.shards,
                "queries": query_stats.rows,
                "query_shards": query_stats.shards,
                "qrels": qrel_stats.rows,
                "qrel_shards": qrel_stats.shards,
            },
            "model": {
                "gguf": "johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF",
                "source_model": "VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m",
                "expected_dim": args.expected_dim,
                "max_doc_vectors": args.max_doc_vectors,
                "max_query_vectors": args.max_query_vectors,
            },
            "postgresql": {
                "database": args.database,
                "extensions": extension_versions(conn),
            },
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "command": {
                "argv": sys.argv,
                "cwd": str(Path.cwd()),
            },
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        write_dataset_card(output_dir, args.repo_id, metadata)
    finally:
        conn.close()

    if args.repo_id and args.upload:
        upload_dataset(output_dir, args.repo_id, args.private)
    return metadata


def upload_dataset(output_dir: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        commit_message="add DBpedia ColBERT multivector dataset",
    )


def resolve_source(source: str, cache_dir: Path) -> Path:
    source_path = Path(source)
    if source_path.exists():
        return source_path.resolve()

    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_dir = cache_dir / source.replace("/", "__")
    snapshot_download(
        repo_id=source,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=[
            "README.md",
            "metadata.json",
            "generation-summary.json",
            "docs/*.parquet",
            "queries/*.parquet",
            "qrels/*.parquet",
        ],
    )
    return local_dir.resolve()


def parquet_files(dataset_dir: Path, rel_dir: str) -> list[Path]:
    files = sorted((dataset_dir / rel_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found in {dataset_dir / rel_dir}")
    return files


def iter_parquet_rows(files: list[Path], batch_size: int) -> Iterable[dict[str, Any]]:
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row


def row_multivector_text(row: dict[str, Any]) -> str:
    text = row.get("colbert_multivector")
    if text is not None:
        return str(text)

    values_f16 = row.get("colbert_values_f16")
    if values_f16 is None:
        raise KeyError("row has neither colbert_multivector nor colbert_values_f16")
    dim = int(row["colbert_dim"])
    count = int(row["colbert_count"])
    values = np.frombuffer(bytes(values_f16), dtype="<f2").astype(np.float32)
    expected = dim * count
    if values.size != expected:
        raise ValueError(f"f16 multivector has {values.size} values, expected {expected}")

    vectors = values.reshape((count, dim))
    vector_texts = [
        "[" + ",".join(format(float(value), ".9g") for value in vector) + "]"
        for vector in vectors
    ]
    return f"turbohybrid_multivector(dim={dim},count={count},values=[{','.join(vector_texts)}])"


def import_documents(
    conn: psycopg.Connection[Any],
    files: list[Path],
    batch_size: int,
    max_docs: int,
) -> int:
    def rows() -> Iterable[tuple[str, str, str, str]]:
        emitted = 0
        for row in iter_parquet_rows(files, batch_size):
            if max_docs > 0 and emitted >= max_docs:
                return
            emitted += 1
            yield (
                str(row["doc_id"]),
                row.get("title") or "",
                row.get("body") or "",
                row_multivector_text(row),
            )

    return copy_rows(
        conn,
        "COPY dbpedia_colbert_docs (doc_id, title, body, colbert) FROM STDIN",
        rows(),
    )


def import_queries(
    conn: psycopg.Connection[Any],
    files: list[Path],
    batch_size: int,
    max_queries: int,
) -> int:
    def rows() -> Iterable[tuple[str, str, str]]:
        emitted = 0
        for row in iter_parquet_rows(files, batch_size):
            if max_queries > 0 and emitted >= max_queries:
                return
            emitted += 1
            yield (
                str(row["query_id"]),
                str(row["query_text"]),
                row_multivector_text(row),
            )

    return copy_rows(
        conn,
        "COPY dbpedia_colbert_queries (query_id, query_text, colbert) FROM STDIN",
        rows(),
    )


def import_qrels(
    conn: psycopg.Connection[Any],
    files: list[Path],
    batch_size: int,
) -> int:
    def rows() -> Iterable[tuple[str, str, int]]:
        for row in iter_parquet_rows(files, batch_size):
            yield str(row["query_id"]), str(row["doc_id"]), int(row["relevance"])

    return copy_rows(
        conn,
        """
        COPY dbpedia_colbert_qrels (query_id, doc_id, relevance)
        FROM STDIN
        """,
        rows(),
    )


def import_precomputed_dataset_to_postgres(
    *,
    conn: psycopg.Connection[Any],
    source: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    batch_size: int = 4096,
    max_docs: int = 0,
    max_queries: int = 0,
    force_reload: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_dir = resolve_source(source, cache_dir)
    setup_schema(conn)
    if force_reload:
        exec_sql(conn, "DROP INDEX IF EXISTS dbpedia_colbert_docs_colbert_idx")
        exec_sql(conn, "TRUNCATE dbpedia_colbert_qrels, dbpedia_colbert_queries, dbpedia_colbert_docs")

    doc_count = import_documents(conn, parquet_files(dataset_dir, "docs"), batch_size, max_docs)
    query_count = import_queries(conn, parquet_files(dataset_dir, "queries"), batch_size, max_queries)
    qrel_count = import_qrels(conn, parquet_files(dataset_dir, "qrels"), batch_size)
    exec_sql(
        conn,
        """
        DELETE FROM dbpedia_colbert_qrels q
        WHERE NOT EXISTS (
          SELECT 1 FROM dbpedia_colbert_docs d WHERE d.doc_id = q.doc_id
        )
        OR NOT EXISTS (
          SELECT 1 FROM dbpedia_colbert_queries bq WHERE bq.query_id = q.query_id
        )
        """,
    )
    filtered_qrels = fetch_one(conn, "SELECT count(*) FROM dbpedia_colbert_qrels")
    dims = fetch_one(
        conn,
        """
        SELECT
          min(turbohybrid_multivector_dims(colbert)),
          max(turbohybrid_multivector_dims(colbert)),
          min(turbohybrid_multivector_count(colbert)),
          max(turbohybrid_multivector_count(colbert))
        FROM dbpedia_colbert_docs
        WHERE colbert IS NOT NULL
        """,
    )
    return {
        "source": str(dataset_dir),
        "documents": doc_count,
        "queries": query_count,
        "qrels": int(filtered_qrels[0]) if filtered_qrels else 0,
        "input_qrels": qrel_count,
        "filtered_qrels": max(qrel_count - (int(filtered_qrels[0]) if filtered_qrels else 0), 0),
        "document_dim_min": int(dims[0]) if dims and dims[0] is not None else 0,
        "document_dim_max": int(dims[1]) if dims and dims[1] is not None else 0,
        "document_vectors_min": int(dims[2]) if dims and dims[2] is not None else 0,
        "document_vectors_max": int(dims[3]) if dims and dims[3] is not None else 0,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def import_dataset(args: argparse.Namespace) -> dict[str, Any]:
    conn = connect(args.database)
    try:
        return import_precomputed_dataset_to_postgres(
            conn=conn,
            source=args.source,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
            max_docs=args.max_docs,
            max_queries=args.max_queries,
            force_reload=not args.reuse_existing,
        )
    finally:
        conn.close()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", DEFAULT_DATABASE))
    parser.add_argument("--batch-size", type=int, default=4096)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export", help="export encoded PostgreSQL rows to a HF dataset directory")
    add_common_args(export)
    export.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    export.add_argument("--force", action="store_true")
    export.add_argument("--progress-every", type=int, default=10000)
    export.add_argument("--repo-id", default=None, help="optional Hugging Face dataset repo id, e.g. owner/name")
    export.add_argument("--upload", action="store_true", help="upload output directory to --repo-id after export")
    export.add_argument("--private", action="store_true", help="create the Hugging Face dataset as private")
    export.add_argument("--expected-dim", type=int, default=128)
    export.add_argument("--max-doc-vectors", type=int, default=256)
    export.add_argument("--max-query-vectors", type=int, default=32)
    export.add_argument(
        "--embedding-format",
        choices=[EMBEDDING_FORMAT_F16, EMBEDDING_FORMAT_TEXT],
        default=EMBEDDING_FORMAT_F16,
        help="embedding storage format for docs/queries parquet files",
    )

    import_cmd = subparsers.add_parser("import", help="import a local or Hugging Face dataset into PostgreSQL")
    add_common_args(import_cmd)
    import_cmd.add_argument("--source", required=True, help="local dataset directory or Hugging Face dataset repo id")
    import_cmd.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    import_cmd.add_argument("--max-docs", type=int, default=0, help="limit imported documents; 0 imports all")
    import_cmd.add_argument("--max-queries", type=int, default=0, help="limit imported queries; 0 imports all")
    import_cmd.add_argument("--reuse-existing", action="store_true", help="do not truncate existing benchmark tables")

    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "export":
        result = export_dataset(args)
    elif args.command == "import":
        result = import_dataset(args)
    else:  # pragma: no cover - argparse enforces this
        raise SystemExit(f"unknown command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
