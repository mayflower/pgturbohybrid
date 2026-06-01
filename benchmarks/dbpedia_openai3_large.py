#!/usr/bin/env python3
"""Run DBPedia OpenAI3-large hybrid and dense-only benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


DIMENSIONS = 3072
FINAL_K = 10
RRF_K = 60
EMBEDDING_MODEL = "text-embedding-3-large"
CORPUS_EMBEDDING_FIELD = "text-embedding-3-large-3072-embedding"
DEFAULT_METHODS = "postgres_sql_rrf_halfvec,pgturbohybrid,pgturbohybrid_quality"
DEFAULT_QUERY_SOURCE = "qdrant-self"
DENSE_ONLY_METHODS = {
    "pgvector_halfvec_dense_only",
    "pgturbohybrid_dense_only",
    "pgturbohybrid_dense_adaptive_off",
    "pgturbohybrid_dense_exact_storage_on",
    "pgturbohybrid_dense_exact_build",
    "pgturbohybrid_dense_backbone",
    "pgturbohybrid_dense_adaptive_auto_1_25",
    "pgturbohybrid_dense_adaptive_auto_1_5",
    "pgturbohybrid_dense_adaptive_auto_2_0",
    "pgturbohybrid_dense_adaptive_on_2_0",
    "pgturbohybrid_dense_local_expansion_on_4",
    "pgturbohybrid_dense_local_expansion_on_8",
    "pgturbohybrid_dense_local_expansion_on_16",
    "pgturbohybrid_dense_local_expansion_auto_8",
    "pgturbohybrid_dense_entry_sidecar_64",
    "pgturbohybrid_dense_entry_sidecar_128",
    "pgturbohybrid_dense_entry_sidecar_256",
    "pgturbohybrid_dense_residual_rerank_16",
    "pgturbohybrid_dense_residual_rerank_32",
    "pgturbohybrid_dense_residual_rerank_64",
}

TURBOHYBRID_DEFAULT_INDEX_METHODS = {
    "pgturbohybrid",
    "pgturbohybrid_dense_only",
    "pgturbohybrid_dense_adaptive_off",
    "pgturbohybrid_dense_adaptive_auto_1_25",
    "pgturbohybrid_dense_adaptive_auto_1_5",
    "pgturbohybrid_dense_adaptive_auto_2_0",
    "pgturbohybrid_dense_adaptive_on_2_0",
    "pgturbohybrid_dense_local_expansion_on_4",
    "pgturbohybrid_dense_local_expansion_on_8",
    "pgturbohybrid_dense_local_expansion_on_16",
    "pgturbohybrid_dense_local_expansion_auto_8",
}


def method_build_key(method: str) -> str:
    if method in {"pgvector_halfvec_dense_only", "postgres_sql_rrf_halfvec"}:
        return method
    if method in TURBOHYBRID_DEFAULT_INDEX_METHODS:
        return "pgturbohybrid_default_4bit_exact_off"
    if method == "pgturbohybrid_dense_exact_storage_on":
        return "pgturbohybrid_default_4bit_exact_on"
    if method == "pgturbohybrid_dense_exact_build":
        return "pgturbohybrid_exact_build_4bit_exact_off"
    if method == "pgturbohybrid_dense_backbone":
        return "pgturbohybrid_backbone_4bit_exact_off"
    if method in {"pgturbohybrid_latency_explicit", "pgturbohybrid_exact_storage_off"}:
        return "pgturbohybrid_default_4bit_exact_off"
    if method == "pgturbohybrid_dense_entry_sidecar_64":
        return "pgturbohybrid_entry_sidecar_64_4bit_exact_off"
    if method == "pgturbohybrid_dense_entry_sidecar_128":
        return "pgturbohybrid_entry_sidecar_128_4bit_exact_off"
    if method == "pgturbohybrid_dense_entry_sidecar_256":
        return "pgturbohybrid_entry_sidecar_256_4bit_exact_off"
    if method == "pgturbohybrid_dense_residual_rerank_16":
        return "pgturbohybrid_residual_rerank_16_4bit_exact_off"
    if method == "pgturbohybrid_dense_residual_rerank_32":
        return "pgturbohybrid_residual_rerank_32_4bit_exact_off"
    if method == "pgturbohybrid_dense_residual_rerank_64":
        return "pgturbohybrid_residual_rerank_64_4bit_exact_off"
    if method == "pgturbohybrid_quality":
        return "pgturbohybrid_quality_4bit_exact_on"
    return method


def method_build_provenance(
    method: str,
    *,
    build_source: str,
    index_stats: dict[str, Any] | None = None,
    reused_from_build_key: str | None = None,
) -> dict[str, Any]:
    """Describe how an index was created, or what can be inferred about it."""
    is_turbohybrid = method.startswith("pgturbohybrid")
    provenance: dict[str, Any] = {
        "build_source": build_source,
        "method": method,
        "build_key": method_build_key(method),
        "reused_from_build_key": reused_from_build_key,
        "provenance_confidence": "fresh_build"
        if build_source == "fresh_build"
        else "shared_build_in_run"
        if build_source == "shared_build_in_run"
        else "failed"
        if build_source == "failed"
        else "existing_index_stats_only",
    }

    if method in {"pgvector_halfvec_dense_only", "postgres_sql_rrf_halfvec"}:
        provenance.update({
            "index_type": "pgvector_halfvec_hnsw",
            "expression_type": "halfvec(3072)",
        })
        return provenance

    if not is_turbohybrid:
        provenance["index_type"] = "unknown"
        return provenance

    sidecar_reps = {
        "pgturbohybrid_dense_entry_sidecar_64": 64,
        "pgturbohybrid_dense_entry_sidecar_128": 128,
        "pgturbohybrid_dense_entry_sidecar_256": 256,
    }.get(method)
    residual_bytes = {
        "pgturbohybrid_dense_residual_rerank_16": 16,
        "pgturbohybrid_dense_residual_rerank_32": 32,
        "pgturbohybrid_dense_residual_rerank_64": 64,
    }.get(method)

    configured_exact_build = method == "pgturbohybrid_dense_exact_build"
    observed_exact_build = index_stats.get("dense_build_exact_distances") if index_stats else None
    configured_backbone = method == "pgturbohybrid_dense_backbone"
    observed_backbone = index_stats.get("graph_backbone") if index_stats else None
    provenance.update({
        "index_type": "pgturbohybrid",
        "quantization_bits": 4,
        "exact_storage": method in {"pgturbohybrid_dense_exact_storage_on", "pgturbohybrid_quality"},
        "dense_build_exact_distances": observed_exact_build
        if observed_exact_build is not None
        else configured_exact_build
        if build_source != "existing_index"
        else None,
        "entry_sidecar": sidecar_reps is not None,
        "entry_sidecar_representatives": sidecar_reps or 0,
        "graph_backbone": observed_backbone
        if observed_backbone is not None
        else configured_backbone
        if build_source != "existing_index"
        else None,
        "residual_rerank": residual_bytes is not None,
        "residual_rerank_bytes": residual_bytes or 0,
    })

    if index_stats:
        provenance.update({
            "index_storage_kind": index_stats.get("storage_kind"),
            "index_exact_storage": index_stats.get("exact_storage"),
            "index_quantization_bits": index_stats.get("quantization_bits"),
            "index_graph_m": index_stats.get("graph_m"),
            "index_graph_ef_construction": index_stats.get("graph_ef_construction"),
            "index_graph_ef_search": index_stats.get("graph_ef_search"),
            "index_dense_build_exact_distances": observed_exact_build,
            "index_graph_backbone": observed_backbone,
            "index_entry_sidecar_count": index_stats.get("entry_sidecar_count"),
            "index_residual_rerank_bytes": index_stats.get("residual_rerank_bytes"),
        })

    return provenance


def run_psql(database: str, sql: str, *, quiet: bool = True) -> str:
    cmd = ["psql", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-d", database, "-c", sql]
    if quiet:
        cmd.insert(1, "-q")
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout.strip()


def run_psql_optional(database: str, sql: str, default: Any = None) -> Any:
    try:
        return run_psql(database, sql)
    except subprocess.CalledProcessError:
        return default


def run_psql_file(database: str, path: Path) -> str:
    cmd = ["psql", "-q", "-X", "-v", "ON_ERROR_STOP=1", "-d", database, "-f", str(path)]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()
    return result.stdout


def portable_path(path: Path) -> str:
    """Prefer repo-relative paths in generated benchmark metadata."""
    try:
        return str(path.absolute().relative_to(Path.cwd().absolute()))
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def portable_argv(argv: list[str]) -> list[str]:
    """Keep command metadata useful without forcing repo-local absolute paths."""
    portable = []
    for item in argv:
        if item.startswith(("/", "./", "../")):
            portable.append(portable_path(Path(item)))
        else:
            portable.append(item)
    return portable


def benchmark_command_metadata() -> dict[str, Any]:
    """Record the benchmark invocation in a reproducible, repo-relative form."""
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
            "DBPEDIA_DATASET": os.environ.get("DBPEDIA_DATASET", ""),
            "OUTPUT": os.environ.get("OUTPUT", ""),
            "JSONL_OUTPUT": os.environ.get("JSONL_OUTPUT", ""),
            "INDEX_OUTPUT": os.environ.get("INDEX_OUTPUT", ""),
            "MARKDOWN_OUTPUT": os.environ.get("MARKDOWN_OUTPUT", ""),
            "PG_CONFIG": os.environ.get("PG_CONFIG", ""),
            "PGVECTOR_REF": os.environ.get("PGVECTOR_REF", ""),
        },
    }


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to read Hugging Face parquet shards. "
            "Install it with: python3 -m pip install pyarrow"
        ) from exc
    return pq


def vector_literal(values: Iterable[float]) -> str:
    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


def validate_embedding(values: Any, label: str) -> list[float]:
    if not isinstance(values, list):
        try:
            values = values.as_py()
        except AttributeError as exc:
            raise ValueError(f"{label} does not contain an embedding array") from exc
    if len(values) != DIMENSIONS:
        raise ValueError(f"{label} has {len(values)} dimensions; expected {DIMENSIONS}")
    return [float(value) for value in values]


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


def find_parquet_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    unique = sorted({path.resolve() for path in files if path.is_file()})
    return unique


def find_corpus_parquet(dataset: Path) -> list[Path]:
    files = find_parquet_files(dataset, ("data/*.parquet", "train/*.parquet", "*.parquet", "default/train/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Qdrant DBPedia parquet shards found under {dataset}")
    return files


def find_query_parquet(beir_dataset: Path) -> list[Path]:
    files = find_parquet_files(beir_dataset, ("queries/*.parquet", "*queries*.parquet", "*.parquet"))
    if not files:
        raise FileNotFoundError(f"no BEIR DBPedia query parquet file found under {beir_dataset}")
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
    raise FileNotFoundError(
        "could not find BEIR DBPedia qrels; pass --qrels or set BEIR_DBPEDIA_QRELS"
    )


def load_queries(beir_dataset: Path) -> dict[str, str]:
    pq = require_pyarrow()
    queries: dict[str, str] = {}
    for path in find_query_parquet(beir_dataset):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["_id", "title", "text"]):
            data = batch.to_pylist()
            for row in data:
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


def read_query_embeddings(path: Path) -> dict[str, list[float]]:
    embeddings: dict[str, list[float]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)
            query_id = str(row.get("id") or row.get("_id"))
            values = row.get("values") or row.get("embedding")
            embeddings[query_id] = validate_embedding(values, f"{path}:{line_no}")
    if not embeddings:
        raise ValueError(f"no query embeddings loaded from {path}")
    return embeddings


def prepare_query_embeddings(args: argparse.Namespace) -> None:
    output = Path(args.query_embeddings)
    if output.is_file() and not args.force_query_embeddings:
        print(f"query embeddings already exist: {output}")
        return
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "the openai package is required for --prepare-query-embeddings. "
            "Install it with: python3 -m pip install openai"
        ) from exc

    queries = load_queries(Path(args.beir_dataset).resolve())
    qrels = read_qrels(resolve_qrels_path(args.qrels, Path(args.beir_dataset).resolve()))
    qids = sorted(qid for qid in qrels if qid in queries)
    if args.max_queries > 0:
        qids = qids[:args.max_queries]

    client = OpenAI()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for offset in range(0, len(qids), args.embedding_batch_size):
            batch_ids = qids[offset:offset + args.embedding_batch_size]
            texts = [queries[qid] for qid in batch_ids]
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
            for qid, item in zip(batch_ids, response.data, strict=True):
                values = validate_embedding(list(item.embedding), qid)
                out.write(json.dumps({"id": qid, "values": values}, separators=(",", ":")) + "\n")
            print(f"embedded {min(offset + len(batch_ids), len(qids))}/{len(qids)} queries", file=sys.stderr)
    print(output)


def choose_query_ids(qrels: dict[str, dict[str, int]], queries: dict[str, str],
                     embeddings: dict[str, list[float]], max_queries: int) -> list[str]:
    qids = [qid for qid in sorted(qrels) if qid in queries and qid in embeddings]
    if max_queries > 0:
        qids = qids[:max_queries]
    if not qids:
        raise ValueError("no benchmark queries have both text, qrels, and embeddings")
    return qids


def copy_from_rows(database: str, table: str, columns: str,
                   rows: Iterable[list[Any]], force_not_null: tuple[str, ...] = ()) -> int:
    options = "FORMAT csv, DELIMITER E'\\t'"
    if force_not_null:
        options += ", FORCE_NOT_NULL (" + ", ".join(force_not_null) + ")"
    sql = f"\\copy {table} ({columns}) FROM STDIN WITH ({options})"
    proc = subprocess.Popen(
        ["psql", "-q", "-X", "-v", "ON_ERROR_STOP=1", "-d", database, "-c", sql],
        stdin=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None
    writer = csv.writer(proc.stdin, delimiter="\t", lineterminator="\n")
    count = 0
    try:
        for row in rows:
            writer.writerow(row)
            count += 1
    except BrokenPipeError:
        pass
    finally:
        proc.stdin.close()
    returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, proc.args)
    return count


def corpus_rows(dataset: Path, max_docs: int) -> Iterable[list[Any]]:
    pq = require_pyarrow()
    emitted = 0
    for path in find_corpus_parquet(dataset):
        parquet = pq.ParquetFile(path)
        columns = ["_id", "title", "text", CORPUS_EMBEDDING_FIELD]
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            for row in batch.to_pylist():
                if max_docs > 0 and emitted >= max_docs:
                    return
                doc_id = str(row["_id"])
                title = row.get("title") or ""
                body = row.get("text") or ""
                embedding = validate_embedding(row[CORPUS_EMBEDDING_FIELD], f"{path}:{doc_id}")
                emitted += 1
                yield [doc_id, title, body, vector_literal(embedding)]


def query_rows(qids: list[str], queries: dict[str, str],
               embeddings: dict[str, list[float]]) -> Iterable[list[Any]]:
    for qid in qids:
        yield [qid, queries[qid], vector_literal(embeddings[qid])]


def qrel_rows(qids: list[str], qrels: dict[str, dict[str, int]]) -> Iterable[list[Any]]:
    for qid in qids:
        for doc_id, score in sorted(qrels.get(qid, {}).items()):
            yield [qid, doc_id, score]


def setup_database_beir(args: argparse.Namespace, qrels: dict[str, dict[str, int]],
                        queries: dict[str, str], embeddings: dict[str, list[float]]) -> dict[str, int]:
    if args.reuse_data:
        exists = run_psql(args.database, "SELECT to_regclass('dbpedia_docs') IS NOT NULL;")
        if exists == "t":
            rows = int(run_psql(args.database, "SELECT count(*) FROM dbpedia_docs;"))
            query_count = int(run_psql(args.database, "SELECT count(*) FROM dbpedia_queries;"))
            qrel_count = int(run_psql(args.database, "SELECT count(*) FROM dbpedia_qrels;"))
            if rows > 0 and query_count > 0 and qrel_count > 0:
                return {"rows": rows, "queries": query_count, "qrels": qrel_count}
            print("ignoring empty reused DBPedia tables and reloading data", file=sys.stderr)

    qids = choose_query_ids(qrels, queries, embeddings, args.max_queries)
    run_psql(args.database, f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
DROP TABLE IF EXISTS dbpedia_run_results;
DROP TABLE IF EXISTS dbpedia_run_timings;
DROP TABLE IF EXISTS dbpedia_qrels;
DROP TABLE IF EXISTS dbpedia_queries;
DROP TABLE IF EXISTS dbpedia_docs;
CREATE TABLE dbpedia_docs (
    doc_id text PRIMARY KEY,
    title text NOT NULL,
    body text NOT NULL,
    embedding vector({DIMENSIONS}) NOT NULL,
    body_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(body, ''))
    ) STORED
);
CREATE TABLE dbpedia_queries (
    query_id text PRIMARY KEY,
    query_text text NOT NULL,
    embedding vector({DIMENSIONS}) NOT NULL
);
CREATE TABLE dbpedia_qrels (
    query_id text NOT NULL,
    doc_id text NOT NULL,
    relevance int NOT NULL,
    PRIMARY KEY (query_id, doc_id)
);
""")

    dataset = Path(args.dataset).resolve()
    doc_count = copy_from_rows(
        args.database,
        "dbpedia_docs",
        "doc_id, title, body, embedding",
        corpus_rows(dataset, args.max_docs),
        force_not_null=("title", "body"),
    )
    query_count = copy_from_rows(
        args.database,
        "dbpedia_queries",
        "query_id, query_text, embedding",
        query_rows(qids, queries, embeddings),
        force_not_null=("query_text",),
    )
    qrel_count = copy_from_rows(
        args.database,
        "dbpedia_qrels",
        "query_id, doc_id, relevance",
        qrel_rows(qids, qrels),
    )
    run_psql(args.database, "ANALYZE dbpedia_docs; ANALYZE dbpedia_queries; ANALYZE dbpedia_qrels;")
    return {"rows": doc_count, "queries": query_count, "qrels": qrel_count}


def setup_database_qdrant_self(args: argparse.Namespace) -> dict[str, int]:
    if args.reuse_data:
        exists = run_psql(args.database, "SELECT to_regclass('dbpedia_docs') IS NOT NULL;")
        if exists == "t":
            rows = int(run_psql(args.database, "SELECT count(*) FROM dbpedia_docs;"))
            query_count = int(run_psql(args.database, "SELECT count(*) FROM dbpedia_queries;"))
            qrel_count = int(run_psql(args.database, "SELECT count(*) FROM dbpedia_qrels;"))
            if rows > 0 and query_count > 0 and qrel_count > 0:
                return {"rows": rows, "queries": query_count, "qrels": qrel_count}
            print("ignoring empty reused DBPedia tables and reloading data", file=sys.stderr)

    run_psql(args.database, f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
DROP TABLE IF EXISTS dbpedia_run_results;
DROP TABLE IF EXISTS dbpedia_run_timings;
DROP TABLE IF EXISTS dbpedia_qrels;
DROP TABLE IF EXISTS dbpedia_queries;
DROP TABLE IF EXISTS dbpedia_docs;
CREATE TABLE dbpedia_docs (
    doc_id text PRIMARY KEY,
    title text NOT NULL,
    body text NOT NULL,
    embedding vector({DIMENSIONS}) NOT NULL,
    body_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(body, ''))
    ) STORED
);
CREATE TABLE dbpedia_queries (
    query_id text PRIMARY KEY,
    query_text text NOT NULL,
    embedding vector({DIMENSIONS}) NOT NULL
);
CREATE TABLE dbpedia_qrels (
    query_id text NOT NULL,
    doc_id text NOT NULL,
    relevance int NOT NULL,
    PRIMARY KEY (query_id, doc_id)
);
""")

    dataset = Path(args.dataset).resolve()
    doc_count = copy_from_rows(
        args.database,
        "dbpedia_docs",
        "doc_id, title, body, embedding",
        corpus_rows(dataset, args.max_docs),
        force_not_null=("title", "body"),
    )
    query_limit = args.max_queries if args.max_queries > 0 else min(doc_count, 1000)
    run_psql(args.database, f"""
INSERT INTO dbpedia_queries(query_id, query_text, embedding)
WITH candidates AS (
    SELECT doc_id,
           CASE
               WHEN btrim(title) <> '' THEN title
               ELSE left(regexp_replace(body, '\\s+', ' ', 'g'), 240)
           END AS query_text,
           embedding
    FROM dbpedia_docs
)
SELECT doc_id, query_text, embedding
FROM candidates
WHERE length(to_tsvector('english', query_text)::text) > 0
ORDER BY doc_id
LIMIT {query_limit};

INSERT INTO dbpedia_qrels(query_id, doc_id, relevance)
SELECT query_id, query_id, 1
FROM dbpedia_queries;

ANALYZE dbpedia_docs;
ANALYZE dbpedia_queries;
ANALYZE dbpedia_qrels;
""")
    query_count = int(run_psql(args.database, "SELECT count(*) FROM dbpedia_queries;"))
    qrel_count = int(run_psql(args.database, "SELECT count(*) FROM dbpedia_qrels;"))
    return {"rows": doc_count, "queries": query_count, "qrels": qrel_count}


def filtered_qrels(database: str) -> dict[str, dict[str, int]]:
    text = run_psql(database, """
SELECT q.query_id, q.doc_id, q.relevance
FROM dbpedia_qrels q
JOIN dbpedia_queries bq ON bq.query_id = q.query_id
JOIN dbpedia_docs d ON d.doc_id = q.doc_id
WHERE q.relevance > 0
ORDER BY q.query_id, q.doc_id;
""")
    filtered: dict[str, dict[str, int]] = {}
    for line in text.splitlines():
        qid, doc_id, relevance = line.split("|")
        filtered.setdefault(qid, {})[doc_id] = int(relevance)
    return filtered


def method_profile(method: str, requested_profile: str) -> str:
    if method == "pgturbohybrid_quality":
        return "quality"
    if method.startswith("pgturbohybrid"):
        return requested_profile
    return ""


def build_index(database: str, method: str) -> dict[str, Any]:
    common_drop = """
DROP INDEX IF EXISTS dbpedia_hnsw_halfvec_idx;
DROP INDEX IF EXISTS dbpedia_fts_idx;
DROP INDEX IF EXISTS dbpedia_turbohybrid_idx;
"""
    run_psql(database, common_drop)
    if method in {"pgvector_halfvec_dense_only", "postgres_sql_rrf_halfvec"}:
        sql = """
CREATE INDEX dbpedia_hnsw_halfvec_idx ON dbpedia_docs
USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops);
"""
        indexes = ["dbpedia_hnsw_halfvec_idx"]
        if method == "postgres_sql_rrf_halfvec":
            sql += "CREATE INDEX dbpedia_fts_idx ON dbpedia_docs USING gin (body_tsv);\n"
            indexes.append("dbpedia_fts_idx")
    elif method in TURBOHYBRID_DEFAULT_INDEX_METHODS | {
        "pgturbohybrid_dense_exact_build",
        "pgturbohybrid_dense_backbone",
        "pgturbohybrid_dense_exact_storage_on",
        "pgturbohybrid_dense_entry_sidecar_64",
        "pgturbohybrid_dense_entry_sidecar_128",
        "pgturbohybrid_dense_entry_sidecar_256",
        "pgturbohybrid_dense_residual_rerank_16",
        "pgturbohybrid_dense_residual_rerank_32",
        "pgturbohybrid_dense_residual_rerank_64",
    }:
        exact_build = method == "pgturbohybrid_dense_exact_build"
        exact_storage = "on" if method == "pgturbohybrid_dense_exact_storage_on" else "off"
        backbone_options = ",\n      graph_backbone = on" if method == "pgturbohybrid_dense_backbone" else ""
        sidecar_reps = {
            "pgturbohybrid_dense_entry_sidecar_64": 64,
            "pgturbohybrid_dense_entry_sidecar_128": 128,
            "pgturbohybrid_dense_entry_sidecar_256": 256,
        }.get(method)
        sidecar_options = ""
        if sidecar_reps is not None:
            sidecar_options = (
                ",\n      entry_sidecar = on,"
                f"\n      entry_sidecar_representatives = {sidecar_reps}"
            )
        residual_bytes = {
            "pgturbohybrid_dense_residual_rerank_16": 16,
            "pgturbohybrid_dense_residual_rerank_32": 32,
            "pgturbohybrid_dense_residual_rerank_64": 64,
        }.get(method)
        residual_options = ""
        if residual_bytes is not None:
            residual_options = (
                ",\n      residual_rerank = on,"
                f"\n      residual_rerank_bytes = {residual_bytes}"
            )
        sql = f"""
SET turbohybrid.dense_build_exact_distances = {'on' if exact_build else 'off'};
CREATE INDEX dbpedia_turbohybrid_idx ON dbpedia_docs
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (
    quantization_bits = 4,
    exact_storage = {exact_storage}{backbone_options}{sidecar_options}{residual_options}
);
RESET turbohybrid.dense_build_exact_distances;
"""
        indexes = ["dbpedia_turbohybrid_idx"]
    elif method in {"pgturbohybrid_latency_explicit", "pgturbohybrid_exact_storage_off"}:
        sql = """
CREATE INDEX dbpedia_turbohybrid_idx ON dbpedia_docs
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = off);
"""
        indexes = ["dbpedia_turbohybrid_idx"]
    elif method == "pgturbohybrid_quality":
        sql = """
CREATE INDEX dbpedia_turbohybrid_idx ON dbpedia_docs
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = on);
"""
        indexes = ["dbpedia_turbohybrid_idx"]
    else:
        raise ValueError(f"unknown method: {method}")

    before_lsn = run_psql(database, "SELECT pg_current_wal_lsn();")
    start = time.perf_counter()
    run_psql(database, sql)
    build_ms = (time.perf_counter() - start) * 1000.0
    after_lsn = run_psql(database, "SELECT pg_current_wal_lsn();")
    wal_bytes = float(run_psql(database, f"SELECT pg_wal_lsn_diff('{after_lsn}', '{before_lsn}');"))
    index_bytes = int(run_psql(
        database,
        "SELECT COALESCE(sum(pg_relation_size(indexrelid)), 0) "
        "FROM pg_index WHERE indexrelid::regclass::text IN ("
        + ",".join(sql_literal(item) for item in indexes)
        + ");",
    ) or "0")
    index_stats = None
    if "dbpedia_turbohybrid_idx" in indexes:
        index_stats_text = run_psql(
            database,
            "SELECT turbohybrid_index_stats('dbpedia_turbohybrid_idx'::regclass)::text;",
        )
        index_stats = json.loads(index_stats_text) if index_stats_text else None
    return {
        "build_ms": round(build_ms, 3),
        "build_wal_bytes": int(wal_bytes),
        "index_bytes": index_bytes,
        "index_stats": index_stats,
        "indexes": indexes,
        "build_provenance": method_build_provenance(
            method,
            build_source="fresh_build",
            index_stats=index_stats,
        ),
    }


def existing_index_info(database: str, method: str) -> dict[str, Any]:
    if method in {"pgvector_halfvec_dense_only", "postgres_sql_rrf_halfvec"}:
        indexes = ["dbpedia_hnsw_halfvec_idx"]
        if method == "postgres_sql_rrf_halfvec":
            indexes.append("dbpedia_fts_idx")
    elif method.startswith("pgturbohybrid"):
        indexes = ["dbpedia_turbohybrid_idx"]
    else:
        indexes = []
    index_bytes = int(run_psql(
        database,
        "SELECT COALESCE(sum(pg_relation_size(indexrelid)), 0) "
        "FROM pg_index WHERE indexrelid::regclass::text IN ("
        + ",".join(sql_literal(item) for item in indexes)
        + ");",
    ) or "0")
    index_stats = None
    if "dbpedia_turbohybrid_idx" in indexes:
        index_stats_text = run_psql(
            database,
            "SELECT turbohybrid_index_stats('dbpedia_turbohybrid_idx'::regclass)::text;",
        )
        index_stats = json.loads(index_stats_text) if index_stats_text else None
    return {
        "build_ms": 0.0,
        "build_wal_bytes": 0,
        "index_bytes": index_bytes,
        "index_stats": index_stats,
        "indexes": indexes,
        "build_reused_existing_index": True,
        "build_provenance": method_build_provenance(
            method,
            build_source="existing_index",
            index_stats=index_stats,
        ),
    }


def method_query(method: str, dense_k: int, bm25_k: int, final_k: int) -> str:
    if method == "pgvector_halfvec_dense_only":
        return f"""
SELECT doc_id
FROM dbpedia_docs
ORDER BY (embedding::halfvec({DIMENSIONS})) <=> %L::halfvec({DIMENSIONS})
LIMIT {final_k}
"""

    if method == "postgres_sql_rrf_halfvec":
        return f"""
WITH dense AS MATERIALIZED (
    SELECT doc_id, row_number() OVER () AS rank
    FROM (
        SELECT doc_id
        FROM dbpedia_docs
        ORDER BY (embedding::halfvec({DIMENSIONS})) <=> %L::halfvec({DIMENSIONS})
        LIMIT {dense_k}
    ) s
),
lexical AS MATERIALIZED (
    SELECT doc_id, row_number() OVER () AS rank
    FROM (
        SELECT doc_id
        FROM dbpedia_docs
        WHERE body_tsv @@ plainto_tsquery('english', %L)
        ORDER BY ts_rank_cd(body_tsv, plainto_tsquery('english', %L)) DESC, doc_id
        LIMIT {bm25_k}
    ) s
)
SELECT COALESCE(dense.doc_id, lexical.doc_id) AS doc_id
FROM dense
FULL OUTER JOIN lexical USING (doc_id)
ORDER BY COALESCE(1.0 / ({RRF_K} + dense.rank), 0.0) +
         COALESCE(1.0 / ({RRF_K} + lexical.rank), 0.0) DESC,
         COALESCE(dense.doc_id, lexical.doc_id)
LIMIT {final_k}
"""

    if method == "pgturbohybrid":
        return f"""
SELECT doc_id
FROM dbpedia_docs
ORDER BY embedding <~> turbohybrid_query(
    vector_query => %L::vector({DIMENSIONS}),
    text_query => plainto_tsquery('english', %L)
)
LIMIT {final_k}
"""

    if method in DENSE_ONLY_METHODS - {"pgvector_halfvec_dense_only"}:
        return f"""
SELECT doc_id
FROM dbpedia_docs
ORDER BY embedding <~> turbohybrid_query(
    vector_query => %L::vector({DIMENSIONS}),
    dense_k => {dense_k},
    final_k => {final_k}
)
LIMIT {final_k}
"""

    return f"""
SELECT doc_id
FROM dbpedia_docs
ORDER BY embedding <~> turbohybrid_query(
    vector_query => %L::vector({DIMENSIONS}),
    text_query => plainto_tsquery('english', %L),
    dense_k => {dense_k},
    bm25_k => {bm25_k},
    rrf_k => {RRF_K},
    final_k => {final_k}
)
LIMIT {final_k}
"""


def method_runtime_settings(method: str) -> dict[str, str]:
    if method == "pgturbohybrid_dense_adaptive_auto_1_25":
        return {
            "turbohybrid.dense_adaptive_widening": "auto",
            "turbohybrid.dense_adaptive_widening_multiplier": "1.25",
            "turbohybrid.dense_local_expansion": "off",
        }
    if method == "pgturbohybrid_dense_adaptive_auto_1_5":
        return {
            "turbohybrid.dense_adaptive_widening": "auto",
            "turbohybrid.dense_adaptive_widening_multiplier": "1.5",
            "turbohybrid.dense_local_expansion": "off",
        }
    if method == "pgturbohybrid_dense_adaptive_off":
        return {
            "turbohybrid.dense_adaptive_widening": "off",
            "turbohybrid.dense_local_expansion": "off",
        }
    if method == "pgturbohybrid_dense_adaptive_auto_2_0":
        return {
            "turbohybrid.dense_adaptive_widening": "auto",
            "turbohybrid.dense_adaptive_widening_multiplier": "2.0",
            "turbohybrid.dense_local_expansion": "off",
        }
    if method == "pgturbohybrid_dense_adaptive_on_2_0":
        return {
            "turbohybrid.dense_adaptive_widening": "on",
            "turbohybrid.dense_adaptive_widening_multiplier": "2.0",
            "turbohybrid.dense_local_expansion": "off",
        }
    local_expansion_variants = {
        "pgturbohybrid_dense_local_expansion_on_4": ("on", "4"),
        "pgturbohybrid_dense_local_expansion_on_8": ("on", "8"),
        "pgturbohybrid_dense_local_expansion_on_16": ("on", "16"),
        "pgturbohybrid_dense_local_expansion_auto_8": ("auto", "8"),
    }
    if method in local_expansion_variants:
        mode, topn = local_expansion_variants[method]
        return {
            "turbohybrid.dense_adaptive_widening": "off",
            "turbohybrid.dense_local_expansion": mode,
            "turbohybrid.dense_local_expansion_topn": topn,
            "turbohybrid.dense_local_expansion_max_neighbors": "256",
        }
    if method.startswith("pgturbohybrid"):
        return {}
    return {}


def run_method_queries(database: str, method: str, label: str, dense_k: int,
                       bm25_k: int, final_k: int, profile: str,
                       hnsw_ef_search: int, measured_runs: int,
                       record_results: bool, force_turbohybrid_index: bool) -> None:
    query_template = method_query(method, dense_k, bm25_k, final_k).replace("'", "''")
    is_turbohybrid = "true" if method.startswith("pgturbohybrid") else "false"
    profile_sql = method_profile(method, profile)
    record_results_sql = "true" if record_results else "false"
    force_turbohybrid_index_sql = "true" if force_turbohybrid_index else "false"
    turbohybrid_settings_sql = "\n".join(
        f"        PERFORM set_config({sql_literal(key)}, {sql_literal(value)}, false);"
        for key, value in method_runtime_settings(method).items()
    )
    sql = f"""
DO $$
DECLARE
    q record;
    r record;
    started timestamptz;
    elapsed float8;
    rank_no int;
    sql text;
    scan_stats jsonb;
    pass_no int;
BEGIN
    IF {hnsw_ef_search} > 0 THEN
        PERFORM set_config('hnsw.ef_search', '{hnsw_ef_search}', false);
    END IF;
    IF {is_turbohybrid} THEN
        PERFORM set_config('turbohybrid.profile', {sql_literal(profile_sql)}, false);
{turbohybrid_settings_sql}
    END IF;
    IF {is_turbohybrid} AND {force_turbohybrid_index_sql} THEN
        PERFORM set_config('enable_seqscan', 'off', true);
    END IF;
    FOR pass_no IN 1..{measured_runs} LOOP
        FOR q IN SELECT query_id, embedding::text AS embedding, query_text FROM dbpedia_queries ORDER BY query_id LOOP
            sql := format('{query_template}', q.embedding, q.query_text, q.query_text);
            started := clock_timestamp();
            rank_no := 0;
            FOR r IN EXECUTE sql LOOP
                rank_no := rank_no + 1;
                IF {record_results_sql} AND pass_no = 1 THEN
                    INSERT INTO dbpedia_run_results(method, query_id, rank, doc_id)
                    VALUES ({sql_literal(label)}, q.query_id, rank_no, r.doc_id);
                END IF;
            END LOOP;
            elapsed := EXTRACT(EPOCH FROM clock_timestamp() - started) * 1000.0;
            IF {is_turbohybrid} THEN
                scan_stats := turbohybrid_last_scan_stats();
            ELSE
                scan_stats := NULL;
            END IF;
            IF {record_results_sql} THEN
                INSERT INTO dbpedia_run_timings(method, pass_no, query_id, elapsed_ms, last_scan_stats)
                VALUES ({sql_literal(label)}, pass_no, q.query_id, elapsed, scan_stats);
            END IF;
        END LOOP;
    END LOOP;
END $$;
"""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as f:
        f.write(sql)
        path = Path(f.name)
    try:
        run_psql_file(database, path)
    finally:
        path.unlink(missing_ok=True)


def run_method_probe(database: str, method: str, label: str, dense_k: int,
                     bm25_k: int, probe_k: int, profile: str,
                     hnsw_ef_search: int, force_turbohybrid_index: bool) -> None:
    if probe_k <= 0:
        return
    query_template = method_query(method, dense_k, bm25_k, probe_k).replace("'", "''")
    is_turbohybrid = "true" if method.startswith("pgturbohybrid") else "false"
    profile_sql = method_profile(method, profile)
    force_turbohybrid_index_sql = "true" if force_turbohybrid_index else "false"
    turbohybrid_settings_sql = "\n".join(
        f"        PERFORM set_config({sql_literal(key)}, {sql_literal(value)}, false);"
        for key, value in method_runtime_settings(method).items()
    )
    hnsw_setting_sql = ""
    if hnsw_ef_search > 0:
        hnsw_setting_sql = f"    PERFORM set_config('hnsw.ef_search', '{hnsw_ef_search}', false);"
    sql = f"""
DO $$
DECLARE
    q record;
    r record;
    query_sql text;
    rank_no int;
BEGIN
{hnsw_setting_sql}
    IF {is_turbohybrid} THEN
        PERFORM set_config('turbohybrid.profile', {sql_literal(profile_sql)}, false);
{turbohybrid_settings_sql}
    END IF;
    IF {is_turbohybrid} AND {force_turbohybrid_index_sql} THEN
        PERFORM set_config('enable_seqscan', 'off', true);
    END IF;

    FOR q IN SELECT query_id, embedding::text AS embedding, query_text FROM dbpedia_queries ORDER BY query_id LOOP
        query_sql := format('{query_template}', q.embedding, q.query_text, q.query_text);
        rank_no := 0;
        FOR r IN EXECUTE query_sql LOOP
            rank_no := rank_no + 1;
            INSERT INTO dbpedia_probe_results(method, query_id, rank, doc_id)
            VALUES ({sql_literal(label)}, q.query_id, rank_no, r.doc_id);
        END LOOP;
    END LOOP;
END $$;
"""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as f:
        f.write(sql)
        path = Path(f.name)
    try:
        run_psql_file(database, path)
    finally:
        path.unlink(missing_ok=True)


def fetch_method_output(database: str, label: str) -> tuple[dict[str, list[str]], list[float], dict[str, Any]]:
    rows = run_psql(database, f"""
SELECT query_id, rank, doc_id
FROM dbpedia_run_results
WHERE method = {sql_literal(label)}
ORDER BY query_id, rank;
""")
    runs: dict[str, list[str]] = {}
    for line in rows.splitlines():
        if not line:
            continue
        qid, _rank, doc_id = line.split("|")
        runs.setdefault(qid, []).append(doc_id)

    timing_rows = run_psql(database, f"""
SELECT elapsed_ms
FROM dbpedia_run_timings
WHERE method = {sql_literal(label)}
ORDER BY pass_no, query_id;
""")
    timings = [float(line) for line in timing_rows.splitlines() if line]
    scan_summary_text = run_psql(database, f"""
SELECT jsonb_build_object(
    'profile', max(last_scan_stats->>'profile'),
    'index_used', bool_or((last_scan_stats->>'index_used')::bool),
    'dense_k_effective_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'dense_k_effective')::float8)
        ),
    'dense_k_effective_max', max((last_scan_stats->>'dense_k_effective')::int),
    'bm25_k_effective_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'bm25_k_effective')::float8)
        ),
    'bm25_k_effective_max', max((last_scan_stats->>'bm25_k_effective')::int),
    'final_k_effective_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'final_k_effective')::float8)
        ),
    'final_k_source', max(last_scan_stats->>'final_k_source'),
    'final_k_inferred_count',
        count(*) FILTER (WHERE (last_scan_stats->>'final_k_inferred')::bool),
    'bm25_cache_hit_count',
        count(*) FILTER (WHERE (last_scan_stats->>'bm25_cache_hit')::bool),
    'graph_visited_nodes_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'graph_visited_nodes')::float8)
        ),
    'graph_visited_nodes_max', max((last_scan_stats->>'graph_visited_nodes')::int),
    'graph_scored_codes_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'graph_scored_codes')::float8)
        ),
    'graph_scored_codes_max', max((last_scan_stats->>'graph_scored_codes')::int),
    'graph_candidate_count_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'graph_candidate_count')::float8)
        ),
    'graph_candidate_count_max', max((last_scan_stats->>'graph_candidate_count')::int),
    'graph_effective_result_target_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'graph_effective_result_target')::float8)
        ),
    'graph_effective_search_ef_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'graph_effective_search_ef')::float8)
        ),
    'dense_adaptive_widening_mode', max(last_scan_stats->>'dense_adaptive_widening_mode'),
    'dense_adaptive_trigger_count',
        count(*) FILTER (WHERE (last_scan_stats->>'dense_adaptive_triggered')::bool),
    'dense_adaptive_trigger_reason', max(last_scan_stats->>'dense_adaptive_trigger_reason'),
    'dense_adaptive_final_result_target_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'dense_adaptive_final_result_target')::float8)
        ),
    'dense_adaptive_final_search_ef_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'dense_adaptive_final_search_ef')::float8)
        ),
    'dense_adaptive_gap_top10_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'dense_adaptive_gap_top10')::float8)
        ),
    'dense_adaptive_gap_boundary_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'dense_adaptive_gap_boundary')::float8)
        ),
    'dense_local_expansion_mode', max(last_scan_stats->>'dense_local_expansion_mode'),
    'dense_local_expansion_trigger_count',
        count(*) FILTER (WHERE (last_scan_stats->>'dense_local_expansion_triggered')::bool),
    'dense_local_expansion_seed_count_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'dense_local_expansion_seed_count')::float8)
        ),
    'dense_local_expansion_neighbors_scored_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'dense_local_expansion_neighbors_scored')::float8)
        ),
    'dense_local_expansion_neighbors_scored_max',
        max((last_scan_stats->>'dense_local_expansion_neighbors_scored')::int),
    'dense_local_expansion_candidates_added_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'dense_local_expansion_candidates_added')::float8)
        ),
    'dense_local_expansion_candidates_added_max',
        max((last_scan_stats->>'dense_local_expansion_candidates_added')::int),
    'dense_local_expansion_us_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'dense_local_expansion_us')::float8)
        ),
    'dense_residual_rerank_count_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'dense_residual_rerank_count')::float8)
        ),
    'dense_residual_rerank_count_max',
        max((last_scan_stats->>'dense_residual_rerank_count')::int),
    'dense_residual_rerank_bytes_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'dense_residual_rerank_bytes')::float8)
        ),
    'dense_residual_rerank_us_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'dense_residual_rerank_us')::float8)
        ),
    'graph_rescore_count_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'graph_rescore_count')::float8)
        ),
    'graph_rescore_count_max', max((last_scan_stats->>'graph_rescore_count')::int),
    'graph_code_pages_read_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'graph_code_pages_read')::float8)
        ),
    'graph_adj_pages_read_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'graph_adj_pages_read')::float8)
        ),
    'graph_entry_point_count_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'graph_entry_point_count')::float8)
        ),
    'graph_entry_sidecar_count_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'graph_entry_sidecar_count')::float8)
        ),
    'graph_entry_sidecar_scored_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'graph_entry_sidecar_scored')::float8)
        ),
    'graph_entry_sidecar_selected_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'graph_entry_sidecar_selected')::float8)
        ),
    'graph_entry_sidecar_us_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'graph_entry_sidecar_us')::float8)
        ),
    'dense_elapsed_us_p50',
        percentile_cont(0.50) WITHIN GROUP (
            ORDER BY ((last_scan_stats->>'dense_elapsed_us')::float8)
        )
)::text
FROM dbpedia_run_timings
WHERE method = {sql_literal(label)} AND last_scan_stats IS NOT NULL;
""")
    scan_summary = json.loads(scan_summary_text) if scan_summary_text else {}
    return runs, timings, scan_summary


def fetch_probe_output(database: str, label: str) -> dict[str, list[str]]:
    exists = run_psql(database, "SELECT to_regclass('dbpedia_probe_results') IS NOT NULL;")
    if exists != "t":
        return {}
    rows = run_psql(database, f"""
SELECT query_id, rank, doc_id
FROM dbpedia_probe_results
WHERE method = {sql_literal(label)}
ORDER BY query_id, rank;
""")
    runs: dict[str, list[str]] = {}
    for line in rows.splitlines():
        if not line:
            continue
        qid, _rank, doc_id = line.split("|")
        runs.setdefault(qid, []).append(doc_id)
    return runs


SCAN_ATTRIBUTION_FIELDS = (
    "graph_visited_nodes",
    "graph_scored_codes",
    "graph_candidate_count",
    "graph_effective_result_target",
    "graph_effective_search_ef",
    "dense_adaptive_final_result_target",
    "dense_adaptive_final_search_ef",
    "dense_adaptive_gap_top10",
    "dense_adaptive_gap_boundary",
    "dense_local_expansion_seed_count",
    "dense_local_expansion_neighbors_scored",
    "dense_local_expansion_candidates_added",
    "dense_local_expansion_us",
    "dense_residual_rerank_count",
    "dense_residual_rerank_bytes",
    "dense_residual_rerank_us",
    "graph_effective_rescore_band",
    "graph_code_pages_read",
    "graph_adj_pages_read",
    "graph_entry_point_count",
    "graph_entry_sidecar_count",
    "graph_entry_sidecar_scored",
    "graph_entry_sidecar_selected",
    "graph_entry_sidecar_us",
    "graph_rescore_count",
    "dense_elapsed_us",
)


def fetch_timing_details(database: str, label: str) -> dict[str, list[dict[str, Any]]]:
    rows = run_psql(database, f"""
SELECT query_id, pass_no, elapsed_ms, COALESCE(last_scan_stats::text, '')
FROM dbpedia_run_timings
WHERE method = {sql_literal(label)}
ORDER BY query_id, pass_no;
""")
    details: dict[str, list[dict[str, Any]]] = {}
    for line in rows.splitlines():
        if not line:
            continue
        query_id, pass_no, elapsed_ms, stats_text = line.split("|", 3)
        stats = json.loads(stats_text) if stats_text else None
        details.setdefault(query_id, []).append({
            "pass_no": int(pass_no),
            "elapsed_ms": float(elapsed_ms),
            "scan_stats": stats,
        })
    return details


def source_rank_for_query(top_docs: list[str], relevant: dict[str, int]) -> int | None:
    ranks = [
        index
        for index, doc_id in enumerate(top_docs, start=1)
        if relevant.get(doc_id, 0) > 0
    ]
    return min(ranks) if ranks else None


def source_rank_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    rank1 = sum(1 for record in records if record["source_rank"] == 1)
    rank2_10 = sum(
        1
        for record in records
        if record["source_rank"] is not None and 2 <= record["source_rank"] <= 10
    )
    top10 = rank1 + rank2_10
    missed = total - top10
    reached_probe_not_top10 = sum(
        1
        for record in records
        if record.get("source_rank") is None
        and record.get("source_probe_rank") is not None
    )
    missed_probe = sum(
        1
        for record in records
        if record.get("source_rank") is None
        and record.get("source_probe_rank") is None
    )
    return {
        "total_queries": total,
        "source_rank_1": rank1,
        "source_rank_2_10": rank2_10,
        "source_top10": top10,
        "source_missed_top10": missed,
        "source_reached_probe_not_top10": reached_probe_not_top10,
        "source_missed_probe": missed_probe,
        "source_top1_rate": round(rank1 / total, 6) if total else 0.0,
        "source_top10_rate": round(top10 / total, 6) if total else 0.0,
        "source_miss_top10_rate": round(missed / total, 6) if total else 0.0,
        "source_reached_probe_not_top10_rate": round(reached_probe_not_top10 / total, 6)
        if total else 0.0,
        "source_missed_probe_rate": round(missed_probe / total, 6) if total else 0.0,
    }


def median_for_field(records: list[dict[str, Any]], field: str) -> float | None:
    values = [
        float(record["scan_stats"][field])
        for record in records
        if isinstance(record.get("scan_stats"), dict)
        and record["scan_stats"].get(field) is not None
    ]
    if not values:
        return None
    return round(statistics.median(values), 3)


def latency_summary_for_records(records: list[dict[str, Any]]) -> dict[str, float]:
    values = [float(record["latency_ms_mean"]) for record in records if record["latency_ms_mean"] is not None]
    return summarize_ms(values)


def failure_attribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    recovered = [record for record in records if record["source_in_top10"]]
    missed = [record for record in records if not record["source_in_top10"]]
    reached_probe = [
        record
        for record in missed
        if record.get("source_probe_rank") is not None
    ]
    missed_probe = [
        record
        for record in missed
        if record.get("source_probe_rank") is None
    ]
    return {
        "recovered_top10": source_rank_summary(recovered),
        "missed_top10": source_rank_summary(missed),
        "reached_probe_not_top10": source_rank_summary(reached_probe),
        "missed_probe": source_rank_summary(missed_probe),
        "latency_recovered_top10": latency_summary_for_records(recovered),
        "latency_missed_top10": latency_summary_for_records(missed),
        "scan_medians_recovered_top10": {
            field: median_for_field(recovered, field)
            for field in SCAN_ATTRIBUTION_FIELDS
        },
        "scan_medians_missed_top10": {
            field: median_for_field(missed, field)
            for field in SCAN_ATTRIBUTION_FIELDS
        },
    }


def adaptive_widening_attribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    triggered = [
        record
        for record in records
        if isinstance(record.get("scan_stats"), dict)
        and record["scan_stats"].get("dense_adaptive_triggered") is True
    ]
    not_triggered = [
        record
        for record in records
        if isinstance(record.get("scan_stats"), dict)
        and record["scan_stats"].get("dense_adaptive_triggered") is not True
    ]
    total = len(triggered) + len(not_triggered)
    return {
        "triggered_queries": len(triggered),
        "not_triggered_queries": len(not_triggered),
        "trigger_rate": round(len(triggered) / total, 6) if total else 0.0,
        "triggered_source_rank_summary": source_rank_summary(triggered),
        "not_triggered_source_rank_summary": source_rank_summary(not_triggered),
        "triggered_latency": latency_summary_for_records(triggered),
        "not_triggered_latency": latency_summary_for_records(not_triggered),
        "triggered_scan_medians": {
            field: median_for_field(triggered, field)
            for field in SCAN_ATTRIBUTION_FIELDS
        },
        "not_triggered_scan_medians": {
            field: median_for_field(not_triggered, field)
            for field in SCAN_ATTRIBUTION_FIELDS
        },
    }


def local_expansion_attribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    triggered = [
        record
        for record in records
        if isinstance(record.get("scan_stats"), dict)
        and record["scan_stats"].get("dense_local_expansion_triggered") is True
    ]
    not_triggered = [
        record
        for record in records
        if isinstance(record.get("scan_stats"), dict)
        and record["scan_stats"].get("dense_local_expansion_triggered") is not True
    ]
    total = len(triggered) + len(not_triggered)
    return {
        "triggered_queries": len(triggered),
        "not_triggered_queries": len(not_triggered),
        "trigger_rate": round(len(triggered) / total, 6) if total else 0.0,
        "triggered_source_rank_summary": source_rank_summary(triggered),
        "not_triggered_source_rank_summary": source_rank_summary(not_triggered),
        "triggered_latency": latency_summary_for_records(triggered),
        "not_triggered_latency": latency_summary_for_records(not_triggered),
        "triggered_scan_medians": {
            field: median_for_field(triggered, field)
            for field in SCAN_ATTRIBUTION_FIELDS
        },
        "not_triggered_scan_medians": {
            field: median_for_field(not_triggered, field)
            for field in SCAN_ATTRIBUTION_FIELDS
        },
    }


def per_query_recovery_overlap(records: list[dict[str, Any]], method_order: list[str]) -> dict[str, Any]:
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_method.setdefault(record["method"], {})[record["query_id"]] = record
    methods = [method for method in method_order if method in by_method]
    if not methods:
        return {}

    recovered_by_method = {
        method: {
            query_id
            for query_id, record in query_records.items()
            if record.get("source_in_top10")
        }
        for method, query_records in by_method.items()
    }
    query_ids_by_method = {
        method: set(query_records)
        for method, query_records in by_method.items()
    }
    missed_by_method = {
        method: query_ids_by_method[method] - recovered_by_method[method]
        for method in by_method
    }
    baseline = methods[0]
    baseline_missed = missed_by_method[baseline]
    baseline_recovered = recovered_by_method[baseline]

    variants: dict[str, Any] = {}
    for method in methods:
        recovered = recovered_by_method[method]
        missed = missed_by_method[method]
        variants[method] = {
            "queries": len(query_ids_by_method[method]),
            "source_recovered": len(recovered),
            "source_missed": len(missed),
            "recovers_baseline_misses": len(recovered & baseline_missed),
            "regresses_baseline_recovered": len(missed & baseline_recovered),
            "same_recovered_as_baseline": len(recovered & baseline_recovered),
            "same_missed_as_baseline": len(missed & baseline_missed),
            "unique_recovered_vs_others": len(
                recovered - set().union(
                    *(recovered_by_method[other] for other in methods if other != method)
                )
            ) if len(methods) > 1 else len(recovered),
        }

    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1:]:
            left_recovered = recovered_by_method[left]
            right_recovered = recovered_by_method[right]
            pairwise[f"{left}__vs__{right}"] = {
                "both_recovered": len(left_recovered & right_recovered),
                "left_only_recovered": len(left_recovered - right_recovered),
                "right_only_recovered": len(right_recovered - left_recovered),
                "both_missed": len(missed_by_method[left] & missed_by_method[right]),
            }

    return {
        "baseline_method": baseline,
        "variants": variants,
        "pairwise": pairwise,
    }


def exception_summary(exc: BaseException) -> dict[str, Any]:
    """Return a compact machine-readable failure summary for benchmark artifacts."""
    summary: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    if isinstance(exc, subprocess.CalledProcessError):
        summary.update({
            "returncode": exc.returncode,
            "cmd": exc.cmd if isinstance(exc.cmd, str) else " ".join(map(str, exc.cmd)),
        })
        stdout = getattr(exc, "stdout", None)
        stderr = getattr(exc, "stderr", None)
        if stdout:
            summary["stdout_tail"] = str(stdout)[-4000:]
        if stderr:
            summary["stderr_tail"] = str(stderr)[-4000:]
    return summary


def build_per_query_records(database: str, label: str, run: dict[str, list[str]],
                            qrels: dict[str, dict[str, int]], k: int,
                            probe_run: dict[str, list[str]] | None = None) -> list[dict[str, Any]]:
    timing_details = fetch_timing_details(database, label)
    probe_run = probe_run or {}
    records: list[dict[str, Any]] = []
    for query_id, relevant in sorted(qrels.items()):
        top_docs = run.get(query_id, [])[:k]
        relevant_doc_ids = sorted(doc_id for doc_id, score in relevant.items() if score > 0)
        source_id = query_id if query_id in relevant_doc_ids else (relevant_doc_ids[0] if relevant_doc_ids else None)
        rank = source_rank_for_query(top_docs, relevant)
        probe_docs = probe_run.get(query_id, [])
        probe_rank = source_rank_for_query(probe_docs, relevant) if probe_docs else None
        query_timings = timing_details.get(query_id, [])
        elapsed_values = [item["elapsed_ms"] for item in query_timings]
        first_stats = next(
            (
                item["scan_stats"]
                for item in query_timings
                if item["pass_no"] == 1 and item["scan_stats"] is not None
            ),
            None,
        )
        records.append({
            "method": label,
            "query_id": query_id,
            "source_id": source_id,
            "relevant_doc_ids": relevant_doc_ids,
            "source_rank": rank,
            "source_probe_rank": probe_rank,
            "source_in_top10": rank is not None and rank <= 10,
            "top10_ids": top_docs,
            "probe_top_ids": probe_docs,
            "latency_ms": round(statistics.mean(elapsed_values), 6) if elapsed_values else None,
            "latency_ms_mean": round(statistics.mean(elapsed_values), 6) if elapsed_values else None,
            "latency_ms_p50": round(percentile(elapsed_values, 50), 6) if elapsed_values else None,
            "latency_ms_p95": round(percentile(elapsed_values, 95), 6) if elapsed_values else None,
            "latency_runs": len(elapsed_values),
            "scan_stats": first_stats,
        })
    return records


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


def overlap_at_k(run: dict[str, list[str]], baseline: dict[str, list[str]], k: int) -> float:
    overlaps = []
    for qid, docs in baseline.items():
        left = set(run.get(qid, [])[:k])
        right = set(docs[:k])
        if right:
            overlaps.append(len(left & right) / len(right))
    return round(statistics.mean(overlaps), 6) if overlaps else 0.0


def explain_probe(database: str, method: str, dense_k: int, bm25_k: int,
                  final_k: int, profile: str, hnsw_ef_search: int,
                  force_turbohybrid_index: bool) -> Any:
    template = method_query(method, dense_k, bm25_k, final_k).strip()
    qrow = run_psql(database, "SELECT embedding::text, query_text FROM dbpedia_queries ORDER BY query_id LIMIT 1;")
    embedding, query_text = qrow.split("|", 1)
    query_sql = template.replace("%L", "{}").format(
        sql_literal(embedding),
        sql_literal(query_text),
        sql_literal(query_text),
    )
    settings = ""
    if hnsw_ef_search > 0:
        settings += f"SET hnsw.ef_search = {hnsw_ef_search}; "
    profile_sql = method_profile(method, profile)
    if method.startswith("pgturbohybrid"):
        settings += f"SET turbohybrid.profile = {sql_literal(profile_sql)}; "
        for key, value in method_runtime_settings(method).items():
            settings += f"SET {key} = {sql_literal(value)}; "
        if force_turbohybrid_index:
            settings += "SET enable_seqscan = off; "
    plan_text = run_psql(
        database,
        settings + "EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS, SETTINGS, VERBOSE) " + query_sql + ";",
    )
    try:
        return json.loads(plan_text)
    except json.JSONDecodeError:
        return plan_text


def benchmark_environment(database: str) -> dict[str, Any]:
    simd_capabilities = None
    simd_text = run_psql_optional(database, "SELECT turbohybrid_simd_capabilities()::text;")
    if simd_text:
        try:
            simd_capabilities = json.loads(simd_text)
        except json.JSONDecodeError:
            simd_capabilities = {"raw": simd_text}

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "postgres_version": run_psql_optional(database, "SHOW server_version;"),
        "pgvector_version": run_psql_optional(
            database,
            "SELECT extversion FROM pg_extension WHERE extname = 'vector';",
        ),
        "pgturbohybrid_version": run_psql_optional(
            database,
            "SELECT COALESCE((SELECT extversion FROM pg_extension WHERE extname = 'pgturbohybrid'), 'not-installed');",
        ),
        "pgturbohybrid_simd_capabilities": simd_capabilities,
        "host_cpu_count": os.cpu_count(),
        "host_load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip(),
        "pg_config": os.environ.get("PG_CONFIG", "pg_config"),
        "pgvector_ref": os.environ.get("PGVECTOR_REF", ""),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Path(args.dataset).resolve()
    if args.query_source == "beir":
        beir_dataset = Path(args.beir_dataset).resolve()
        query_embedding_path = Path(args.query_embeddings).resolve()
        if not query_embedding_path.is_file():
            raise FileNotFoundError(
                f"query embeddings not found: {query_embedding_path}; "
                "run with --prepare-query-embeddings --prepare-only first"
            )
        queries = load_queries(beir_dataset)
        qrels = read_qrels(resolve_qrels_path(args.qrels, beir_dataset))
        embeddings = read_query_embeddings(query_embedding_path)
        data_counts = setup_database_beir(args, qrels, queries, embeddings)
    else:
        beir_dataset = None
        query_embedding_path = None
        data_counts = setup_database_qdrant_self(args)
    scoring_qrels = filtered_qrels(args.database)
    if not scoring_qrels:
        raise ValueError("filtered qrels are empty; no benchmark queries have a loaded positive document")

    run_psql(args.database, """
DROP TABLE IF EXISTS dbpedia_run_results;
DROP TABLE IF EXISTS dbpedia_run_timings;
DROP TABLE IF EXISTS dbpedia_probe_results;
CREATE TABLE dbpedia_run_results(method text, query_id text, rank int, doc_id text);
CREATE TABLE dbpedia_probe_results(method text, query_id text, rank int, doc_id text);
CREATE TABLE dbpedia_run_timings(
    method text,
    pass_no int,
    query_id text,
    elapsed_ms float8,
    last_scan_stats jsonb
);
""")

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    results = []
    plan_checks = {}
    runs_by_label: dict[str, dict[str, list[str]]] = {}
    per_query_records: list[dict[str, Any]] = []
    build_cache: dict[str, dict[str, Any]] = {}

    for method in methods:
        dense_k = args.dense_k
        bm25_k = args.bm25_k
        profile = method_profile(method, args.profile)
        hnsw_ef_search = args.hnsw_ef_search
        if method == "pgturbohybrid_quality":
            dense_k = max(dense_k, 400)
            bm25_k = max(bm25_k, 400)
            hnsw_ef_search = max(hnsw_ef_search, 400)
        label = method
        build_key = method_build_key(method)
        try:
            if args.skip_build:
                build = existing_index_info(args.database, method)
                build["build_key"] = build_key
            elif build_key in build_cache:
                build = {**build_cache[build_key], "build_reused_from": build_key}
                build["build_provenance"] = {
                    **build.get("build_provenance", {}),
                    "build_source": "shared_build_in_run",
                    "method": method,
                    "build_key": build_key,
                    "reused_from_build_key": build_key,
                    "provenance_confidence": "shared_build_in_run",
                }
            else:
                build = build_index(args.database, method)
                build["build_key"] = build_key
                build_cache[build_key] = build
            if args.explain:
                plan_checks[label] = explain_probe(
                    args.database, method, dense_k, bm25_k, args.final_k,
                    args.profile, hnsw_ef_search, args.force_turbohybrid_index,
                )
            if args.warmup > 0:
                run_method_queries(
                    args.database, method, label, dense_k, bm25_k, args.final_k,
                    args.profile, hnsw_ef_search, args.warmup, False,
                    args.force_turbohybrid_index,
                )
            run_method_queries(
                args.database, method, label, dense_k, bm25_k, args.final_k,
                args.profile, hnsw_ef_search, args.measured_runs, True,
                args.force_turbohybrid_index,
            )
            if args.failure_probe_k > 0:
                run_method_probe(
                    args.database, method, label, dense_k, bm25_k,
                    args.failure_probe_k, args.profile, hnsw_ef_search,
                    args.force_turbohybrid_index,
                )
            run, timings, scan_summary = fetch_method_output(args.database, label)
            probe_run = fetch_probe_output(args.database, label) if args.failure_probe_k > 0 else {}
            runs_by_label[label] = run
            query_records = build_per_query_records(
                args.database,
                label,
                run,
                scoring_qrels,
                min(args.final_k, 10),
                probe_run,
            )
            per_query_records.extend(query_records)
            metrics = metrics_for_run(run, scoring_qrels, min(args.final_k, 10))
            result = {
                "method": label,
                "base_method": method,
                "variant": method,
                "status": "ok",
                "retrieval_mode": "dense_only" if method in DENSE_ONLY_METHODS else "hybrid",
                "profile": profile,
                "dense_k": dense_k,
                "bm25_k": bm25_k,
                "rrf_k": RRF_K,
                "final_k": args.final_k,
                "failure_probe_k": args.failure_probe_k,
                "hnsw_ef_search": hnsw_ef_search if method in {"pgvector_halfvec_dense_only", "postgres_sql_rrf_halfvec"} else None,
                **build,
                "bytes_per_vector": round(build["index_bytes"] / data_counts["rows"], 3)
                if data_counts["rows"] else 0.0,
                "build_seconds": round(build["build_ms"] / 1000.0, 3),
                "latency": summarize_ms(timings),
                "metrics": metrics,
                "source_rank_summary": source_rank_summary(query_records),
                "failure_attribution": failure_attribution(query_records),
            }
            if method.startswith("pgturbohybrid"):
                result["scan_summary"] = scan_summary
                result["adaptive_widening_attribution"] = adaptive_widening_attribution(query_records)
                result["local_expansion_attribution"] = local_expansion_attribution(query_records)
            results.append(result)
        except (subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as exc:
            results.append({
                "method": label,
                "base_method": method,
                "variant": method,
                "status": "failed",
                "retrieval_mode": "dense_only" if method in DENSE_ONLY_METHODS else "hybrid",
                "profile": profile,
                "dense_k": dense_k,
                "bm25_k": bm25_k,
                "rrf_k": RRF_K,
                "final_k": args.final_k,
                "failure_probe_k": args.failure_probe_k,
                "hnsw_ef_search": hnsw_ef_search if method in {"pgvector_halfvec_dense_only", "postgres_sql_rrf_halfvec"} else None,
                "build_key": build_key,
                "build_provenance": method_build_provenance(
                    method,
                    build_source="failed",
                    index_stats=None,
                ),
                "error": exception_summary(exc),
            })

    baseline = runs_by_label.get("postgres_sql_rrf_halfvec")
    if baseline:
        for result in results:
            run = runs_by_label.get(result["method"], {})
            result.setdefault("metrics", {})["overlap@10_vs_postgres_sql_rrf_halfvec"] = overlap_at_k(run, baseline, 10)
    dense_baseline = runs_by_label.get("pgvector_halfvec_dense_only")
    if dense_baseline:
        for result in results:
            run = runs_by_label.get(result["method"], {})
            result.setdefault("metrics", {})["overlap@10_vs_pgvector_halfvec_dense_only"] = overlap_at_k(
                run, dense_baseline, 10
            )

    payload: dict[str, Any] = {
        "suite": "pgturbohybrid_dbpedia_openai3_large_1m",
        "layer": "ir_quality_and_systems",
        "generated_at_unix": int(time.time()),
        "command": benchmark_command_metadata(),
        "dataset_name": "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M",
        "dataset_path": portable_path(dataset),
        "query_source": args.query_source,
        "embedding_model": EMBEDDING_MODEL,
        "rows": data_counts["rows"],
        "queries": data_counts["queries"],
        "qrels_loaded": data_counts["qrels"],
        "qrels_filtered_positive": sum(len(items) for items in scoring_qrels.values()),
        "queries_with_filtered_positive_qrels": len(scoring_qrels),
        "split": "test",
        "dimensions": DIMENSIONS,
        "dense_k": args.dense_k,
        "bm25_k": args.bm25_k,
        "final_k": args.final_k,
        "failure_probe_k": args.failure_probe_k,
        "rrf_k": RRF_K,
        "warmup": args.warmup,
        "measured_runs": args.measured_runs,
        "profile": args.profile,
        "max_docs": args.max_docs,
        "max_queries": args.max_queries,
        "force_turbohybrid_index": args.force_turbohybrid_index,
        "skip_build": args.skip_build,
        "methods": methods,
        "results": results,
        "per_query_recovery_overlap": per_query_recovery_overlap(per_query_records, methods),
        "_per_query_records": per_query_records,
        "environment": benchmark_environment(args.database),
    }
    if beir_dataset is not None:
        payload["beir_dataset_name"] = "BeIR/dbpedia-entity"
        payload["beir_dataset_path"] = portable_path(beir_dataset)
    if query_embedding_path is not None:
        payload["query_embeddings_path"] = portable_path(query_embedding_path)
    if args.explain:
        payload["plan_checks"] = plan_checks
    return payload


def write_output(path: str, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path == "-":
        print(text)
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out)


def write_jsonl_output(path: str, records: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    print(out)


def write_index_output(path: str, payload: dict[str, Any]) -> None:
    rows = []
    for result in payload["results"]:
        rows.append({
            "method": result["method"],
            "status": result.get("status", "ok"),
            "build_key": result.get("build_key"),
            "build_provenance": result.get("build_provenance"),
            "index_bytes": result.get("index_bytes"),
            "bytes_per_vector": result.get("bytes_per_vector"),
            "build_ms": result.get("build_ms"),
            "build_seconds": result.get("build_seconds"),
            "build_wal_bytes": result.get("build_wal_bytes"),
            "index_stats": result.get("index_stats"),
            "error": result.get("error"),
        })
    write_output(path, {
        "suite": payload["suite"],
        "generated_at_unix": payload.get("generated_at_unix"),
        "dataset_name": payload["dataset_name"],
        "rows": payload["rows"],
        "methods": payload["methods"],
        "results": rows,
        "environment": payload["environment"],
    })


def write_markdown_output(path: str, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DBPedia OpenAI3-large dense quality benchmark",
        "",
        f"- Dataset: `{payload['dataset_name']}`",
        f"- Rows: {payload['rows']}",
        f"- Queries: {payload['queries']}",
        f"- Query source: `{payload['query_source']}`",
        f"- Warmup passes: {payload['warmup']}",
        f"- Measured passes: {payload['measured_runs']}",
        f"- Command: `{payload['command']['shell']}`",
        "",
        "| Method | nDCG@10 | recall@10 | source top1 | source top10 | widened | expanded | p50 ms | p95 ms | p99 ms | visited p50 | scored p50 | index bytes | bytes/vector | build ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["results"]:
        if result.get("status", "ok") != "ok":
            lines.append(
                "| {method} | failed | failed | failed | failed | 0.000 | 0.000 | "
                "0.000 | 0.000 | 0.000 | 0 | 0 | 0 | 0.000 | 0.000 |".format(
                    method=result["method"],
                )
            )
            continue
        metrics = result["metrics"]
        ranks = result["source_rank_summary"]
        latency = result["latency"]
        scan = result.get("scan_summary") or {}
        adaptive = result.get("adaptive_widening_attribution") or {}
        expansion = result.get("local_expansion_attribution") or {}
        lines.append(
            "| {method} | {ndcg:.6f} | {recall:.6f} | {top1:.6f} | {top10:.6f} | {widened:.3f} | {expanded:.3f} | "
            "{p50:.3f} | {p95:.3f} | {p99:.3f} | {visited:.0f} | {scored:.0f} | {index_bytes} | {bytes_per_vector:.3f} | {build_ms:.3f} |".format(
                method=result["method"],
                ndcg=metrics.get("ndcg@10", 0.0),
                recall=metrics.get("recall@10", 0.0),
                top1=ranks["source_top1_rate"],
                top10=ranks["source_top10_rate"],
                widened=adaptive.get("trigger_rate", 0.0),
                expanded=expansion.get("trigger_rate", 0.0),
                p50=latency["p50_ms"],
                p95=latency["p95_ms"],
                p99=latency["p99_ms"],
                visited=scan.get("graph_visited_nodes_p50") or 0.0,
                scored=scan.get("graph_scored_codes_p50") or 0.0,
                index_bytes=result["index_bytes"],
                bytes_per_vector=result["bytes_per_vector"],
                build_ms=result["build_ms"],
            )
        )
    lines.extend([
        "",
        "## Failure attribution",
        "",
    ])
    for result in payload["results"]:
        if result.get("status", "ok") != "ok":
            error = result.get("error") or {}
            lines.extend([
                f"### {result['method']}",
                "",
                f"- Status: failed",
                f"- Error: {error.get('type', 'unknown')}: {error.get('message', '')}",
                "",
            ])
            continue
        summary = result["source_rank_summary"]
        lines.extend([
            f"### {result['method']}",
            "",
            f"- Source rank 1: {summary['source_rank_1']}",
            f"- Source rank 2-10: {summary['source_rank_2_10']}",
            f"- Source missed top 10: {summary['source_missed_top10']}",
            f"- Source reached only by failure probe: {summary['source_reached_probe_not_top10']}",
            f"- Source missed by failure probe: {summary['source_missed_probe']}",
            "",
        ])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


def default_output_path() -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return str(Path("benchmarks/results") / f"dbpedia_openai3_large_{stamp}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", "pgturbohybrid_dbpedia_1m"))
    parser.add_argument("--dataset", default=os.environ.get("DBPEDIA_DATASET"),
                        help="Qdrant DBPedia dataset directory, or set DBPEDIA_DATASET")
    parser.add_argument("--query-source", default=DEFAULT_QUERY_SOURCE,
                        choices=("qdrant-self", "beir"),
                        help="query source; qdrant-self uses sampled Qdrant rows and their existing embeddings")
    parser.add_argument("--beir-dataset", default=os.environ.get("BEIR_DBPEDIA_DATASET"),
                        help="BEIR/dbpedia-entity directory, or set BEIR_DBPEDIA_DATASET")
    parser.add_argument("--qrels", default=os.environ.get("BEIR_DBPEDIA_QRELS"),
                        help="BEIR/dbpedia-entity-qrels test.tsv path, or set BEIR_DBPEDIA_QRELS")
    parser.add_argument("--query-embeddings", default=os.environ.get("DBPEDIA_QUERY_EMBEDDINGS"),
                        help="cached query embeddings JSONL, or set DBPEDIA_QUERY_EMBEDDINGS")
    parser.add_argument("--prepare-query-embeddings", action="store_true",
                        help="generate query embeddings with OpenAI text-embedding-3-large")
    parser.add_argument("--prepare-only", action="store_true",
                        help="exit after preparing query embeddings")
    parser.add_argument("--force-query-embeddings", action="store_true",
                        help="overwrite an existing query embedding cache")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--dense-k", type=int, default=100)
    parser.add_argument("--bm25-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=FINAL_K)
    parser.add_argument("--failure-probe-k", type=int, default=0,
                        help="optional deeper result depth used only for source-row failure attribution")
    parser.add_argument("--profile", default="latency",
                        choices=("balanced", "latency", "quality", "matched_recall", "debug"))
    parser.add_argument("--hnsw-ef-search", type=int, default=100)
    parser.add_argument("--max-docs", type=int, default=0,
                        help="deterministic corpus subset size; 0 loads the full 1M corpus")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="deterministic query subset size; 0 loads all qrels-backed queries")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--skip-build", action="store_true",
                        help="reuse existing benchmark indexes instead of dropping and rebuilding them")
    parser.add_argument("--force-turbohybrid-index", action="store_true",
                        help="set enable_seqscan=off for TurboHybrid queries; useful for tiny smoke subsets")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--output", default=os.environ.get("OUTPUT", default_output_path()),
                        help="result JSON path, or set OUTPUT; defaults to benchmarks/results/")
    parser.add_argument("--jsonl-output", default=os.environ.get("JSONL_OUTPUT"),
                        help="optional per-query JSONL path")
    parser.add_argument("--index-output", default=os.environ.get("INDEX_OUTPUT"),
                        help="optional index-size/build-time JSON path")
    parser.add_argument("--markdown-output", default=os.environ.get("MARKDOWN_OUTPUT"),
                        help="optional aggregate Markdown report path")
    args = parser.parse_args()

    if args.query_source == "beir" and not args.beir_dataset:
        parser.error("--beir-dataset is required with --query-source beir unless BEIR_DBPEDIA_DATASET is set")
    if args.query_source == "beir" and not args.query_embeddings:
        parser.error("--query-embeddings is required with --query-source beir unless DBPEDIA_QUERY_EMBEDDINGS is set")
    if args.prepare_query_embeddings:
        prepare_query_embeddings(args)
        if args.prepare_only:
            return
    if not args.dataset:
        parser.error("--dataset is required unless DBPEDIA_DATASET is set")
    if args.final_k != FINAL_K:
        print("warning: primary DBPedia benchmark metrics should include k=10", file=sys.stderr)

    try:
        payload = run_benchmark(args)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"dataset validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    per_query_records = payload.pop("_per_query_records", [])
    if args.jsonl_output:
        write_jsonl_output(args.jsonl_output, per_query_records)
        payload["per_query_jsonl"] = args.jsonl_output
    if args.index_output:
        write_index_output(args.index_output, payload)
        payload["index_report"] = args.index_output
    if args.markdown_output:
        write_markdown_output(args.markdown_output, payload)
        payload["markdown_report"] = args.markdown_output
    write_output(args.output, payload)


if __name__ == "__main__":
    main()
