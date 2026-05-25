#!/usr/bin/env python3
"""Run the FIQA OpenAI-embedding benchmark against pgturbohybrid."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


DIMENSIONS = 1536
FINAL_K = 10
RRF_K = 60
DEFAULT_BUDGET_MATRIX = ((50, 50), (100, 100), (400, 400))
REQUIRED_DATASET_FILES = (
    "corpus.jsonl",
    "corpus_embeddings.jsonl",
    "queries.jsonl",
    "query_embeddings.jsonl",
    "qrels/test.tsv",
)


def run_psql(database: str, sql: str, *, quiet: bool = True) -> str:
    cmd = ["psql", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-d", database, "-c", sql]
    if quiet:
        cmd.insert(1, "-q")
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout.strip()


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


def run_copy(database: str, table: str, columns: str, path: Path,
             force_not_null: tuple[str, ...] = ()) -> None:
    options = "FORMAT csv, DELIMITER E'\\t'"
    if force_not_null:
        options += ", FORCE_NOT_NULL (" + ", ".join(force_not_null) + ")"
    sql = f"\\copy {table} ({columns}) FROM '{path}' WITH ({options})"
    subprocess.run(["psql", "-q", "-X", "-v", "ON_ERROR_STOP=1", "-d", database, "-c", sql],
                   check=True, text=True)


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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


def read_jsonl_by_id(path: Path, text_fields: tuple[str, ...]) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            parts = [row.get(field, "") for field in text_fields]
            rows[str(row["_id"])] = " ".join(part for part in parts if part).strip()
    return rows


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as f:
        header = f.readline()
        if "query-id" not in header:
            raise ValueError(f"unexpected qrels header in {path}: {header!r}")
        for line in f:
            qid, docid, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[docid] = int(score)
    return qrels


def validate_dataset_files(dataset: Path) -> None:
    missing = [name for name in REQUIRED_DATASET_FILES if not (dataset / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "missing real FIQA/OpenAI dataset files: " +
            ", ".join(str(dataset / name) for name in missing)
        )


def validate_embedding_row(row: dict[str, Any], path: Path) -> None:
    values = row.get("values")
    if not isinstance(values, list):
        raise ValueError(f"embedding row in {path} does not contain a values array")
    if len(values) != DIMENSIONS:
        raise ValueError(
            f"embedding row {row.get('id')!r} in {path} has {len(values)} dimensions; "
            f"expected {DIMENSIONS}"
        )


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(format(value, ".9g") for value in values) + "]"


def choose_subset(qrels: dict[str, dict[str, int]], max_queries: int,
                  max_docs: int) -> tuple[set[str], set[str] | None]:
    qids = sorted(qrels)
    if max_queries > 0:
        qids = qids[:max_queries]
    selected_qids = set(qids)

    if max_docs <= 0:
        return selected_qids, None

    selected_docids: set[str] = set()
    for qid in qids:
        selected_docids.update(qrels[qid])
    return selected_qids, selected_docids


def prepare_copy_files(dataset: Path, qids: set[str], tmpdir: Path,
                       selected_docids: set[str] | None,
                       max_docs: int) -> tuple[Path, Path]:
    corpus_text = read_jsonl_by_id(dataset / "corpus.jsonl", ("title", "text"))
    query_text = read_jsonl_by_id(dataset / "queries.jsonl", ("text",))
    docs_path = tmpdir / "fiqa_docs.tsv"
    queries_path = tmpdir / "fiqa_queries.tsv"

    docs_written = 0
    with docs_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        with (dataset / "corpus_embeddings.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                docid = str(row["id"])
                if selected_docids is not None:
                    if docid in selected_docids:
                        pass
                    elif docs_written < max_docs:
                        selected_docids.add(docid)
                    else:
                        continue
                validate_embedding_row(row, dataset / "corpus_embeddings.jsonl")
                writer.writerow([docid, vector_literal(row["values"]), corpus_text[docid]])
                docs_written += 1

    with queries_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        with (dataset / "query_embeddings.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                qid = str(row["id"])
                if qid in qids:
                    validate_embedding_row(row, dataset / "query_embeddings.jsonl")
                    writer.writerow([qid, vector_literal(row["values"]), query_text[qid]])

    return docs_path, queries_path


def prepare_qrels_file(qrels: dict[str, dict[str, int]], qids: set[str],
                       selected_docids: set[str] | None, tmpdir: Path) -> Path:
    qrels_path = tmpdir / "fiqa_qrels.tsv"
    with qrels_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        for qid in sorted(qids):
            for docid, relevance in sorted(qrels.get(qid, {}).items()):
                if selected_docids is None or docid in selected_docids:
                    writer.writerow([qid, docid, relevance])
    return qrels_path


def setup_database(database: str, dataset: Path, qrels: dict[str, dict[str, int]], reuse: bool,
                   max_docs: int, max_queries: int, extension_mode: str) -> dict[str, int]:
    if reuse:
        count = run_psql(database, "SELECT to_regclass('fiqa_docs') IS NOT NULL;")
        if count == "t":
            return {
                "rows": int(run_psql(database, "SELECT count(*) FROM fiqa_docs;")),
                "queries": int(run_psql(database, "SELECT count(*) FROM fiqa_queries;")),
            }

    selected_qids, selected_docids = choose_subset(qrels, max_queries, max_docs)

    extension_sql = "CREATE EXTENSION IF NOT EXISTS vector;"
    if extension_mode == "pgturbohybrid":
        extension_sql += "\nCREATE EXTENSION IF NOT EXISTS pgturbohybrid;"

    run_psql(database, f"""
{extension_sql}
DROP TABLE IF EXISTS fiqa_queries;
DROP TABLE IF EXISTS fiqa_qrels;
DROP TABLE IF EXISTS fiqa_docs;
CREATE TABLE fiqa_docs (
    doc_id text PRIMARY KEY,
    embedding vector({DIMENSIONS}) NOT NULL,
    body text NOT NULL,
    body_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
);
CREATE TABLE fiqa_queries (
    query_id text PRIMARY KEY,
    embedding vector({DIMENSIONS}) NOT NULL,
    query_text text NOT NULL
);
CREATE TABLE fiqa_qrels (
    query_id text NOT NULL REFERENCES fiqa_queries(query_id),
    doc_id text NOT NULL REFERENCES fiqa_docs(doc_id),
    relevance int NOT NULL,
    PRIMARY KEY (query_id, doc_id)
);
""")

    with tempfile.TemporaryDirectory(prefix="pgturbohybrid-fiqa-") as td:
        docs_path, queries_path = prepare_copy_files(dataset, selected_qids, Path(td),
                                                     selected_docids, max_docs)
        qrels_path = prepare_qrels_file(qrels, selected_qids, selected_docids, Path(td))
        run_copy(database, "fiqa_docs", "doc_id, embedding, body", docs_path,
                 force_not_null=("body",))
        run_copy(database, "fiqa_queries", "query_id, embedding, query_text", queries_path,
                 force_not_null=("query_text",))
        run_copy(database, "fiqa_qrels", "query_id, doc_id, relevance", qrels_path)

    run_psql(database, "ANALYZE fiqa_docs; ANALYZE fiqa_queries; ANALYZE fiqa_qrels;")
    return {
        "rows": int(run_psql(database, "SELECT count(*) FROM fiqa_docs;")),
        "queries": int(run_psql(database, "SELECT count(*) FROM fiqa_queries;")),
    }


def filtered_qrels(database: str, qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    query_rows = run_psql(database, "SELECT query_id FROM fiqa_queries ORDER BY query_id;")
    doc_rows = run_psql(database, "SELECT doc_id FROM fiqa_docs ORDER BY doc_id;")
    query_ids = {line for line in query_rows.splitlines() if line}
    doc_ids = {line for line in doc_rows.splitlines() if line}

    filtered: dict[str, dict[str, int]] = {}
    for qid in sorted(query_ids):
        relevant = {docid: score for docid, score in qrels.get(qid, {}).items() if docid in doc_ids}
        if relevant:
            filtered[qid] = relevant
    return filtered


def build_index(database: str, method: str) -> dict[str, Any]:
    if method == "pgvector_hnsw_dense_only":
        sql = """
DROP INDEX IF EXISTS fiqa_hnsw_idx;
DROP INDEX IF EXISTS fiqa_fts_idx;
DROP INDEX IF EXISTS fiqa_turbohybrid_idx;
CREATE INDEX fiqa_hnsw_idx ON fiqa_docs USING hnsw (embedding vector_cosine_ops);
"""
        indexes = ["fiqa_hnsw_idx"]
    elif method == "postgres_sql_rrf":
        sql = """
DROP INDEX IF EXISTS fiqa_hnsw_idx;
DROP INDEX IF EXISTS fiqa_fts_idx;
DROP INDEX IF EXISTS fiqa_turbohybrid_idx;
CREATE INDEX fiqa_hnsw_idx ON fiqa_docs USING hnsw (embedding vector_cosine_ops);
CREATE INDEX fiqa_fts_idx ON fiqa_docs USING gin (body_tsv);
"""
        indexes = ["fiqa_hnsw_idx", "fiqa_fts_idx"]
    elif method == "pgturbohybrid":
        sql = """
DROP INDEX IF EXISTS fiqa_hnsw_idx;
DROP INDEX IF EXISTS fiqa_fts_idx;
DROP INDEX IF EXISTS fiqa_turbohybrid_idx;
CREATE INDEX fiqa_turbohybrid_idx ON fiqa_docs
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);
"""
        indexes = ["fiqa_turbohybrid_idx"]
    elif method in {"pgturbohybrid_recovered_explicit", "pgturbohybrid_exact_storage_off"}:
        sql = """
DROP INDEX IF EXISTS fiqa_hnsw_idx;
DROP INDEX IF EXISTS fiqa_fts_idx;
DROP INDEX IF EXISTS fiqa_turbohybrid_idx;
CREATE INDEX fiqa_turbohybrid_idx ON fiqa_docs
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = off);
"""
        indexes = ["fiqa_turbohybrid_idx"]
    elif method == "pgturbohybrid_exact_storage_on":
        sql = """
DROP INDEX IF EXISTS fiqa_hnsw_idx;
DROP INDEX IF EXISTS fiqa_fts_idx;
DROP INDEX IF EXISTS fiqa_turbohybrid_idx;
CREATE INDEX fiqa_turbohybrid_idx ON fiqa_docs
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = on);
"""
        indexes = ["fiqa_turbohybrid_idx"]
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
        + ",".join("'" + item + "'" for item in indexes)
        + ");",
    ) or "0")
    index_stats = None
    if "fiqa_turbohybrid_idx" in indexes:
        index_stats_text = run_psql(
            database,
            "SELECT turbohybrid_index_stats('fiqa_turbohybrid_idx'::regclass)::text;",
        )
        index_stats = json.loads(index_stats_text) if index_stats_text else None
    return {
        "build_ms": round(build_ms, 3),
        "build_wal_bytes": int(wal_bytes),
        "index_bytes": index_bytes,
        "index_stats": index_stats,
        "indexes": indexes,
    }


def install_dev_diagnostics(database: str, path: str) -> None:
    if not path:
        return
    sql_path = Path(path)
    if not sql_path.is_file():
        return
    run_psql_file(database, sql_path)


def derive_bm25_probe_cases(database: str) -> list[dict[str, str]]:
    rows = run_psql(database, """
WITH doc_count AS (
    SELECT count(*)::float8 AS n FROM fiqa_docs
),
term_stats AS MATERIALIZED (
    SELECT word, ndoc
    FROM ts_stat('SELECT body_tsv FROM fiqa_docs')
    WHERE word ~ '^[a-z0-9][a-z0-9_-]*$'
      AND length(word) >= 3
),
qrel_term_stats AS MATERIALIZED (
    SELECT word, ndoc
    FROM ts_stat(
        'SELECT d.body_tsv FROM fiqa_docs d ' ||
        'JOIN fiqa_qrels r ON r.doc_id = d.doc_id ' ||
        'WHERE r.relevance > 0'
    )
    WHERE word ~ '^[a-z0-9][a-z0-9_-]*$'
      AND length(word) >= 3
),
rare_case AS (
    SELECT 'rare' AS category, word AS query_expr
    FROM qrel_term_stats
    ORDER BY ndoc ASC, word
    LIMIT 1
),
common_case AS (
    SELECT 'common' AS category, word AS query_expr
    FROM term_stats, doc_count
    WHERE ndoc < doc_count.n
    ORDER BY ndoc DESC, word
    LIMIT 1
),
or_case AS (
    SELECT 'or' AS category, string_agg(word, ' | ' ORDER BY ndoc DESC, word) AS query_expr
    FROM (
        SELECT word, ndoc
        FROM term_stats, doc_count
        WHERE ndoc < doc_count.n
        ORDER BY ndoc DESC, word
        LIMIT 2
    ) s
),
and_doc AS (
    SELECT d.doc_id
    FROM fiqa_docs d
    JOIN fiqa_qrels r ON r.doc_id = d.doc_id
    WHERE r.relevance > 0
    GROUP BY d.doc_id
    ORDER BY count(*) DESC, d.doc_id
    LIMIT 1
),
and_case AS (
    SELECT 'and' AS category, string_agg(word, ' & ' ORDER BY word) AS query_expr
    FROM (
        SELECT s.word
        FROM and_doc d
        CROSS JOIN LATERAL ts_stat(
            format('SELECT body_tsv FROM fiqa_docs WHERE doc_id = %L', d.doc_id)
        ) AS s
        WHERE s.word ~ '^[a-z0-9][a-z0-9_-]*$'
          AND length(s.word) >= 3
        ORDER BY s.nentry DESC, s.word
        LIMIT 2
    ) selected
)
SELECT category, query_expr
FROM rare_case
UNION ALL SELECT category, query_expr FROM common_case
UNION ALL SELECT category, query_expr FROM or_case
UNION ALL SELECT category, query_expr FROM and_case
ORDER BY category;
""")
    cases = []
    for line in rows.splitlines():
        category, query_expr = line.split("|", 1)
        if not query_expr or category in {"or", "and"} and "|" not in query_expr and "&" not in query_expr:
            raise ValueError(f"could not derive BM25 {category} probe terms from real FIQA data")
        cases.append({"category": category, "query_expr": query_expr})
    expected = {"rare", "common", "or", "and"}
    found = {case["category"] for case in cases}
    if found != expected:
        raise ValueError(f"could not derive all BM25 probe cases from real FIQA data: {sorted(found)}")
    return cases


def run_bm25_probe_phase(database: str, probe_id: str, method: str, category: str,
                         query_expr: str, bm25_k: int, final_k: int,
                         profile: str, force_turbohybrid_index: bool) -> None:
    force_index = "true" if force_turbohybrid_index else "false"
    sql = f"""
CREATE TABLE IF NOT EXISTS fiqa_bm25_cache_probe_results (
    probe_id text NOT NULL,
    method text NOT NULL,
    category text NOT NULL,
    phase text NOT NULL,
    query_expr text NOT NULL,
    elapsed_ms float8 NOT NULL,
    last_scan_stats jsonb NOT NULL,
    debug_scan_stats jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

DO $$
DECLARE
    query_sql text;
    started timestamptz;
    elapsed float8;
    i int;
BEGIN
    PERFORM set_config('turbohybrid.profile', {sql_literal(profile)}, false);
    IF {force_index} THEN
        PERFORM set_config('enable_seqscan', 'off', true);
    END IF;
    query_sql := format(
        'SELECT doc_id FROM fiqa_docs ORDER BY embedding <~> turbohybrid_query(text_query => to_tsquery(''english'', %L), bm25_k => {bm25_k}, final_k => {final_k}) LIMIT {final_k}',
        {sql_literal(query_expr)}
    );

    started := clock_timestamp();
    EXECUTE query_sql;
    elapsed := EXTRACT(EPOCH FROM clock_timestamp() - started) * 1000.0;
    INSERT INTO fiqa_bm25_cache_probe_results
    SELECT {sql_literal(probe_id)}, {sql_literal(method)}, {sql_literal(category)}, 'cold',
           {sql_literal(query_expr)}, elapsed, turbohybrid_last_scan_stats(),
           turbohybrid_last_scan_stats();

    started := clock_timestamp();
    EXECUTE query_sql;
    elapsed := EXTRACT(EPOCH FROM clock_timestamp() - started) * 1000.0;
    INSERT INTO fiqa_bm25_cache_probe_results
    SELECT {sql_literal(probe_id)}, {sql_literal(method)}, {sql_literal(category)}, 'warm_after_one',
           {sql_literal(query_expr)}, elapsed, turbohybrid_last_scan_stats(),
           turbohybrid_last_scan_stats();

    FOR i IN 1..10 LOOP
        EXECUTE query_sql;
    END LOOP;

    started := clock_timestamp();
    EXECUTE query_sql;
    elapsed := EXTRACT(EPOCH FROM clock_timestamp() - started) * 1000.0;
    INSERT INTO fiqa_bm25_cache_probe_results
    SELECT {sql_literal(probe_id)}, {sql_literal(method)}, {sql_literal(category)}, 'warm_after_ten',
           {sql_literal(query_expr)}, elapsed, turbohybrid_last_scan_stats(),
           turbohybrid_last_scan_stats();
END $$;
"""
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as f:
        f.write(sql)
        path = Path(f.name)
    try:
        run_psql_file(database, path)
    finally:
        path.unlink(missing_ok=True)


def run_bm25_cache_probe(database: str, method: str, bm25_k: int, final_k: int,
                         profile: str, force_turbohybrid_index: bool) -> list[dict[str, Any]]:
    cases = derive_bm25_probe_cases(database)
    probe_id = f"{method}-{int(time.time() * 1000)}"
    for case in cases:
        run_bm25_probe_phase(database, probe_id, method, case["category"],
                             case["query_expr"], bm25_k, final_k, profile,
                             force_turbohybrid_index)

    text = run_psql(database, f"""
SELECT jsonb_agg(
    jsonb_build_object(
        'method', method,
        'category', category,
        'phase', phase,
        'query_expr', query_expr,
        'elapsed_ms', round(elapsed_ms::numeric, 3),
        'bm25_first_query', debug_scan_stats->'bm25_first_query',
        'bm25_cache_build_us', debug_scan_stats->'bm25_cache_build_us',
        'bm25_cache_hit', debug_scan_stats->'bm25_cache_hit',
        'bm25_hot_postings_cache_hit', debug_scan_stats->'bm25_hot_postings_cache_hit',
        'bm25_impact_loaded_from_storage', debug_scan_stats->'bm25_impact_loaded_from_storage',
        'bm25_impact_built_lazily', debug_scan_stats->'bm25_impact_built_lazily',
        'bm25_strategy', debug_scan_stats->'bm25_strategy',
        'bm25_elapsed_us', debug_scan_stats->'bm25_elapsed_us',
        'bm25_postings_decoded', debug_scan_stats->'bm25_postings_decoded'
    )
    ORDER BY method, category,
        CASE phase WHEN 'cold' THEN 1 WHEN 'warm_after_one' THEN 2 ELSE 3 END
)::text
FROM fiqa_bm25_cache_probe_results
WHERE probe_id = {sql_literal(probe_id)};
""")
    return json.loads(text) if text else []


def method_query(method: str, dense_k: int, bm25_k: int, final_k: int) -> str:
    if method == "pgvector_hnsw_dense_only":
        return f"""
SELECT doc_id
FROM fiqa_docs
ORDER BY embedding <=> %L::vector({DIMENSIONS})
LIMIT {final_k}
"""

    if method == "postgres_sql_rrf":
        return f"""
WITH dense AS MATERIALIZED (
    SELECT doc_id, row_number() OVER () AS rank
    FROM (
        SELECT doc_id
        FROM fiqa_docs
        ORDER BY embedding <=> %L::vector({DIMENSIONS})
        LIMIT {dense_k}
    ) s
),
lexical AS MATERIALIZED (
    SELECT doc_id, row_number() OVER () AS rank
    FROM (
        SELECT doc_id
        FROM fiqa_docs
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
FROM fiqa_docs
ORDER BY embedding <~> turbohybrid_query(
    vector_query => %L::vector({DIMENSIONS}),
    text_query => plainto_tsquery('english', %L)
)
LIMIT {final_k}
"""

    return f"""
SELECT doc_id
FROM fiqa_docs
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


def run_method_queries(database: str, method: str, result_label: str, dense_k: int, bm25_k: int,
                       final_k: int, record: bool, profile: str,
                       force_turbohybrid_index: bool, extension_mode: str) -> None:
    query_template = method_query(method, dense_k, bm25_k, final_k)
    escaped_template = query_template.replace("'", "''")
    record_sql = "true" if record else "false"
    is_turbohybrid = "true" if method.startswith("pgturbohybrid") else "false"
    force_index_sql = "true" if force_turbohybrid_index else "false"
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
BEGIN
    IF '{extension_mode}' = 'pgturbohybrid' THEN
        PERFORM set_config('turbohybrid.profile', '{profile}', false);
    END IF;
    IF {is_turbohybrid} AND {force_index_sql} THEN
        PERFORM set_config('enable_seqscan', 'off', true);
    END IF;
    FOR q IN SELECT query_id, embedding::text AS embedding, query_text FROM fiqa_queries ORDER BY query_id LOOP
        sql := format('{escaped_template}', q.embedding, q.query_text, q.query_text);
        started := clock_timestamp();
        rank_no := 0;
        FOR r IN EXECUTE sql LOOP
            rank_no := rank_no + 1;
            IF {record_sql} THEN
                INSERT INTO fiqa_run_results(method, query_id, rank, doc_id)
                VALUES ('{result_label}', q.query_id, rank_no, r.doc_id);
            END IF;
        END LOOP;
        elapsed := EXTRACT(EPOCH FROM clock_timestamp() - started) * 1000.0;
        IF {is_turbohybrid} THEN
            scan_stats := turbohybrid_last_scan_stats();
        ELSE
            scan_stats := NULL;
        END IF;
        IF {record_sql} THEN
            INSERT INTO fiqa_run_timings(method, query_id, elapsed_ms, last_scan_stats)
            VALUES ('{result_label}', q.query_id, elapsed, scan_stats);
        END IF;
    END LOOP;
END $$;
"""
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as f:
        f.write(sql)
        path = Path(f.name)
    try:
        run_psql_file(database, path)
    finally:
        path.unlink(missing_ok=True)


def fetch_method_output(database: str, method: str) -> tuple[dict[str, list[str]], list[float], dict[str, Any]]:
    rows = run_psql(database, f"""
SELECT query_id, rank, doc_id
FROM fiqa_run_results
WHERE method = '{method}'
ORDER BY query_id, rank;
""")
    runs: dict[str, list[str]] = {}
    if rows:
        for line in rows.splitlines():
            qid, _rank, docid = line.split("|")
            runs.setdefault(qid, []).append(docid)

    timing_rows = run_psql(database, f"""
SELECT elapsed_ms
FROM fiqa_run_timings
WHERE method = '{method}'
ORDER BY query_id;
""")
    timings = [float(line) for line in timing_rows.splitlines() if line]
    scan_summary_text = run_psql(database, f"""
SELECT jsonb_build_object(
    'profile',
    max(last_scan_stats->>'profile'),
    'index_used',
    bool_or((last_scan_stats->>'index_used')::bool),
    'scan_orchestration',
    max(last_scan_stats->>'scan_orchestration'),
    'dense_k_effective_p50',
    percentile_cont(0.50) WITHIN GROUP (
        ORDER BY ((last_scan_stats->>'dense_k_effective')::float8)
    ),
    'dense_k_effective_max',
    max((last_scan_stats->>'dense_k_effective')::int),
    'bm25_k_effective_p50',
    percentile_cont(0.50) WITHIN GROUP (
        ORDER BY ((last_scan_stats->>'bm25_k_effective')::float8)
    ),
    'bm25_k_effective_max',
    max((last_scan_stats->>'bm25_k_effective')::int),
    'final_k_effective_p50',
    percentile_cont(0.50) WITHIN GROUP (
        ORDER BY ((last_scan_stats->>'final_k_effective')::float8)
    ),
    'detected_sql_limit_p50',
    percentile_cont(0.50) WITHIN GROUP (
        ORDER BY ((last_scan_stats->>'detected_sql_limit')::float8)
    ),
    'final_k_source',
    max(last_scan_stats->>'final_k_source'),
    'final_k_inferred_count',
    count(*) FILTER (WHERE (last_scan_stats->>'final_k_inferred')::bool),
    'bm25_cache_hit_count',
    count(*) FILTER (WHERE (last_scan_stats->>'bm25_cache_hit')::bool),
    'strict_vector_validations_max',
    max((last_scan_stats->>'strict_vector_validations')::bigint),
    'fast_vector_checks_max',
    max((last_scan_stats->>'fast_vector_checks')::bigint),
    'graph_rescore_count_p50',
    percentile_cont(0.50) WITHIN GROUP (
        ORDER BY ((last_scan_stats->>'graph_rescore_count')::float8)
    ),
    'graph_rescore_count_max',
    max((last_scan_stats->>'graph_rescore_count')::int),
    'graph_effective_rescore_band_p50',
    percentile_cont(0.50) WITHIN GROUP (
        ORDER BY ((last_scan_stats->>'graph_effective_rescore_band')::float8)
    ),
    'graph_effective_rescore_band_max',
    max((last_scan_stats->>'graph_effective_rescore_band')::int)
)::text
FROM fiqa_run_timings
WHERE method = '{method}' AND last_scan_stats IS NOT NULL;
""")
    scan_summary = json.loads(scan_summary_text) if scan_summary_text else {}
    return runs, timings, scan_summary


def dcg(rels: list[int]) -> float:
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))


def metrics_for_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int) -> dict[str, float]:
    ndcg_total = 0.0
    recall_total = 0.0
    mrr_total = 0.0
    map_total = 0.0
    count = 0

    for qid, relevant in qrels.items():
        if not relevant:
            continue
        docs = run.get(qid, [])[:k]
        rels = [relevant.get(docid, 0) for docid in docs]
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
        return {
            f"ndcg@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"mrr@{k}": 0.0,
            f"map@{k}": 0.0,
        }

    return {
        f"ndcg@{k}": round(ndcg_total / count, 6),
        f"recall@{k}": round(recall_total / count, 6),
        f"mrr@{k}": round(mrr_total / count, 6),
        f"map@{k}": round(map_total / count, 6),
    }


def explain_probe(database: str, method: str, dense_k: int, bm25_k: int, final_k: int,
                  profile: str, extension_mode: str) -> Any:
    template = method_query(method, dense_k, bm25_k, final_k).strip()
    qrow = run_psql(database, "SELECT embedding::text, query_text FROM fiqa_queries ORDER BY query_id LIMIT 1;")
    embedding, query_text = qrow.split("|", 1)
    sql_literal = template.replace("%L", "{}").format(
        "'" + embedding.replace("'", "''") + "'",
        "'" + query_text.replace("'", "''") + "'",
        "'" + query_text.replace("'", "''") + "'",
    )
    settings = "SET enable_seqscan = off; "
    if extension_mode == "pgturbohybrid":
        settings += f"SET turbohybrid.profile = '{profile}'; "
    plan_text = run_psql(
        database,
        settings + "EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS, SETTINGS, VERBOSE) " + sql_literal + ";",
    )
    try:
        return json.loads(plan_text)
    except json.JSONDecodeError:
        return plan_text


def benchmark_variants(args: argparse.Namespace) -> list[dict[str, Any]]:
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    budgets = DEFAULT_BUDGET_MATRIX if args.budget_matrix else ((args.dense_k, args.bm25_k),)
    variants: list[dict[str, Any]] = []

    for method in methods:
        for dense_k, bm25_k in budgets:
            label = method
            if args.budget_matrix:
                label = f"{method}_dense{dense_k}_bm25{bm25_k}"
            variants.append({
                "method": method,
                "label": label,
                "dense_k": dense_k,
                "bm25_k": bm25_k,
                "final_k": args.final_k,
            })

    return variants


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Path(args.dataset).resolve()
    validate_dataset_files(dataset)
    qrels = read_qrels(dataset / "qrels" / "test.tsv")
    data_counts = setup_database(args.database, dataset, qrels, args.reuse_data,
                                 args.max_docs, args.max_queries,
                                 args.extension_mode)
    if args.extension_mode == "pgturbohybrid":
        install_dev_diagnostics(args.database, args.dev_diagnostics_sql)
    scoring_qrels = filtered_qrels(args.database, qrels)
    run_psql(args.database, "DROP TABLE IF EXISTS fiqa_run_results; DROP TABLE IF EXISTS fiqa_run_timings; "
                           "CREATE TABLE fiqa_run_results(method text, query_id text, rank int, doc_id text); "
                           "CREATE TABLE fiqa_run_timings("
                           "method text, query_id text, elapsed_ms float8, last_scan_stats jsonb);")

    results = []
    plan_checks = {}
    variants = benchmark_variants(args)
    for method in sorted({variant["method"] for variant in variants}):
        build = build_index(args.database, method)

        for variant in [item for item in variants if item["method"] == method]:
            label = variant["label"]
            dense_k = variant["dense_k"]
            bm25_k = variant["bm25_k"]
            final_k = variant["final_k"]
            if args.explain:
                plan_checks[label] = explain_probe(args.database, method, dense_k, bm25_k,
                                                   final_k, args.profile,
                                                   args.extension_mode)
            if args.warmup:
                run_method_queries(args.database, method, label, dense_k, bm25_k, final_k,
                                   False, args.profile, args.force_turbohybrid_index,
                                   args.extension_mode)
            run_method_queries(args.database, method, label, dense_k, bm25_k, final_k,
                               True, args.profile, args.force_turbohybrid_index,
                               args.extension_mode)
            run, timings, scan_summary = fetch_method_output(args.database, label)
            build_for_result = dict(build)
            if method.startswith("pgturbohybrid"):
                build_for_result["scan_summary"] = scan_summary
                if args.bm25_cache_probe:
                    build_for_result["bm25_cache_probe"] = run_bm25_cache_probe(
                        args.database, method, bm25_k, final_k, args.profile,
                        args.force_turbohybrid_index,
                    )
            results.append({
                "method": label,
                "base_method": method,
                "profile": args.profile,
                "dense_k": dense_k,
                "bm25_k": bm25_k,
                "final_k": final_k,
                **build_for_result,
                "latency": summarize_ms(timings),
                "metrics": metrics_for_run(run, scoring_qrels, final_k),
            })

    payload = {
        "suite": "pgturbohybrid_real_rag_fiqa",
        "layer": "ir_quality_and_systems",
        "dataset_name": "fiqa-openai",
        "dataset_path": str(dataset),
        "embedding_model": "text-embedding-3-small",
        "rows": data_counts["rows"],
        "queries": data_counts["queries"],
        "qrels": sum(len(items) for items in scoring_qrels.values()),
        "split": "test",
        "dimensions": DIMENSIONS,
        "dense_k": args.dense_k,
        "bm25_k": args.bm25_k,
        "final_k": args.final_k,
        "rrf_k": RRF_K,
        "warmup": args.warmup,
        "profile": args.profile,
        "budget_matrix": args.budget_matrix,
        "max_docs": args.max_docs,
        "max_queries": args.max_queries,
        "force_turbohybrid_index": args.force_turbohybrid_index,
        "bm25_cache_probe": args.bm25_cache_probe,
        "extension_mode": args.extension_mode,
        "results": results,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "postgres_version": run_psql(args.database, "SHOW server_version;"),
            "pgvector_version": run_psql(args.database, "SELECT extversion FROM pg_extension WHERE extname = 'vector';"),
            "pgturbohybrid_version": run_psql(args.database, "SELECT COALESCE((SELECT extversion FROM pg_extension WHERE extname = 'pgturbohybrid'), 'not-installed');"),
            "pgturbohybrid_simd_capabilities": json.loads(run_psql(args.database, "SELECT turbohybrid_simd_capabilities()::text;")) if args.extension_mode == "pgturbohybrid" else None,
            "host_cpu_count": os.cpu_count(),
            "host_load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip(),
            "pg_config": os.environ.get("PG_CONFIG", "pg_config"),
            "pgvector_ref": os.environ.get("PGVECTOR_REF", ""),
        },
    }

    if args.explain:
        payload["plan_checks"] = plan_checks

    return payload


def write_output(path: str, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path == "-":
        print(text)
    else:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(out)


def default_output_path() -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return str(Path("benchmarks/results") / f"fiqa_openai_{stamp}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", "pgturbohybrid_fiqa"))
    parser.add_argument("--dataset", default=os.environ.get("FIQA_DATASET"),
                        help="FIQA/OpenAI dataset directory, or set FIQA_DATASET")
    parser.add_argument("--methods",
                        default="pgvector_hnsw_dense_only,postgres_sql_rrf,pgturbohybrid,pgturbohybrid_recovered_explicit")
    parser.add_argument("--dense-k", type=int, default=400)
    parser.add_argument("--bm25-k", type=int, default=400)
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--profile", default="balanced",
                        choices=("balanced", "latency", "quality", "debug"))
    parser.add_argument("--budget-matrix", action="store_true",
                        help="run dense/bm25 budget variants 50/50, 100/100, and 400/400")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="deterministic real-corpus subset size; 0 loads the full corpus")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="deterministic qrels-backed query subset size; 0 loads all qrels queries")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--force-turbohybrid-index", action="store_true",
                        help="set enable_seqscan=off for pgturbohybrid query timing; useful for tiny real subsets")
    parser.add_argument("--bm25-cache-probe", action="store_true",
                        help="measure real FIQA-derived BM25 cold/warm cache behavior")
    parser.add_argument("--dev-diagnostics-sql", default="",
                        help="optional path to sql/pgturbohybrid_dev_diagnostics.sql for BM25 cache probes")
    parser.add_argument("--extension-mode", default="pgturbohybrid",
                        choices=("pgturbohybrid", "patched_pgvector"),
                        help="pgturbohybrid creates vector plus pgturbohybrid; patched_pgvector creates only vector and expects TurboHybrid SQL objects there")
    parser.add_argument("--output", default=os.environ.get("OUTPUT", default_output_path()),
                        help="result JSON path, or set OUTPUT; defaults to benchmarks/results/")
    args = parser.parse_args()

    if not args.dataset:
        parser.error("--dataset is required unless FIQA_DATASET is set")
    if args.final_k != FINAL_K:
        print("warning: FIQA metrics are usually reported at k=10", file=sys.stderr)
    if not args.budget_matrix and args.dense_k == 400 and args.bm25_k == 400:
        print(
            "warning: this run uses dense_k=400 and bm25_k=400; old fast-path "
            "numbers used smaller explicit budgets such as 100/100",
            file=sys.stderr,
        )

    try:
        payload = run_benchmark(args)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"dataset validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    write_output(args.output, payload)


if __name__ == "__main__":
    main()
