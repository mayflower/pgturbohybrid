#!/usr/bin/env python3
"""Run a Turbovec dense-only comparison on the DBPedia OpenAI3-large 1M set."""

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
import time
from pathlib import Path
from typing import Any, Iterable


DIMENSIONS = 3072
FINAL_K = 10
CORPUS_EMBEDDING_FIELD = "text-embedding-3-large-3072-embedding"
DATASET_NAME = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M"
EMBEDDING_MODEL = "text-embedding-3-large"


def require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "numpy is required for the Turbovec benchmark. "
            "Install it with: python3 -m pip install numpy"
        ) from exc
    return np


def require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to read Hugging Face parquet shards. "
            "Install it with: python3 -m pip install pyarrow"
        ) from exc
    return pq


def require_turbovec() -> tuple[Any, str]:
    try:
        import turbovec
        from turbovec import TurboQuantIndex
    except ImportError as exc:
        raise SystemExit(
            "turbovec is required for this optional benchmark. "
            "Install it with: python3 -m pip install turbovec"
        ) from exc
    version = getattr(turbovec, "__version__", "")
    return TurboQuantIndex, version


def run_psql(database: str, sql: str) -> str:
    cmd = ["psql", "-q", "-X", "-A", "-t", "-v", "ON_ERROR_STOP=1", "-d", database, "-c", sql]
    return subprocess.run(cmd, check=True, text=True, capture_output=True).stdout.strip()


def portable_path(path: Path) -> str:
    """Prefer repo-relative paths in generated benchmark metadata."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def copy_query(database: str, sql: str) -> list[list[str]]:
    cmd = ["psql", "-q", "-X", "-v", "ON_ERROR_STOP=1", "-d", database, "-c", f"\\copy ({sql}) TO STDOUT WITH (FORMAT csv, DELIMITER E'\\t')"]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return list(csv.reader(proc.stdout.splitlines(), delimiter="\t"))


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


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parse_vector_text(text: str, np: Any) -> Any:
    values = json.loads(text)
    if len(values) != DIMENSIONS:
        raise ValueError(f"query vector has {len(values)} dimensions; expected {DIMENSIONS}")
    return np.asarray(values, dtype=np.float32)


def validate_embedding(values: Any, label: str) -> list[float]:
    if not isinstance(values, list):
        try:
            values = values.as_py()
        except AttributeError as exc:
            raise ValueError(f"{label} does not contain an embedding array") from exc
    if len(values) != DIMENSIONS:
        raise ValueError(f"{label} has {len(values)} dimensions; expected {DIMENSIONS}")
    return [float(value) for value in values]


def iter_corpus_batches(dataset: Path, max_docs: int, batch_size: int, np: Any) -> Iterable[tuple[list[str], Any]]:
    pq = require_pyarrow()
    emitted = 0
    columns = ["_id", CORPUS_EMBEDDING_FIELD]
    for path in find_corpus_parquet(dataset):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            doc_ids: list[str] = []
            vectors: list[list[float]] = []
            for row in batch.to_pylist():
                if max_docs > 0 and emitted >= max_docs:
                    break
                doc_id = str(row["_id"])
                embedding = validate_embedding(row[CORPUS_EMBEDDING_FIELD], f"{path}:{doc_id}")
                doc_ids.append(doc_id)
                vectors.append(embedding)
                emitted += 1
            if doc_ids:
                yield doc_ids, np.asarray(vectors, dtype=np.float32)
            if max_docs > 0 and emitted >= max_docs:
                return


def load_queries(database: str, max_queries: int, np: Any) -> tuple[list[str], Any]:
    limit_sql = f" LIMIT {max_queries}" if max_queries > 0 else ""
    rows = copy_query(
        database,
        "SELECT query_id, embedding::text FROM dbpedia_queries ORDER BY query_id" + limit_sql,
    )
    if not rows:
        raise ValueError("no rows found in dbpedia_queries; run the DBPedia PostgreSQL harness first")
    query_ids = [row[0] for row in rows]
    vectors = np.stack([parse_vector_text(row[1], np) for row in rows]).astype(np.float32, copy=False)
    return query_ids, vectors


def load_qrels(database: str, query_ids: list[str]) -> dict[str, dict[str, int]]:
    if not query_ids:
        return {}
    query_list = ",".join(sql_literal(query_id) for query_id in query_ids)
    rows = copy_query(
        database,
        "SELECT query_id, doc_id, relevance "
        "FROM dbpedia_qrels "
        f"WHERE relevance > 0 AND query_id IN ({query_list}) "
        "ORDER BY query_id, doc_id",
    )
    qrels: dict[str, dict[str, int]] = {}
    for query_id, doc_id, relevance in rows:
        qrels.setdefault(query_id, {})[doc_id] = int(relevance)
    if not qrels:
        raise ValueError("no positive qrels found for selected dbpedia_queries")
    return qrels


def load_baseline_run(database: str, method: str, query_ids: list[str]) -> dict[str, list[str]]:
    exists = run_psql(database, "SELECT to_regclass('dbpedia_run_results') IS NOT NULL;")
    if exists != "t":
        return {}
    query_list = ",".join(sql_literal(query_id) for query_id in query_ids)
    rows = copy_query(
        database,
        "SELECT query_id, rank, doc_id "
        "FROM dbpedia_run_results "
        f"WHERE method = {sql_literal(method)} AND query_id IN ({query_list}) "
        "ORDER BY query_id, rank",
    )
    run: dict[str, list[str]] = {}
    for query_id, _rank, doc_id in rows:
        run.setdefault(query_id, []).append(doc_id)
    return run


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


def dcg(rels: list[int]) -> float:
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))


def metrics_for_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int) -> dict[str, float]:
    ndcg_total = 0.0
    recall_total = 0.0
    mrr_total = 0.0
    map_total = 0.0
    count = 0
    for query_id, relevant in qrels.items():
        docs = run.get(query_id, [])[:k]
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
    for query_id, docs in baseline.items():
        left = set(run.get(query_id, [])[:k])
        right = set(docs[:k])
        if right:
            overlaps.append(len(left & right) / len(right))
    return round(statistics.mean(overlaps), 6) if overlaps else 0.0


def normalize_search_output(scores: Any, indices: Any, np: Any) -> tuple[Any, Any]:
    scores_arr = np.asarray(scores)
    indices_arr = np.asarray(indices)
    if indices_arr.ndim == 1:
        indices_arr = indices_arr.reshape(1, -1)
    if scores_arr.ndim == 1:
        scores_arr = scores_arr.reshape(1, -1)
    return scores_arr, indices_arr


def search_once(index: Any, queries: Any, final_k: int, np: Any) -> tuple[list[list[int]], list[float]]:
    timings: list[float] = []
    all_indices: list[list[int]] = []
    for query in queries:
        started = time.perf_counter()
        _scores, indices = index.search(query.reshape(1, -1), k=final_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings.append(elapsed_ms)
        _scores_arr, indices_arr = normalize_search_output(_scores, indices, np)
        all_indices.append([int(value) for value in indices_arr[0].tolist() if int(value) >= 0])
    return all_indices, timings


def build_index(args: argparse.Namespace, TurboQuantIndex: Any, np: Any) -> tuple[Any, list[str], float]:
    dataset = Path(args.dataset).resolve()
    index = TurboQuantIndex(dim=DIMENSIONS, bit_width=args.bit_width)
    doc_ids: list[str] = []
    started = time.perf_counter()
    total = 0
    for batch_doc_ids, vectors in iter_corpus_batches(dataset, args.max_docs, args.batch_size, np):
        index.add(vectors)
        doc_ids.extend(batch_doc_ids)
        total += len(batch_doc_ids)
        if args.progress and total % args.progress == 0:
            print(f"indexed {total} rows", file=sys.stderr)
    if not doc_ids:
        raise ValueError(f"no corpus rows loaded from {dataset}")
    build_ms = (time.perf_counter() - started) * 1000.0
    return index, doc_ids, build_ms


def index_bytes(path: str | None) -> int | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    if target.is_file():
        return target.stat().st_size
    return sum(item.stat().st_size for item in target.rglob("*") if item.is_file())


def write_index(index: Any, path: str | None) -> int | None:
    if not path:
        return None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    index.write(str(target))
    return index_bytes(str(target))


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    np = require_numpy()
    TurboQuantIndex, turbovec_version = require_turbovec()
    query_ids, query_vectors = load_queries(args.database, args.max_queries, np)
    qrels = load_qrels(args.database, query_ids)
    baseline_run = load_baseline_run(args.database, args.baseline_method, query_ids)

    index, doc_ids, build_ms = build_index(args, TurboQuantIndex, np)
    saved_index_bytes = write_index(index, args.index_output)

    if args.warmup > 0:
        for _ in range(args.warmup):
            search_once(index, query_vectors, args.final_k, np)

    result_indices: list[list[int]] = []
    timings: list[float] = []
    for pass_no in range(args.measured_runs):
        indices, pass_timings = search_once(index, query_vectors, args.final_k, np)
        timings.extend(pass_timings)
        if pass_no == 0:
            result_indices = indices

    run: dict[str, list[str]] = {}
    for query_id, indices in zip(query_ids, result_indices, strict=True):
        run[query_id] = [doc_ids[index] for index in indices if 0 <= index < len(doc_ids)]

    metrics = metrics_for_run(run, qrels, min(args.final_k, 10))
    if baseline_run:
        metrics[f"overlap@10_vs_{args.baseline_method}"] = overlap_at_k(run, baseline_run, 10)

    latency = summarize_ms(timings)
    index_output = portable_path(Path(args.index_output)) if args.index_output else None
    result_row: dict[str, Any] = {
        "method": "turbovec_turboquant",
        "variant": f"turbovec_turboquant_{args.bit_width}bit",
        "status": "ok",
        "retrieval_mode": "dense_only",
        "bit_width": args.bit_width,
        "final_k": args.final_k,
        "warmup": args.warmup,
        "measured_runs": args.measured_runs,
        "index_bytes": saved_index_bytes,
        "bytes_per_vector": round(saved_index_bytes / len(doc_ids), 3)
        if saved_index_bytes is not None and doc_ids else None,
        "build_ms": round(build_ms, 3),
        "build_seconds": round(build_ms / 1000.0, 3),
        "latency": latency,
        "metrics": metrics,
        "index_output": index_output,
        "comparison_note": (
            "Turbovec is an in-process dense vector library. Compare this row "
            "with PostgreSQL-backed dense-only rows as a systems reference, "
            "not as a hybrid retrieval or SQL executor comparison."
        ),
    }
    if baseline_run:
        result_row["baseline_method"] = args.baseline_method

    payload: dict[str, Any] = {
        "suite": "turbovec_dbpedia_openai3_large_1m",
        "layer": "external_dense_vector_library",
        "generated_at_unix": int(time.time()),
        "dataset_name": DATASET_NAME,
        "dataset_path": portable_path(Path(args.dataset)),
        "query_source": "qdrant-self-from-postgresql",
        "embedding_model": EMBEDDING_MODEL,
        "dimensions": DIMENSIONS,
        "rows": len(doc_ids),
        "queries": len(query_ids),
        "qrels_filtered_positive": sum(len(items) for items in qrels.values()),
        "queries_with_filtered_positive_qrels": len(qrels),
        "method": "turbovec_turboquant",
        "retrieval_mode": "dense_only",
        "bit_width": args.bit_width,
        "final_k": args.final_k,
        "warmup": args.warmup,
        "measured_runs": args.measured_runs,
        "max_docs": args.max_docs,
        "max_queries": args.max_queries,
        "build_ms": result_row["build_ms"],
        "build_seconds": result_row["build_seconds"],
        "index_bytes": result_row["index_bytes"],
        "bytes_per_vector": result_row["bytes_per_vector"],
        "index_output": index_output,
        "latency": latency,
        "metrics": metrics,
        "results": [result_row],
        "run": run,
        "comparison_note": result_row["comparison_note"],
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "turbovec": turbovec_version,
            "postgres_version": run_psql(args.database, "SHOW server_version;"),
            "host_cpu_count": os.cpu_count(),
            "host_load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True).stdout.strip(),
        },
    }
    if baseline_run:
        payload["baseline_method"] = args.baseline_method
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


def default_output_path() -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return str(Path("benchmarks/results") / f"dbpedia_openai3_large_turbovec_{stamp}.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", "pgturbohybrid_dbpedia_dense_1m"))
    parser.add_argument("--dataset", default=os.environ.get("DBPEDIA_DATASET"),
                        help="Qdrant DBPedia dataset directory, or set DBPEDIA_DATASET")
    parser.add_argument("--output", default=os.environ.get("OUTPUT", default_output_path()),
                        help="result JSON path, or set OUTPUT; defaults to benchmarks/results/")
    parser.add_argument("--index-output", default=os.environ.get("TURBOVEC_INDEX_OUTPUT"),
                        help="optional ignored path for a persisted .tq index")
    parser.add_argument("--bit-width", type=int, default=4, choices=(2, 3, 4))
    parser.add_argument("--final-k", type=int, default=FINAL_K)
    parser.add_argument("--max-docs", type=int, default=0,
                        help="deterministic corpus subset size; 0 loads the full 1M corpus")
    parser.add_argument("--max-queries", type=int, default=0,
                        help="query subset size from dbpedia_queries; 0 uses all loaded queries")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--baseline-method", default="pgvector_halfvec_dense_only",
                        help="optional dbpedia_run_results method for overlap@10")
    parser.add_argument("--progress", type=int, default=100000,
                        help="print indexing progress every N rows; 0 disables progress")
    args = parser.parse_args()

    if not args.dataset:
        parser.error("--dataset is required unless DBPEDIA_DATASET is set")
    if args.final_k != FINAL_K:
        print("warning: primary DBPedia benchmark metrics should include k=10", file=sys.stderr)

    payload = run_benchmark(args)
    write_output(args.output, payload)


if __name__ == "__main__":
    main()
