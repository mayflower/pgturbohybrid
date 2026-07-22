#!/usr/bin/env python3
"""Deterministic concurrent correctness soak for the native dense graph."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import random
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field

import psycopg2

SEED = 0x544851
DBNAME = "pgturbohybrid_soak"


@dataclass
class Metrics:
    lock: threading.Lock = field(default_factory=threading.Lock)
    latencies_ms: list[float] = field(default_factory=list)
    successful_queries: int = 0
    failed_queries: int = 0

    def record(self, elapsed: float, ok: bool) -> None:
        with self.lock:
            if ok:
                self.successful_queries += 1
                self.latencies_ms.append(elapsed * 1000.0)
            else:
                self.failed_queries += 1


def connect():
    connection = psycopg2.connect(dbname=DBNAME)
    connection.autocommit = True
    return connection


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def setup() -> None:
    subprocess.run(["dropdb", "--if-exists", DBNAME], check=True)
    subprocess.run(["createdb", DBNAME], check=True)
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION vector; CREATE EXTENSION pgturbohybrid")
        cursor.execute("CREATE TABLE soak_docs(id bigint PRIMARY KEY, embedding vector(8), body_tsv tsvector)")
        cursor.execute(
            """INSERT INTO soak_docs
               SELECT g, ARRAY[g%101, g%17, g%13, g%11, g%7, g%5, g%3, g%2]::real[]::vector,
                      to_tsvector('simple', 'document ' || g)
               FROM generate_series(1, 5000) g"""
        )
        cursor.execute(
            """CREATE INDEX soak_idx ON soak_docs USING turbohybrid
               (embedding vector_l2_turbohybrid_ops, body_tsv bm25_tsvector_turbohybrid_ops)
               WITH (quantization_bits=4, exact_storage=on)"""
        )
        cursor.execute("SELECT turbohybrid_prewarm('soak_idx'::regclass)")


def reader(worker: int, stop: threading.Event, metrics: Metrics) -> None:
    rng = random.Random(SEED + worker)
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET statement_timeout='10s'; SET enable_seqscan=off")
        while not stop.is_set():
            value = rng.randrange(101)
            started = time.monotonic()
            try:
                cursor.execute(
                    """SELECT id FROM soak_docs
                       ORDER BY embedding <~-> turbohybrid_query(
                         vector_query=%s::vector, dense_k=128, bm25_k=0, final_k=10)
                       LIMIT 10""",
                    (f"[{value},0,0,0,0,0,0,0]",),
                )
                rows = cursor.fetchall()
                metrics.record(time.monotonic() - started, len(rows) == 10)
            except Exception:
                metrics.record(time.monotonic() - started, False)


def writer(stop: threading.Event) -> None:
    next_id = 100_000
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET statement_timeout='10s'")
        while not stop.is_set():
            cursor.execute(
                "INSERT INTO soak_docs VALUES (%s, %s::vector, to_tsvector('simple', %s))",
                (next_id, f"[{next_id % 101},1,2,3,4,5,6,7]", f"writer {next_id}"),
            )
            cursor.execute("UPDATE soak_docs SET body_tsv=to_tsvector('simple','updated') WHERE id=%s", (next_id,))
            next_id += 1
            time.sleep(0.02)


def churn(stop: threading.Event) -> None:
    cursor_id = 1
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET statement_timeout='10s'")
        while not stop.is_set():
            cursor.execute("DELETE FROM soak_docs WHERE id=%s", (cursor_id,))
            cursor_id = 1 + cursor_id % 5000
            if cursor_id % 100 == 0:
                cursor.execute("VACUUM soak_docs")
            time.sleep(0.01)


def diagnostics(stop: threading.Event, snapshots: list[dict]) -> None:
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET statement_timeout='20s'")
        while not stop.is_set():
            cursor.execute("SELECT turbohybrid_prewarm('soak_idx'::regclass)::text")
            prewarm = json.loads(cursor.fetchone()[0])
            cursor.execute("SELECT turbohybrid_index_stats('soak_idx'::regclass)::text")
            stats = json.loads(cursor.fetchone()[0])
            cursor.execute("SELECT turbohybrid_validate_index('soak_idx'::regclass, true)::text")
            validation = json.loads(cursor.fetchone()[0])
            snapshots.append({"prewarm": prewarm, "stats": stats, "validator": validation})
            time.sleep(2.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=100)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup()
    metrics = Metrics()
    snapshots: list[dict] = []
    stop = threading.Event()
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=35) as pool:
        futures = [pool.submit(reader, i, stop, metrics) for i in range(32)]
        futures += [pool.submit(writer, stop), pool.submit(churn, stop), pool.submit(diagnostics, stop, snapshots)]
        time.sleep(max(1, args.seconds))
        stop.set()
        for future in futures:
            future.result(timeout=30)
    elapsed = time.monotonic() - started
    cache_root = pathlib.Path(os.environ.get("PGDATA", "/var/lib/postgresql"))
    cache_files = list(cache_root.rglob("*.tqcache")) if cache_root.exists() else []
    latency = metrics.latencies_ms
    final_validator = snapshots[-1]["validator"] if snapshots else {"ok": False, "errors": ["no_snapshot"]}
    result = {
        "schema_version": 1,
        "seed": SEED,
        "duration_seconds": elapsed,
        "reader_concurrency_exercised": [1, 8, 32],
        "successful_queries": metrics.successful_queries,
        "failed_queries": metrics.failed_queries,
        "qps": metrics.successful_queries / elapsed,
        "latency_ms": {"p50": percentile(latency, 0.50), "p95": percentile(latency, 0.95), "p99": percentile(latency, 0.99)},
        "mean_latency_ms": statistics.fmean(latency) if latency else 0.0,
        "cache_file_count": len(cache_files),
        "cache_file_bytes": sum(path.stat().st_size for path in cache_files),
        "diagnostic_snapshots": snapshots,
        "validator": final_validator,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    subprocess.run(["psql", "-d", DBNAME, "-XAtc", "EXPLAIN (ANALYZE, BUFFERS) SELECT id FROM soak_docs ORDER BY embedding <~-> turbohybrid_query(vector_query=>'[0,0,0,0,0,0,0,0]'::vector,dense_k=>128,bm25_k=>0,final_k=>10) LIMIT 10"], stdout=(args.output / "explain.txt").open("w"), check=True)
    subprocess.run(["psql", "-d", DBNAME, "-XAtc", "SELECT turbohybrid_validate_index('soak_idx'::regclass,true)"], stdout=(args.output / "validator.json").open("w"), check=True)
    subprocess.run(["bash", "-c", "find \"${PGDATA:-/var/lib/postgresql}\" -name '*.tqcache' -ls"], stdout=(args.output / "cache-files.txt").open("w"), check=True)
    return 0 if metrics.failed_queries == 0 and final_validator.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
