# pgturbohybrid Benchmarks

The benchmark harness treats pgvector as an upstream dependency and installs
`pgturbohybrid` as a separate extension. It does not require a patched pgvector
checkout.

## Baselines

Every publishable run should include at least:

- `postgres_sql_rrf`: pgvector HNSW over `vector`, PostgreSQL full-text search
  over `tsvector`, and SQL reciprocal-rank fusion.
- `pgturbohybrid`: one pgturbohybrid index over the same `vector` and
  `tsvector` columns.

When relevant, also compare `pgturbohybrid_exact_storage_off` to document the
latency, storage, and quality tradeoff from disabling exact vector storage.

## Local Smoke Run

Install pgvector and pgturbohybrid first:

```sh
PG_CONFIG=pg_config PGVECTOR_REF=v0.8.2 scripts/dev-install.sh
```

Then run the small synthetic benchmark:

```sh
python3 benchmarks/suite.py run-system-synthetic \
  --database pgturbohybrid_dev \
  --rows 1000 \
  --dimensions 16 \
  --runs 3 \
  --methods postgres_sql_rrf,pgturbohybrid \
  --output /tmp/pgturbohybrid_smoke.json
```

The smoke run creates a synthetic table, builds both baseline indexes, runs a
hybrid query, and reports build time, index size, and p50/p95/p99 latency.
Latency is measured inside PostgreSQL with `EXPLAIN (ANALYZE, FORMAT JSON,
TIMING OFF)` so results exclude Python and `psql` process startup time.

The SQL RRF baseline intentionally keeps the dense query vector as a literal in
the `ORDER BY` clause. This lets PostgreSQL pick the pgvector HNSW index path;
passing the vector through a materialized CTE can make the planner choose a
sequential scan and invalidates the comparison.

## Latest Standalone Synthetic Result

This result was run from the standalone `pgturbohybrid` extension architecture,
not from the earlier patched pgvector tree.

Environment:

- Date: 2026-05-24
- Hardware: Apple M4, 10 cores, 24 GiB memory
- OS: macOS 26.5 arm64
- PostgreSQL: 16.13 (Homebrew)
- pgvector: 0.8.2
- pgturbohybrid: 0.1.0
- Python: 3.14.5

Command:

```sh
dropdb --if-exists pgturbohybrid_benchmark
createdb pgturbohybrid_benchmark
python3 benchmarks/suite.py run-system-synthetic \
  --database pgturbohybrid_benchmark \
  --rows 100000 \
  --dimensions 1536 \
  --runs 30 \
  --warmup 5 \
  --dense-k 100 \
  --bm25-k 100 \
  --final-k 10 \
  --methods postgres_sql_rrf,pgturbohybrid,pgturbohybrid_exact_storage_off \
  --output /tmp/pgturbohybrid_benchmark_100k_1536_standalone.json
```

Plan check:

- `postgres_sql_rrf` used `pgturbohybrid_bench_hnsw_idx` for dense vector
  search and `pgturbohybrid_bench_fts_idx` for full-text search.
- `pgturbohybrid` and `pgturbohybrid_exact_storage_off` used
  `pgturbohybrid_bench_idx` through the `turbohybrid` access method.

Results:

| Method | Build s | Index MiB | p50 ms | p95 ms | p99 ms | QPS |
|---|---:|---:|---:|---:|---:|---:|
| pgvector HNSW + PostgreSQL FTS SQL RRF | 17.692 | 191.3 | 18.407 | 21.675 | 22.408 | 53.26 |
| pgturbohybrid, 4-bit, exact storage on | 132.511 | 687.0 | 41.834 | 51.643 | 59.286 | 23.22 |
| pgturbohybrid, 4-bit, exact storage off | 166.342 | 96.5 | 51.655 | 64.689 | 72.456 | 19.25 |

Interpretation:

- This synthetic standalone result does not reproduce the earlier patched-tree
  latency advantage. The corrected HNSW-backed baseline is faster on p50, p95,
  and p99.
- `pgturbohybrid` with `exact_storage = off` is about 49.5% smaller than the
  HNSW plus GIN baseline in this run.
- This is a systems benchmark only. It does not report recall, nDCG, MRR, or
  MAP. Use a real RAG dataset such as FIQA or BEIR before making quality
  claims.

## Publishable Run Metadata

Do not commit generated benchmark outputs. Store JSON/Markdown under `/tmp`, an
external artifact store, or ignored directories such as `benchmarks/results/`.

Record the following with any published result:

- hardware and CPU governor
- operating system
- PostgreSQL version and settings
- pgvector ref or release
- pgturbohybrid commit
- dataset and embedding model
- exact commands
- warmup policy and measured run count
- p50, p95, p99, QPS
- index size, build time, and WAL generated
- recall, nDCG, MRR, and MAP for quality datasets
- baseline definitions and index options

## Dataset Notes

`config/datasets.json` lists intended quality and systems datasets. The local
smoke benchmark is intentionally synthetic so it can run quickly in CI and does
not require external downloads. FIQA, BEIR, MS MARCO, MIRACL, LoTTE, and RAG
sets should be run as separate reproducible experiments with committed commands
but external result artifacts.

## Commands

List configured datasets and methods:

```sh
python3 benchmarks/suite.py list
```

Show the reproducibility checklist:

```sh
python3 benchmarks/suite.py plan
```

Run the synthetic smoke benchmark:

```sh
python3 benchmarks/suite.py run-system-synthetic --database pgturbohybrid_dev
```
