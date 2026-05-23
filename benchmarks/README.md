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
