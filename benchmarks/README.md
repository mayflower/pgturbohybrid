# pgturbohybrid Benchmarks

Benchmark comparisons for `pgturbohybrid` must use real embedding datasets and
documented relevance metrics. The benchmark setup treats pgvector as an
upstream dependency and installs `pgturbohybrid` as a separate extension. It
does not require a patched pgvector checkout.

## Baselines

Every publishable run should include at least:

- `postgres_sql_rrf`: pgvector HNSW over `vector`, PostgreSQL full-text search
  over `tsvector`, and SQL reciprocal-rank fusion.
- `pgturbohybrid`: one pgturbohybrid index over the same `vector` and
  `tsvector` columns.

When relevant, also compare `pgturbohybrid_exact_storage_off` to document the
latency, storage, and quality tradeoff from disabling exact vector storage.

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

`config/datasets.json` lists intended quality and systems datasets. FIQA, BEIR,
MS MARCO, MIRACL, LoTTE, and RAG sets should be run as reproducible
experiments with committed commands and external result artifacts.

Synthetic vector generators are intentionally not part of the benchmark suite.
They are too far from real retrieval workloads for project performance claims.
