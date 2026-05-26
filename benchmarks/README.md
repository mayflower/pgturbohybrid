# pgturbohybrid Benchmarks

Benchmark comparisons for `pgturbohybrid` must use real embedding datasets and
documented relevance metrics. The benchmark setup treats pgvector as an
upstream dependency and installs `pgturbohybrid` as a separate extension. It
does not require a patched pgvector checkout.

Public benchmark explanations live under `docs/benchmarks/`. This directory is
for reproducible tooling, acceptance thresholds, and developer benchmark
helpers. Developer-only workflows are documented in `benchmarks/dev/README.md`.

The DBPedia OpenAI3-large benchmark spec lives in
[`dbpedia_openai3_large.md`](dbpedia_openai3_large.md). It covers the
1M-row Qdrant DBPedia corpus, BEIR DBPedia queries/qrels, the native pgvector
`halfvec` + PostgreSQL full-text SQL RRF baseline, and the TurboHybrid runs.

## Baselines

Every publishable run should include at least:

- `postgres_sql_rrf`: pgvector HNSW over `vector`, PostgreSQL full-text search
  over `tsvector`, and SQL reciprocal-rank fusion.
- `pgturbohybrid`: one pgturbohybrid index over the same `vector` and
  `tsvector` columns.

When relevant, also compare `pgturbohybrid_recovered_explicit` to document the
latency, storage, and quality tradeoff from the recovered fast settings: 4-bit,
`exact_storage = off`, 100/100/60, and `final_k = 10`. The harness still accepts
`pgturbohybrid_exact_storage_off` as a legacy alias for older artifacts.

## Profile Matrix

Publishable FIQA/OpenAI results should include all of:

- `pgturbohybrid`: latency profile, default index options, omitted query
  budgets, and LIMIT-inferred `final_k`.
- `pgturbohybrid_recovered_explicit`: latency profile, 4-bit index,
  `exact_storage = off`, effective `dense_k = 100`, `bm25_k = 100`, and
  `final_k = 10`.
- `pgturbohybrid_quality`: quality profile, effective `dense_k = 400`,
  `bm25_k = 400`, exact-safe BM25 paths, SIMD enabled, and a documented
  `exact_storage` choice. Prefer `exact_storage = on` when evaluating final
  quality-sensitive settings.
- `postgres_sql_rrf`: pgvector HNSW plus PostgreSQL full-text search fused in
  SQL.

The result summary must compare quality profile against latency profile for
both relevance and latency. Do not infer quality-profile relevance from the
latency-profile artifact.

Validate a generated matrix artifact with:

```sh
FIQA_DATASET=/path/to/fiqa
PGDATABASE=pgturbohybrid_fiqa
OUTPUT=benchmarks/results/pgturbohybrid-fiqa.json
python3 benchmarks/tools/check_acceptance.py \
  "$OUTPUT" \
  --suite fiqa_openai_profile_matrix
```

## Publishable Run Metadata

Do not commit generated benchmark outputs. Store JSON/Markdown in an external
artifact store or ignored directories such as `benchmarks/results/`.

Record the following with any published result:

- hardware and CPU governor
- operating system
- PostgreSQL version and settings
- pgvector ref or release
- pgturbohybrid commit
- dataset and embedding model
- corpus size, query count, and embedding dimensions
- retrieval profile
- index settings and reloptions
- candidate budgets, fusion settings, and final result target
- exact commands
- warmup policy and measured run count
- p50, p95, p99, QPS
- index size, build time, and WAL generated
- quality metrics such as recall, nDCG, MRR, MAP, or overlap
- baseline definitions and index options
- note that results vary by dataset and hardware

## Acceptance Checks

Fast defaults must pass the FIQA/OpenAI quality gate before they are used for a
published claim. The gate is configured in
`config/acceptance_thresholds.json` and is intended for a full manual or
nightly FIQA run, not for per-PR perf smoke.

Run it against the generated full benchmark artifact:

```sh
OUTPUT=benchmarks/results/pgturbohybrid-fiqa.json
python3 benchmarks/tools/check_acceptance.py \
  "$OUTPUT" \
  --suite fiqa_openai_fast_defaults
```

The artifact must include `pgturbohybrid_recovered_explicit` latency-profile
results and the SQL RRF baseline with `nDCG@10`, `MRR@10`, `p95_ms`, and either
`Recall@10` or `overlap@10` versus SQL RRF. If the gate fails, evaluate
`quality` profile, `exact_storage = on`, or larger dense/BM25 budgets before
publishing the fast default result.

## Dataset Notes

`config/datasets.json` lists intended quality and systems datasets. FIQA, BEIR,
MS MARCO, MIRACL, LoTTE, and RAG sets should be run as reproducible
experiments with committed commands and external result artifacts.

For 3,072-dimensional DBPedia/OpenAI3-large runs, use
`benchmarks/dbpedia_openai3_large.py` rather than the FIQA harness. The native
PostgreSQL hybrid baseline uses pgvector `halfvec(3072)` HNSW plus PostgreSQL
full-text search because standard pgvector `vector` HNSW is not the intended
ANN path for this dimensionality.

Synthetic vector generators are intentionally not part of the benchmark suite.
They are too far from real retrieval workloads for project performance claims.
