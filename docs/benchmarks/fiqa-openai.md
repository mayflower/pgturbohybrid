# FIQA/OpenAI Benchmark Snapshot

This file helps you understand and reproduce the public FIQA/OpenAI benchmark
snapshot for `v0.1.0-alpha.2`.

It is a useful result, but it is still one benchmark on one dataset and one
machine. Results vary by dataset and hardware. Treat it as evidence to
reproduce, not a universal performance claim.

## Short Summary

- Dataset: FIQA
- Embeddings: OpenAI `text-embedding-3-small`
- Corpus rows: 57,638
- Queries: 648
- Embedding dimensions: 1,536
- Warmup: three passes
- Measurement: one measured pass
- Retrieval profile: `latency`

Terms used below:

- RRF means reciprocal-rank fusion.
- HNSW means Hierarchical Navigable Small World, pgvector's graph index type
  for approximate nearest-neighbor vector search.
- BM25 means Best Matching 25, a text-ranking method for exact term matches.
- GIN means Generalized Inverted Index, PostgreSQL's common index type for
  full-text search.
- nDCG means normalized discounted cumulative gain, a relevance metric for
  ranked results.
- MRR means mean reciprocal rank, another ranked-result quality metric.

## Results

| Method | Settings | p50 | p95 | p99 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| pgturbohybrid default | `latency` profile, default index (`quantization_bits=4`, `exact_storage=off`), adaptive widening off, effective `dense_k=100`, `bm25_k=100`, LIMIT-inferred `final_k=10` | 0.759 ms | 1.628 ms | 1.982 ms | 0.415535 |
| pgturbohybrid adaptive auto 2.0 | Same index definition and budgets, with adaptive dense widening explicitly set to `auto` and multiplier `2.0` | 1.139 ms | 1.552 ms | 1.708 ms | 0.421465 |
| SQL RRF baseline | pgvector HNSW default reloptions plus PostgreSQL GIN full-text search, `dense_k=100`, `bm25_k=100` | 1.333 ms | 2.009 ms | 3.603 ms | 0.423430 |
| pgvector HNSW dense-only baseline | pgvector HNSW default reloptions, dense vector search only | 0.767 ms | 1.412 ms | 2.372 ms | 0.442786 |

## How To Read The Comparisons

The direct hybrid comparison is `pgturbohybrid` versus SQL RRF. RRF means
reciprocal-rank fusion: ranked dense and lexical result lists are merged by rank
position rather than by comparing raw scores. Both rows use dense and lexical
retrieval, but `pgturbohybrid` fuses candidates inside the `turbohybrid` index
access method while SQL RRF combines two retrieval paths at the SQL level.

The pgvector HNSW dense-only row is a useful speed and quality reference, but it
is not a hybrid retrieval baseline. The dense-only row does not use the
lexical/BM25 branch, so it can behave differently on names, IDs, rare terms,
and keyword-heavy queries.

Any benchmark claim based on this page must include:

- dataset and embedding model
- dimensions and query count
- PostgreSQL version and settings
- pgvector release or commit
- `pgturbohybrid` commit
- retrieval profile
- index settings
- query budgets
- warmup and measured pass counts
- quality metric
- baseline definitions

Do not summarize this result as a global pgvector comparison. The measured
hybrid claim here is narrower: on this FIQA/OpenAI setup, with 1,536-dimensional
OpenAI embeddings, 648 queries, the `latency` profile, 4-bit quantization,
`exact_storage = off`, 100 dense candidates, 100 BM25 candidates, and nDCG@10 as
the quality metric, the explicit adaptive-auto `pgturbohybrid` row had lower
p95 latency than the SQL RRF hybrid baseline using pgvector HNSW default
reloptions plus PostgreSQL GIN full-text search while keeping nDCG@10 within
the configured acceptance threshold. The package default keeps adaptive dense
widening off; that row is faster to configure and evaluate, but it is below the
current FIQA/OpenAI quality guard for this artifact.

## Reproduction

Set the benchmark environment:

```sh
export FIQA_DATASET=/path/to/beir/fiqa
export PGDATABASE=pgturbohybrid_fiqa
export OUTPUT=/tmp/pgturbohybrid-fiqa.json
```

`FIQA_DATASET` must point to a local FIQA/OpenAI dataset directory containing:

- `corpus.jsonl`
- `corpus_embeddings.jsonl`
- `queries.jsonl`
- `query_embeddings.jsonl`
- `qrels/test.tsv`

Then run:

```sh
python3 benchmarks/fiqa_openai.py \
  --database "$PGDATABASE" \
  --dataset "$FIQA_DATASET" \
  --profile latency \
  --dense-k 100 \
  --bm25-k 100 \
  --final-k 10 \
  --warmup 3 \
  --reuse-data \
  --methods pgturbohybrid,pgturbohybrid_recovered_explicit,pgturbohybrid_adaptive_auto_2_0,postgres_sql_rrf,pgvector_hnsw_dense_only \
  --explain \
  --bm25-cache-probe \
  --output "$OUTPUT"
```

If you want a fresh database for a reproduction run, create or reset
`$PGDATABASE` before running the benchmark. The output location is controlled by
`$OUTPUT`; the path above is only an example.

The generated artifact is an input for review. The current numbers recover the
FIQA quality gap with explicit adaptive widening, but they do not satisfy the
older 2.0x p95 speedup gate versus SQL RRF, so do not publish this as a
fast-default acceptance pass.

## Quality Guard

The release acceptance gate is defined in
[`benchmarks/config/acceptance_thresholds.json`](../../benchmarks/config/acceptance_thresholds.json).
For the FIQA/OpenAI fast-default suite (`fiqa_openai_fast_defaults`), the
generated artifact must include the configured `pgturbohybrid_recovered_explicit`
method row and the `postgres_sql_rrf` baseline row (the run above includes both
in `--methods`).
The current default-off versus adaptive-auto artifact should be read as a
recovery diagnostic unless the threshold file is intentionally updated and the
gate passes.

The guard checks:

- `nDCG@10`: maximum allowed delta versus SQL RRF is `0.005`
- `MRR@10`: maximum allowed delta versus SQL RRF is `0.02`
- `Recall@10`: maximum allowed delta versus SQL RRF is `0.03`
- `overlap@10`: accepted as an alternative or companion recall guard; minimum
  overlap versus SQL RRF is `0.70`
- p95 latency: minimum accepted ratio versus SQL RRF is `2.0x` for this suite

If the default row misses the quality guard, evaluate safer settings before
publishing a fast-default result: explicit adaptive dense widening,
`SET turbohybrid.profile = 'quality';`, an index with `exact_storage = on`, or
larger dense and BM25 candidate budgets.

## Example Run

The table above comes from one Apple M4 development run. These numbers are not
portable promises; CPU, storage, PostgreSQL settings, pgvector version, compiler
flags, cache state, and corpus shape can all change the result.

Recorded metadata for the run:

- CPU: Apple M4
- OS: macOS 26.5 arm64, Darwin 25.5.0
- PostgreSQL: 16.13, Homebrew build
- pgvector: 0.8.2
- `pgturbohybrid` commit: `1cb7fe794ab4c32635ca7026a5ec468a8e4f3791`
- compiler: Apple clang 21.0.0 with PostgreSQL PGXS `pg_config --cflags`
  defaults. PGXS is PostgreSQL's extension build system.
- SIMD: portable build, ARM dotprod and i8mm compiled, SIMD enabled. SIMD means
  single instruction, multiple data.

When publishing a new run, replace this metadata with the machine and software
that produced that artifact.

## Artifact Policy

Generated benchmark JSON, CSV, and Markdown summaries are not committed to the
repository. Write them outside the checkout or under ignored directories such
as `benchmarks/results/` or `benchmarks/output/`.

Attach full generated artifacts to GitHub releases, pull requests, or CI runs.
That keeps the repository small while preserving the evidence needed to inspect
MRR, recall, overlap, latency distributions, query plans, and diagnostic JSON.
