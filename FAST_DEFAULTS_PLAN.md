# Fast Defaults Plan

This plan defines what "fast by default" means for `pgturbohybrid` and what the
explicit quality escape hatch must do.

## Recovered Benchmark Result

The target default is based on the recovered FIQA/OpenAI benchmark run:

| Property | Value |
|---|---:|
| Dataset | FIQA/OpenAI |
| Corpus rows | 57,638 |
| Queries | 648 |
| Embedding dimensions | 1,536 |
| Profile | `latency` |
| Quantization | 4-bit |
| Exact storage | off |
| Dense candidates | 100 |
| BM25 candidates | 100 |
| Final k | 10 |
| p50 latency | 0.770 ms |
| p95 latency | 1.006 ms |
| p99 latency | 1.265 ms |
| SQL RRF p95 | 3.988 ms |
| pgvector HNSW dense-only p95 | 3.361 ms |

Quality in this run stayed close to the SQL RRF baseline: nDCG@10 essentially
matched SQL RRF while p95 latency was materially lower.

## 2026-05-25 Validation Result

The final validation run used the same real FIQA/OpenAI dataset and measured
both the easy default path and the explicit recovered settings:

| Method | Settings | p50 | p95 | p99 | nDCG@10 |
|---|---|---:|---:|---:|---:|
| pgturbohybrid default | default index, omitted budgets, LIMIT-inferred final_k | 0.729 ms | 0.910 ms | 1.096 ms | 0.421540 |
| pgturbohybrid explicit recovered | 4-bit, exact_storage=off, dense_k=100, bm25_k=100, rrf_k=60, final_k=10 | 0.793 ms | 1.030 ms | 1.361 ms | 0.421540 |
| SQL RRF | pgvector HNSW + Postgres FTS, 100/100 | 1.635 ms | 3.254 ms | 5.579 ms | 0.423341 |
| pgvector HNSW dense-only | baseline | 1.129 ms | 2.718 ms | 4.052 ms | 0.439863 |

The canonicalized default path beats the `1.006 ms` p95 target on this
machine. The omitted-budget default path is structurally correct and reports
`profile = latency`, `dense_k_effective = 100`, `bm25_k_effective = 100`,
`final_k_source = limit`, and `scan_orchestration = graph_native`. After
canonicalizing latency-profile defaults before execution, the default path is
close to the explicit recovered path while preserving LIMIT/default diagnostics.

## New Default Contract

The default user experience should match the recovered latency-oriented setup:

- `turbohybrid.profile = 'latency'`
- `dense_k = 100`
- `bm25_k = 100`
- `rrf_k = 60`
- `final_k` defaults to the SQL `LIMIT` when the planner/executor context makes
  it available.
- If no SQL `LIMIT` is available, `final_k` uses a small safe default, currently
  planned as `10`.
- `quantization_bits = 4`
- `exact_storage = off`
- BM25 WAND is enabled.
- BM25 impact/head fast paths are enabled where exact-safe, or where explicitly
  covered by the latency profile contract.
- BM25 hot postings cache is enabled by default.
- Approximate latency fast paths are enabled only where documented and covered
  by relevance metrics.
- SIMD is enabled when supported by the build and host CPU.

Defaults must be visible in `turbohybrid_index_stats()` and scan diagnostics so
benchmark reports can prove which behavior was active.

## Escape Hatches

Users must be able to opt back into safer or higher-quality behavior explicitly:

```sql
SET turbohybrid.profile = 'balanced';
```

```sql
SET turbohybrid.profile = 'quality';
```

Quality profile means:

- `dense_k = 400`
- `bm25_k = 400`
- `rrf_k = 60`
- BM25 impact OR mode is `exact_only`
- BM25 hybrid bound mode is `safe`, not `approx`
- BM25 automatic budget reduction is disabled
- BM25-only exact rescore is preferred where available
- SIMD remains enabled

Exact vectors can be retained for final exact rescoring:

```sql
CREATE INDEX documents_turbohybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (exact_storage = on);
```

Candidate budgets and final result count remain explicit query controls:

```sql
SELECT *
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => $1,
    text_query => plainto_tsquery('english', $2),
    dense_k => 400,
    bm25_k => 400,
    final_k => 20
)
LIMIT 20;
```

## Risks

- `exact_storage = off` can reduce exact-rescore quality on some datasets.
- Approximate BM25 paths must be benchmarked with relevance metrics before they
  become part of the default contract.
- The on-disk format is still alpha and may change before a stable release.
- Fast defaults are tuned from FIQA/OpenAI results; entity-heavy, code-heavy,
  and domain-specific corpora may need different quality settings.
- Enabling BM25 hot postings cache by default consumes backend memory and needs
  a conservative default size.

## Validation Requirements

Before changing code defaults, run and record:

- regression tests for default profile, reloptions, query budgets, and escape
  hatches,
- full FIQA/OpenAI benchmark with 57,638 corpus rows and 648 queries,
- perf smoke with the canonical real-data runner,
- BM25 first-query cold/warm probe,
- vector-validation overhead guard,
- planner/index-use validation,
- README and benchmark documentation updates.

The implementation should not claim a new fast default until the benchmark
report states the exact profile, reloptions, query budgets, dataset, warmup
policy, quality metrics, index size, and build time.
