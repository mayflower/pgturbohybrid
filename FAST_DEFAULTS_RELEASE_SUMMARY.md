# Fast Defaults Release Summary

This release makes the latency-optimized retrieval path the normal first
experience for `pgturbohybrid`.

## What Changed

- `turbohybrid.profile` now defaults to `latency`.
- Default query budgets are `dense_k = 100`, `bm25_k = 100`, and `rrf_k = 60`.
- `final_k` is inferred from the SQL `LIMIT` when possible, with a small default
  target when no limit is available.
- Default indexes use `quantization_bits = 4`.
- `exact_storage` defaults to `off` for smaller indexes and faster retrieval.
- BM25 latency fast paths are enabled by default, including WAND, impact/head
  paths where allowed, hot postings cache, and approximate latency paths covered
  by the FIQA/OpenAI quality guard.
- The latency profile canonicalizes omitted-budget queries internally after
  LIMIT/default resolution, so the easy query shape follows the same hot path as
  the recovered explicit setup.
- Scan and index diagnostics now expose the effective profile, budgets, final-k
  source, index use, timing breakdowns, quantization, exact-storage state, and
  validation counters.
- README, fast-start examples, benchmark docs, quality thresholds, perf-smoke,
  and recovery notes were updated around the fast default path.

## Why

The recovered FIQA/OpenAI benchmark showed that the useful fast setup was simple:
4-bit quantization, exact storage off, 100 dense candidates, 100 BM25 candidates,
RRF `k = 60`, and `final_k = 10`.

The previous user path required passing those settings explicitly. This release
makes the recovered setup the default so users can create a normal index and run
a normal `ORDER BY ... LIMIT n` query without knowing every tuning knob first.

## Benchmark Result

Dataset: FIQA/OpenAI, 57,638 corpus rows, 648 qrels-backed queries,
1,536-dimensional OpenAI `text-embedding-3-small` embeddings, one warmup pass,
one measured pass.

| Method | Settings | p50 | p95 | p99 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| pgturbohybrid default | latency profile, default index, omitted budgets, LIMIT-inferred final_k | 0.729 ms | 0.910 ms | 1.096 ms | 0.421540 |
| pgturbohybrid explicit recovered | 4-bit, `exact_storage=off`, 100/100/60, `final_k=10` | 0.793 ms | 1.030 ms | 1.361 ms | 0.421540 |
| SQL RRF baseline | pgvector HNSW + PostgreSQL FTS, 100/100 | 1.635 ms | 3.254 ms | 5.579 ms | 0.423341 |
| pgvector HNSW dense-only baseline | dense vector baseline | 1.129 ms | 2.718 ms | 4.052 ms | 0.439863 |

Generated benchmark JSON is not committed. The accepted local validation artifact
was `/tmp/pgturbohybrid-fiqa-canonicalized-with-baseline.json`.

## Quality Safeguards

Fast defaults are not the only supported mode. Use these escape hatches when
exact recall or dataset-specific relevance matters more than latency:

```sql
SET turbohybrid.profile = 'quality';
```

```sql
CREATE INDEX documents_turbohybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (exact_storage = on);
```

```sql
SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => $1,
    text_query => websearch_to_tsquery('english', $2),
    dense_k => 400,
    bm25_k => 400,
    final_k => 10
)
LIMIT 10;
```

The benchmark acceptance gate checks nDCG@10, MRR@10, recall or overlap versus
SQL RRF, and requires the full benchmark p95 to stay at least 2x faster than SQL
RRF for published fast-default claims.

## Troubleshooting

Check that the query used the intended index path:

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => $1,
    text_query => websearch_to_tsquery('english', $2)
)
LIMIT 10;
```

Check the active profile:

```sql
SHOW turbohybrid.profile;
```

Check the last scan:

```sql
SELECT turbohybrid_last_scan_stats();
```

Useful fields include `profile`, `index_used`, `scan_orchestration`,
`dense_candidates_effective`, `bm25_candidates_effective`, `final_k_effective`,
`final_k_source`, `dense_elapsed_us`, `bm25_elapsed_us`, `fusion_elapsed_us`,
and `elapsed_us`.

Check the index settings:

```sql
SELECT turbohybrid_index_stats('documents_turbohybrid_idx'::regclass);
```

Useful fields include `quantization_bits`, `exact_storage`, `graph_ef_search`,
`graph_oversampling`, and `routing`.

## Risks

- The extension is still alpha and the on-disk format is not stable.
- Relevance can vary by dataset. Validate fast defaults against your own qrels or
  a trusted SQL RRF baseline before relying on production settings.
- `exact_storage = off` can reduce exact-rescore quality on some datasets.
- The extension depends on pgvector's SQL `vector` type and compatibility with
  pgvector's installed type layout. pgvector ABI or layout changes may require a
  `pgturbohybrid` compatibility update.
- Approximate BM25 latency paths are enabled by the latency profile and should be
  evaluated with relevance metrics for quality-sensitive workloads.

## Recommended User SQL

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;

CREATE INDEX documents_turbohybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);

SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => $1,
    text_query => websearch_to_tsquery('english', $2)
)
LIMIT 10;
```
