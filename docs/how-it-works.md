# How TurboHybrid Works

This file helps PostgreSQL, pgvector, and application engineers understand the
basic TurboHybrid query path without reading extension internals.

It starts with the query shape, then explains why dense vector search and
keyword search are useful together.

## One-Minute Explanation

A TurboHybrid query has two inputs:

- a vector query, usually an embedding from the same model used for your stored
  document embeddings
- a text query, usually a PostgreSQL `tsquery` such as
  `websearch_to_tsquery(...)` or `plainto_tsquery(...)`

At scan time, TurboHybrid does three practical things:

1. It retrieves dense candidates from the pgvector `vector` side.
2. It retrieves BM25/text candidates from the `tsvector` side. BM25 means Best
   Matching 25, a classic keyword-ranking method for text search.
3. It fuses those candidate lists with reciprocal-rank fusion, or RRF. RRF
   combines ranked lists by position instead of comparing unlike raw scores.

The fused result is returned through an index-backed PostgreSQL
`ORDER BY ... LIMIT` path:

```sql
SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => $1::vector,
    text_query => websearch_to_tsquery('english', $2)
)
LIMIT 10;
```

In normal use, the SQL `LIMIT` becomes the final top-k target. You ask
PostgreSQL for the top 10, and TurboHybrid uses that shape to keep the work
bounded.

## Why Hybrid Search?

Dense vector search is good at meaning. It can find documents that say the same
thing in different words, which is why it is useful for RAG
(retrieval-augmented generation), semantic search, and question answering.

Lexical search is good at exactness. It finds names, IDs, rare terms, keywords,
model numbers, function names, and all the oddly specific strings users care
about.

Each approach has failure modes:

- dense-only search can miss exact terms that were not represented strongly in
  the embedding
- lexical-only search can miss paraphrases, synonyms, and concept matches

Hybrid search tries to reduce both sets of misses. Dense search gets you the
meaning. Lexical search keeps the exact words honest.

## What pgturbohybrid Adds

`pgturbohybrid` is a standalone PostgreSQL extension. It depends on pgvector's
SQL `vector` type, but it does not modify pgvector.

It adds the TurboHybrid feature surface:

- access method: `turbohybrid`
- query constructor: `turbohybrid_query(...)`
- vector operator classes:
  - `vector_l2_turbohybrid_ops`
  - `vector_ip_turbohybrid_ops`
  - `vector_cosine_turbohybrid_ops`
- BM25/text operator class:
  - `bm25_tsvector_turbohybrid_ops`
- diagnostics:
  - `turbohybrid_last_scan_stats()`
  - `turbohybrid_index_stats(regclass)`
  - `turbohybrid_simd_capabilities()`

A typical index combines one dense column and one generated text-search column:

```sql
CREATE INDEX documents_turbohybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);
```

## Default Fast Path

Fresh sessions use the `latency` profile:

```sql
SHOW turbohybrid.profile;
```

The default fast path is designed to be small and predictable:

- `turbohybrid.profile = 'latency'`
- 4-bit dense-vector quantization
- exact vector storage off in the index
- 100 dense candidates
- 100 BM25/text candidates
- RRF constant 60
- SQL `LIMIT` inferred as the internal `final_k` target when possible

That means the common query shape is intentionally simple: create the default
index, omit manual `dense_k`, `bm25_k`, and `final_k` arguments, and let
`ORDER BY ... LIMIT n` drive the final result count.

## Quality Path

Use the `quality` profile when relevance matters more than the lowest latency:

```sql
SET turbohybrid.profile = 'quality';
```

Quality mode uses larger dense and BM25 candidate budgets and safer BM25 paths.
It is the right first switch when the latency profile is too aggressive for
your dataset.

For quality-sensitive evaluation, also benchmark an exact-storage index:

```sql
CREATE INDEX documents_turbohybrid_quality_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (exact_storage = on);
```

The tradeoff is straightforward: expect more work per query, and measure both
latency and relevance before choosing production settings.

## What It Does Not Do

`pgturbohybrid` is deliberately scoped.

- It does not modify pgvector.
- It does not replace pgvector HNSW or IVFFlat indexes. HNSW means
  Hierarchical Navigable Small World, pgvector's graph index type for
  approximate nearest-neighbor vector search.
- It is not an official pgvector project.
- It is not stable production software yet.
- It does not remove the need for dataset-specific relevance evaluation.

If dense-only pgvector search is the right tool for a table, keep using it. If
PostgreSQL full-text search is enough, keep that simple too. TurboHybrid is for
cases where both semantic and lexical signals matter in the same top-k query.

## Mental Model

The hot hatch analogy is useful if you keep it modest:

- the PostgreSQL table is the practical hatchback
- pgvector is the proven engine for dense search
- the BM25 branch is extra grip for exact words
- TurboHybrid is sport suspension and a turbo, not a trailer full of magic

You still drive on normal PostgreSQL roads: tables, indexes, `ORDER BY`, and
`LIMIT`. TurboHybrid just gives one retrieval path a sharper setup.
