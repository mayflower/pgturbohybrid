# Easy Fast Setup

This guide creates the default fast `pgturbohybrid` path: latency profile,
4-bit quantized index, exact storage off, SQL `LIMIT` as the final result
target, and no manual candidate-budget arguments.

## Install

Install pgvector first, then build and install `pgturbohybrid` with PGXS:

```sh
git clone https://github.com/mayflower/pgturbohybrid.git
cd pgturbohybrid
make
make install
```

## Database Setup

Create pgvector and pgturbohybrid in the database:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
```

The default profile is `latency`. You can verify it with:

```sql
SHOW turbohybrid.profile;
```

## Table Schema

Use a pgvector `vector` column for embeddings and a generated `tsvector` column
for lexical search. This local smoke test uses `vector(3)` so it can be run by
hand; use `vector(1536)` for OpenAI `text-embedding-3-small` data.

```sql
CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    embedding vector(3) NOT NULL,
    body text NOT NULL,
    body_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', body)
    ) STORED
);

INSERT INTO documents (embedding, body)
VALUES
    ('[1,0,0]', 'postgres vector search'),
    ('[1,1,0]', 'hybrid search with bm25'),
    ('[0,1,0]', 'lexical search in postgres'),
    ('[0,0,1]', 'unrelated document');
```

Run `ANALYZE` after loading data:

```sql
ANALYZE documents;
```

## Default Fast Index

Create the index without manual fast options. The defaults use 4-bit
quantization and `exact_storage = off`.

```sql
CREATE INDEX documents_turbohybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);

ANALYZE documents;
```

## Default Fast Query

Omit `dense_k`, `bm25_k`, and `final_k`. The latency profile supplies
100 dense candidates, 100 BM25 candidates, and RRF constant 60. SQL `LIMIT`
becomes the final result target when available.

```sql
SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres hybrid search')
)
LIMIT 10;
```

Use the correct vector dimension for your table in real data, for example
`$1::vector(1536)` for OpenAI `text-embedding-3-small` embeddings.

## Verify The Fast Path

First inspect the plan:

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres hybrid search')
)
LIMIT 10;
```

Then inspect the stable diagnostics for the last scan and the index:

```sql
SELECT turbohybrid_last_scan_stats();
SELECT turbohybrid_index_stats('documents_turbohybrid_idx'::regclass);
SELECT turbohybrid_simd_capabilities();
```

Useful fast-path indicators:

```json
{
  "profile": "latency",
  "index_used": true,
  "scan_orchestration": "graph_native",
  "quantization_bits": 4,
  "exact_storage": false,
  "dense_candidates_effective": 100,
  "bm25_candidates_effective": 100,
  "final_k_source": "limit"
}
```

## Common Fixes

| Problem | Check | Fix |
| --- | --- | --- |
| Plan does not use the index | `EXPLAIN (ANALYZE, BUFFERS)` and `scan_orchestration` | Ensure the `turbohybrid` index exists and the query uses `ORDER BY embedding <~> turbohybrid_query(...) LIMIT n`. |
| Planner has stale table stats | Last `ANALYZE` time and row estimates in `EXPLAIN` | Run `ANALYZE documents;` after loading data and after large changes. |
| `final_k_source` is `default` | Query has no SQL `LIMIT` | Add `LIMIT 10` or another explicit top-k limit. |
| `profile` is not `latency` | `SHOW turbohybrid.profile;` | Run `SET turbohybrid.profile = 'latency';` or reset role/database settings. |
| SIMD is unavailable or disabled | `SELECT turbohybrid_simd_capabilities();` | Build with the portable default and avoid `SIMD_BUILD=none`; check host CPU support. |
| Relevance drops on your dataset | Compare against SQL RRF or a labeled benchmark | Use `SET turbohybrid.profile = 'quality';`, rebuild with `WITH (exact_storage = on)`, or pass larger `dense_k` and `bm25_k`. |

Quality mode is the simple relevance-first escape hatch:

```sql
SET turbohybrid.profile = 'quality';
```

For quality-sensitive production evaluation, benchmark an exact-storage index:

```sql
CREATE INDEX documents_turbohybrid_quality_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (exact_storage = on);
```

## FIQA/OpenAI Validation

The final fast-default validation used the real FIQA/OpenAI benchmark: 57,638
corpus rows, 648 qrels-backed queries, OpenAI `text-embedding-3-small`
embeddings, 1,536 dimensions, one warmup pass, and one measured pass.

| Method | Settings | p50 | p95 | p99 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| pgturbohybrid default | default index, omitted budgets, LIMIT-inferred final_k | 0.729 ms | 0.910 ms | 1.096 ms | 0.421540 |
| pgturbohybrid explicit recovered | 4-bit, exact_storage=off, 100/100/60, final_k=10 | 0.793 ms | 1.030 ms | 1.361 ms | 0.421540 |
| SQL RRF | pgvector HNSW + PostgreSQL FTS, 100/100 | 1.635 ms | 3.254 ms | 5.579 ms | 0.423341 |

The easy default path is index-backed and uses the intended effective settings.
It is canonicalized internally after LIMIT/default resolution so it follows the
same hot-path shape as the explicit recovered query while still reporting
`final_k_source = limit` in diagnostics.
