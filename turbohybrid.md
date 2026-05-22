# TurboHybrid SQL Usage

TurboHybrid is a PostgreSQL index access method that stores dense vector search
and BM25 lexical search in one index. This page shows the SQL needed to create
and query TurboHybrid indexes.

## Create A Table

Use one `vector` column for embeddings and one `tsvector` column for BM25. A
stored generated column is the simplest way to keep the lexical column updated.

```sql
CREATE EXTENSION vector;

CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    embedding vector(3) NOT NULL,
    body text NOT NULL,
    body_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
);

INSERT INTO documents (embedding, body)
VALUES
    ('[1,0,0]', 'postgres vector search'),
    ('[1,1,0]', 'hybrid search with bm25'),
    ('[0,1,0]', 'lexical search in postgres');
```

For production data, replace `vector(3)` with your embedding dimension, for
example `vector(1536)`.

## Create A Cosine TurboHybrid Index

Use `vector_cosine_hybrid_ops` for cosine distance and `bm25_tsvector_ops` for
the lexical side.

```sql
CREATE INDEX documents_turbohybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_hybrid_ops,
    body_tsv bm25_tsvector_ops
)
WITH (
    tq_bits = 4,
    tq_exact_storage = off,
    graph_m = 16,
    graph_ef_construction = 128,
    graph_ef_search = 64,
    graph_oversampling = 4
);
```

Use `tq_exact_storage = on` if you want exact vectors retained in the index for
safer dense rescoring. Use `tq_exact_storage = off` for a smaller index and
lower-latency quantized-only dense scoring.

## Query Dense And BM25 Together

Use `hybrid_query(...)` with the hybrid cosine operator `<~>`.

```sql
SELECT id, body
FROM documents
ORDER BY embedding <~> hybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres search'),
    dense_k => 100,
    bm25_k => 100,
    rrf_k => 60,
    final_k => 10
)
LIMIT 10;
```

`dense_k` is the number of dense candidates gathered before fusion. `bm25_k` is
the number of BM25 candidates gathered before fusion. `final_k` is the desired
final result count and should usually match the SQL `LIMIT`.

## Dense-Only Query

Pass only `vector_query`.

```sql
SELECT id, body
FROM documents
ORDER BY embedding <~> hybrid_query(
    vector_query => '[1,0,0]'::vector,
    dense_k => 100,
    final_k => 10
)
LIMIT 10;
```

## BM25-Only Query

Pass only `text_query`, and set `dense_k` to `0` when you want the query to be
pure lexical retrieval.

```sql
SELECT id, body
FROM documents
ORDER BY embedding <~> hybrid_query(
    text_query => websearch_to_tsquery('english', 'postgres search'),
    dense_k => 0,
    bm25_k => 100,
    final_k => 10
)
LIMIT 10;
```

## Require A Lexical Match

Use `require_bm25_match => true` when dense candidates must also match the text
query.

```sql
SELECT id, body
FROM documents
ORDER BY embedding <~> hybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres search'),
    dense_k => 100,
    bm25_k => 100,
    final_k => 10,
    require_bm25_match => true
)
LIMIT 10;
```

## Fusion Modes

The default fusion mode is reciprocal rank fusion.

```sql
SELECT id, body
FROM documents
ORDER BY embedding <~> hybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres search'),
    fusion => 'rrf',
    rrf_k => 60,
    dense_k => 100,
    bm25_k => 100,
    final_k => 10
)
LIMIT 10;
```

Weighted fusion is also available.

```sql
SELECT id, body
FROM documents
ORDER BY embedding <~> hybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres search'),
    fusion => 'weighted',
    dense_weight => 0.7,
    bm25_weight => 0.3,
    alpha => 0.5,
    dense_k => 100,
    bm25_k => 100,
    final_k => 10
)
LIMIT 10;
```

## L2 And Inner Product Indexes

For L2 distance, use `vector_l2_hybrid_ops` and the `<~->` operator.

```sql
CREATE INDEX documents_turbohybrid_l2_idx ON documents
USING turbohybrid (
    embedding vector_l2_hybrid_ops,
    body_tsv bm25_tsvector_ops
);

SELECT id, body
FROM documents
ORDER BY embedding <~-> hybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres search'),
    dense_k => 100,
    bm25_k => 100,
    final_k => 10
)
LIMIT 10;
```

For inner product, use `vector_ip_hybrid_ops` and the `<~#>` operator.

```sql
CREATE INDEX documents_turbohybrid_ip_idx ON documents
USING turbohybrid (
    embedding vector_ip_hybrid_ops,
    body_tsv bm25_tsvector_ops
);

SELECT id, body
FROM documents
ORDER BY embedding <~#> hybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres search'),
    dense_k => 100,
    bm25_k => 100,
    final_k => 10
)
LIMIT 10;
```

## Include Columns

Use `INCLUDE` for columns that should be available to the index scan as payload
or filter context.

```sql
CREATE INDEX documents_turbohybrid_tenant_idx ON documents
USING turbohybrid (
    embedding vector_cosine_hybrid_ops,
    body_tsv bm25_tsvector_ops
)
INCLUDE (id)
WITH (
    tq_bits = 4,
    tq_exact_storage = off
);
```

## Session Settings

These settings affect query execution and can be set per session.

```sql
SET hybrid.default_dense_k = 100;
SET hybrid.default_bm25_k = 100;
SET hybrid.default_rrf_k = 60;
```

These defaults are used only when the corresponding `hybrid_query(...)`
arguments are omitted or set to `NULL`. Pass `dense_k`, `bm25_k`, and `rrf_k`
in the query when you want statement-local control.

## Index Options

Common reloptions for `CREATE INDEX ... WITH (...)`:

| Option | Example | Purpose |
| --- | --- | --- |
| `routing` | `graph` or `hnsw` | Dense routing implementation for the index. |
| `tq_bits` | `4` | Number of quantization bits for dense codes. |
| `tq_exact_storage` | `on` or `off` | Store exact vectors in the index or use quantized-only dense scoring. |
| `graph_m` | `16` | Dense graph connectivity. |
| `graph_ef_construction` | `128` | Build-time graph search width. |
| `graph_ef_search` | `64` | Query-time graph search width. |
| `graph_oversampling` | `4` | Dense candidate oversampling before final selection. |

Changes to index options require rebuilding the index.

## Check The Plan

Use `EXPLAIN` to confirm PostgreSQL is using the TurboHybrid index.

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, body
FROM documents
ORDER BY embedding <~> hybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres search'),
    dense_k => 100,
    bm25_k => 100,
    final_k => 10
)
LIMIT 10;
```

On very small tables PostgreSQL may prefer a sequential scan. For local testing,
temporarily disable sequential scans:

```sql
SET enable_seqscan = off;
-- run EXPLAIN or the test query
RESET enable_seqscan;
```

## Rebuild An Index

Use `REINDEX` after changing reloptions that affect index storage.

```sql
REINDEX INDEX documents_turbohybrid_idx;
```

Or recreate the index explicitly:

```sql
DROP INDEX documents_turbohybrid_idx;

CREATE INDEX documents_turbohybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_hybrid_ops,
    body_tsv bm25_tsvector_ops
)
WITH (
    tq_bits = 4,
    tq_exact_storage = off
);
```
