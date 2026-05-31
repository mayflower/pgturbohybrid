# Fast Setup

This file helps you get from a fresh checkout to one working fast TurboHybrid
query in a few minutes.

It uses tiny `vector(3)` example data so you can copy and paste the SQL by hand.
For real embeddings, use the dimension from your model, such as `vector(1536)`.
TurboHybrid combines dense vector search with BM25-style keyword search; BM25
means Best Matching 25, a common text-ranking method for exact terms.

## 1. Install pgvector

Install pgvector first. `pgturbohybrid` depends on the pgvector SQL `vector`
type.

```sh
git clone --depth 1 --branch v0.8.2 https://github.com/pgvector/pgvector.git ../pgvector
make -C ../pgvector
make -C ../pgvector install
```

## 2. Install pgturbohybrid

Build and install this extension with PGXS, PostgreSQL's extension build
system:

```sh
git clone https://github.com/mayflower/pgturbohybrid.git
cd pgturbohybrid
make
make install
```

For repeatable local setup, the helper script wraps the same idea:

```sh
PG_CONFIG=pg_config PGVECTOR_REF=v0.8.2 scripts/dev-install.sh
```

## 3. Create Extensions

Create pgvector first, then `pgturbohybrid`:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
```

The default TurboHybrid profile is `latency`:

```sql
SHOW turbohybrid.profile;
```

## 4. Choose Index Shape

For dense-only vector search, the table and index only need the embedding key:

```sql
CREATE TABLE dense_documents (
    id bigserial PRIMARY KEY,
    embedding vector(3) NOT NULL,
    body text NOT NULL
);

CREATE INDEX dense_documents_turbohybrid_idx ON dense_documents
USING turbohybrid (embedding vector_cosine_turbohybrid_ops);

SELECT id, body
FROM dense_documents
ORDER BY embedding <~> turbohybrid_query(vector_query => $1::vector)
LIMIT 10;
```

For hybrid vector+text search, add a generated `tsvector` column and include it
as the second index key:

```sql
CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    embedding vector(3) NOT NULL,
    body text NOT NULL,
    body_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', body)
    ) STORED
);
```

`text_query` requires the second `tsvector` key. Dense-only indexes support
vector queries only.

## 5. Insert Tiny Example Data

```sql
INSERT INTO documents (embedding, body)
VALUES
    ('[1,0,0]', 'postgres vector search'),
    ('[1,1,0]', 'hybrid search with bm25'),
    ('[0,1,0]', 'lexical search in postgres'),
    ('[0,0,1]', 'unrelated document');

ANALYZE documents;
```

## 6. Create Default Hybrid TurboHybrid Index

Create the index without manual tuning options. The default fast path uses the
`latency` profile, 4-bit quantization, exact storage off, adaptive dense
widening off, and SQL `LIMIT` as the final result target when possible.

```sql
CREATE INDEX documents_turbohybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);

ANALYZE documents;
```

Existing two-key indexes remain valid. Dense-only users can create smaller
one-key indexes. Changing between dense-only and hybrid index shapes requires
rebuilding the index with the desired key list.

## 7. Query

Omit `dense_k`, `bm25_k`, and `final_k` for the normal fast path:

```sql
SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres hybrid search')
)
LIMIT 10;
```

Use the vector dimension that matches your table in real code, for example
`$1::vector(1536)` for OpenAI `text-embedding-3-small` embeddings.

## 8. Verify Fast Path

First inspect the query plan:

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

Then inspect TurboHybrid diagnostics:

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

If those values line up, you are on the normal fast path: PostgreSQL used the
TurboHybrid index, the latency profile is active, and the SQL `LIMIT` drove the
final top-k result size.

## 9. Common Fixes

| Problem | Check | Fix |
| --- | --- | --- |
| Plan does not use the index | `EXPLAIN (ANALYZE, BUFFERS)` and `scan_orchestration` | Ensure the `turbohybrid` index exists and the query uses `ORDER BY embedding <~> turbohybrid_query(...) LIMIT n`. |
| Planner has stale table stats | Row estimates in `EXPLAIN` | Run `ANALYZE documents;` after loading data and after large changes. |
| `final_k_source` is `default` | Query has no SQL `LIMIT` | Add `LIMIT 10` or another explicit top-k limit. |
| `profile` is not `latency` | `SHOW turbohybrid.profile;` | Run `SET turbohybrid.profile = 'latency';` or reset role/database settings. |
| SIMD is unavailable or disabled | `SELECT turbohybrid_simd_capabilities();` | SIMD means single instruction, multiple data; build with the portable default and avoid `SIMD_BUILD=none`; check `runtime_*`, `enabled_*`, and `simd_force` fields. |
| Relevance drops on your dataset | Compare against a SQL RRF (reciprocal-rank fusion) baseline or labeled relevance data | Use quality mode, try `SET turbohybrid.dense_heap_rescore = 'topk'`, rebuild with `WITH (exact_storage = on)`, or pass larger `dense_k` and `bm25_k`. |

## 10. Quality Mode

Use quality mode when relevance matters more than the lowest latency:

```sql
SET turbohybrid.profile = 'quality';
```

For quality-sensitive evaluation, also benchmark an exact-storage index:

```sql
CREATE INDEX documents_turbohybrid_quality_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (exact_storage = on);
```

For exact-free 4-bit indexes, you can also benchmark heap-backed exact rescore:

```sql
SET turbohybrid.dense_heap_rescore = 'topk'; -- or 'band'
```

This fetches candidate heap tuples and computes exact vector distance at query
time, so it is off by default in the latency profile.

Quality mode does more work per query. Measure latency and relevance on your
own data before choosing production settings.
