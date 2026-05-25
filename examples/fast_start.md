# pgturbohybrid Fast Start

This example creates a small hybrid retrieval table, builds a default
TurboHybrid index, runs an index-backed vector-plus-text query, and prints the
diagnostics you normally need when checking that the fast path is active.

Run it after installing `pgvector` and `pgturbohybrid`:

```sh
psql -d your_database -f examples/fast_start.sql
```

The SQL file creates both extensions if needed:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
```

It then creates a table with:

- `id`
- `embedding vector(3)`
- `body text`
- `body_tsv`, a generated `tsvector` column

The index intentionally uses no manual fast options:

```sql
CREATE INDEX turbohybrid_fast_start_docs_idx
ON turbohybrid_fast_start_docs
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);
```

## Defaults

Fresh sessions use the latency profile by default:

```sql
SHOW turbohybrid.profile;
```

Under the latency profile, omitted query budgets default to:

- `dense_k = 100`
- `bm25_k = 100`
- `rrf_k = 60`

You normally do not need to pass `final_k`. When possible, pgturbohybrid infers
the final result target from the SQL `LIMIT`, so this is the preferred query
shape:

```sql
SELECT id, body
FROM turbohybrid_fast_start_docs
ORDER BY embedding <~> turbohybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres vector hybrid search')
)
LIMIT 10;
```

The example sets `enable_seqscan = off` only around the demo query because the
sample table is deliberately tiny. Normal application sessions should let the
planner choose.

## Diagnostics

After the query, inspect:

```sql
SELECT turbohybrid_last_scan_stats();
SELECT turbohybrid_index_stats('turbohybrid_fast_start_docs_idx'::regclass);
SELECT turbohybrid_simd_capabilities();
```

For an index-backed run, `turbohybrid_last_scan_stats()` should show an indexed
scan orchestration such as `graph_native`, along with the active profile and
effective final result target.

## Quality Mode

Use the default latency profile for fast RAG retrieval. Switch to quality mode
when you want larger default candidate budgets and more conservative fast-path
choices:

```sql
SET turbohybrid.profile = 'quality';
```

You can still override budgets per query:

```sql
SELECT id, body
FROM turbohybrid_fast_start_docs
ORDER BY embedding <~> turbohybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres vector hybrid search'),
    dense_k => 400,
    bm25_k => 400
)
LIMIT 10;
```

## Cleanup

The SQL file leaves the table in place so you can inspect it. Remove it with:

```sql
DROP TABLE IF EXISTS turbohybrid_fast_start_docs;
```
