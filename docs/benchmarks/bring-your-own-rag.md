# Bring Your Own RAG Benchmark

This file helps you try `pgturbohybrid` against an existing PostgreSQL RAG
database without downloading a public benchmark dataset.

The goal is simple: compare TurboHybrid with the retrieval SQL you already use,
on your own rows and your own query embeddings. The output is useful for local
evaluation, not for broad performance claims.

## What You Need

- PostgreSQL with `vector` and `pgturbohybrid` installed
- a document table with:
  - an id column
  - a pgvector `vector` embedding column
  - a `tsvector` column for lexical search
- query text and query vectors from the same embedding model as your documents
- a baseline SQL file that represents your current RAG retrieval query

The benchmark does not call an embedding API. It uses query vectors you provide.

## Query Input

The easiest input is JSONL:

```json
{"query_id":"q1","text_query":"refund policy","vector_query":"[0.1,0.2,0.3]"}
{"query_id":"q2","text_query":"reset password","vector_query":"[0.4,0.5,0.6]"}
```

`vector_query` can also be an array of numbers:

```json
{"query_id":"q1","text_query":"refund policy","vector_query":[0.1,0.2,0.3]}
```

You can also read queries from a PostgreSQL table with `--query-table`,
`--query-id-column`, `--query-text-column`, and `--query-vector-column`.

## Baseline SQL

Write the query you want to compare against TurboHybrid in a SQL file. It must
be a single `SELECT` or `WITH ... SELECT` statement and return exactly one
column named `doc_id`, castable to text, in rank order.

Dense-only example:

```sql
SELECT id AS doc_id
FROM documents
ORDER BY embedding <=> {vector_query}::vector
LIMIT {final_k}
```

Simple SQL RRF example:

```sql
WITH dense AS (
    SELECT id, row_number() OVER () AS rank
    FROM documents
    ORDER BY embedding <=> {vector_query}::vector
    LIMIT {dense_k}
), lexical AS (
    SELECT id, row_number() OVER () AS rank
    FROM documents
    WHERE body_tsv @@ websearch_to_tsquery('english', {text_query})
    ORDER BY ts_rank_cd(body_tsv, websearch_to_tsquery('english', {text_query})) DESC
    LIMIT {bm25_k}
), fused AS (
    SELECT id, 1.0 / ({rrf_k} + rank) AS score FROM dense
    UNION ALL
    SELECT id, 1.0 / ({rrf_k} + rank) AS score FROM lexical
)
SELECT id AS doc_id
FROM fused
GROUP BY id
ORDER BY sum(score) DESC
LIMIT {final_k}
```

Supported placeholders are `{vector_query}`, `{text_query}`, `{final_k}`,
`{dense_k}`, `{bm25_k}`, and `{rrf_k}`.

Baseline checklist:

- the query returns only `doc_id`
- the final `ORDER BY` is the order you want to compare
- the query uses the same filters you would use in your application
- `{vector_query}` is cast to `vector`
- `{text_query}` is passed to your text-search expression

## Distance Metric

The default example uses cosine distance:

- TurboHybrid opclass: `vector_cosine_turbohybrid_ops`
- TurboHybrid operator: `<~>`
- pgvector baseline operator: `<=>`

If your existing index uses another distance metric, pass the matching
TurboHybrid options:

| Metric | TurboHybrid opclass | TurboHybrid operator | pgvector baseline operator |
| --- | --- | --- | --- |
| cosine | `vector_cosine_turbohybrid_ops` | `<~>` | `<=>` |
| L2 | `vector_l2_turbohybrid_ops` | `<~->` | `<->` |
| inner product | `vector_ip_turbohybrid_ops` | `<~#>` | `<#>` |

## Run It

Minimal file layout:

```text
baseline.sql
queries.jsonl
```

Example with a JSONL query file:

```sh
export PGDATABASE=my_rag_database
export OUTPUT=benchmarks/results/my-rag-turbohybrid.json

python3 benchmarks/rag_existing.py \
  --database "$PGDATABASE" \
  --table documents \
  --id-column id \
  --vector-column embedding \
  --tsvector-column body_tsv \
  --queries-jsonl queries.jsonl \
  --baseline-sql baseline.sql \
  --create-turbohybrid-index \
  --index-name documents_turbohybrid_bench_idx \
  --final-k 10 \
  --warmup 1 \
  --measured-runs 3 \
  --explain \
  --output "$OUTPUT"
```

Run a preflight check before creating an index on a large table:

```sh
python3 benchmarks/rag_existing.py \
  --database "$PGDATABASE" \
  --table documents \
  --id-column id \
  --vector-column embedding \
  --tsvector-column body_tsv \
  --queries-jsonl queries.jsonl \
  --baseline-sql baseline.sql \
  --check
```

The helper sets `enable_seqscan = off` for the TurboHybrid query by default.
That keeps small evaluation samples on the index-backed path instead of falling
back to scalar operator evaluation. Use `--no-force-index` if you specifically
want to observe the planner's natural choice.

Example with a query table:

```sh
python3 benchmarks/rag_existing.py \
  --database "$PGDATABASE" \
  --table documents \
  --id-column id \
  --vector-column embedding \
  --tsvector-column body_tsv \
  --query-table rag_eval_queries \
  --query-id-column id \
  --query-text-column query_text \
  --query-vector-column embedding \
  --baseline-sql baseline.sql \
  --max-queries 100 \
  --output "$OUTPUT"
```

## What It Reports

The JSON output includes:

- p50, p95, p99, mean latency, and QPS for your baseline SQL
- the same latency summary for TurboHybrid
- overlap@k between TurboHybrid and your baseline
- TurboHybrid scan diagnostics from `turbohybrid_last_scan_stats()`
- index diagnostics from `turbohybrid_index_stats(...)`
- PostgreSQL, pgvector, and pgturbohybrid versions
- optional `EXPLAIN (ANALYZE, BUFFERS)` plans

Latency is measured as client-observed wall-clock time around each `psql`
execution. That includes a small amount of command/session overhead, so use it
as a practical local comparison signal rather than a publishable microbenchmark.
The optional `EXPLAIN` plans are better when you need server-side executor
timing.

Overlap is not relevance. It only answers how similar TurboHybrid results are
to your current retrieval path. If you have human labels or qrels, evaluate
quality separately before making production choices.

## Safety Notes

`--create-turbohybrid-index` creates an index in your database. That can use
disk, write-ahead log, CPU, and I/O. Run it first on a staging copy or a
maintenance window if the table is large.

The benchmark uses `CREATE INDEX CONCURRENTLY IF NOT EXISTS` by default. To
clean up:

```sql
DROP INDEX CONCURRENTLY IF EXISTS documents_turbohybrid_bench_idx;
```

Generated JSON output belongs in `benchmarks/results/` or another ignored
artifact directory. Do not commit private benchmark outputs.

## Interpreting Results

Use the result as a local evaluation signal:

- If TurboHybrid is faster and overlap is high, try a larger query sample.
- If latency is good but overlap is low, test `quality` profile or explicit
  larger budgets.
- If the index is not used, inspect `plan_checks` and
  `turbohybrid_last_scan_stats()`.
- If your baseline includes app-specific filters, make sure the TurboHybrid
  comparison reflects whether those filters are required before or after
  retrieval.

Results vary by schema, corpus, query mix, filters, PostgreSQL settings, CPU,
storage, and embedding model. Describe all of those when sharing numbers.
