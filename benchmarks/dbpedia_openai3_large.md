# DBPedia OpenAI3 Large Benchmark

This file helps you implement and run 1M-row DBPedia retrieval benchmarks for
`pgturbohybrid`, including both hybrid retrieval and a dense-only default
comparison against pgvector.

The benchmark uses Qdrant's DBPedia entity corpus with OpenAI
`text-embedding-3-large` embeddings and compares PostgreSQL-native hybrid
retrieval against TurboHybrid. It is a developer benchmark spec, not a public
performance claim by itself.

## Goal

Measure hybrid dense-vector plus lexical retrieval on a large, real entity
corpus without generating corpus embeddings:

- corpus: `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M`
- dense model: OpenAI `text-embedding-3-large`
- dimensions: 3,072
- corpus size: 1,000,000 rows
- default query source: deterministic self-queries sampled from Qdrant rows
- primary result size: `final_k = 10`

The direct hybrid comparison is:

- native PostgreSQL hybrid retrieval: pgvector approximate nearest-neighbor
  search plus PostgreSQL full-text search, fused with SQL reciprocal-rank
  fusion, or RRF
- `pgturbohybrid`: one `turbohybrid` index that retrieves and fuses dense and
  BM25/text candidates inside the index access method

BM25 means Best Matching 25, a text-ranking method for exact term matching.
FTS means PostgreSQL full-text search. ANN means approximate nearest-neighbor
search. HNSW means Hierarchical Navigable Small World, pgvector's graph index
type for ANN search.

The dense-only comparison is separate. It compares pgvector HNSW dense retrieval
against TurboHybrid with only `vector_query` supplied to `turbohybrid_query(...)`.
No text query is passed, so PostgreSQL full-text search, BM25, and SQL RRF are
not part of that run.

## Why the pgvector baseline uses halfvec

The Qdrant corpus uses 3,072-dimensional embeddings. Standard pgvector HNSW
over `vector` is not the right ANN baseline at this dimensionality. Use
pgvector `halfvec(3072)` HNSW for the native dense branch, then combine that
with PostgreSQL GIN full-text search and SQL RRF.

This makes the native PostgreSQL baseline:

```sql
CREATE INDEX dbpedia_hnsw_halfvec_idx ON dbpedia_docs
USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops);

CREATE INDEX dbpedia_fts_idx ON dbpedia_docs USING gin (body_tsv);
```

The TurboHybrid benchmark keeps the stored column as pgvector `vector(3072)`:

```sql
CREATE INDEX dbpedia_turbohybrid_idx ON dbpedia_docs
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);
```

## Data

Download the Qdrant corpus outside the repository or into an ignored data
directory:

```sh
export DBPEDIA_DATASET=/path/to/qdrant-dbpedia-openai3-large-1m

hf download Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M \
  --repo-type dataset \
  --local-dir "$DBPEDIA_DATASET"
```

The Qdrant dataset stores the corpus in Parquet shards with these fields:

- `_id`
- `title`
- `text`
- `text-embedding-3-large-3072-embedding`

By default, the harness uses `--query-source qdrant-self`. It samples rows from
the loaded Qdrant corpus, uses each row's existing embedding as the query
embedding, uses the title or a short body excerpt as the text query, and marks
the same row as the positive qrel. This is a systems benchmark for large
hybrid retrieval over the Qdrant embeddings. It is not a broad relevance
benchmark.

## Optional BEIR queries

The harness can also run BEIR DBPedia Entity queries and qrels with
`--query-source beir`. Use this only when you explicitly want BEIR relevance
labels and have generated query embeddings with the same OpenAI model.

```sh
export BEIR_DBPEDIA_DATASET=/path/to/beir-dbpedia-entity
export BEIR_DBPEDIA_QRELS=/path/to/beir-dbpedia-entity-qrels/test.tsv
export DBPEDIA_QUERY_EMBEDDINGS=/path/to/dbpedia-query-embeddings.jsonl
export OPENAI_API_KEY=...

hf download BeIR/dbpedia-entity \
  --repo-type dataset \
  --local-dir "$BEIR_DBPEDIA_DATASET"

hf download BeIR/dbpedia-entity-qrels \
  --repo-type dataset \
  --local-dir "$(dirname "$BEIR_DBPEDIA_QRELS")"

python3 benchmarks/dbpedia_openai3_large.py \
  --beir-dataset "$BEIR_DBPEDIA_DATASET" \
  --query-embeddings "$DBPEDIA_QUERY_EMBEDDINGS" \
  --prepare-query-embeddings \
  --prepare-only
```

The cache is JSONL with one row per query:

```json
{"id":"INEX_LD-20120112","values":[0.001,0.002]}
```

The harness validates that every query embedding has 3,072 dimensions.

The BEIR corpus contains more documents than the Qdrant 1M subset. The harness
filters qrels to documents that were actually loaded into PostgreSQL and drops
queries with no positive relevance labels left after filtering. Do not report
nDCG or recall against unfiltered qrels.

## Benchmark methods

Run at least these methods:

- `postgres_sql_rrf_halfvec`
- `pgturbohybrid`
- `pgturbohybrid_quality`

`postgres_sql_rrf_halfvec` uses:

- pgvector HNSW expression index over `embedding::halfvec(3072)`
- PostgreSQL GIN index over the generated `tsvector`
- SQL-level RRF with `rrf_k = 60`

`pgturbohybrid` uses the default TurboHybrid index and the `latency` profile.
It omits explicit candidate budgets so the extension can use its current fast
defaults and infer `final_k` from SQL `LIMIT`.

`pgturbohybrid_quality` uses:

- `turbohybrid.profile = quality`
- `dense_k = 400`
- `bm25_k = 400`
- `rrf_k = 60`
- `final_k = 10`
- `exact_storage = on`

For the dense-only default comparison, run:

- `pgvector_halfvec_dense_only`
- `pgturbohybrid_dense_only`

`pgvector_halfvec_dense_only` uses only the pgvector HNSW expression index over
`embedding::halfvec(3072)`.

`pgturbohybrid_dense_only` uses the default TurboHybrid index definition, but
the query supplies only `vector_query`:

```sql
SELECT doc_id
FROM dbpedia_docs
ORDER BY embedding <~> turbohybrid_query(
    vector_query => $1::vector(3072)
)
LIMIT 10;
```

The current TurboHybrid access method still expects a vector column plus a
`tsvector` column in the index definition. That does not make this a hybrid
retrieval run: without `text_query`, the benchmark is measuring the dense
TurboHybrid path.

## Primary values

Use these values for the main latency run:

```text
final_k = 10
dense_k = 100
bm25_k = 100
rrf_k = 60
warmup = 1 full pass
measured_runs = 3
hnsw.ef_search = 100
```

Use these values for the quality run:

```text
final_k = 10
dense_k = 400
bm25_k = 400
rrf_k = 60
warmup = 1 full pass
measured_runs = 3
hnsw.ef_search = 400
```

For budget sweeps, use:

```text
50/50
100/100
200/200
400/400
1000/1000
```

## Commands

Dense-only default 1M comparison:

```sh
export PGDATABASE=pgturbohybrid_dbpedia_dense_1m
export OUTPUT=benchmarks/results/dbpedia-openai3-large-dense-defaults.json

python3 benchmarks/dbpedia_openai3_large.py \
  --database "$PGDATABASE" \
  --dataset "$DBPEDIA_DATASET" \
  --methods pgvector_halfvec_dense_only,pgturbohybrid_dense_only \
  --final-k 10 \
  --warmup 1 \
  --measured-runs 3 \
  --hnsw-ef-search 0 \
  --reuse-data \
  --explain \
  --output "$OUTPUT"
```

For a quick dense-only smoke run before the full 1M pass:

```sh
export PGDATABASE=pgturbohybrid_dbpedia_dense_smoke
export OUTPUT=benchmarks/results/dbpedia-openai3-large-dense-smoke.json

python3 benchmarks/dbpedia_openai3_large.py \
  --database "$PGDATABASE" \
  --dataset "$DBPEDIA_DATASET" \
  --max-docs 10000 \
  --max-queries 25 \
  --methods pgvector_halfvec_dense_only,pgturbohybrid_dense_only \
  --final-k 10 \
  --warmup 1 \
  --measured-runs 1 \
  --hnsw-ef-search 0 \
  --force-turbohybrid-index \
  --explain \
  --output "$OUTPUT"
```

Use `--hnsw-ef-search 0` for the default pgvector runtime setting. If you also
want an equalized-depth experiment, run a separate artifact with an explicit
value such as `--hnsw-ef-search 100` and label it separately.

Smoke run:

```sh
export PGDATABASE=pgturbohybrid_dbpedia_smoke
export OUTPUT=benchmarks/results/dbpedia-openai3-large-smoke.json

python3 benchmarks/dbpedia_openai3_large.py \
  --database "$PGDATABASE" \
  --dataset "$DBPEDIA_DATASET" \
  --max-docs 10000 \
  --max-queries 25 \
  --methods postgres_sql_rrf_halfvec,pgturbohybrid \
  --dense-k 100 \
  --bm25-k 100 \
  --final-k 10 \
  --warmup 1 \
  --measured-runs 1 \
  --force-turbohybrid-index \
  --explain \
  --output "$OUTPUT"
```

Full latency run:

```sh
export PGDATABASE=pgturbohybrid_dbpedia_1m
export OUTPUT=benchmarks/results/dbpedia-openai3-large-latency.json

python3 benchmarks/dbpedia_openai3_large.py \
  --database "$PGDATABASE" \
  --dataset "$DBPEDIA_DATASET" \
  --methods postgres_sql_rrf_halfvec,pgturbohybrid \
  --dense-k 100 \
  --bm25-k 100 \
  --final-k 10 \
  --warmup 1 \
  --measured-runs 3 \
  --hnsw-ef-search 100 \
  --reuse-data \
  --explain \
  --output "$OUTPUT"
```

Full quality run:

```sh
export PGDATABASE=pgturbohybrid_dbpedia_1m
export OUTPUT=benchmarks/results/dbpedia-openai3-large-quality.json

python3 benchmarks/dbpedia_openai3_large.py \
  --database "$PGDATABASE" \
  --dataset "$DBPEDIA_DATASET" \
  --methods postgres_sql_rrf_halfvec,pgturbohybrid_quality \
  --dense-k 400 \
  --bm25-k 400 \
  --final-k 10 \
  --warmup 1 \
  --measured-runs 3 \
  --hnsw-ef-search 400 \
  --reuse-data \
  --explain \
  --output "$OUTPUT"
```

## Metrics

Every artifact must include:

- p50, p95, p99, mean latency, and QPS
- nDCG@10
- MRR@10
- MAP@10
- Recall@10
- overlap@10 versus the native SQL RRF baseline when both methods are present
- overlap@10 versus `pgvector_halfvec_dense_only` when the dense-only baseline
  is present
- loaded row count
- loaded query count
- filtered positive qrel count
- index build time
- index size
- WAL bytes generated during index build
- query plan checks when `--explain` is used
- `turbohybrid_last_scan_stats()` summary for TurboHybrid rows
- `turbohybrid_index_stats(...)` for TurboHybrid indexes

For a separate depth-quality run, set `--final-k 100` and report Recall@100.
Do not mix top-10 latency claims with top-100 latency claims.

## Artifact policy

Do not commit generated benchmark JSON, CSV, or Markdown output. Write artifacts
to `benchmarks/results/`, `benchmarks/output/`, or another ignored directory.

Any public claim must include:

- dataset names and revisions, if available
- embedding model and dimensions
- corpus rows, query count, and filtered qrel count
- PostgreSQL version and settings
- pgvector version or ref
- `pgturbohybrid` commit
- hardware, storage, and operating system
- method definitions
- index settings
- candidate budgets
- warmup and measured pass counts
- quality metrics
- latency distribution
- a note that results vary by dataset and hardware

Do not summarize this benchmark as a global comparison against pgvector or
Qdrant. The direct claim is limited to this DBPedia/OpenAI3 setup, these
settings, and the native SQL RRF halfvec baseline.

For dense-only runs, the direct claim is limited to the Qdrant DBPedia/OpenAI3
setup, 3,072-dimensional embeddings, 1M loaded rows, qdrant-self queries,
default index definitions, the stated pgvector `hnsw.ef_search` setting, and
the recorded hardware. Do not describe it as a hybrid retrieval comparison.
