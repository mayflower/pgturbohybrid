# Performance Recovery Summary

## Final FIQA/OpenAI Validation

This is the final local validation result for the fast-default work. The
dataset is FIQA with OpenAI `text-embedding-3-small` embeddings: 57,638 corpus
rows, 648 test queries, and 1,536 dimensions. Timings are measured inside
PostgreSQL around each retrieval query after one warmup pass.

| Method | Profile/settings | p50 | p95 | p99 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| pgturbohybrid default | latency, default index, omitted budgets, LIMIT-inferred final_k | 0.729 ms | 0.910 ms | 1.096 ms | 0.421540 |
| pgturbohybrid explicit recovered | latency, 4-bit, exact_storage=off, 100/100/60, final_k=10 | 0.793 ms | 1.030 ms | 1.361 ms | 0.421540 |
| SQL RRF | pgvector HNSW + Postgres FTS, 100/100 | 1.635 ms | 3.254 ms | 5.579 ms | 0.423341 |
| pgvector HNSW dense-only | baseline | 1.129 ms | 2.718 ms | 4.052 ms | 0.439863 |

The dense-only baseline has a higher nDCG@10 on this FIQA run, but it is not a
hybrid retrieval method and does not evaluate the lexical branch. The primary
hybrid comparison is pgturbohybrid explicit recovered settings versus SQL RRF,
where nDCG@10 is effectively unchanged and p95 is lower.

The default omitted-budget path is structurally correct and index-backed. It
reported `dense_k_effective = 100`, `bm25_k_effective = 100`,
`final_k_source = limit`, `final_k_inferred_count = 648`, and
`scan_orchestration = graph_native`. After canonicalizing latency-profile
defaults before execution, the focused rerun measured `0.729 ms` p50,
`0.910 ms` p95, and `1.096 ms` p99 with the same nDCG@10, MRR@10, and recall@10
as the explicit recovered row.

The original recovered target remains:

| Method | Profile/settings | p50 | p95 | p99 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| pgturbohybrid recovered target | latency, 4-bit, exact_storage=off, 100/100, final_k=10 | 0.770 ms | 1.006 ms | 1.265 ms | 0.421540 |
| SQL RRF recovered target | pgvector HNSW + Postgres FTS | 1.978 ms | 3.988 ms | 7.493 ms | 0.421322 |
| pgvector HNSW dense-only recovered target | baseline | 1.734 ms | 3.361 ms | 5.640 ms | 0.441554 |

## Quality Profile Matrix

Quality profile is the explicit safer comparison point for users who can trade
latency for relevance. It must be measured as its own benchmark row because it
changes candidate budgets and BM25 safety policy.

| Method | Profile/settings | p50 | p95 | p99 | nDCG@10 | Summary |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| pgturbohybrid latency default | latency, default index, omitted budgets, LIMIT-inferred final_k | 0.729 ms | 0.910 ms | 1.096 ms | 0.421540 | easy default path |
| pgturbohybrid latency explicit | latency, 4-bit, exact_storage=off, 100/100/60, final_k=10 | 0.793 ms | 1.030 ms | 1.361 ms | 0.421540 | recovered explicit path |
| pgturbohybrid quality | quality, exact-safe BM25, 400/400, final_k=10; evaluate exact_storage=on | full run required | full run required | full run required | full run required | expected to cost more latency for safer relevance |
| SQL RRF | pgvector HNSW + Postgres FTS | 1.635 ms | 3.254 ms | 5.579 ms | 0.423341 | baseline hybrid retrieval |

The quality-profile result summary must state whether relevance improves,
matches, or regresses versus the latency profile, and report the exact p95/p99
cost of the safer settings. A generated quality-profile JSON artifact is not
committed yet.

Use this matrix command shape for the full FIQA/OpenAI run:

```sh
FIQA_DATASET=/Volumes/CrucialMusic/src/pgvector/.cache/beir/fiqa \
PGDATABASE=pgturbohybrid_fiqa_profile_matrix \
MAX_DOCS=0 \
MAX_QUERIES=0 \
FINAL_K=10 \
WARMUP=1 \
METHODS=postgres_sql_rrf,pgturbohybrid_recovered_explicit,pgturbohybrid_quality \
OUTPUT=/tmp/fiqa_full_profile_matrix.json \
benchmarks/dev/run_fiqa_quick.sh
```

For the quality row, the runner must set `turbohybrid.profile = 'quality'`,
use effective `dense_k = 400` and `bm25_k = 400`, keep SIMD enabled, keep
approximate hybrid bound disabled by using the `safe` bound mode, and record
whether the index was built with `exact_storage = on` or `off`.

Validate the generated profile matrix artifact with:

```sh
python3 benchmarks/tools/check_acceptance.py \
  /tmp/fiqa_full_profile_matrix.json \
  --suite fiqa_openai_profile_matrix
```

## Fast Default Quality Guard

The latency profile is accepted only when the full FIQA/OpenAI benchmark keeps
quality close to the SQL RRF baseline. This is a manual or nightly benchmark
gate, not a per-PR smoke check.

Thresholds live in
`benchmarks/config/acceptance_thresholds.json` under
`fiqa_openai_fast_defaults`:

- `nDCG@10` must be no more than `0.005` below SQL RRF.
- `MRR@10` must be no more than `0.02` below SQL RRF.
- `Recall@10` must be no more than `0.03` below SQL RRF, or `overlap@10`
  versus SQL RRF must be at least `0.70`.
- p95 latency must be at least `2x` faster than SQL RRF in the full benchmark.

Run the acceptance check against the generated full benchmark artifact:

```sh
python3 benchmarks/tools/check_acceptance.py \
  /tmp/pgturbohybrid-fiqa-canonicalized-with-baseline.json \
  --suite fiqa_openai_fast_defaults
```

The artifact must include both `pgturbohybrid_recovered_explicit` and
`postgres_sql_rrf` results with `nDCG@10`, `MRR@10`, `p95_ms`, and either
`Recall@10` for both methods or `overlap@10` for pgturbohybrid versus SQL RRF.
Generated JSON artifacts remain outside the repository.

If the quality guard fails, prefer safer settings before treating the fast
default as acceptable for that dataset:

```sql
SET turbohybrid.profile = 'quality';
```

or build with exact storage:

```sql
CREATE INDEX ... WITH (exact_storage = on);
```

or pass larger retrieval budgets:

```sql
SELECT ...
ORDER BY embedding <~> turbohybrid_query(
    vector_query => $1::vector(1536),
    text_query => plainto_tsquery('english', $2),
    dense_k => 400,
    bm25_k => 400
)
LIMIT 10;
```

The command used for the final validation artifact was:

```sh
dropdb --if-exists pgturbohybrid_fiqa_final
createdb pgturbohybrid_fiqa_final
python3 benchmarks/fiqa_openai.py \
  --database pgturbohybrid_fiqa_final \
  --dataset /Volumes/CrucialMusic/src/pgvector/.cache/beir/fiqa \
  --profile latency \
  --dense-k 100 \
  --bm25-k 100 \
  --final-k 10 \
  --warmup 1 \
  --reuse-data \
  --methods postgres_sql_rrf,pgturbohybrid,pgturbohybrid_recovered_explicit \
  --explain \
  --bm25-cache-probe \
  --output /tmp/pgturbohybrid-fiqa-canonicalized-with-baseline.json
```

`--bm25-cache-probe` records extra cold/warm BM25 diagnostics using the public
scan diagnostics.

Equivalent runner arguments:

```sh
python3 benchmarks/fiqa_openai.py \
  --database pgturbohybrid_fiqa_final \
  --dataset /Volumes/CrucialMusic/src/pgvector/.cache/beir/fiqa \
  --profile latency \
  --dense-k 100 \
  --bm25-k 100 \
  --final-k 10 \
  --warmup 1 \
  --reuse-data \
  --methods postgres_sql_rrf,pgturbohybrid,pgturbohybrid_recovered_explicit \
  --bm25-cache-probe \
  --explain \
  --output /tmp/pgturbohybrid-fiqa-canonicalized-with-baseline.json
```

The default pgturbohybrid index and query shape were:

```sql
SET turbohybrid.profile = 'latency';

CREATE INDEX fiqa_turbohybrid_idx ON fiqa_docs
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);

SELECT doc_id
FROM fiqa_docs
ORDER BY embedding <~> turbohybrid_query(
    vector_query => $1::vector(1536),
    text_query => plainto_tsquery('english', $2)
)
LIMIT 10;
```

The explicit recovered pgturbohybrid query shape was:

```sql
CREATE INDEX fiqa_turbohybrid_idx ON fiqa_docs
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = off);

SELECT doc_id
FROM fiqa_docs
ORDER BY embedding <~> turbohybrid_query(
    vector_query => $1::vector(1536),
    text_query => plainto_tsquery('english', $2),
    dense_k => 100,
    bm25_k => 100,
    rrf_k => 60,
    final_k => 10
)
LIMIT 10;
```

The SQL RRF baseline used the same dense and lexical budgets:

```sql
WITH dense AS MATERIALIZED (
    SELECT doc_id, row_number() OVER () AS rank
    FROM (
        SELECT doc_id
        FROM fiqa_docs
        ORDER BY embedding <=> $1::vector(1536)
        LIMIT 100
    ) s
),
lexical AS MATERIALIZED (
    SELECT doc_id, row_number() OVER () AS rank
    FROM (
        SELECT doc_id
        FROM fiqa_docs
        WHERE body_tsv @@ plainto_tsquery('english', $2)
        ORDER BY ts_rank_cd(body_tsv, plainto_tsquery('english', $2)) DESC, doc_id
        LIMIT 100
    ) s
)
SELECT COALESCE(dense.doc_id, lexical.doc_id) AS doc_id
FROM dense
FULL OUTER JOIN lexical USING (doc_id)
ORDER BY COALESCE(1.0 / (60 + dense.rank), 0.0) +
         COALESCE(1.0 / (60 + lexical.rank), 0.0) DESC,
         COALESCE(dense.doc_id, lexical.doc_id)
LIMIT 10;
```

Run metadata:

- CPU: Apple M4
- OS: macOS 26.5 arm64, Darwin 25.5.0
- PostgreSQL: 16.13 (Homebrew)
- pgvector: 0.8.2
- pgturbohybrid commit: `edf1b5eb4f0a7ed3320d786596ff692a9881186d`
- compiler: Apple clang 21.0.0
- compiler flags: PostgreSQL PGXS defaults from `pg_config --cflags`
  (`-Wall ... -O2`)
- SIMD: portable build, ARM dotprod and i8mm compiled, SIMD enabled
- result artifact: `/tmp/pgturbohybrid-fiqa-canonicalized-with-baseline.json`

Generated full JSON/Markdown benchmark artifacts are not committed. Attach them
to releases or CI runs, and keep only reproducible commands and methodology in
the repository.

## Validation Overhead

Before this pass, the shared pgvector compatibility helpers used one strict
path for both SQL boundary calls and internal index execution. Internal calls to
`PgturbohybridVectorDims`, `PgturbohybridCheckSameDims`, and query payload
access could repeat checks that are only needed when accepting external SQL
input:

- vector extension/type lookup through PostgreSQL catalogs
- full vector header and payload validation
- finite-value scans across all dimensions
- strict `turbohybrid_query` payload validation during index scan setup

That is correct at SQL and ingestion boundaries, but it is too expensive for
the latency profile if it is repeated while scanning dense candidates.

After this pass, validation is split into:

- strict SQL-boundary validation for `turbohybrid_query(...)`, SQL-callable
  distance/support functions, and index/insert ingestion paths
- fast internal validation for already-constructed query payloads and candidate
  scoring paths

The fast path checks cheap structural invariants only: non-null pointer,
varlena size, dimensions, reserved header field, and expected payload size. It
does not look up extension metadata, scan dimensions for finite values, or
revalidate the full query object per candidate.

The vector type OID lookup is cached per backend. Diagnostics now expose:

- `strict_vector_validations`
- `fast_vector_checks`
- `vector_type_cache_hits`
- `vector_type_cache_misses`

Regression coverage includes a guard that runs an indexed dense scan returning
100 candidates and fails if strict vector validations grow with the candidate
count. This protects the recovered fast default from being lost to compatibility
validation overhead.

The guard exposed two hot-path leaks during this pass:

- before the split, the 100-result indexed scan added about 304 strict vector
  validations because scan normalization and resjunk ORDER BY evaluation used
  SQL-callable wrappers
- after internal fast normalization and fast index-orderby-only distance
  evaluation, the same guard still added about 101 strict validations because
  the `turbohybrid_query(...)` constructor was re-run for a constant expression
- after caching all-constant constructor results per expression, the guard adds
  2 strict vector validations and 605 fast vector checks for the same indexed
  scan (`graph_native`, 148 graph candidates)
