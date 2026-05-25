<p align="center">
  <img src="logo.png" alt="pgturbohybrid" width="720">
</p>

# pgturbohybrid

`pgturbohybrid` is an experimental PostgreSQL extension for dense vector and
BM25 hybrid retrieval on top of pgvector's `vector` type. It installs as its
own extension, builds its own shared library, and requires an already installed
unmodified `vector` extension.

## Status

This project is alpha software. The SQL API, on-disk index format, and
performance profile may change before a stable release. Use it for evaluation
and controlled experiments until the compatibility and recovery matrix is
green on your target PostgreSQL and pgvector versions.

## Requirements

- PostgreSQL 14 through 18 are tested in CI. PostgreSQL 19 is included when the
  setup action provides it.
- pgvector 0.8.2 and pgvector `master` are tested in CI.
- PGXS build tooling for the target PostgreSQL installation.
- A C compiler supported by PostgreSQL. Portable SIMD is the default; the build
  does not use `-march=native` unless requested.

`pgturbohybrid` uses pgvector's SQL `vector` type but does not link against
private pgvector C symbols.

## Why This Exists

`pgturbohybrid` is a companion extension for pgvector. It is not a fork of
pgvector and is not an official pgvector project. The goal is to experiment with
hybrid dense-vector and BM25 retrieval while keeping pgvector itself
unmodified.

## Fast defaults

`pgturbohybrid` is fast by default. Fresh sessions use the `latency` profile:

- default candidate budgets are 100 dense candidates and 100 BM25 candidates
- the default RRF constant is 60
- default indexes use 4-bit quantization
- exact vector storage in the index is off by default
- `ORDER BY ... LIMIT n` is used as the final result target when possible

The normal query shape is therefore to omit `dense_k`, `bm25_k`, and `final_k`
and let the SQL `LIMIT` drive top-k retrieval.

For a one-page walkthrough from installation to diagnostics, see
[Easy fast setup](docs/fast_setup.md).

## Installation

Install pgvector first:

```sh
git clone --depth 1 --branch v0.8.2 https://github.com/pgvector/pgvector.git /tmp/pgvector
make -C /tmp/pgvector
make -C /tmp/pgvector install
```

Then build and install `pgturbohybrid`:

```sh
git clone https://github.com/mayflower/pgturbohybrid.git
cd pgturbohybrid
make
make install
```

Create both extensions in the database:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
```

For a repeatable local setup:

```sh
PG_CONFIG=pg_config PGVECTOR_REF=v0.8.2 scripts/dev-install.sh
```

## Minimal Example

```sql
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

CREATE INDEX documents_pgturbohybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);

SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres hybrid search')
)
LIMIT 10;
```

Use `<~->` for L2, `<~#>` for negative inner product, and `<~>` for cosine
hybrid ordering.

For a copy-paste runnable version with diagnostics, see
[examples/fast_start.sql](examples/fast_start.sql) and
[examples/fast_start.md](examples/fast_start.md). For the full setup flow, see
[Easy fast setup](docs/fast_setup.md).

## Index Options

The public index options are:

- `graph_m`: maximum graph connections. Default: `16`.
- `graph_ef_construction`: graph candidate list size during build. Default: `128`.
- `graph_ef_search`: graph candidate list size during scans. Default: `64`.
- `graph_oversampling`: graph candidate oversampling multiplier. Default: `4`.
- `quantization_bits`: quantized dense-vector code width. Default: `4`.
- `exact_storage`: store exact vectors in the index for final exact rescoring.
  Default: `off`.
- `routing`: dense routing mode. Default: `auto`, which selects the native
  graph path for supported TurboHybrid opclasses.

## Query Parameters

`turbohybrid_query(...)` accepts:

- `vector_query`: dense pgvector query vector.
- `text_query`: PostgreSQL `tsquery` for BM25 retrieval.
- `fusion`: currently `rrf` for reciprocal-rank fusion.
- `dense_weight` and `bm25_weight`: branch weights for score fusion.
- `rrf_k`: RRF constant.
- `dense_k`: dense branch candidate budget.
- `bm25_k`: BM25 branch candidate budget.
- `final_k`: internal final fusion target. Normally omit this and let
  `ORDER BY ... LIMIT n` provide the final result target. Pass it only when
  you want a different internal fusion target than the SQL LIMIT.
- `require_bm25_match`: only return rows with a BM25 lexical match.

## Settings

The public GUCs are intentionally small:

- `turbohybrid.profile`: retrieval profile (`latency`, `balanced`, `quality`, or `debug`). The default is `latency`.
- `turbohybrid.default_dense_k`
- `turbohybrid.default_bm25_k`
- `turbohybrid.default_rrf_k`
- `turbohybrid.enable_wand`
- `turbohybrid.bm25_strategy`
- `turbohybrid.bm25_impact_or_mode`
- `turbohybrid.bm25_hot_postings_cache_mb`
- `turbohybrid.bm25_hot_postings_cache_min_df`
- `turbohybrid.bm25_hybrid_bound`
- `turbohybrid.bm25_accumulator_mode`
- `turbohybrid.max_union_candidates`
- `turbohybrid.simd`

The `latency` profile defaults to 4-bit exact-free indexes, dense and BM25
candidate budgets of 100, RRF constant 60, WAND enabled, safe BM25 impact fast
paths plus documented approximate latency paths enabled, hot postings cache
enabled, and SIMD enabled when available.
Explicit per-query arguments and explicitly set GUCs override profile defaults.

The `quality` profile is the explicit alternative when relevance matters more
than the lowest latency. Its effective defaults are:

| Setting | Latency profile | Quality profile |
| --- | --- | --- |
| `dense_k` | `100` | `400` |
| `bm25_k` | `100` | `400` |
| `rrf_k` | `60` | `60` |
| `turbohybrid.bm25_impact_or_mode` | `approx` | `exact_only` |
| `turbohybrid.bm25_hybrid_bound` | `approx` | `safe` |
| BM25 auto budget reduction | enabled | disabled |
| BM25-only exact rescore preference | off | on where available |
| SIMD | enabled | enabled |

## Choosing a profile

Use latency mode for the fast default path:

```sql
SET turbohybrid.profile = 'latency';
```

Use balanced mode when you want a more conservative middle ground:

```sql
SET turbohybrid.profile = 'balanced';
```

### When to use quality profile

Use quality mode when:

- exact recall matters more than latency
- dataset-specific relevance drops with the latency profile
- you are evaluating final production settings

```sql
SET turbohybrid.profile = 'quality';
```

Quality mode uses larger dense and lexical candidate budgets and exact-safe BM25
paths. Expect higher p95/p99 latency than the latency profile. For final
production evaluation, also benchmark an index with exact storage enabled:

```sql
CREATE INDEX documents_turbohybrid_quality_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (exact_storage = on);
```

Explicit per-query `dense_k`, `bm25_k`, `rrf_k`, and `final_k` values still
override profile defaults.

## Advanced Tuning

The `latency` profile enables the fast BM25 path automatically:

```sql
SHOW turbohybrid.bm25_strategy;                 -- auto
SHOW turbohybrid.bm25_impact_or_mode;           -- approx
SHOW turbohybrid.bm25_hot_postings_cache_mb;    -- 32
SHOW turbohybrid.bm25_hot_postings_cache_min_df; -- 1024
SHOW turbohybrid.bm25_hybrid_bound;             -- approx
SHOW turbohybrid.bm25_accumulator_mode;         -- auto
```

Use these only when reproducing a benchmark or disabling a fast path:

```sql
SET turbohybrid.bm25_strategy = 'wand';
SET turbohybrid.bm25_impact_or_mode = 'exact_only';
SET turbohybrid.bm25_hybrid_bound = 'safe';
SET turbohybrid.bm25_hot_postings_cache_mb = 0;
```

`balanced` and `quality` keep approximate BM25 shortcuts off by default.
Explicit `SET` values survive profile changes until reset.

## Diagnostics

The installed diagnostic API is limited to:

- `turbohybrid_index_stats(regclass)`
- `turbohybrid_last_scan_stats()`
- `turbohybrid_simd_capabilities()`

These return stable JSONB summaries for index metadata, the last scan in the
current backend, and build/host SIMD capability information.
`turbohybrid_last_scan_stats()` includes `final_k_requested`,
`final_k_effective`, `detected_sql_limit`, and `final_k_source` to show how the
final result target was selected. It also reports whether an index-backed scan
was observed, the scan orchestration mode, quantization bits, exact-storage
state when known, effective candidate budgets, phase timings, the effective
BM25 strategy, impact OR mode, hot postings cache size, hybrid bound mode, and
accumulator mode.

After a query, check whether it used the expected fast path:

```sql
SELECT turbohybrid_last_scan_stats();
SELECT turbohybrid_index_stats('documents_turbohybrid_idx'::regclass);
```

Example scan output, shortened to the most useful fields:

```json
{
  "version": 1,
  "profile": "latency",
  "index_used": true,
  "scan_orchestration": "graph_native",
  "quantization_bits": 4,
  "exact_storage": false,
  "dense_candidates_effective": 100,
  "bm25_candidates_effective": 100,
  "final_k_effective": 10,
  "final_k_source": "limit",
  "dense_elapsed_us": 420,
  "bm25_elapsed_us": 360,
  "fusion_elapsed_us": 40,
  "elapsed_us": 820
}
```

Troubleshooting fast-path stats:

| Symptom | What to check | Common fix |
| --- | --- | --- |
| `profile` is not `latency` | A session or role changed `turbohybrid.profile`. | `SET turbohybrid.profile = 'latency';` or reset the role/database setting. |
| `exact_storage` is `true` | The index stores exact vectors for safer rescoring. | Rebuild the latency index without `WITH (exact_storage = on)`. |
| `dense_candidates_effective` is too high | Explicit `dense_k`, quality profile, or planner LIMIT inference changed the budget. | Omit `dense_k` for defaults or reset `turbohybrid.default_dense_k`. |
| `scan_orchestration` is `none` | No pgturbohybrid graph path was observed for the last scan. | Check `EXPLAIN`, operator class, `ORDER BY ... LIMIT`, and whether index scans are enabled. |
| `elapsed_us` is dominated by `bm25_elapsed_us` | Lexical branch or postings/cache work is the bottleneck. | Check BM25 strategy, hot postings cache, term selectivity, and consider larger warmup. |
| `elapsed_us` is dominated by `dense_elapsed_us` | Dense graph traversal or rescoring is the bottleneck. | Check quantization bits, exact storage, `graph_ef_search`, result LIMIT, and profile. |

## WAL And Recovery

`pgturbohybrid` uses PostgreSQL generic WAL for index page changes. It does not
register a custom resource manager and does not require
`shared_preload_libraries = 'pgturbohybrid'`.

## Benchmarks

Benchmark methodology and reproducibility requirements live in
[benchmarks/README.md](benchmarks/README.md). Performance claims should use
real embedding datasets with relevance labels, not synthetic vector generators.
Generated result JSON and Markdown files should be written outside the
repository or under ignored benchmark output directories.

FIQA/OpenAI fast-default validation snapshot:

- dataset: FIQA, 57,638 corpus rows and 648 test queries
- embeddings: OpenAI `text-embedding-3-small`, 1,536 dimensions
- measurement: one warmup pass, one measured pass, 648 measured queries
- profile: `turbohybrid.profile = 'latency'`

| Method | Profile/settings | p50 | p95 | p99 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| pgturbohybrid default | default index, omitted budgets, LIMIT-inferred final_k | 0.729 ms | 0.910 ms | 1.096 ms | 0.421540 |
| pgturbohybrid explicit recovered | 4-bit, exact_storage=off, 100/100/60, final_k=10 | 0.793 ms | 1.030 ms | 1.361 ms | 0.421540 |
| SQL RRF | pgvector HNSW + Postgres FTS, 100/100 | 1.635 ms | 3.254 ms | 5.579 ms | 0.423341 |
| pgvector HNSW dense-only | baseline | 1.129 ms | 2.718 ms | 4.052 ms | 0.439863 |

The dense-only baseline is included as a speed/quality reference, but it is not
a hybrid retrieval method. The direct hybrid comparison is pgturbohybrid versus
SQL RRF with the same dense and lexical candidate budgets.

The default omitted-budget path is structurally correct and index-backed. It
reports latency profile, 4-bit quantization, `exact_storage = false`,
100/100 effective candidates, and LIMIT-inferred `final_k`. The executor
canonicalizes these latency defaults before retrieval, so the easy default path
uses the same hot-path shape as the explicit recovered settings while retaining
diagnostics that show where values came from.

The quality profile must be published as a separate matrix row, not inferred
from the latency run:

| Method | Required settings | p50 | p95 | p99 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| pgturbohybrid latency default | `profile=latency`, default index, omitted budgets, LIMIT-inferred final_k | 0.729 ms | 0.910 ms | 1.096 ms | 0.421540 |
| pgturbohybrid latency explicit | `profile=latency`, 4-bit, `exact_storage=off`, 100/100/60, `final_k=10` | 0.793 ms | 1.030 ms | 1.361 ms | 0.421540 |
| pgturbohybrid quality | `profile=quality`, exact-safe BM25, 400/400, `final_k=10`; evaluate `exact_storage=on` | full run required | full run required | full run required | full run required |
| SQL RRF | pgvector HNSW + Postgres FTS | 1.635 ms | 3.254 ms | 5.579 ms | 0.423341 |

Result summaries should report whether quality profile improves or preserves
relevance versus latency profile, and the exact p95/p99 cost paid for that
improvement. No generated quality-profile JSON artifact is committed.

Reproduction command:

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

Equivalent lower-level command:

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

Default pgturbohybrid SQL shape:

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

Explicit recovered pgturbohybrid SQL shape:

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

Run metadata:

- CPU: Apple M4
- OS: macOS 26.5 arm64, Darwin 25.5.0
- PostgreSQL: 16.13 (Homebrew)
- pgvector: 0.8.2
- pgturbohybrid commit: `edf1b5eb4f0a7ed3320d786596ff692a9881186d`
- compiler: Apple clang 21.0.0 with PostgreSQL PGXS `pg_config --cflags` defaults
  (`-Wall ... -O2`)
- SIMD: portable build, ARM dotprod and i8mm compiled, SIMD enabled

Benchmark claims must include the retrieval profile and exact index/query
settings. The generated full JSON artifact is not committed; attach generated
artifacts to releases or CI runs instead of storing host-specific outputs in
the repository.

## Compatibility Notes

`pgturbohybrid` requires pgvector at build and install time. It refers to
pgvector's `vector` type through the extension dependency search path, so
pgvector can be installed outside `public` on supported PostgreSQL versions.

The compatibility layer validates that the `vector` extension is installed and
that vector dimensions are valid before accessing vector payloads. If pgvector's
type layout changes, the build or runtime checks should fail with a direct
compatibility error rather than silently reading malformed data.

See [COMPATIBILITY.md](COMPATIBILITY.md) for the tested PostgreSQL and pgvector
matrix.

## Known Limitations

- Alpha status: SQL APIs and the on-disk index format may change before a
  stable release.
- On-disk format: incompatible storage changes require `REINDEX`; see
  [RELEASE.md](RELEASE.md) for the current policy.
- pgvector ABI: the extension depends on pgvector's SQL `vector` type and uses
  a private compatibility copy of its varlena layout. Tested compatibility
  starts at pgvector 0.8.2.
- Dimensions: the compatibility layer supports pgvector vector payloads up to
  16000 dimensions; index build paths validate their supported limits before
  writing index pages.
- Text-aware scalar fallback: hybrid queries with a `text_query` are intended
  for indexed `ORDER BY ... LIMIT` retrieval. Scalar projection without an
  index path fails with a `feature_not_supported` error.
- Production support: treat this as evaluation software until the release
  checklist is green for your target platform and PostgreSQL version.

## Attribution

This project depends on pgvector and contains code derived from pgvector's HNSW
implementation. pgvector is an excellent PostgreSQL vector search extension;
`pgturbohybrid` is a separate experimental extension built on top of it and is
not an official pgvector project.

The license is preserved in [LICENSE](LICENSE), with attribution notes in
[NOTICE](NOTICE).
