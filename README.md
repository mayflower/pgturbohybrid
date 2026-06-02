<p align="center">
  <img src="docs/assets/logo.png" alt="pgturbohybrid logo with a compact hot hatchback theme" width="720">
</p>

# pgturbohybrid

This README helps you understand what `pgturbohybrid` does, when hybrid search
is useful, how to install it, how to create your first index, and how to check
whether the fast path is working.

> The hot hatch of pgvector hybrid search: practical PostgreSQL retrieval with
> a surprising turn of speed.

[![build](https://github.com/mayflower/pgturbohybrid/actions/workflows/build.yml/badge.svg)](https://github.com/mayflower/pgturbohybrid/actions/workflows/build.yml)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange)
![PostgreSQL 14–19](https://img.shields.io/badge/PostgreSQL-14--19-336791)
![PostgreSQL extension](https://img.shields.io/badge/PostgreSQL-extension-336791)

`pgturbohybrid` is a PostgreSQL extension for hybrid dense-vector + lexical
retrieval on top of pgvector. It combines pgvector dense nearest-neighbor search
with PostgreSQL text search using BM25-style ranking. BM25, short for Best
Matching 25, is a classic keyword-ranking method that rewards exact term
matches without requiring embeddings.

It is a standalone companion extension for pgvector: install pgvector first,
then install `pgturbohybrid` alongside it. The logo nods to hot hatchbacks:
compact, practical, quick, and daily-driver friendly. Not a million-euro
hypercar, but still has room for groceries.

## Status

`pgturbohybrid` is alpha software.

- APIs and on-disk index formats may change.
- Benchmarks are early and should be repeated on your own data.
- It is a good fit for evaluation, prototypes, and controlled experiments.
- Treat production use as something to validate carefully, not assume.

## Benchmarks

Dense (vector-only) retrieval on **dbpedia-openai-1M** (1,000,000 × 1536-d,
cosine), top-10, run with the [vector-db-benchmark][vdbb] harness at
`parallel=8` on a single node (AWS c6i, Intel Xeon Platinum 8375C). Every engine
is measured at **steady state**: an untimed warm-up pass precedes the timed run
so each engine's cache/buffers are hot. (Without this, engines that keep their
working set in a separate cache populated on first access — including
pgturbohybrid's native scan cache — are unfairly penalized against engines whose
index is already warm in shared buffers from the build.)

| engine | recall@10 | queries/s | mean latency |
|---|---:|---:|---:|
| **pgturbohybrid `dense`** | 0.836 | **5739** | 1.27 ms |
| **pgturbohybrid `high_recall`** | **0.983** | **2800** | 2.71 ms |
| weaviate | 0.977 | 2633 | 2.90 ms |
| pgvector (HNSW) | 0.979 | 1770 | 4.37 ms |
| qdrant | 0.986 | 853 | 9.24 ms |
| milvus | 0.988 | 750 | 10.39 ms |

- `dense` is the speed profile (4-bit, no rescore): the highest throughput here,
  ~3.2× pgvector, at recall 0.84 — use it when approximate recall is acceptable.
- `high_recall` is the exact-free high-recall profile (4-bit + heap-band
  rescore): **0.983 recall at 2800 q/s** — the best recall-per-throughput in this
  set. It beats pgvector on both recall *and* throughput, and delivers ~3× the
  throughput of qdrant/milvus at near-equal recall.

Single machine, single dataset — repeat on your own data and hardware. 4-bit
quantization is strongest on cosine / inner-product embeddings; high-dimensional
L2 (e.g. GIST-960) is a weaker case where recall holds up but latency does not.

[vdbb]: https://github.com/johannhartmann/vector-db-benchmark

## What It Does

TurboHybrid, the feature provided by `pgturbohybrid`, aims to make hybrid search
feel like a normal PostgreSQL index:

- combines pgvector dense retrieval with PostgreSQL text search/BM25-style
  retrieval
- fuses dense and lexical candidates inside one `turbohybrid` index access
  method
- uses reciprocal-rank fusion, or RRF, by default. RRF combines ranked result
  lists by position instead of trying to compare unlike score scales directly.
- aims to avoid the overhead of SQL-level two-index RRF plans for common
  top-k queries
- keeps pgvector unmodified

The basic idea: dense search helps with meaning, lexical search helps with exact
words, product names, IDs, and the weird little terms users actually type.
RRF is the fusion method that combines the two ranked candidate lists.

## When It Is Useful

Try `pgturbohybrid` when you are evaluating:

- RAG, or retrieval-augmented generation, over documents, tickets, support
  content, or knowledge bases
- semantic + lexical product or documentation search
- queries where dense-only retrieval misses exact terms
- queries where full-text search misses paraphrases
- PostgreSQL-first systems that want hybrid retrieval without adding another
  search service on day one

## When Not To Use It Yet

Wait, or isolate it carefully, if you need:

- stable on-disk index compatibility across releases
- official pgvector support
- production workloads without your own benchmark and relevance validation
- a mature operational story for every PostgreSQL and pgvector combination

Alpha means the paint is dry enough to touch, not enough to take through a car
wash.

## Install

Install pgvector first:

```sh
git clone --depth 1 --branch v0.8.2 https://github.com/pgvector/pgvector.git ../pgvector
make -C ../pgvector
make -C ../pgvector install
```

Then build and install `pgturbohybrid`:

```sh
git clone https://github.com/mayflower/pgturbohybrid.git
cd pgturbohybrid
make
make install
```

Create both extensions in your database:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
```

For a repeatable local setup, you can also use:

```sh
PG_CONFIG=pg_config PGVECTOR_REF=v0.8.2 scripts/dev-install.sh
```

## Quick Start

### Dense-Only Vector Search

Use a one-column `turbohybrid` index when you only need vector retrieval:

```sql
CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    embedding vector(3) NOT NULL,
    body text NOT NULL
);

CREATE INDEX documents_dense_idx ON documents
USING turbohybrid (embedding vector_cosine_turbohybrid_ops);

SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(vector_query => $1::vector)
LIMIT 10;
```

### Hybrid Vector + Text Search

Add a `tsvector` key when queries use `text_query`:

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

CREATE INDEX documents_turbohybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);

ANALYZE documents;

SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres hybrid search')
)
LIMIT 10;
```

`text_query` requires a turbohybrid index with a `tsvector` key. A dense-only
index accepts vector queries and rejects text or vector+text queries with a clear
error.

Use `<~->` for L2, `<~#>` for negative inner product, and `<~>` for cosine
hybrid ordering. A longer copy-paste walkthrough lives in
[examples/fast_start.sql](examples/fast_start.sql) and
[docs/fast_setup.md](docs/fast_setup.md).

Migration note: existing two-key hybrid indexes remain valid. Dense-only users
can now create smaller one-key indexes. To change an index from hybrid to
dense-only, or from dense-only to hybrid, rebuild it with `DROP INDEX` /
`CREATE INDEX` or `REINDEX` after changing the index definition.

## Fast Defaults

Fresh sessions use the `latency` profile:

```sql
SHOW turbohybrid.profile;
```

The default path uses a 4-bit quantized index, exact vector storage off,
adaptive dense widening off, dense and BM25 candidate budgets of 100, RRF
constant 60, and the SQL `LIMIT` as the final result target when possible. In
plain terms: create the default index, query with
`ORDER BY ... turbohybrid_query(...) LIMIT n`, then inspect the scan stats
below.

Public candidate and cache settings are intentionally capped in this alpha so a
user cannot set runaway per-query budgets in a shared PostgreSQL server. If you
hit a cap during evaluation, please open an issue with the dataset size, query
shape, and the settings you tried.

For the normal fast path, keep the query simple:

```sql
SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => $1::vector,
    text_query => websearch_to_tsquery('english', $2)
)
LIMIT 10;
```

That is the "daily driver" mode: compact settings, fewer knobs, enough speed to
be interesting, and no need to pack a racing helmet.

## Matched-Recall And Quality Modes

Use `matched_recall` when you want a compact 4-bit, exact-free index but need a
single named setting for pgvector/Qdrant-like dense recall comparisons:

```sql
SET turbohybrid.profile = 'matched_recall';
```

This profile keeps `quantization_bits = 4` and `exact_storage = off`, but new
low-dimensional indexes use heuristic graph neighbor selection, exact build
distances during graph construction, `ef_construction = 192`,
`ef_search = 128`, `graph_oversampling = 8`, final top-k heap rescore, and one
native segment unless `native_segments` is explicitly set. Exact build
distances are build-time only; they do not store full vectors in the index.

Use `high_recall` when you want near-exact dense recall from a compact 4-bit,
**exact-free** index and have latency headroom to spend:

```sql
SET turbohybrid.profile = 'high_recall';
```

`high_recall` keeps `quantization_bits = 4` and `exact_storage = off`, reuses
`matched_recall`'s candidate budgets, and additionally turns on
`dense_heap_rescore = band` (exact rescore of the full final candidate band by
re-reading vectors from the heap), turns `dense_adaptive_widening` off, and
defaults new indexes to `ef_construction = 256`, `ef_search = 192`,
`graph_oversampling = 12`. Band rescore is what recovers near-exact ordering:
4-bit code scoring finds the right neighbors but mis-ranks them, and the heap
rescore fixes the ranking without storing full vectors in the index. On
DBPedia/OpenAI (1536-d) this reaches ~0.99 recall while staying faster than
pgvector and Qdrant. Pair it with a heuristic build for best results:

```sql
SET turbohybrid.profile = 'high_recall';
SET turbohybrid.dense_build_neighbor_select = 'heuristic';
SET turbohybrid.dense_build_distance = 'code';

CREATE INDEX documents_turbohybrid_high_recall_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (
    quantization_bits = 4,
    exact_storage = off,
    graph_ef_construction = 256,
    graph_ef_search = 192,
    graph_oversampling = 12,
    native_segments = 1
);
```

Use `quality` when relevance matters more than lowest latency:

```sql
SET turbohybrid.profile = 'quality';
```

Quality mode is stronger and slower: it uses larger dense and lexical candidate
budgets, conservative BM25 paths, higher default graph search windows, and
heuristic dense-neighbor selection for new indexes. It usually costs more build
CPU and query CPU than `latency` or `matched_recall`, so compare it at matched
recall/precision instead of only comparing raw p95. For quality-sensitive
production evaluation, benchmark an exact-storage index too:

```sql
CREATE INDEX documents_turbohybrid_quality_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (exact_storage = on);
```

For exact-free 4-bit indexes that need a precision check without storing a full
copy of every vector in the index, heap-backed exact rescore can rescore only
the final top-k from the heap:

```sql
SET turbohybrid.dense_heap_rescore = 'topk'; -- or 'band'
```

`topk` fetches heap tuples for the final top-k and recomputes exact vector
distance with the same SIMD kernels used by index-exact rescore. `band` rescans
the full final candidate band. Both add heap I/O. The `latency` profile keeps
heap rescore off; the `high_recall` profile defaults it to `band`; `balanced`,
`matched_recall`, and `quality` may enable `topk` automatically for
low-dimensional exact-free indexes, and an explicit
`SET turbohybrid.dense_heap_rescore = 'off'|'topk'|'band'` always wins over the
profile default; set it to `auto` to return to profile-driven behavior.
Residual rerank is the lower-I/O middle ground:
`WITH (residual_rerank = on, residual_rerank_bytes = 16|32|64)`.

Do not treat either profile as universally best. Measure latency and relevance
on your dataset. Adaptive dense widening stays off for the `latency` profile,
but `balanced`, `matched_recall`, and `quality` can use a conservative `auto`
mode on low-dimensional exact-free 4-bit indexes when `final_k` is small and the
score boundary looks ambiguous. Local expansion, entry sidecars, and residual
rerank remain opt-in knobs for benchmark work, not release defaults.
`native_segments` is a build/concurrency lever rather than a free quality knob:
the default is one segment, quality/exact-build auto segmenting resolves to one
segment, and multi-segment indexes should be benchmarked with
`turbohybrid.native_segment_budget = sqrt|linear` before using them for
quality-sensitive comparisons.

For a dense-only profile sweep against a glove-like workload, use
`benchmarks/glove100_recall_latency_grid.sql`. It reports build time, index
size, precision@K against exact pgvector ordering, p50/p95/p99, and scan stats
for `default`, `balanced`, `matched_recall`, `quality`, `exact_storage`,
`residual_rerank`, and heap-rescore/adaptive-widening configurations. Treat
that grid as the authority for deciding whether `matched_recall` should become a
workload default.

## Diagnostics

After a query, check whether PostgreSQL used the expected TurboHybrid path:

```sql
SELECT turbohybrid_last_scan_stats();
SELECT turbohybrid_index_stats('documents_turbohybrid_idx'::regclass);
SELECT turbohybrid_estimate_memory('documents_turbohybrid_idx'::regclass);
SELECT turbohybrid_simd_capabilities();
```

Use `turbohybrid_estimate_memory(index)` before a query or prewarm step when
sizing native graph cache memory. It reports native code/adjacency/exact-vector
cache estimates, the current native cache policy, shared-cache per-backend view
bytes, and BM25 reader-cache arrays without building the native graph cache or
loading BM25 doc/liveness caches.

Native cache scope matters under concurrency:

- `per_backend` stores the resident native graph cache in each active PostgreSQL
  backend. A 400 MB native cache is roughly 400 MB at 1 backend, 4 GB at 10
  backends, and 40 GB at 100 backends before BM25 reader caches and other
  backend memory.
- `shared` stores the large immutable native graph arenas once in the mmap-backed
  shared cache, but each backend still allocates view/scratch metadata and BM25
  reader caches.
- `off` avoids resident native graph cache memory, but scans load graph pages
  through PostgreSQL shared buffers and can do more per-scan I/O/CPU work.

Use the estimator to compute concrete 1/10/100-backend projections for an index:

```sql
WITH estimate AS (
  SELECT turbohybrid_estimate_memory('documents_turbohybrid_idx'::regclass) AS m
),
backends(n) AS (VALUES (1), (10), (100))
SELECT
  n AS active_backends,
  pg_size_pretty(
    n * (
      (m->'concurrency'->>'per_backend_total_bytes_per_backend')::bigint +
      (m->'concurrency'->>'bm25_total_bytes_per_backend')::bigint
    )
  ) AS per_backend_scope_total,
  pg_size_pretty(
    (m->'concurrency'->>'shared_cache_total_bytes')::bigint +
    n * (
      (m->'concurrency'->>'shared_backend_view_bytes_per_backend')::bigint +
      (m->'concurrency'->>'bm25_total_bytes_per_backend')::bigint
    )
  ) AS shared_scope_total
FROM estimate, backends
ORDER BY n;
```

`turbohybrid.native_cache_warn_mb` emits a non-fatal DEBUG1 message when a
per-backend native cache build crosses the configured resident-size threshold.
Set it to `0` to disable the warning.

For a slow query, `turbohybrid_last_scan_diagnosis()` reduces the full stats
JSON to the key dense hot-path fields, a few derived ratios, and a single
`diagnosis` label (for example `healthy_u8_x4`, `traversal_dominated`,
`rescore_dominated`, or `scalar_lut_fallback`):

```sql
SELECT turbohybrid_last_scan_diagnosis();
SELECT turbohybrid_last_scan_diagnosis() ->> 'diagnosis';
```

Useful fields include:

- `profile`
- `index_used`
- `scan_orchestration`
- `quantization_bits`
- `exact_storage`
- SIMD fields such as `runtime_avx2`, `enabled_avx512vnni`, and
  `simd_force`
- `dense_candidates_effective`
- `bm25_candidates_effective`
- `final_k_source`
- `elapsed_us`

For troubleshooting examples, see [docs/fast_setup.md](docs/fast_setup.md).

## Benchmark Snapshot

These are local benchmark snapshots, not global claims. Results vary by
dataset, hardware, PostgreSQL settings, cache state, and query workload. nDCG
means normalized discounted cumulative gain, a relevance metric for ranked
results. HNSW means Hierarchical Navigable Small World, pgvector's graph index
type for approximate nearest-neighbor vector search.

On this FIQA/OpenAI setup, the run used 57,638 corpus rows, 648 qrels-backed
queries, 1,536-dimensional OpenAI `text-embedding-3-small` embeddings, the
`latency` profile, 100 dense candidates, 100 BM25 candidates, `final_k = 10`,
three warmup passes, and one measured pass. The TurboHybrid index used 4-bit
quantization with `exact_storage = off`.

| Method | Settings | p95 | nDCG@10 |
| --- | --- | ---: | ---: |
| pgturbohybrid default | default 4-bit exact-free index, adaptive widening off, LIMIT-inferred `final_k` | 1.628 ms | 0.415535 |
| pgturbohybrid adaptive auto 2.0 | explicit diagnostic setting, adaptive dense widening opt-in | 1.552 ms | 0.421465 |
| SQL RRF baseline | pgvector HNSW plus PostgreSQL GIN full-text search, 100/100 candidates | 2.009 ms | 0.423430 |
| pgvector dense-only reference | pgvector HNSW, no lexical branch | 1.412 ms | 0.442786 |

On a larger DBPedia/OpenAI3-large 1M qdrant-self setup, the run used 1,000,000
rows, 1,000 sampled self-queries, 3,072-dimensional
`text-embedding-3-large` embeddings, the `latency` profile, 100 dense
candidates, 100 BM25 candidates, `final_k = 10`, one warmup pass, and three
measured passes.

| Method | p95 | p99 | nDCG@10 | recall@10 | Index size |
| --- | ---: | ---: | ---: | ---: | ---: |
| pgturbohybrid | 3.978 ms | 7.386 ms | 0.940439 | 0.980 | 2.38 GB |
| pgvector halfvec HNSW + GIN FTS SQL RRF | 79.799 ms | 219.576 ms | 0.971086 | 0.992 | 8.41 GB |

That DBPedia result is a tradeoff, not a victory lap: TurboHybrid was much
faster and smaller on this machine, while the pgvector + FTS SQL RRF baseline
kept higher nDCG@10 and recall@10. The package default latency profile keeps
adaptive dense widening off; adaptive widening variants remain available in the
benchmark harness for controlled experiments. Benchmark details, baselines, and
reproduction notes are in
[docs/benchmarks/fiqa-openai.md](docs/benchmarks/fiqa-openai.md),
[benchmarks/dbpedia_openai3_large.md](benchmarks/dbpedia_openai3_large.md), and
[benchmarks/README.md](benchmarks/README.md).

The same DBPedia/OpenAI3-large corpus can also be used as a dense-only systems
comparison. This is not a hybrid-search benchmark: it uses the dataset's
existing 3,072-dimensional embeddings, no BM25 branch, no full-text search, and
no SQL RRF fusion. Turbovec is an in-process dense vector library that runs a
flat (brute-force) 4-bit scan over all rows, so treat this as a useful
reference point rather than a PostgreSQL access-method comparison. The run below
is a fresh local snapshot (Ice Lake Xeon, current build, 200 qdrant-self
queries, one warmup pass plus measured passes); the pgturbohybrid percentiles
are end-to-end SQL latency while the Turbovec percentiles are in-process,
single-threaded `index.search()` timings.

| Dense-only method | p50 | p95 | p99 | nDCG@10 | recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pgturbohybrid dense-only | 1.848 ms | 2.287 ms | 2.443 ms | 0.965 | 0.965 |
| Turbovec TurboQuant 4-bit | 174.176 ms | 181.312 ms | 190.260 ms | 1.000 | 1.000 |

In this qdrant-self setup each query has a single positive qrel, so recall@10 is
"is the source document in the top 10." Turbovec's flat scan is exact over the
4-bit codes and recovers that document for every query (recall@10 = 1.000), but
pays O(n) latency (~174 ms p50). `pgturbohybrid` uses a sub-linear graph index:
~94x lower latency (1.848 ms p50) while still recovering the source document for
96.5% of queries. The top-10 overlap between the two runs was 0.922 — i.e. the
graph reproduces ~92% of Turbovec's exact-over-codes top-10. As with the hybrid
numbers above, repeat this on your own hardware and query mix before drawing
conclusions.

If you already have a PostgreSQL RAG database, the bring-your-own benchmark
compares TurboHybrid with your existing retrieval SQL on your own rows and
query embeddings. See
[docs/benchmarks/bring-your-own-rag.md](docs/benchmarks/bring-your-own-rag.md).

## How It Works, Short Version

`pgturbohybrid` defines a `turbohybrid` PostgreSQL index access method over:

- one pgvector `vector` column for dense retrieval
- optionally, one `tsvector` column for lexical/BM25 retrieval

Dense-only indexes run the dense branch. Hybrid indexes can gather dense and
BM25-style lexical candidates, fuse them with reciprocal-rank fusion, and return
rows through PostgreSQL's normal `ORDER BY ... LIMIT` index-scan shape.

It depends on pgvector's SQL `vector` type but does not require pgvector to be
patched. Some graph/index code is derived from pgvector's HNSW implementation;
see [NOTICE](NOTICE) and [docs/architecture.md](docs/architecture.md).

## Documentation

- [How TurboHybrid works](docs/how-it-works.md)
- [Easy fast setup](docs/fast_setup.md)
- [FIQA/OpenAI benchmark snapshot](docs/benchmarks/fiqa-openai.md)
- [Bring-your-own RAG benchmark](docs/benchmarks/bring-your-own-rag.md)
- [DBPedia OpenAI3-large benchmark spec](benchmarks/dbpedia_openai3_large.md)
- [Benchmark methodology](benchmarks/README.md)
- [Compatibility notes](docs/compatibility.md)
- [Architecture notes](docs/architecture.md)
- [Release policy](RELEASE.md)
- [v0.1.0-alpha.2 release notes](docs/release-notes/v0.1.0-alpha.2.md)
- [Release hygiene summary](RELEASE_HYGIENE.md)

## Compatibility

The release target is PostgreSQL 14 through 19 (CI builds and runs the
regression tests on every version; PostgreSQL 19 is tested against pgvector
`master`). The pgvector compatibility target is pgvector 0.8.2 through current
pgvector `master`.

See [docs/compatibility.md](docs/compatibility.md) for the tested matrix and
boundary notes. If pgvector changes its internal vector layout, `pgturbohybrid`
should fail directly rather than silently reading malformed data.

## Contributing

Contributions are welcome, especially:

- real benchmark reports with complete context
- compatibility results across PostgreSQL, pgvector, CPU, and OS versions
- bug reports with `EXPLAIN` output and diagnostic JSON
- documentation fixes that make the alpha easier to evaluate safely

Start with [CONTRIBUTING.md](CONTRIBUTING.md). For community expectations,
see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), [SUPPORT.md](SUPPORT.md), and
[SECURITY.md](SECURITY.md).

## Attribution

`pgturbohybrid` depends on pgvector and contains code derived from pgvector's
HNSW implementation. pgvector is an excellent PostgreSQL vector search
extension; `pgturbohybrid` is a separate experimental companion project, not an
official pgvector project.

`pgturbohybrid` is distributed under the PostgreSQL License. See
[LICENSE](LICENSE) for the license text and [NOTICE](NOTICE) for pgvector
attribution.
