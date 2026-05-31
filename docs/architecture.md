# pgturbohybrid Standalone Extension Architecture

This document describes the current `pgturbohybrid` alpha architecture. The
extension is built and installed beside pgvector, without patching pgvector or
claiming pgvector-owned release metadata.

## Package Identity

- Extension name: `pgturbohybrid`
- Shared library name: `pgturbohybrid`
- Control file: `pgturbohybrid.control`
- Initial SQL install script: `sql/pgturbohybrid--0.1.0.sql`
- Extension dependency: `requires = 'vector'`
- Build model: PGXS build against an already-installed PostgreSQL and pgvector

The extension must install as:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
```

## Repository Boundary

The repository must not modify pgvector source files. pgvector is an external
dependency, not a patch target.

The standalone extension must not create or replace pgvector-owned SQL objects,
including:

- types: `vector`, `halfvec`, `sparsevec`, `bit`
- existing pgvector operators
- existing pgvector operator classes
- existing pgvector access methods or support functions

Any copied or derived implementation that remains in this repository must be
renamed, scoped, and built as `pgturbohybrid` code. The build must not compile
pgvector source files into the `pgturbohybrid` shared library.

## Dependency Model

`pgturbohybrid` uses pgvector's SQL type `vector` as its dense embedding type.
It should treat pgvector as a SQL and ABI dependency, not as a private C library.

The preferred model is:

- require the pgvector extension at SQL install time with `requires = 'vector'`
- reference pgvector's SQL type `vector` in SQL declarations
- use pgvector's installed header only for the `Vector` varlena layout, if that
  header is available in the target pgvector installation
- do not link against non-public pgvector C symbols
- do not call pgvector-internal distance functions, index functions, or support
  routines from C
- reimplement required vector distance helpers inside `pgturbohybrid` with
  `pgturbohybrid_*` C names

If a pgvector header is used, the supported pgvector versions and ABI risk must
be documented in the compatibility and release notes. The current compatibility
target is pgvector 0.8.2 through current pgvector `master`, backed by CI. The
build should include compile-time checks for the expected `Vector` layout where
possible. If the header is unavailable, any local compatibility struct must be
explicitly gated and documented as ABI-sensitive.

SQL-level pgvector behavior, such as the existence of the `vector` type, is
safe to depend on through `CREATE EXTENSION vector`. Private pgvector C symbols
are not part of the contract and must not be used.

## SQL Object Naming

All new SQL objects must be prefixed or schema-scoped to avoid collisions with
pgvector and other extensions.

The package remains `pgturbohybrid`, while the SQL feature surface uses the
shorter `turbohybrid` name:

- access method: `turbohybrid`
- query constructor: `turbohybrid_query(...)`
- SQL functions: `turbohybrid_*`
- hybrid query operators: `<~>` for cosine, `<~->` for L2, and `<~#>` for
  inner product
- operator classes: `vector_l2_turbohybrid_ops`,
  `vector_ip_turbohybrid_ops`, `vector_cosine_turbohybrid_ops`
- text/BM25 operator class: `bm25_tsvector_turbohybrid_ops`

The extension must not create generic names such as `hybrid_query` or opclasses
whose names could reasonably be mistaken for pgvector-owned objects.

If a dedicated schema is introduced later, SQL objects may instead be scoped in
that schema, but the extension should still keep externally visible names
unambiguous.

## C Symbol Naming

All PostgreSQL-visible C functions must be renamed away from pgvector, HNSW, and
prototype naming.

Required naming rules:

- `PG_FUNCTION_INFO_V1` functions use the `pgturbohybrid_*` prefix
- access method handlers use names such as `pgturbohybrid_handler`
- support functions use names such as `pgturbohybrid_vector_l2_support`
- no exported or SQL-visible C function may use `vector_*`, `hnsw_*`, `tq_*`, or
  generic `hybrid_*` names
- private `static` helpers may use shorter local names, but should avoid names
  that imply ownership by pgvector

This keeps the dynamic symbol table and SQL declarations clearly separated from
pgvector.

## GUCs and Reloptions

All GUCs must use the `turbohybrid.*` prefix. The extension must not create
`pgturbohybrid.*`, `hybrid.*`, or `hnsw.tq_*` GUCs.

The current alpha exposes these user-facing GUCs:

- `turbohybrid.profile`: retrieval profile. Current values are `latency`,
  `balanced`, `quality`, and `debug`; the default profile is `latency`.
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

The DBPedia dense-quality work also exposes experimental diagnostics. These
diagnostics remain off by default in the public `latency` profile so the package
default stays compact and predictable. Enable them explicitly for controlled
benchmark runs:

- `turbohybrid.dense_build_exact_distances`: use exact vector distances while
  building dense graph edges, while still allowing `exact_storage = off`.
- `turbohybrid.dense_adaptive_widening`: `off`, `auto`, or `on`; defaults to
  `off` and controls one bounded second graph-search pass for ambiguous
  dense-only scans when explicitly enabled.
- `turbohybrid.dense_adaptive_widening_multiplier`
- `turbohybrid.dense_adaptive_widening_max_multiplier`
- `turbohybrid.dense_adaptive_min_gap`
- `turbohybrid.dense_local_expansion`: `off`, `auto`, or `on`; controls bounded
  one-hop scoring from top approximate dense candidates.
- `turbohybrid.dense_local_expansion_topn`
- `turbohybrid.dense_local_expansion_max_neighbors`

The native dense-scoring hot path also exposes SIMD-tier, scoring-kernel,
rescore, and cache diagnostics. These pick the same results by default; they
exist to force a specific kernel for parity testing or to measure tiers:

- `turbohybrid.dense_graph_avx512vnni`, `turbohybrid.dense_graph_avxvnni`:
  allow the AVX-512 VNNI / AVX-VNNI dense scorers (default on where the CPU
  supports them; turn off to force a lower SIMD tier).
- `turbohybrid.dense_query_split_impl`: `signed`, `unsigned`, or `auto` 4-bit
  query-split representation (`auto` picks the unsigned-codebook maddubs/VPDPBUSD
  split on capable x86).
- `turbohybrid.dense_u8_split`: `auto`, `on`, or `off` for the unsigned-codebook
  (u8) 4-bit split scorer.
- `turbohybrid.dense_u8_batch_x4`: scores each batch of four codes with the true
  4-candidate (x4) u8 kernel in a single pass, sharing the query `[low|high]`
  loads and issuing the four scattered code loads together so their memory
  latency overlaps. This is the default (`on`); turning it `off` falls back to
  four single-node u8 passes, kept as a parity/benchmark escape hatch (not the
  default). `turbohybrid_last_scan_stats()` reports which path a scan actually
  took as `graph_u8_batch_mode` (`x4`, `single`, or `none`), with
  `dense_u8_batch_x4_enabled` echoing the knob.
- `turbohybrid.dense_rescore_band`: exact f32 rescore policy (`auto`, `off`,
  `exact`, `limited`); the latency profile resolves to exact-free for 4-bit
  code-only indexes.
- `turbohybrid.native_cache_max_mb`: per-backend in-memory native scan cache
  cap (default 512 MB); indexes whose working set fits are fully resident so
  warm scans read zero code pages.

The batch scorer adapts its prefetch to index size automatically (no GUC). Once
the estimated code working set (`tqNodeCount * codeBytes`) exceeds 64 MB -- large
enough that each code is a scattered RAM read rather than cache-resident -- the
scorer prefetches every cache line of each code ("whole-code prefetch") to hide
the full-code latency; below that threshold it touches only the first line, since
the codes are already hot. `turbohybrid_last_scan_stats()` surfaces this as
`graph_large_code_arena` (was the arena over the threshold),
`graph_whole_code_prefetch_active` (whole-code prefetch actually ran, i.e. the
arena was large and `turbohybrid.dense_graph_prefetch` was on), `graph_code_bytes`
(per-code width), and `graph_code_arena_estimated_bytes` (the working-set
estimate used for the decision).

Candidate-budget and cache GUCs have conservative public caps in this alpha:
`default_dense_k` and `default_bm25_k` are capped at 10,000, `default_rrf_k` at
100,000, `max_union_candidates` at 1,000,000, and
`bm25_hot_postings_cache_mb` at 1,024 MB. These are resource-safety limits, not
quality recommendations.

Profile assignment updates the dynamic defaults for the candidate budgets, BM25
strategy knobs, hot postings cache, hybrid bound mode, and SIMD setting unless a
GUC has been explicitly set by the user. The latency profile is the default fast
path documented in the README and setup guide.

Reloptions are scoped to the `turbohybrid` index access method, but should
still use stable descriptive names. The current alpha reloptions are:

- `graph_m`: maximum graph connections. Default: `16`.
- `graph_ef_construction`: graph candidate list size during build. Default:
  `128`.
- `graph_ef_search`: graph candidate list size during scans. Default: `64`.
- `graph_oversampling`: graph candidate oversampling multiplier. Default: `4`.
- `quantization_bits`: quantized dense-vector code width. Default: `4`.
- `exact_storage`: store exact vectors in the index for final exact rescoring.
  Default: `off`.
- `routing`: dense routing mode. Default: `auto`; current values are `auto`,
  `graph`, and `flat`.

The DBPedia dense-quality work adds experimental reloptions. They are explicit,
off by default where applicable, and should not be documented as stable public
storage format without a release note and `REINDEX` guidance:

- `graph_backbone`: force adjacent level-0 graph edges during build. Default:
  `off`.
- `entry_sidecar`: store a tiny list of representative node IDs in metadata.
  Default: `off`.
- `entry_sidecar_representatives`: maximum representative node IDs when
  `entry_sidecar` is enabled. Default: `128`; maximum: `256`.
- `residual_rerank`: store small per-vector sketches for final-band dense
  reranking. Default: `off`.
- `residual_rerank_bytes`: sketch bytes per vector when `residual_rerank` is
  enabled. Default: `32`; maximum: `64`.

Prototype names such as `tq_*` should not appear in user-facing reloptions.

## Access Method Storage

The standalone index access method is named `turbohybrid` and is installed by
the SQL handler `turbohybrid_handler(internal)`, which maps to the C symbol
`pgturbohybrid_handler`. Its on-disk identity is owned by this extension and
must not collide with pgvector access methods:

- `PGTURBOHYBRID_MAGIC_NUMBER`
- `PGTURBOHYBRID_PAGE_ID`
- `PGTURBOHYBRID_VERSION`

Block 0 is the metapage. It stores the access-method identity, format version,
index dimensions, dense graph options, quantization options, bounded routing
entry IDs, optional entry-sidecar IDs, entry/start block pointers, and BM25
metadata pointers. Older metapage tails are zero-filled by metadata readers so
default compact indexes do not get misread as having sidecars.

Current page kinds are:

- graph tuples and graph metadata inherited from the standalone graph storage
- quantized code pages
- quantized adjacency pages
- optional exact-vector pages for final rescoring
- quantization correction pages
- optional residual-rerank bytes embedded in quantized code tuples
- BM25 metadata, document statistics, lexicon, postings, block-max, delta,
  impact, and delta-term pages

Index page changes are WAL-logged with PostgreSQL generic WAL. pgturbohybrid
does not register a custom resource manager and does not require
`shared_preload_libraries`. Crash recovery and replicas recover index changes
from generic WAL and new-page WAL records alone.

## Executor Hooks

pgturbohybrid installs executor hooks because the index scan needs planner
context that PostgreSQL does not pass directly to an access method callback:

- the graph controller wraps `turbohybrid` index scans so LIMIT-aware scan
  budgets can be derived from the surrounding plan;
- the access method records the current `PlannedStmt` so planner-aware hybrid
  distance functions can reject scalar fallback when a query is not using the
  intended index-backed `ORDER BY` path.

One hook manager is installed during `_PG_init()` after graph and access-method
initialization. It calls any previously installed executor hook exactly once,
then records the current `PlannedStmt` and wraps eligible graph index scans.
Executor end clears scan wrapper state, pops the planned-statement stack, and
then delegates to the previous end hook or PostgreSQL's standard executor end.
Transaction and subtransaction abort callbacks clear the planned-statement stack
so failed statements do not leave backend-global state behind.

## Build Layout

The PGXS build uses:

```make
EXTENSION = pgturbohybrid
MODULE_big = pgturbohybrid
DATA = sql/pgturbohybrid--0.1.0.sql
PG_CONFIG ?= pg_config
```

The build may use `PG_CPPFLAGS` to include pgvector's installed server header
directory when available, but it must not compile pgvector `.c` files or depend
on pgvector private object files.

Source ownership is kept explicit with `pgturbohybrid_*` file names, including:

- `src/pgturbohybrid.c`
- `src/pgturbohybrid_am.c`
- `src/pgturbohybrid_build.c`
- `src/pgturbohybrid_scan.c`
- `src/pgturbohybrid_insert.c`
- `src/pgturbohybrid_vacuum.c`
- `src/pgturbohybrid_graph.c`
- `src/pgturbohybrid_quant.c`
- `src/pgturbohybrid_bm25.c`
- `src/pgturbohybrid_bm25_build.c`
- `src/pgturbohybrid_bm25_query.c`
- `src/pgturbohybrid_query.c`
- `src/pgturbohybrid_stats.c`
- `src/pgturbohybrid_vector_compat.c`

Windows and Unix builds must compile the same required source objects and expose
the same SQL install contract.

## Install SQL Contract

`sql/pgturbohybrid--0.1.0.sql` is the first install script for this extension.
It may reference pgvector's `vector` type because `pgturbohybrid.control`
requires `vector`, but it must not create that type.

The script intentionally uses the unqualified `vector` type. PostgreSQL makes
required extension schemas available during `CREATE EXTENSION`, and regression
tests cover `vector` and `pgturbohybrid` installed in different schemas.

The install script may create:

- `pgturbohybrid` access method objects
- prefixed support functions
- prefixed operator classes for the new access method
- prefixed query constructors and distance helpers
- minimal diagnostics, if they are intended as stable public API

It must not edit or replace pgvector install scripts. Future upgrades belong in
scripts named like:

```text
sql/pgturbohybrid--0.1.0--0.1.1.sql
```

They must not use pgvector upgrade script names.

## Testing And CI

The minimum test matrix should validate:

- PGXS build against an installed PostgreSQL and installed pgvector
- `CREATE EXTENSION vector`
- `CREATE EXTENSION pgturbohybrid`
- `DROP EXTENSION pgturbohybrid` without dropping pgvector-owned objects
- no creation of `vector`, `halfvec`, `sparsevec`, or `bit`
- no replacement of pgvector operators or opclasses
- all public GUCs use the `turbohybrid.*` prefix
- index creation and scans using `USING turbohybrid`
- insert, update, delete, vacuum, and restart behavior for the new access method
- supported pgvector versions across the documented compatibility range

Regression tests should validate user-visible behavior, not internal counters or
benchmark-only diagnostics.

## Benchmarks

Benchmark scripts and reproducibility configuration may live in the repository
when they are deterministic and useful for users. Generated benchmark JSON, MD,
CSV, logs, host-specific outputs, and result directories must not be vendored.

Benchmark methodology lives in [benchmarks/README.md](../benchmarks/README.md).
The public FIQA/OpenAI validation snapshot lives in
[docs/benchmarks/fiqa-openai.md](benchmarks/fiqa-openai.md). Architecture docs
should describe benchmark surfaces and artifact policy, not include benchmark
results.

## Non-goals

- no patching pgvector
- no pgvector release version bump
- no changes to `vector.control`
- no changes to pgvector SQL scripts
- no upstream pgvector README changes
- no vendored benchmark result artifacts
