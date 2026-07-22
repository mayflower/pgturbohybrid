# pgturbohybrid Standalone Extension Architecture

This document describes the current `pgturbohybrid` alpha architecture. The
extension is built and installed beside pgvector, without patching pgvector or
claiming pgvector-owned release metadata.

## Package Identity

- Extension name: `pgturbohybrid`
- Shared library name: `pgturbohybrid`
- Control file: `pgturbohybrid.control`
- SQL install scripts: `sql/pgturbohybrid--0.2.0.sql` (current default),
  `sql/pgturbohybrid--0.1.1.sql`, `sql/pgturbohybrid--0.1.0.sql`, and upgrades
  `sql/pgturbohybrid--0.1.0--0.1.1.sql` and
  `sql/pgturbohybrid--0.1.1--0.1.2.sql`
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

The access method supports two index shapes:

```sql
-- Dense-only: vector queries only.
CREATE INDEX documents_dense_idx ON documents
USING turbohybrid (embedding vector_cosine_turbohybrid_ops);

-- Hybrid: vector and/or text queries.
CREATE INDEX documents_hybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);
```

`text_query` requires the hybrid shape with a `tsvector` key. A one-key dense
index must not build or scan BM25 metadata.

The access method also supports two further key types, each usable standalone or
combined with a `tsvector` BM25 key:

```sql
-- Learned-sparse (SPLADE-style) retrieval, native sparse postings store.
CREATE INDEX documents_sparse_idx ON documents
USING turbohybrid (sparse_embedding sparse_ip_turbohybrid_ops);

-- Multivector (ColBERT-style late interaction).
CREATE INDEX documents_mv_idx ON documents
USING turbohybrid (colbert multivector_cosine_turbohybrid_ops);
```

See [feature-matrix.md](feature-matrix.md) for the maturity of each shape.

The extension must not create generic names such as `hybrid_query` or opclasses
whose names could reasonably be mistaken for pgvector-owned objects.

If a dedicated schema is introduced later, SQL objects may instead be scoped in
that schema, but the extension should still keep externally visible names
unambiguous.

## Native graph deletion and topology repair

Native graph node IDs are append-only between REINDEX operations. PostgreSQL
heap visibility remains authoritative: deletion marks the node dead, and every
result, payload, fusion, and rerank path excludes dead nodes. Traversal treats
liveness differently from return eligibility. A dead node can still be scored
and expanded so that deleting a hub does not disconnect live regions.

VACUUM repairs topology under the graph scan lock, independently at every graph
level. Its bounded candidate pool contains a dead node's live neighbors and
their live neighbors. A dead incoming reference is removed only when the repair
can install a same-level reciprocal live edge; otherwise the dead bridge is
preserved. Adjacency and entry-point changes are WAL logged, and the graph
generation advances once for the mutation phase. Global and segment entry
points are repaired deterministically by highest level and then smallest node
ID within the permitted range. REINDEX is the compaction operation.

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

The current alpha groups its `turbohybrid.*` GUCs by how safe each is to tune in
production. Every GUC keeps the same name across categories; the
developer/benchmark knobs additionally carry a `(developer/benchmark)` tag in
their `pg_settings` description, so the category is visible from `SHOW` /
`pg_settings` as well as here.

### Stable public

Stable names and semantics -- this is the supported tuning surface. The
`latency` / `balanced` / `matched_recall` / `quality` profiles set sensible
defaults for these, so override only for a specific need.

- `turbohybrid.profile`: retrieval profile. Current values are `latency`,
  `balanced`, `matched_recall`, `high_recall`, `quality`, and `debug`; the
  default profile is `latency`. `latency` is the smallest/fastest compact
  4-bit profile. `matched_recall` is the compact exact-free profile intended
  for comparisons against full-vector HNSW defaults in pgvector/Qdrant.
  `high_recall` spends more scan-time CPU on heap-band rescore while keeping
  exact vectors out of the index. `quality` is stronger but slower. `balanced`,
  `matched_recall`, `high_recall`, and `quality` use larger default graph/search
  windows and heuristic dense neighbor selection for new indexes when index
  reloptions do not override those values.

  Profile choice (guidance — validate on real data, not synthetic benchmarks):
  `latency` is the fast default and a good fit for easy corpora; `matched_recall`
  is the compact comparison baseline meant to approximate full-vector HNSW recall
  (validate its recall on your workload); `high_recall` is for compact 4-bit
  exact-free high recall when latency headroom exists — on hard/ambiguous queries
  its recall gain comes from wider `graph_ef_search` / `graph_oversampling` (and
  heuristic build), not from the opt-in features; `quality` is relevance-oriented
  but should be benchmarked before being made a default. The residual rerank,
  dense uncertainty retry, BM25 heap-tsvector rerank, and final-diversity features
  are opt-in or profile-gated, off in the default profiles, and should be
  benchmarked separately. No profile default should change from synthetic
  benchmarks alone.
- `turbohybrid.default_dense_k`, `turbohybrid.default_bm25_k`,
  `turbohybrid.default_rrf_k`: default dense/BM25 candidate budgets and the RRF
  fusion constant for `turbohybrid_query` callers.
- `turbohybrid.max_union_candidates`: cap on candidates retained while fusing the
  dense and BM25 branches.
- `turbohybrid.enable_wand`: WAND pruning for BM25 candidate generation.
- `turbohybrid.simd`: master switch for SIMD kernels; turn off for the portable
  scalar fallback (troubleshooting, or a non-SIMD baseline).
- `turbohybrid.native_cache_scope`: native dense graph cache scope. `auto`
  prefers the mmap-backed shared immutable cache on supported platforms when the
  working set fits `turbohybrid.native_cache_max_mb`; `per_backend` forces the
  backend-local arena; `shared` forces the shared mmap cache; `off` uses
  scan-local page loading through shared buffers.
- `turbohybrid.native_cache_max_mb`: native scan cache cap (default 2048 MB);
  indexes whose working set fits are fully resident, so warm scans read zero code
  pages. This is a per-backend allocation only when
  `native_cache_scope=per_backend`, which should be sized to host RAM and
  connection count.
- `turbohybrid.native_cache_warn_mb`: DEBUG1 warning threshold for per-backend
  native cache builds (default 512 MB). Set it to `0` to disable the warning.
  The warning is diagnostic only; it does not change cache policy.

For concurrency sizing, prefer `turbohybrid_estimate_memory(index)` before
running the workload. `per_backend` duplicates native resident cache bytes per
active PostgreSQL backend. `shared` stores the large immutable native arenas
once, but still has per-backend view/scratch overhead and BM25 cache arrays.
`off` avoids resident native cache memory but increases graph page loading per
scan. To project 1, 10, and 100 active backends from the estimator:

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
- `turbohybrid.bm25_strategy`, `turbohybrid.bm25_impact_or_mode`,
  `turbohybrid.bm25_hybrid_bound`, `turbohybrid.bm25_accumulator_mode`: BM25
  candidate-generation strategy and bound/accumulator modes. `auto` is robust;
  the profiles set these.
- `turbohybrid.bm25_hot_postings_cache_mb`,
  `turbohybrid.bm25_hot_postings_cache_min_df`: hot-postings cache size and the
  minimum document frequency for a cached entry.

### Experimental public

Real query-time behavior controls that are `off` or `auto` by default and change
recall/latency. Their names and semantics may still change, so they are not part
of the stable surface; enable them after measuring on your data. They stay off
in the public `latency` profile so the package default stays compact and
predictable.

- `turbohybrid.dense_rescore_band`: exact f32 rescore policy (`auto`, `off`,
  `exact`, `limited`); the latency profile resolves to exact-free for 4-bit
  code-only indexes.
- `turbohybrid.dense_heap_rescore`: exact-free heap rescore policy (`auto`,
  `off`, `topk`, `band`). `topk` fetches only final-k heap tuples; `band`
  fetches the full candidate band. Both read the indexed vector column and
  compute exact vector distance with the existing exact SIMD kernels. The
  default setting is `off`, which resolves to off in the `latency` profile when
  it has not been explicitly set; use `auto` to force profile-driven behavior.
  `balanced`, `matched_recall`, and `quality` may resolve to `topk` for
  low-dimensional exact-free indexes. Explicit `off`, `topk`, and `band` values
  always override profile defaults.
- `turbohybrid.dense_adaptive_widening` (`off` / `auto` / `on`, default `off`):
  controls conservative query-time widening of dense candidate targets for
  ambiguous dense-only scans. The `latency` profile resolves `auto` to off.
  `balanced`, `matched_recall`, and `quality` may resolve `auto` to a wider
  target for low-dimensional (`<= 256`) exact-free 4-bit indexes when
  `final_k <= 20` and the top-10 gap, final-k boundary gap, or filled candidate
  band says the query is ambiguous. The default cap is conservative:
  `balanced` and `matched_recall` widen by at most 1.5x and `quality` widens by
  at most 2.0x.
  The knobs are `turbohybrid.dense_adaptive_widening_multiplier`,
  `turbohybrid.dense_adaptive_widening_max_multiplier`, and
  `turbohybrid.dense_adaptive_min_gap`.
- `turbohybrid.dense_uncertainty_retry` (`off` / `auto` / `on`, default `off`):
  controls a bounded second traversal pass after the first dense result band has
  been scored and inspected. Named production profiles keep it off until
  retrieval-quality benchmark rows show a stable recall/precision win with low
  p95/p99 impact. Use `benchmarks/dev/retrieval_quality_grid.sql` to compare
  `matched_recall_uncertainty_retry_off`, `_auto`, and `_on` before promoting
  it. This is query-time only and does not require `REINDEX`.
- Heap filters that are not mapped to native graph `INCLUDE` int4 payload
  columns cannot be applied inside graph traversal.  PostgreSQL selectivity can
  therefore widen dense candidate collection toward
  `turbohybrid.max_scan_tuples`; `turbohybrid_last_scan_stats()` reports this
  as `dense_filter_unmapped`, `dense_linear_fallback_warning`, and
  `dense_linear_fallback_ratio`.  `turbohybrid.warn_linear_fallback` (default
  on) emits a DEBUG1 diagnostic when the ratio crosses
  `turbohybrid.linear_fallback_notice_threshold_ratio` (default 0.25).  Add the
  filter column as an `INCLUDE` int4 payload where possible, or lower candidate
  budgets / `max_scan_tuples`.
- `turbohybrid.dense_local_expansion` (`off` / `auto` / `on`, default `off`):
  bounded one-hop scoring from the top approximate dense candidates, tuned by
  `turbohybrid.dense_local_expansion_topn` and
  `turbohybrid.dense_local_expansion_max_neighbors`.

### Developer / benchmark

These force a specific SIMD tier or scoring-kernel implementation, or change
build-time graph construction, so kernels can be compared for parity and
benchmarking. The defaults already auto-select the best path for the host;
setting these does not improve production performance, and they are tagged
`(developer/benchmark)` in `pg_settings`. They pick identical results by default
-- only the kernel that produces them changes. See
[Benchmarking dense kernels](../benchmarks/README.md#benchmarking-dense-kernels).

- `turbohybrid.dense_graph_avx512vnni`, `turbohybrid.dense_graph_avxvnni`: allow
  the AVX-512 VNNI / AVX-VNNI dense scorers (default `on` where the CPU supports
  them; turn off to force a lower SIMD tier for parity testing).
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
- `turbohybrid.dense_build_exact_distances`: use exact f32 vector distances while
  building dense graph edges, while still allowing `exact_storage = off`. Can
  improve graph topology for experiments without storing exact vectors at scan
  time. This is a compatibility override for `turbohybrid.dense_build_distance`.
- `turbohybrid.dense_build_distance`: controls the distance source used while
  building dense graph topology. `code` builds in the compact quantized-code
  domain. `exact` keeps original vectors only during graph construction and
  frees them before writing compact indexes when `exact_storage = off`. `auto`
  uses exact build distances for low-dimensional (`dimensions <= 256`)
  `balanced`, `matched_recall`, and `quality` builds, and code-domain build
  distances for latency or high-dimensional builds. Use
  `benchmarks/glove100_recall_latency_grid.sql` to measure the build-time,
  index-size, precision, and p95 cost before promoting a build-distance setting
  for a workload.
- `turbohybrid.dense_build_neighbor_select`: dense graph neighbor selection for
  native graph builds. `fast` uses the simple nearest-neighbor selector used by
  the code-only build path. `heuristic` uses the diversified HNSW-style selector
  even when the build can encode vectors code-only. `auto` uses `heuristic` for
  low-dimensional indexes (`dimensions <= 256`) and for `balanced`,
  `matched_recall`, `quality`, and `debug`; it uses `fast` only for
  high-dimensional latency-profile builds.

The batch scorer adapts its prefetch to index size automatically (no GUC). Once
the estimated code working set (`tqNodeCount * codeBytes`) exceeds 64 MB -- large
enough that each code is a scattered RAM read rather than cache-resident -- the
scorer prefetches every cache line of each code ("whole-code prefetch") to hide
the full-code latency; below that threshold it touches only the first line, since
the codes are already hot. `turbohybrid_last_scan_stats()` surfaces this as
`graph_large_code_arena` (was the arena over the threshold),
`graph_whole_code_prefetch_active` (whole-code prefetch actually ran, i.e. the
arena was large; prefetch is always enabled, with no GUC), `graph_code_bytes`
(per-code width), and `graph_code_arena_estimated_bytes` (the working-set
estimate used for the decision).

Alongside the flat keys, `turbohybrid_last_scan_stats()` also emits a grouped
view of the same numbers under nested sections so bottleneck diagnosis reads
top-down: `dense` (with `dense.kernels`, `dense.cache`, `dense.traversal`, and
`dense.timing_us` sub-objects), `bm25`, `fusion`, and `query`. The hot-path flags
above appear inside `dense.kernels` / `dense.cache`; the nested values are built
from a single consolidated struct, so they always agree with their flat
counterparts. The flat keys remain for backwards compatibility.

Candidate-budget and cache GUCs have conservative public caps in this alpha:
`default_dense_k` and `default_bm25_k` are capped at 10,000, `default_rrf_k` at
100,000, `max_union_candidates` at 1,000,000, and
`bm25_hot_postings_cache_mb` at 1,024 MB. These are resource-safety limits, not
quality recommendations.

Profile assignment updates the dynamic defaults for candidate budgets, BM25
strategy knobs, hot postings cache, hybrid bound mode, SIMD setting, and
query-time quality knobs unless a GUC has been explicitly set by the user.
During dense index builds without explicit graph reloptions, the profile also
selects graph construction/search defaults: `latency` uses
`ef_construction=128`, `ef_search=64`, `oversampling=4`, and heuristic build
edges for low-dimensional indexes while keeping fast build edges for
high-dimensional latency builds; `balanced` uses `ef_construction=192`,
`ef_search=96`, `oversampling=4`, and heuristic build edges; `matched_recall`
uses `ef_construction=192`, `ef_search=128`, `oversampling=8`, heuristic build
edges, low-dimensional exact build distances, and final top-k heap rescore for
exact-free indexes; `high_recall` uses `ef_construction=256`,
`ef_search=192`, `oversampling=12`, heuristic build edges, one segment, and
heap-band rescore at scan time; `quality` uses `ef_construction=256`,
`ef_search=192`, `oversampling=8`, and heuristic build edges. The latency
profile is the default fast path documented in the README and setup guide.

Build-time graph choices affect only newly created indexes. Changing
`graph_ef_construction`, `graph_ef_search`, `graph_oversampling`,
`dense_build_distance`, `dense_build_neighbor_select`, `native_segments`,
`entry_sidecar*`, or `residual_rerank*` for an existing index requires
`REINDEX` to materialize the new topology or stored sketches. Query-time GUCs
such as `dense_heap_rescore`, `dense_adaptive_widening`,
`dense_uncertainty_retry`, `dense_residual_rerank_mode`, calibrated fusion, and
BM25 heap tsvector rerank do not change index storage, though residual mode only
has an effect when residual sketches were built.

Multivector late-interaction indexes use the same native graph storage for
subvector nodes, but result identity remains the heap TID/document. A build over
`turbohybrid_multivector` expands each row into one graph node per token vector
and keeps in-memory build maps from node ID to document ID/token ordinal.
Incremental insert/update uses the same expansion semantics for the new tuple
and appends one BM25 delta per inserted document when a lexical key is present.
The scan path runs one bounded graph traversal per query token, assigns
scan-local `TqDocId` values to touched documents, and accumulates document
scores with approximate MaxSim before final ordering. The `<~>` operator
therefore still orders by smaller-is-better distance, using `-MaxSim` at the
document level. Hybrid multivector + BM25 is supported for document-level RRF
fusion; score-level fusion modes remain unsupported unless explicitly
documented.

The scan-time candidate budgets are controlled by
`turbohybrid.multivector_subvector_k`,
`turbohybrid.multivector_unique_docs_per_token`,
`turbohybrid.multivector_max_raw_hits_per_token`, and
`turbohybrid.multivector_doc_candidate_k`. Exact final ordering for retained
document candidates is controlled by `turbohybrid.multivector_exact_rerank`
(`topk` by default) and `turbohybrid.multivector_exact_rerank_k`; this fetches
heap multivector values and does not store full f32 vectors in the index. These
GUCs do not alter index storage. Build/query safety caps are
`turbohybrid.multivector_max_doc_vectors`,
`turbohybrid.multivector_max_query_vectors`, and
`turbohybrid.multivector_max_dim`. The scan-time accumulator memory estimate is
bounded by `turbohybrid.multivector_max_accumulator_mb`.

The exact MaxSim kernel is dispatched below the reference scalar implementation:
scalar remains mandatory, while SIMD-enabled builds may use AVX-512F or AVX2 on
x86 and NEON on ARM when `turbohybrid.simd` is on. The SQL semantics do not
depend on SIMD; the regression parity test compares scalar and SIMD-enabled
MaxSim across tail-heavy dimensions.

Reloptions are scoped to the `turbohybrid` index access method, but should
still use stable descriptive names. The current alpha reloptions are:

- `graph_m`: maximum graph connections. Default: `16`.
- `graph_ef_construction`: graph candidate list size during build. Default:
  `128`.
- `graph_ef_search`: graph candidate list size during scans. Default: `64`.
- `graph_oversampling`: graph candidate oversampling multiplier. Default: `4`.
- `native_segments`: number of independent native dense graph segments to build.
  Valid values are `1`, `2`, `4`, `8`, and `auto`. Default: `1` for
  single-graph compatibility. `auto` resolves from PostgreSQL's parallel
  maintenance worker setting and is capped internally, except quality-sensitive
  builds (`profile = matched_recall`, `profile = quality`, or explicit exact
  build distances) resolve auto to one segment. More segments can reduce
  build-edge work, but they are a build/concurrency lever: query search must
  scale the segment budget or recall can drop.
- `quantization_bits`: quantized dense-vector code width. Default: `4`.
- `exact_storage`: store exact vectors in the index for final exact rescoring.
  Default: `off`. When it is off, users can instead benchmark profile-driven or
  explicit `turbohybrid.dense_heap_rescore = topk|band` and residual sketches as
  query-time quality paths.
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
- `entry_sidecar_strategy`: representative selection strategy. Default:
  `hash`, which preserves the original code-hash bucket behavior. Other
  supported values are `farthest_code`, `level_covering`, and
  `hybrid_level_covering`. All strategies write node IDs into the existing
  metapage sidecar array; no sidecar storage pages are added. Since the chosen
  representatives are materialized at build time, changing this option requires
  `REINDEX` to alter an existing index's sidecar contents.
- `residual_rerank`: store small per-vector sketches for final-band dense
  reranking. Default: `off`.
- `residual_rerank_bytes`: sketch bytes per vector when `residual_rerank` is
  enabled. Default: `32`; supported presets are `16`, `32`, and `64` bytes
  (maximum: `64`).
- `turbohybrid.dense_residual_rerank_mode`: scan-time residual-sketch policy.
  `off` ignores stored sketches, `fixed` preserves the original hardcoded
  adjustment, and `calibrated` scales and clamps the adjustment by the observed
  final-band distance spread. This GUC does not change index storage.
- `turbohybrid.dense_residual_rerank_weight` and
  `turbohybrid.dense_residual_rerank_max_adjust_ratio`: calibrated residual
  rerank controls. The default weight `-1` uses the built-in automatic weight;
  the max-adjust ratio bounds the absolute adjustment as a fraction of the
  candidate-band spread.

Prototype names such as `tq_*` should not appear in user-facing reloptions.

## Access Method Storage

The full on-disk format contract (identity/version constants, page kinds,
metapage layout, corruption-rejection rules, and which changes require
`REINDEX`) lives in [storage-format.md](storage-format.md); this section is the
overview.

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
- sparse retrieval pages: sparse metapage, per-term lexicon, postings chunks,
  block-max (WAND) directory, delta-chain, and node-map pages. A sparse-only
  index (no dense graph) maps node IDs to heap TIDs through the node-map chain
  and delegates liveness to heap MVCC visibility

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
DATA = sql/pgturbohybrid--0.1.0.sql sql/pgturbohybrid--0.1.1.sql sql/pgturbohybrid--0.1.2.sql sql/pgturbohybrid--0.2.0.sql sql/pgturbohybrid_experimental--0.2.0.sql sql/pgturbohybrid--0.1.0--0.1.1.sql sql/pgturbohybrid--0.1.1--0.1.2.sql sql/pgturbohybrid--0.1.2--0.2.0.sql
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
- `src/pgturbohybrid_bm25_build.c`
- `src/pgturbohybrid_bm25_query.c`
- `src/pgturbohybrid_multivector.c`
- `src/pgturbohybrid_sparse_build.c`
- `src/pgturbohybrid_sparse_primary.c`
- `src/pgturbohybrid_sparse_query.c`
- `src/pgturbohybrid_sparse_score.c`
- `src/pgturbohybrid_query.c`
- `src/pgturbohybrid_diagnostics.c`
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
- one-key dense-only and two-key hybrid index creation and scans using
  `USING turbohybrid`
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
