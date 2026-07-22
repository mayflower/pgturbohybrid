# Operations Guide

How to evaluate and run `pgturbohybrid` safely. This consolidates the cache,
memory, concurrency, maintenance, and upgrade guidance that was previously
scattered across the README and `architecture.md`.

> **Alpha software.** `pgturbohybrid` is alpha: on-disk formats and APIs may
> change, and you should validate recall, latency, and stability on your own
> data before any production use. Treat everything below as evaluation guidance,
> not a production-readiness guarantee. See the
> [feature & maturity matrix](feature-matrix.md).

## Native cache scope

Dense graph scans can keep a resident "native cache" of the compact codes and
adjacency. The scope is chosen by `turbohybrid.native_cache_scope`:

| Scope | Behavior | Memory |
| --- | --- | --- |
| `auto` (default) | prefers the shared mmap cache when the working set fits `turbohybrid.native_cache_max_mb`, else per-backend | shared where possible |
| `per_backend` | each backend builds its own resident cache | **multiplies per active backend** |
| `shared` | the large immutable arenas are stored once in an mmap-backed shared cache | once, plus per-backend view/scratch + BM25 reader caches |
| `off` | no resident native cache; scans load graph pages through shared buffers | none resident, but more per-scan I/O/CPU |

**Per-backend memory multiplication is the main sizing trap.** Under
`per_backend`, a 400 MB native cache is roughly 400 MB at 1 backend, 4 GB at 10,
and 40 GB at 100 — before BM25 reader caches and other per-backend memory. Size
`per_backend` to host RAM × connection count, or use `shared`/`auto`.

Related GUCs:

- `turbohybrid.native_cache_max_mb` (default 2048): working sets that fit are
  fully resident, so warm scans read zero code pages.
- `turbohybrid.native_cache_warn_mb` (default 512): emits a non-fatal DEBUG1
  message when a per-backend build crosses this resident size; set `0` to
  disable. Diagnostic only — it does not change policy.

Shared files are rebuildable acceleration data, not index storage. They are
partitioned by database and tablespace and identified by physical relation,
fork, native format, and a WAL-protected 64-bit graph generation. Their headers
also bind PostgreSQL major version, block size, endianness, and serialized C
layout. Attach validates the complete identity and every segment bound before
using the mapping. A mismatch is discarded as `shared_invalidated`; queries do
not score from a partial or foreign cache.

## Memory sizing with `turbohybrid_estimate_memory()`

Estimate cache memory **before** running a workload or `turbohybrid_prewarm()`.
It reports native code/adjacency/exact-vector estimates, the cache policy,
shared-cache per-backend view bytes, and BM25 reader-cache arrays without
building the native cache or loading BM25 caches:

```sql
SELECT turbohybrid_estimate_memory('documents_turbohybrid_idx'::regclass);
```

Project resident memory at 1, 10, and 100 active backends:

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

## Maintenance: VACUUM, REINDEX, CREATE INDEX

- **VACUUM / autovacuum** mark native graph nodes dead and perform bounded,
  WAL-logged local topology repair. Dead nodes are never SQL results, but their
  adjacency remains available as a routing bridge until enough reciprocal live
  replacement edges can be installed. Sparse-primary indexes
  delegate liveness to heap MVCC, so dead rows are filtered at scan time and
  reclaimed on vacuum.
- **`turbohybrid_sparse_compact(index)`** compacts a sparse index in place,
  reclaiming dead postings between vacuums (experimental).
- **`CREATE INDEX CONCURRENTLY`** is supported; the access method does **not**
  advertise parallel index build in this alpha (parallel maintenance workers are
  not used for the build).
- **`REINDEX` is required** to change build-time index contents:
  `quantization_bits`, `exact_storage`, `graph_*` windows,
  `dense_build_distance`, `dense_build_neighbor_select`, `native_segments`,
  `entry_sidecar*`, and `residual_rerank*`. Query-time GUCs (heap rescore,
  adaptive widening, uncertainty retry, fusion, residual mode) take effect
  without `REINDEX`.

## Upgrades and on-disk compatibility

This is alpha: the on-disk index format can change between alpha tags. **Plan to
`REINDEX` (or drop/recreate) indexes when upgrading.** Format/version mismatches
fail with a clear `ERRCODE_DATA_CORRUPTED` error and a REINDEX hint rather than
misreading. Extension SQL upgrades, when introduced, follow the standard
`pgturbohybrid--<from>--<to>.sql` pattern (see [RELEASE.md](../RELEASE.md)).
Native graph format 2 introduced the 64-bit cache generation; native indexes
from the previous format therefore require `REINDEX`. Legacy non-native
storage is checked by its own format contract.

Shared native-cache files are rebuildable serving artifacts. On POSIX systems,
builders use advisory kernel locks and crash-safe atomic publication; a killed
builder cannot leave permanent ownership behind. Lazy garbage collection runs
during cache access and removes only recognized stale temporary files,
superseded generations, and artifacts for dropped/reindexed objects. Unknown
files are preserved. Explicit `shared` configuration fails visibly to
uncached scans and does not silently allocate a per-backend copy; `auto` may
fall back to a fitting per-backend cache after a transient shared failure.

## Benchmark before production

Recall and latency are dataset- and hardware-specific. Before relying on any
profile, measure on your data — see [profile-tuning.md](profile-tuning.md) and
the benchmark methodology in [../benchmarks/README.md](../benchmarks/README.md).
No profile's compiled defaults should be chosen from synthetic benchmarks alone.

## Diagnostics to collect for support

When filing a [performance or bug report](../CONTRIBUTING.md#reporting-issues-and-opening-prs),
include:

```sql
SELECT turbohybrid_last_scan_stats();      -- after the slow query
SELECT turbohybrid_last_scan_diagnosis();  -- single bottleneck label + key fields
SELECT turbohybrid_index_stats('idx'::regclass);
SELECT turbohybrid_estimate_memory('idx'::regclass);
SELECT turbohybrid_simd_capabilities();
SELECT turbohybrid_validate_index('idx'::regclass, true);
```

plus the `CREATE INDEX` statement, the query, `EXPLAIN (ANALYZE, BUFFERS)`, and
the PostgreSQL/pgvector/pgturbohybrid versions. The stable vs experimental keys
of the stats JSON are documented in [diagnostics-schema.md](diagnostics-schema.md).

`turbohybrid_graph_repair_dry_run(index, sample_nodes, search_ef,
candidate_limit)` is a read-only (`AccessShareLock`, no WAL) native-graph
neighborhood diagnostic: it samples nodes, compares each one's level-0
neighborhood with a stronger bounded local pool, and reports `avg_overlap`,
`weak_nodes`, `missed_neighbor_count`, and `suggested_edges` — useful for
deciding whether a graph rebuild is worthwhile.

`turbohybrid_validate_index(index, deep := false)` is the read-only mechanical
integrity check. It takes `AccessShareLock`, writes no WAL, and returns JSON
instead of throwing for detected corruption. Sampled mode checks the metapage,
page envelopes and chains, dense tuples, and enabled BM25, sparse, and
multivector sidecars. Deep mode additionally checks every adjacency tuple and
performs graph reachability from the entry, routing, segment, and sidecar roots;
dead nodes remain traversable because they may be required bridges.

Treat `ok = false` as an index-integrity incident. Preserve the returned issue
codes, verify the heap separately, and `REINDEX` the affected index before
returning it to service. `recommendation = reindex_recommended` is a churn
signal rather than proof of corruption; schedule a REINDEX to compact the
append-only node space. Run deep validation after crash recovery, a suspected
storage fault, or a REINDEX. Sampled validation is appropriate for routine
health polling.
- Native graph storage is append-mostly: inserts allocate new node IDs and
  VACUUM does not reuse them. `turbohybrid_index_stats()` reports `live_nodes`,
  `dead_nodes`, `dead_node_ratio`, `live_entry_node`, `dead_bridge_nodes`,
  `avg_live_degree_level0`, `dead_neighbor_refs`, `reindex_recommended`, and
  `reindex_reason`. A dead-node ratio of 0.20 or dead-neighbor-reference ratio
  of 0.25 is the fixed recommendation threshold. VACUUM emits at most one
  recommendation warning; use REINDEX to compact the append-only node space.
