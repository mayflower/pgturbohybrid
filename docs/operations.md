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

- **VACUUM / autovacuum** reclaim dead tuples normally; the access method
  participates in `ambulkdelete`/`amvacuumcleanup`. Sparse-primary indexes
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
