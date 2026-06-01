# Native Shared Cache

Native dense graph scans can serve immutable graph data from three scopes:

- `shared`: an mmap-backed cache file keyed by the index relfilenode and graph
  generation metadata. One backend builds it, other backends attach it.
- `per_backend`: the historical backend-local arena. This can have good
  single-session latency but duplicates resident graph data across every
  PostgreSQL backend.
- `off`: scan-local page loading through PostgreSQL buffers.

`turbohybrid.native_cache_scope=auto` resolves to `shared` on supported
platforms when the native working set fits `turbohybrid.native_cache_max_mb`.
On platforms where the mmap-backed cache is not enabled, auto keeps the
per-backend behavior.

## Prewarm

Use:

```sql
SELECT turbohybrid_prewarm('items_turbohybrid_idx'::regclass);
```

The function builds or attaches the shared cache and returns JSON with:

- `native_cache_built`
- `native_cache_attached`
- `native_cache_attach_us`
- `native_cache_build_us`
- `native_cache_wait_us`
- `native_cache_bytes`
- `native_cache_code_bytes`
- `native_cache_adj_bytes`
- `native_cache_exact_bytes`
- `native_cache_reason`

This moves the cold shared-cache build away from the first user query. It does
not change index contents or query semantics.

## CREATE INDEX Materialization

Materializing the shared serving cache as part of `CREATE INDEX` is intentionally
deferred. Doing it safely requires a clear lifecycle for cache files across
`REINDEX`, relation forks, tablespace moves, crash cleanup, and platform-specific
file mapping behavior. The current low-risk path is to keep the index storage
unchanged and let operators call `turbohybrid_prewarm(regclass)` after index
creation or before opening a connection pool to traffic.

## Diagnostics

`turbohybrid_last_scan_stats()` reports the cache scope and cold-start counters.
`turbohybrid_last_scan_diagnosis()` emits `native_cache_cold_build` when the
previous native dense scan was dominated by cache materialization rather than
graph traversal or scoring.
