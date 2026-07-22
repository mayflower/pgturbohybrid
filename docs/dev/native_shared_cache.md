# Native Shared Cache

Native dense graph scans can serve immutable graph data from three scopes:

- `shared`: an mmap-backed, rebuildable cache file keyed by database OID,
  tablespace OID, physical relation number, fork, native graph format, and a
  64-bit graph generation. One backend builds it, other backends attach it.
- `per_backend`: the historical backend-local arena. This can have good
  single-session latency but duplicates resident graph data across every
  PostgreSQL backend.
- `off`: scan-local page loading through PostgreSQL buffers.

`turbohybrid.native_cache_scope=auto` resolves to `shared` on supported
platforms when the native working set fits `turbohybrid.native_cache_max_mb`.
If shared-cache publication fails transiently, `auto` falls back to a fitting
per-backend cache and exposes the shared failure in scan diagnostics. Explicit
`shared` never falls back: it uses an uncached scan and warns once per
backend/index. Explicit `per_backend` uses an uncached scan when it exceeds the
cap. `off` is always uncached. On Windows, where shared mmap files are disabled,
`auto` selects `per_backend` when it fits.

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
- `native_cache_invalidated`
- `built`, `attached`, `reused`, and `invalidated`
- `failure_reason`
- `gc_removed_files`
- `effective_bytes`
- `backend_mappings` (current backend, for lifecycle diagnostics)

This moves the cold shared-cache build away from the first user query. It does
not change index contents or query semantics.

## Identity and validation

Cache files live below
`pg_turbohybrid_cache/<database OID>/<tablespace OID>/`; their names include
the physical relation number, fork, generation, and native graph format. The
header repeats that identity and also records the shared-cache format,
PostgreSQL major version, `BLCKSZ`, byte order, and the size/alignment of every
fixed C structure serialized into the file. Relation OID is diagnostic only;
it is not used as the physical identity.

All segment offsets and lengths come from one checked layout calculation used
for admission, memory estimates, writing, and attaching. Attach validates the
exact file size, counts, alignment, non-overlap, and every segment bound before
constructing any pointer. An identity, ABI, or layout mismatch is treated as a
cache miss and reported as `shared_invalidated`; the file is never partially
used. This only discards rebuildable cache data and does not alter the index.

The native graph format is version 2 and stores a WAL-protected 64-bit
generation initialized to 1. Inserts, document-map changes, and vacuum
mutations advance it; generation exhaustion and older native graph formats
fail closed with a `REINDEX` hint.

## Crash safety and cleanup

Builders coordinate with a persistent POSIX advisory-lock file. The lock is
owned by the open file descriptor, so process exit or `SIGKILL` releases it;
the file itself is not an ownership marker. Publication uses a unique
`O_EXCL` temporary file in the destination directory, validates the completed
image, synchronizes and unmaps it, fsyncs the file, renames it atomically, and
fsyncs the parent directory. Readers therefore see either the previous valid
generation or the complete new generation, never a partial image.

The first shared-cache access in a backend, and later accesses after a bounded
interval, opportunistically take a separate kernel GC lock. GC removes only
recognized stale temp files, cache files whose database/relation identity no
longer exists, and superseded generations. Unknown filenames are deliberately
left untouched. Normal publication also removes older generations for the
same physical index, which bounds cache-file growth across insert, vacuum, and
reindex cycles without a background worker.

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
