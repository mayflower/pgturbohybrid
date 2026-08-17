# Changelog

All notable changes for `pgturbohybrid` are documented here. Release tags may
use alpha suffixes while the PostgreSQL extension SQL version remains `0.1.0`.

## Unreleased
### Added
### Fixed

- **BM25 delta term directory no longer leaks one or more pages per INSERT.**
  `PgturbohybridBm25WriteDeltaTermDirectory` always allocated a fresh page
  for the (fixed-size, 776-byte) directory snapshot and repointed the meta,
  abandoning the previous page -- measured on a 300-doc hybrid index:
  50 delta inserts leaked 84 `BM25_DELTA_TERM` pages (~1.7 pages/insert,
  quadratic bytes since each rewrite carried the whole accumulated
  directory). The directory tuple is now updated in place on its head page
  (buffer-exclusive + WAL); readers, which read the first directory tuple of
  the head page, are unchanged. After the fix: 150 inserts, 0 abandoned
  pages (was on track for ~250). Compaction still rebuilds the directory
  from scratch (unchanged).
- **`turbohybrid_index_stats()` now reports page accounting.** New keys:
  `pages_total`, `pages_in_use`, `pages_abandoned` plus a per-kind histogram
  (`pages_quant_code`, `bm25_postings_pages`, `bm25_delta_pages`,
  `pages_bm25_delta_term`, ...). In-use pages are computed by walking every
  chain anchored in the graph meta, the BM25 meta (including the delta term
  directory's per-bucket chains) and the sparse meta, then compared with a
  one-pass whole-relation kind histogram. This makes compaction bloat and
  any future leak measurable from SQL. Regression whitelist updated.

- **Per-tenant BM25 statistics (`bm25Version 2`).** The BM25 meta tuple now
  stores `tenantStatsStartBlkno`/`tenantCount`, and new per-tenant aggregate
  pages (page kind 22, tuple type `0x6a`) hold `(tenant, docCount,
  totalDocLen)` entries. When an index is built with
  `INCLUDE (tenant_col int4)` (reloption `bm25_tenant_payload_slot`, default
  slot 0, `-1` disables) and `turbohybrid.bm25_tenant_stats = on` (default),
  the scorer resolves the tenant from the scan qual and rewrites
  `docCount`/`avgDocLen` in the metadata snapshot before scoring, so idf and
  length normalization stop leaking across tenants. Delta inserts increment
  the aggregates in place under the BM25 delta lock; compaction rebuilds
  exact aggregates. New SRF `turbohybrid_bm25_tenant_stats(index regclass)`
  reports the live per-tenant aggregates for operations and verification.
  Older `bm25Version 1` indexes keep global statistics; `REINDEX` rebuilds
  them at version 2. Regression coverage:
  `test/sql/pgturbohybrid_bm25_tenant.sql` (build, delta insert, scoped and
  unscoped queries, plain-index fallback).

### Fixed
- **Scan storage is now bounded by its metapage snapshot, not the live meta.**
  Every code-page/adjacency loader validated tuple node ids against the
  caller's freshly-read `meta->tqNodeCount` while indexing `storage->nodes[]`
  and the arenas, which are sized from the metapage snapshot taken when the
  scan storage (or native cache) was built. A concurrent insert appends new
  code/adjacency tuples to pages an in-flight scan still loads, so the loader
  could index nodes past the snapshot capacity -- an out-of-bounds heap write
  and the reader SIGSEGV class observed under concurrent insert on 2026-07-28.
  `PgturbohybridGraphScanStorage` now carries `nodeCapacity`/
  `adjRecordCapacity` snapshot bounds set at every sizing site (uncached
  scans, shared mmap view, per-backend cache, insert-cache growth), and all
  loaders skip tuples newer than the snapshot: newly inserted nodes are
  simply invisible to an in-flight scan, which matches its traversal state.
  Verified: th-installcheck 32/32; stress run 3 writers x 80 inserts + 3
  concurrent hybrid readers on one index: 740/740 rows, node_count 740
  (no lost nodes), no reader crash.
- **Concurrent insert no longer loses graph nodes and no longer crashes readers.**
  `aminsert` and `tqgraphinsert` take `ExclusiveLock` on the index update lock
  again, reverting the lock downgrade of 2026-07-11. Measured with
  `stress-turbohybrid-concurrency.sh` (8 writers, 8 readers, same index): with
  `ShareLock`, 480 concurrent inserts produced `node_count` 237 against
  `bm25_document_count` 260 — documents reachable by the lexical half of the
  index and unreachable by the vector half, with nothing in the log — and
  readers hit `SIGSEGV` in `PgturbohybridGraphLoadCodePage`
  (`storage->codePagesLoaded[pageNo]`) because the mmap-backed shared native
  cache is grown by writers while readers keep the old mapping. Two cluster
  crashes in two minutes of testing. After the change: `rows == node_count ==
  bm25_document_count`, no crash. Concurrent insert can come back once the graph
  metadata update is atomic and the native cache carries a generation the reader
  revalidates.
- **`PgturbohybridBm25AppendDelta` no longer holds the BM25 metadata page across
  the delta chain walk.** It held `BUFFER_LOCK_EXCLUSIVE` on the meta page while
  reading the delta directory and walking the whole delta chain. Every hybrid
  search starts by reading that page in share mode, so a single insert stalled
  every search for the length of the walk. Observed in production on 2026-07-28:
  one `INSERT` held it for 21 minutes with three searches queued behind it;
  buffer content locks are LWLocks, so there was no deadlock detector, no
  timeout and no way to cancel — the cluster had to be stopped with
  `-m immediate`. The metadata snapshot is now taken in share mode and released
  before any other page is touched.
- **The BM25 delta append is serialized by a heavyweight page lock**
  (`PGTURBOHYBRID_GRAPH_BM25_DELTA_LOCK`) instead of relying on buffer locks.
  Heavyweight locks have a deadlock detector, are interruptible, and are visible
  in `pg_locks`. The lock is released before returning, so a transaction that
  inserts many rows does not hold other backends for its whole duration.
- **A corrupted BM25 page chain now errors instead of hanging.**
  `PgturbohybridBm25CountChainPagesAndTail` had no iteration bound and no
  `CHECK_FOR_INTERRUPTS`, so a chain that links back on itself looped forever
  and could not be cancelled. It is now bounded by the relation size and reports
  the corruption with a `REINDEX` hint.
- **Bulk delete on a hybrid index no longer breaks hybrid search, and the BM25
  corpus statistics no longer go stale.** Two defects with one symptom.
  `PgturbohybridGraphMaybeCompactPageChains` rewrites the page chains assigning
  new dense node IDs, but every BM25 structure — postings, doc stats, block-max,
  impact tiers — is keyed by node ID and none of them is rewritten. Reproduced
  with a 2 000-row table, `DELETE` of half, `VACUUM`: the graph reported 1 000
  nodes while the BM25 metapage still said 2 000 documents, and the invariant
  check then failed **every** hybrid scan with
  `BM25 metadata document count is invalid`. Vector search kept working; hybrid
  search was dead until `REINDEX`. Page compaction is now skipped for indexes
  that have a BM25 branch (dead nodes stay marked; `REINDEX` reclaims the space)
  until the remap knows how to carry the lexical side.
  Separately, `PgturbohybridBm25MaybeCompact` only fired on delta accumulation,
  so a delete-only workload never rebuilt the base and the metapage kept counting
  documents that no longer exist — an inflated `N` in the idf and an inflated
  `avgdl` in the length normalization, which leaks across tenants because the
  lexical index is shared. Measured in production after a bulk purge: the index
  counted 12 480 documents against 9 489 in the table (`avgdl` 1 359 vs 1 040)
  and a *different* tenant's scenario ablation dropped two of eight. It now also
  fires when the recorded document count exceeds the live node count by the same
  percentage threshold, so `VACUUM` alone keeps the statistics true.
- **The maximum item size for a graph page now accounts for the line pointer.**
  `PgturbohybridGraphAppendTuple` computed the usable page space as
  `BLCKSZ - header - opaque`, which yields 8160, while `PageGetFreeSpace` on a
  fresh page reports 8156 because `PageAddItem` also writes an `ItemIdData`. A
  delta chunk sized 8160 therefore passed the capacity check and then failed to
  be added — three times on 2026-07-28. The item was silently dropped from the
  index (lost postings) and, worse, the failure path took the leak below. The
  arithmetic now lives in one place, `PgturbohybridGraphMaxItemSize()`, and
  matches what `PageGetFreeSpace` will allow.
- **`PgturbohybridGraphAppendTuple` no longer leaks the previous page's lock on
  the `PageAddItem` failure path.** `linkbuf` — the previous page of the chain —
  is held in exclusive mode while the new page is linked. The failure path
  released only `buf`, so the `linkbuf` content lock stayed held for the rest of
  the session; buffer content locks are not released at commit. This is what
  stopped hybrid search for 21 minutes in production on 2026-07-28: every reader
  of page 9067 blocked forever, with no holder visible in `pg_locks` and no
  response to cancellation.
- **`ambuild` no longer leaks locks when the sparse build fails.** The sparse
  collection step was wrapped in `PG_CATCH { FlushErrorState(); }`, which
  swallows the error without unwinding, leaving behind whatever the failed call
  held. A leaked exclusive buffer content lock is not released at commit and
  makes every later reader of that page hang forever. The handler now releases
  LWLocks and reports the skipped build as a `WARNING` with the original message.

### Added

- `high_recall` retrieval profile: an exact-free, near-exact-recall operating
  point. Reuses `matched_recall`'s candidate budgets but defaults
  `dense_heap_rescore = band`, `dense_adaptive_widening = off`, and the graph
  topology `ef_construction = 256` / `ef_search = 192` / `oversampling = 12`
  (used when the index does not set those reloptions). Keeps
  `exact_storage = off`. On DBPedia/OpenAI (1536-d) it reaches ~0.99 recall
  while staying faster than pgvector and Qdrant. Explicit GUCs still override
  the profile defaults.

## v0.1.0-alpha.2

This alpha changes the default retrieval behavior to the recovered fast setup.
Users from earlier alpha builds should expect a different latency/relevance
tradeoff unless they explicitly select a conservative profile.

### Changed

- Fast defaults are enabled for new sessions.
- `turbohybrid.profile` now defaults to `latency`.
- Default query budgets are now 100 dense candidates and 100 BM25 candidates.
- The default RRF constant remains 60.
- SQL `ORDER BY ... LIMIT n` is used as the final result target when possible.
- New indexes default to 4-bit quantization with `exact_storage = off`.
- BM25 latency fast paths are enabled by default under the latency profile,
  including WAND, impact/head fast paths where applicable, hot postings cache,
  and approximate latency paths documented for the profile.
- SIMD remains enabled when supported by the build and host CPU.

### Added

- `quality` profile for users who prefer safer relevance behavior over minimum
  latency.
- Improved `turbohybrid_last_scan_stats()` diagnostics for fast-path
  verification, including profile, index-use indicator, scan orchestration,
  quantization bits, exact-storage state, effective candidate budgets, final-k
  source, and phase timings.
- Perf-smoke coverage for the default fast path.
- Easy fast setup documentation in `docs/fast_setup.md`.
- FIQA/OpenAI profile-matrix acceptance configuration for comparing latency
  profile, quality profile, and SQL RRF.

### Benchmark Snapshot

On the FIQA/OpenAI validation setup documented in
`docs/benchmarks/fiqa-openai.md` (57,638 corpus rows, 648 queries,
1,536-dimensional OpenAI `text-embedding-3-small` embeddings, one warmup pass,
one measured pass), the `latency` profile with 4-bit quantization,
`exact_storage = off`, 100 dense candidates, and 100 BM25 candidates reported
0.910 ms p95 and 0.421540 nDCG@10 for `pgturbohybrid` default, compared with
3.254 ms p95 and 0.423341 nDCG@10 for the SQL RRF hybrid baseline using
pgvector HNSW default reloptions plus PostgreSQL GIN FTS. Results vary by
dataset and hardware.

Generated benchmark artifacts are not committed. Reproduction guidance lives in
`docs/benchmarks/fiqa-openai.md`; publish dataset, dimensions, query count,
profile, index options, query budgets, baseline, and relevance metrics with any
benchmark claim.

### Migration Notes

Earlier alpha users may see lower latency and a changed relevance/storage
tradeoff because the default profile is now latency-oriented and exact vector
storage is off by default.

To restore more conservative behavior for evaluation:

```sql
SET turbohybrid.profile = 'balanced';
```

For quality-sensitive evaluation:

```sql
SET turbohybrid.profile = 'quality';
```

When exact vector rescoring matters, rebuild indexes with exact storage:

```sql
CREATE INDEX ... USING turbohybrid (...) WITH (exact_storage = on);
```

Explicit `dense_k`, `bm25_k`, `rrf_k`, and `final_k` arguments still override
profile defaults.

## v0.1.0-alpha.1

Initial alpha packaging for the standalone `pgturbohybrid` extension that
depends on unmodified pgvector through `requires = 'vector'`.
