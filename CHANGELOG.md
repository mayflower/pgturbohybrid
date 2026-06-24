# Changelog

All notable changes for `pgturbohybrid` are documented here. Git release tags
may use alpha suffixes (`v0.1.0-alpha.N`) independently of the PostgreSQL
extension SQL version.

## 0.1.1

Maintenance release. The exported SQL catalog is unchanged from `0.1.0` (the
`0.1.0--0.1.1` upgrade applies no DDL); the substance is in the C library,
tests, and release tooling. This is the first version with an upgrade script.

### Added

- `high_recall` retrieval profile: an exact-free, near-exact-recall operating
  point. Reuses `matched_recall`'s candidate budgets but defaults
  `dense_heap_rescore = band`, `dense_adaptive_widening = off`, and the graph
  topology `ef_construction = 256` / `ef_search = 192` / `oversampling = 12`
  (used when the index does not set those reloptions). Keeps
  `exact_storage = off`. On DBPedia/OpenAI (1536-d) it reaches ~0.99 recall
  while staying faster than pgvector and Qdrant. Explicit GUCs still override
  the profile defaults.
- Binary `RECEIVE` negative/fuzz test coverage for `turbohybrid_multivector`
  (`test/t/005_multivector_recv_binary.pl`).
- First extension upgrade script, `sql/pgturbohybrid--0.1.0--0.1.1.sql`, with an
  `ALTER EXTENSION ... UPDATE` regression test
  (`test/t/006_extension_upgrade.pl`).
- Keyless (Sigstore/OIDC) cosign signatures for release source archives and
  published Docker images.

### Changed

- The exact-MaxSim block dot-product kernel is resolved once and cached instead
  of re-probing CPU features on every call.

### Fixed

- A diagnostics memory-context variable modified in `PG_TRY` and read in
  `PG_CATCH` is now `volatile`, as required across the error-handling longjmp.
- Removed a dead, empty translation unit (`pgturbohybrid_bm25.c`) and corrected
  stale `dim>=1024` GUC help text and `pgturbohybrid_stats.c` doc references.

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
