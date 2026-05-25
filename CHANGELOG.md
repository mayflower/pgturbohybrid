# Changelog

All notable changes for `pgturbohybrid` are documented here. Release tags may
use alpha suffixes while the PostgreSQL extension SQL version remains `0.1.0`.

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

Recovered FIQA/OpenAI latency-profile result:

- dataset: FIQA/OpenAI
- corpus rows: 57,638
- queries: 648
- embedding dimensions: 1,536
- profile/settings: `latency`, 4-bit, `exact_storage = off`, `dense_k = 100`,
  `bm25_k = 100`, `final_k = 10`
- p50: 0.770 ms
- p95: 1.006 ms
- p99: 1.265 ms
- nDCG@10: 0.421540
- SQL RRF baseline p95: 3.988 ms
- pgvector HNSW dense-only p95: 3.361 ms

Generated benchmark artifacts are not committed. Reproduce results using the
commands in `PERF_RECOVERY_SUMMARY.md` and publish profile, index options,
query budgets, and relevance metrics with any benchmark claim.

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
