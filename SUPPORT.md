# Support

This file helps you choose the right support channel and include the details
maintainers need to reproduce bugs, compatibility issues, and performance
reports.

`pgturbohybrid` is alpha software. APIs, defaults, benchmark behavior, and the
on-disk index format may change before a stable release. It is suitable for
evaluation and controlled experiments; production use needs your own testing.

## Where To Ask For Help

Open a GitHub issue for:

- reproducible bugs
- install or compatibility problems
- documentation gaps
- performance reports with enough context to reproduce
- feature requests that fit the `pgturbohybrid` scope

Before opening an issue, please check the README, [docs/fast_setup.md](docs/fast_setup.md),
[docs/how-it-works.md](docs/how-it-works.md), [docs/operations.md](docs/operations.md)
(cache/memory sizing, VACUUM/REINDEX, production evaluation), and
[docs/compatibility.md](docs/compatibility.md).

## Security Issues

Do not report security issues in public issues. Use the repository's GitHub
security advisory workflow, or email security@mayflower.de if the advisory
workflow is unavailable. See [SECURITY.md](SECURITY.md).

## Bug Reports

Please include:

- PostgreSQL version
- pgvector version or commit
- `pgturbohybrid` version or commit
- operating system and CPU architecture
- build flags and relevant environment variables
- exact SQL needed to reproduce the issue
- `EXPLAIN (ANALYZE, BUFFERS)` output when query behavior is involved
- output from `turbohybrid_last_scan_stats()` when available

## Performance Reports

Performance reports are most useful when they include:

- dataset size and source
- vector dimensions and embedding model
- query count
- index definition and reloptions
- `turbohybrid.profile`
- explicit `dense_k`, `bm25_k`, `rrf_k`, and `final_k` values, if used
- benchmark command
- baseline definition
- p50, p95, and p99 latency
- relevance metrics such as nDCG@10, MRR@10, Recall@10, or overlap@10. nDCG is
  normalized discounted cumulative gain, and MRR is mean reciprocal rank.
- `turbohybrid_last_scan_stats()` JSON

Generated benchmark artifacts should be attached to issues, releases, or CI
runs. Do not add full benchmark outputs to the repository.

## Resource Limits

The public `turbohybrid.*` candidate-budget and BM25 cache settings are capped
on purpose in this alpha. The caps protect shared PostgreSQL servers from
accidental or unprivileged resource exhaustion while the extension is still
maturing. If a legitimate workload needs higher limits, include the exact
settings, `work_mem`, dataset size, and benchmark command in the report.

## Supported Versions

The current public compatibility target is:

- PostgreSQL 14 through 19 (CI builds and tests every version; PostgreSQL 19 is
  tested against pgvector `master`)
- pgvector 0.8.2 and newer compatible 0.8.x releases
- pgvector `master` as a forward-compatibility signal

See [docs/compatibility.md](docs/compatibility.md) for the current matrix.
