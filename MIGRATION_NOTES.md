# Migration Notes

This branch was converted from the cleaned prompt-12 TurboHybrid pgvector fork
into a standalone PostgreSQL extension layout for `pgturbohybrid`.

## Copied Files

The standalone repository keeps:

- `LICENSE`
- `.editorconfig`
- `.github/workflows/build.yml`
- `README.md`
- `ARCHITECTURE.md`
- `Makefile`
- `Makefile.win`
- `pgturbohybrid.control`
- `sql/pgturbohybrid--0.1.0.sql`
- `sql/uninstall-dev.sql`
- `benchmarks/README.md`
- `benchmarks/suite.py`
- `benchmarks/config/*.json`
- `benchmarks/examples/*.trec`
- `benchmarks/sql/*.sql`
- focused extension and query tests under `test/sql`, `test/expected`, and
  `test/t`

No generated benchmark result files were copied.

## Renamed Files

Core implementation files retained from the cleaned branch were renamed into
the `pgturbohybrid` namespace:

- `src/tqgraphcontrol.c` -> `src/pgturbohybrid.c`
- `src/tqhybrid.c` -> `src/pgturbohybrid_am.c`
- `src/tqhybrid.h` -> `src/pgturbohybrid_am.h`
- `src/hybrid_query.c` -> `src/pgturbohybrid_query.c`
- `src/hybrid_query.h` -> `src/pgturbohybrid_query.h`
- `src/hnsw.h` -> `src/pgturbohybrid.h`
- `src/hnsw.c` -> `src/pgturbohybrid_graph.c`
- `src/hnswbuild.c` -> `src/pgturbohybrid_build.c`
- `src/hnswinsert.c` -> `src/pgturbohybrid_insert.c`
- `src/hnswscan.c` -> `src/pgturbohybrid_scan.c`
- `src/hnswvacuum.c` -> `src/pgturbohybrid_vacuum.c`
- `src/hnswutils.c` -> `src/pgturbohybrid_graph_utils.c`
- `src/tqgraph.c` -> `src/pgturbohybrid_quant.c`
- `src/tqgraph.h` -> `src/pgturbohybrid_quant.h`
- `src/tqgraph_cache.c` -> `src/pgturbohybrid_quant_cache.c`
- `src/tqgraph_exact.c` -> `src/pgturbohybrid_quant_exact.c`
- `src/tqgraph_insert.c` -> `src/pgturbohybrid_quant_insert.c`
- `src/tqgraph_psquare.c` -> `src/pgturbohybrid_quant_psquare.c`
- `src/tqgraph_psquare.h` -> `src/pgturbohybrid_quant_psquare.h`
- `src/tqgraph_scan_cache.c` -> `src/pgturbohybrid_quant_scan_cache.c`
- `src/tqgraph_score.c` -> `src/pgturbohybrid_quant_score.c`
- `src/tqgraph_score.h` -> `src/pgturbohybrid_quant_score.h`
- `src/tqgraph_storage.c` -> `src/pgturbohybrid_quant_storage.c`
- `src/tqhybrid_bm25.h` -> `src/pgturbohybrid_bm25.h`
- `src/tqhybrid_bm25_build.c` -> `src/pgturbohybrid_bm25_build.c`
- `src/tqhybrid_bm25_query.c` -> `src/pgturbohybrid_bm25_query.c`
- `src/tqstats.c` -> `src/pgturbohybrid_stats.c`
- `src/vector.h` -> `src/pgturbohybrid_vector_compat.h`

Test files were also renamed from `turbohybrid*`, `turboquant*`, and
`hybrid_query*` names to `pgturbohybrid*` names.

## Removed Files

The conversion removed pgvector-owned packaging and source files:

- `vector.control`
- `sql/vector.sql`
- all `sql/vector--*.sql` install and upgrade scripts
- pgvector type implementations: `src/vector.c`, `src/halfvec.*`,
  `src/sparsevec.*`, `src/bitvec.*`, `src/bitutils.*`, `src/halfutils.*`
- pgvector IVFFlat implementation: `src/ivf*.c`, `src/ivfflat.h`
- pgvector README, changelog, Dockerfile, and PGXN metadata content
- pgvector regression tests for vector, halfvec, sparsevec, bit, HNSW, and
  IVFFlat behavior
- stale dense-only TurboQuant regression files that no longer match the
  standalone hybrid access method surface
- generated build outputs, object files, temp regression clusters, and generated
  benchmark result files

## Remaining pgvector-derived Code

The current `pgturbohybrid` implementation still contains code derived from
pgvector's HNSW access method. It is retained as source under the
`src/pgturbohybrid*` names because the cleaned TurboHybrid implementation still
depends on that graph/index scaffolding.

Remaining pgvector-derived areas include:

- graph page and tuple structures
- build, insert, scan, and vacuum code paths
- HNSW-style memory structures and neighbor maintenance
- cost estimation and access method callbacks
- portions of vector layout compatibility copied from pgvector's `vector.h`

This is acceptable for the standalone fork only with preserved license and
clear attribution. It is not intended to patch pgvector upstream.

## Remaining Dependency Risks

- `pgturbohybrid_vector_compat.h` encodes the pgvector `Vector` varlena layout.
  The install script requires pgvector 0.8.2 or newer and the compatibility
  layer performs runtime layout checks, but pgvector ABI drift remains the main
  dependency risk.
- The Unix build, install, SQL regression tests, and TAP entrypoint are covered
  locally. Windows build parity is represented in `Makefile.win` and CI, but it
  still needs confirmation on a real Windows runner.
- The implementation contains pgvector-derived graph/index code under
  `pgturbohybrid` source names with preserved license and attribution.

## Final Rename Sweep Notes

The remaining matches for pgvector and historical names are intentional:

- tests and docs use `CREATE EXTENSION vector` because pgvector is the required
  dependency.
- SQL signatures use `@extschema:vector@.vector` to reference pgvector's type.
- migration notes mention removed files such as `vector.control`,
  `sql/vector.sql`, and `sql/vector--*.sql` as historical inventory.
- architecture notes mention forbidden prefixes only to document the naming
  rules.
- SIMD intrinsic names such as `vcntq_u8` contain the text `tq_` as part of ARM
  intrinsic spelling and are not pgturbohybrid API names.
- the `make dist` target stages the current tracked and untracked non-ignored
  working tree before creating release archives, so the standalone archive can
  be validated before the conversion branch is committed.
- final source and archive sweeps found no remaining `vector.control`,
  `sql/vector--*.sql`, `sql/vector.sql`, `vector.c`, `hnsw.c`, or other
  pgvector-owned source files outside historical documentation and the ignored
  upstream checkout under `.deps/`.
