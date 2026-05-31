# Migration Notes

These internal notes summarize the conversion from an experimental
TurboHybrid-in-pgvector tree into the standalone `pgturbohybrid` PostgreSQL
extension layout.

## Standalone Package Shape

The standalone repository keeps:

- `LICENSE`
- `.editorconfig`
- `.github/workflows/build.yml`
- `README.md`
- `docs/architecture.md`
- `Makefile`
- `Makefile.win`
- `pgturbohybrid.control`
- `sql/pgturbohybrid--0.1.0.sql`
- `sql/uninstall-dev.sql`
- `benchmarks/README.md`
- `benchmarks/config/*.json`
- `benchmarks/examples/*.trec`
- `benchmarks/sql/*.sql`
- focused extension and query tests under `test/sql`, `test/expected`, and
  `test/t`

Generated benchmark result files are not part of the source tree.

## Renamed Implementation Files

Core implementation files retained from the earlier prototype were renamed into
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

Test files were also renamed from prototype names to `pgturbohybrid*` names.

## Dense-Only Index Shape

TurboHybrid now supports one-key dense-only indexes:

```sql
CREATE INDEX documents_dense_idx ON documents
USING turbohybrid (embedding vector_cosine_turbohybrid_ops);
```

Existing two-key indexes remain valid:

```sql
CREATE INDEX documents_hybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);
```

Dense-only users can create smaller one-key indexes and avoid BM25 metadata.
Queries with `text_query` require the two-key hybrid shape. Changing an existing
index from hybrid to dense-only, or from dense-only to hybrid, requires
rebuilding the index with the desired key list; use `DROP INDEX` / `CREATE
INDEX`, or `REINDEX` after the definition has been changed by the migration
procedure.

## Removed pgvector-Owned Files

The conversion removed pgvector-owned packaging and source files:

- `vector.control`
- `sql/vector.sql`
- `sql/vector--*.sql` install and upgrade scripts
- pgvector type implementations such as `src/vector.c`, `src/halfvec.*`,
  `src/sparsevec.*`, `src/bitvec.*`, `src/bitutils.*`, and `src/halfutils.*`
- pgvector IVFFlat implementation files
- pgvector README, changelog, Dockerfile, and PGXN metadata content
- pgvector regression tests outside the standalone pgturbohybrid surface
- generated build outputs, regression clusters, and benchmark result files

## Remaining pgvector-Derived Code

`pgturbohybrid` still contains code derived from pgvector's HNSW access method.
It is retained under `src/pgturbohybrid*` names because the extension owns its
graph page layout, traversal, build, insert, scan, and vacuum behavior.

This is acceptable only with preserved license and attribution. The project
does not patch pgvector upstream and must not claim to be an official pgvector
project.

## Remaining Dependency Risks

- `pgturbohybrid_vector_compat.h` encodes the pgvector `Vector` varlena layout.
- The install script requires pgvector 0.8.2 or newer.
- Runtime checks validate dimensions and extension presence, but pgvector ABI
  drift remains the main dependency risk.
- Windows build parity is represented in `Makefile.win` and CI.
- pgvector-derived graph/index code remains under pgturbohybrid source names
  with preserved license and attribution.
