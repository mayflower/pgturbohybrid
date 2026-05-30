# pgvector Dependency Audit

This audit records the current standalone `pgturbohybrid` boundary after
converting the cleaned TurboHybrid work out of the pgvector source tree.

## Summary

`pgturbohybrid` is now structured as a standalone PostgreSQL extension that
depends on an installed pgvector extension. It does not build `vector.so`, does
not install pgvector SQL scripts, and does not define pgvector-owned SQL types,
operators, access methods, aggregates, or opclasses.

The extension keeps renamed implementation code derived from pgvector's HNSW
access method because the pgturbohybrid graph index still uses that storage,
build, scan, insert, and vacuum scaffolding. The retained code is compiled only
into `pgturbohybrid` and is namespaced as pgturbohybrid-owned source.

## Removed pgvector-owned Files

The current filesystem no longer contains copied source files with these
pgvector-owned names:

- `src/vector.c`
- `src/hnsw.c`
- `src/hnsw.h`
- `src/ivfflat.c`
- `src/halfvec.c`
- `src/sparsevec.c`
- `src/bitvec.c`

The standalone tree also removes pgvector packaging files and install scripts:

- `vector.control`
- `sql/vector.sql`
- `sql/vector--*.sql`
- pgvector type, IVFFlat, halfvec, sparsevec, bit, and upstream pgvector test
  files that are not part of the standalone pgturbohybrid extension

## Remaining pgvector-derived Code

The remaining pgvector-derived code is retained under pgturbohybrid-owned file
names and is built as part of `pgturbohybrid` only:

- `src/pgturbohybrid_graph.c`
- `src/pgturbohybrid_graph_utils.c`
- `src/pgturbohybrid_build.c`
- `src/pgturbohybrid_insert.c`
- `src/pgturbohybrid_scan.c`
- `src/pgturbohybrid_vacuum.c`
- graph-related structures in `src/pgturbohybrid.h`

This code remains because pgturbohybrid currently implements its own graph
access method and needs local ownership of page layout, graph traversal, index
build, insert, scan, and vacuum behavior. It must not call or link against
private pgvector C symbols.

## pgvector Type Compatibility

`pgturbohybrid` uses pgvector's SQL `vector` type through the extension
dependency declared in `pgturbohybrid.control`:

```text
requires = 'vector'
```

The local compatibility layer is:

- `src/pgturbohybrid_vector_compat.h`
- `src/pgturbohybrid_vector_compat.c`

It provides pgturbohybrid-prefixed helpers for vector layout checks and distance
calculations. It does not expose pgvector-owned function names such as
`vector_in`, `vector_out`, `l2_distance`, or `cosine_distance`.

## Removed pgvector SQL Objects

The standalone install script does not define pgvector-owned objects such as:

- `vector`, `halfvec`, `sparsevec`, or `bit` types
- `vector_in`, `vector_out`, `vector_recv`, or `vector_send`
- `ivfflathandler`
- `hnswhandler`
- pgvector IVFFlat or HNSW access methods
- pgvector operators
- pgvector aggregate functions
- pgvector operator classes

The remaining opclasses are pgturbohybrid-owned opclasses for the
`pgturbohybrid` access method over pgvector's `vector` SQL type:

- `vector_l2_turbohybrid_ops`
- `vector_ip_turbohybrid_ops`
- `vector_cosine_turbohybrid_ops`
- `bm25_tsvector_turbohybrid_ops`

These do not replace pgvector's own opclasses because they use the
`pgturbohybrid` access method and pgturbohybrid-prefixed support functions.

## Symbol Renaming

SQL-visible and exported PostgreSQL C functions are renamed to the
`pgturbohybrid_*` namespace. Examples include:

- `pgturbohybrid_handler`
- `turbohybrid_query`
- `turbohybrid_query_in`
- `turbohybrid_query_out`
- `turbohybrid_last_scan_stats`
- `turbohybrid_index_stats`
- `turbohybrid_simd_capabilities`

Renamed graph and quantization code uses pgturbohybrid-owned file names and
project-prefixed SQL-visible symbols. Some remaining internal quantization
helper names and private struct fields still carry historical short names, but
they are not pgvector SQL objects and are not exported as pgvector-owned dynamic
symbols. They are covered by the later final rename sweep.

## Licensing And Attribution

This repository uses the PostgreSQL License in `LICENSE`. Code derived from
pgvector remains under that license, with attribution preserved in `NOTICE`.
Documentation and release materials should continue to state that pgturbohybrid
depends on pgvector and contains code derived from pgvector's HNSW
implementation.

## Compatibility Testing Plan

CI exercises pgvector compatibility with a matrix that installs pgvector first,
then builds and installs `pgturbohybrid` with PGXS. The matrix
(`.github/workflows/build.yml`) covers PostgreSQL 14–19 against both pgvector
`v0.8.2` and `master` (PostgreSQL 19 against `master` only), plus the macOS,
linux/i386, valgrind, and Windows jobs. The steps below describe one matrix
cell:

1. Install supported PostgreSQL and pgvector versions.
2. Build `pgturbohybrid` with `make clean && make`.
3. Run `make install`.
4. In a clean database, run:

   ```sql
   CREATE EXTENSION vector;
   CREATE EXTENSION pgturbohybrid;
   ```

5. Verify `DROP EXTENSION pgturbohybrid` leaves `vector` installed and usable.
6. Run regression and TAP tests that cover index creation, scan behavior,
   insert/update/delete, vacuum, and dependency handling.
7. Repeat across the documented pgvector compatibility range.

The compatibility target is pgvector `0.8.2` through current `master`, which is
the range CI exercises.
