# pgturbohybrid Standalone Extension Architecture

This document describes the target design for converting the cleaned
pgturbohybrid work into a standalone PostgreSQL extension. The goal is a package
that can be built and installed beside pgvector, without patching pgvector or
claiming pgvector-owned release metadata.

## Target Package Identity

- Extension name: `pgturbohybrid`
- Shared library name: `pgturbohybrid`
- Control file: `pgturbohybrid.control`
- Initial SQL install script: `sql/pgturbohybrid--0.1.0.sql`
- Extension dependency: `requires = 'vector'`
- Build model: PGXS build against an already-installed PostgreSQL and pgvector

The extension must install as:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
```

## Repository Boundary

The repository must not modify pgvector source files. pgvector is an external
dependency, not a patch target.

The standalone extension must not create or replace pgvector-owned SQL objects,
including:

- types: `vector`, `halfvec`, `sparsevec`, `bit`
- existing pgvector operators
- existing pgvector operator classes
- existing pgvector access methods or support functions

Any copied or derived implementation that remains in this repository must be
renamed, scoped, and built as `pgturbohybrid` code. The build must not compile
pgvector source files into the `pgturbohybrid` shared library.

## Dependency Model

`pgturbohybrid` uses pgvector's SQL type `vector` as its dense embedding type.
It should treat pgvector as a SQL and ABI dependency, not as a private C library.

The preferred model is:

- require the pgvector extension at SQL install time with `requires = 'vector'`
- reference pgvector's SQL type `vector` in SQL declarations
- use pgvector's installed header only for the `Vector` varlena layout, if that
  header is available in the target pgvector installation
- do not link against non-public pgvector C symbols
- do not call pgvector-internal distance functions, index functions, or support
  routines from C
- reimplement required vector distance helpers inside `pgturbohybrid` with
  `pgturbohybrid_*` C names

If a pgvector header is used, the supported pgvector versions and ABI risk must
be documented in the release notes and build documentation. The initial support
target should be a narrow pgvector version range, such as pgvector `0.8.x`, until
CI proves a wider range. The build should include compile-time checks for the
expected `Vector` layout where possible. If the header is unavailable, any local
compatibility struct must be explicitly gated and documented as ABI-sensitive.

SQL-level pgvector behavior, such as the existence of the `vector` type, is
safe to depend on through `CREATE EXTENSION vector`. Private pgvector C symbols
are not part of the contract and must not be used.

## SQL Object Naming

All new SQL objects must be prefixed or schema-scoped to avoid collisions with
pgvector and other extensions.

The package remains `pgturbohybrid`, while the SQL feature surface uses the
shorter `turbohybrid` name:

- access method: `turbohybrid`
- query constructor: `turbohybrid_query(...)`
- SQL functions: `turbohybrid_*`
- hybrid query operators: `<~>` for cosine, `<~->` for L2, and `<~#>` for
  inner product
- operator classes: `vector_l2_turbohybrid_ops`,
  `vector_ip_turbohybrid_ops`, `vector_cosine_turbohybrid_ops`
- optional text/BM25 operator classes, if retained in this extension:
  `bm25_tsvector_turbohybrid_ops`

The extension must not create generic names such as `hybrid_query` or opclasses
whose names could reasonably be mistaken for pgvector-owned objects.

If a dedicated schema is introduced later, SQL objects may instead be scoped in
that schema, but the extension should still keep externally visible names
unambiguous.

## C Symbol Naming

All PostgreSQL-visible C functions must be renamed away from pgvector, HNSW, and
prototype naming.

Required naming rules:

- `PG_FUNCTION_INFO_V1` functions use the `pgturbohybrid_*` prefix
- access method handlers use names such as `pgturbohybrid_handler`
- support functions use names such as `pgturbohybrid_vector_l2_support`
- no exported or SQL-visible C function may use `vector_*`, `hnsw_*`, `tq_*`, or
  generic `hybrid_*` names
- private `static` helpers may use shorter local names, but should avoid names
  that imply ownership by pgvector

This keeps the dynamic symbol table and SQL declarations clearly separated from
pgvector.

## GUCs and Reloptions

All GUCs must use the `turbohybrid.*` prefix. The extension must not create
`pgturbohybrid.*`, `hybrid.*`, or `hnsw.tq_*` GUCs.

Only stable user-facing settings should be exposed as GUCs. The initial public
set is:

- `turbohybrid.default_dense_k`
- `turbohybrid.default_bm25_k`
- `turbohybrid.default_rrf_k`
- `turbohybrid.enable_wand`
- `turbohybrid.max_union_candidates`
- `turbohybrid.simd`

Internal tuning, debug switches, benchmark controls, SIMD forcing, cache knobs,
and temporary experiments should be constants or private implementation choices
unless there is a clear user story and test coverage.

Reloptions are scoped to the `turbohybrid` index access method, but should
still use stable descriptive names. Candidate names:

- `quantization_bits`
- `exact_storage`
- `m`
- `ef_construction`
- `dense_k`
- `bm25_k`
- `rrf_k`

Prototype names such as `tq_*` should not appear in user-facing reloptions.

## Access Method Storage

The standalone index access method is named `turbohybrid` and is installed by
the SQL handler `turbohybrid_handler(internal)`, which maps to the C symbol
`pgturbohybrid_handler`. Its on-disk identity is owned by this extension and
must not collide with pgvector access methods:

- `PGTURBOHYBRID_MAGIC_NUMBER`
- `PGTURBOHYBRID_PAGE_ID`
- `PGTURBOHYBRID_VERSION`

Block 0 is the metapage. It stores the access-method identity, format version,
index dimensions, dense graph options, quantization options, entry/start block
pointers, and BM25 metadata pointers when the text branch is present. All other
pages carry the pgturbohybrid page identifier and a page-kind tag before their
format-specific payload.

Current page kinds are:

- graph tuples and graph metadata inherited from the standalone graph storage
- quantized code pages
- quantized adjacency pages
- optional exact-vector pages for final rescoring
- quantization correction pages
- BM25 metadata, document statistics, lexicon, postings, block-max, delta,
  impact, and delta-term pages

Index page changes are WAL-logged with PostgreSQL generic WAL. pgturbohybrid
does not register a custom resource manager and does not require
`shared_preload_libraries`. Crash recovery and replicas recover index changes
from generic WAL and new-page WAL records alone.

## Build Layout

The standalone PGXS build should use:

```make
EXTENSION = pgturbohybrid
MODULE_big = pgturbohybrid
DATA = sql/pgturbohybrid--0.1.0.sql
PG_CONFIG ?= pg_config
```

The build may use `PG_CPPFLAGS` to include pgvector's installed server header
directory when available, but it must not compile pgvector `.c` files or depend
on pgvector private object files.

Source files should be renamed or wrapped so their ownership is clear, for
example:

- `src/pgturbohybrid.c`
- `src/pgturbohybrid_index.c`
- `src/pgturbohybrid_scan.c`
- `src/pgturbohybrid_quant.c`
- `src/pgturbohybrid_bm25.c`, if BM25 remains part of this extension
- `src/pgturbohybrid_distance.c`

Windows and Unix builds must compile the same required source objects and expose
the same SQL install contract.

## Install SQL Contract

`sql/pgturbohybrid--0.1.0.sql` is the first install script for this extension.
It may reference pgvector's `vector` type because `pgturbohybrid.control`
requires `vector`, but it must not create that type.

The script intentionally uses the unqualified `vector` type. PostgreSQL makes
required extension schemas available during `CREATE EXTENSION`, and regression
tests cover `vector` and `pgturbohybrid` installed in different schemas.

The install script may create:

- `pgturbohybrid` access method objects
- prefixed support functions
- prefixed operator classes for the new access method
- prefixed query constructors and distance helpers
- minimal diagnostics, if they are intended as stable public API

It must not edit or replace pgvector install scripts. Future upgrades belong in
scripts named like:

```text
sql/pgturbohybrid--0.1.0--0.1.1.sql
```

They must not use pgvector upgrade script names.

## Migration Plan From The Cleaned Tree

1. Add `pgturbohybrid.control` with `requires = 'vector'`.
2. Add `sql/pgturbohybrid--0.1.0.sql`.
3. Change the PGXS package identity from `vector` to `pgturbohybrid`.
4. Remove pgvector source files from the `pgturbohybrid` build.
5. Rename SQL-visible functions, C symbols, access method objects, operator
   classes, and GUCs to the `pgturbohybrid` namespace.
6. Replace any pgvector private C symbol usage with internal
   `pgturbohybrid_*` helpers.
7. Keep only tests that install `vector`, install `pgturbohybrid`, and validate
   standalone behavior.
8. Keep benchmark scripts and reproducibility notes only; do not commit
   generated benchmark result artifacts.

## Testing And CI

The minimum test matrix should validate:

- PGXS build against an installed PostgreSQL and installed pgvector
- `CREATE EXTENSION vector`
- `CREATE EXTENSION pgturbohybrid`
- `DROP EXTENSION pgturbohybrid` without dropping pgvector-owned objects
- no creation of `vector`, `halfvec`, `sparsevec`, or `bit`
- no replacement of pgvector operators or opclasses
- all public GUCs use the `turbohybrid.*` prefix
- index creation and scans using `USING turbohybrid`
- insert, update, delete, vacuum, and restart behavior for the new access method
- supported pgvector versions, initially focused on the documented compatibility
  range

Regression tests should validate user-visible behavior, not internal counters or
benchmark-only diagnostics.

## Benchmarks

Benchmark scripts and reproducibility configuration may live in the repository
when they are deterministic and useful for users. Generated benchmark JSON, MD,
CSV, logs, host-specific outputs, and result directories must not be vendored.

Benchmark claims should live in a reproducible appendix such as `BENCHMARKS.md`,
including hardware, OS, PostgreSQL version, pgvector version, dataset, commands,
warmup policy, run count, latency, build time, index size, WAL generated, and
quality metrics.

## Non-goals

- no patching pgvector
- no pgvector release version bump
- no changes to `vector.control`
- no changes to pgvector SQL scripts
- no upstream pgvector README changes
- no vendored benchmark result artifacts
