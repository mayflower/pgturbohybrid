# Security Code Audit

This file tracks release-hardening work for `v0.1.0-alpha.2`.

## Scope

The audit covers release-blocking correctness and security risks identified
before the public alpha tag. The focus is PostgreSQL extension behavior that can
affect shared-database safety, backend stability, index correctness, and release
evidence.

## Findings

| Finding | Severity | Status |
| --- | --- | --- |
| Stateful diagnostics and planner-aware hybrid distance functions are declared too parallel-safe | Release blocker | Fixed |
| Public USERSET candidate/cache GUCs allow extreme resource use | Release blocker | Fixed |
| Graph scan page locks need local error cleanup | Release blocker | Fixed |
| `turbohybrid_query(...)` constructor caching can hide changed GUC defaults | Release blocker | Fixed |
| BM25 dynamic allocation and metadata-derived allocation paths need overflow/corruption guards | Release blocker | Fixed |
| Release check, source archive verification, and CI gates need final evidence | Release blocker | Fixed for branch |
| Executor hook ownership and cleanup need a deliberate audit pass | High priority | Fixed for alpha |
| `turbohybrid_query` varlena validation should reject trailing bytes and overflowed sizes | High priority | Fixed |
| Global fast-math compiler flags should become opt-in | Medium priority | Fixed |
| Parallel and concurrent build behavior needs proof or explicit restriction | Medium priority | Parallel build disabled for alpha; concurrent build tested |
| Insert/delta node-ID and crash behavior needs stronger evidence | Medium priority | Fixed for alpha |
| BM25 large-tsquery bitmask handling needs a defined limit or dynamic bitmap | Medium priority | Fixed with 64-term cap |
| Diagnostics JSON construction should move away from manual string assembly | Alpha accepted risk | Fixed |
| On-disk metadata/page readers (graph metapage, sparse node-map, sparse postings) need defensive corruption guards and crash-resistance evidence | High priority | Fixed |
| User-controlled varlena/array/query inputs (type text input, sparse/multivector constructors, query-vector payloads) need negative-test coverage proving clean errors and continued session usability | High priority | Fixed |

## Severity

- **Release blocker**: must be fixed or explicitly proven safe before tagging.
- **High priority**: should be fixed before tag unless evidence shows it is not
  on a release-facing path.
- **Medium priority**: should be bounded, documented, or scheduled with a clear
  alpha limitation.
- **Alpha accepted risk**: acceptable only with documented rationale and tests
  for the stable public behavior.

## Files Touched

This section is updated as fixes land.

- `SECURITY_CODE_AUDIT.md`
- `.github/workflows/release.yml`
- `.github/workflows/hardening.yml`
- `CONTRIBUTING.md`
- `Makefile`
- `Makefile.win`
- `README.md`
- `SUPPORT.md`
- `docs/architecture.md`
- `sql/pgturbohybrid--0.1.0.sql`
- `src/pgturbohybrid_am.c`
- `src/pgturbohybrid_am.h`
- `src/pgturbohybrid_bm25_query.c`
- `src/pgturbohybrid.c`
- `src/pgturbohybrid_quant.h`
- `src/pgturbohybrid_quant_insert.c`
- `src/pgturbohybrid_query.c`
- `src/pgturbohybrid_scan.c`
- `src/pgturbohybrid_quant.c`
- `src/pgturbohybrid_bm25_build.c`
- `src/pgturbohybrid_sparse_query.c`
- `test/t/004_metadata_corruption.pl`
- `test/sql/pgturbohybrid_fuzz.sql`
- `test/expected/pgturbohybrid_fuzz.out`
- `test/sql/pgturbohybrid_query.sql`
- `test/expected/pgturbohybrid_query.out`
- `test/sql/security.sql`
- `test/expected/security.out`

## Tests Added

This section is updated as tests land.

- SQL regression checks for GUC caps.
- SQL regression checks for parallel-safety catalog markings.
- Prepared-statement regression for `turbohybrid_query(...)` constructor cache
  invalidation after GUC changes.
- Security regression for the 64-term BM25 query cap and continued query
  usability after a handled error.
- Security regression for diagnostics JSON shape.
- Security regression for inserted-row hybrid visibility after build.
- Security regression for `CREATE INDEX` with parallel maintenance settings and
  `CREATE INDEX CONCURRENTLY`; the access method does not advertise parallel
  build for alpha.
- Existing `installcheck` suite now includes `security`.
- TAP restart tests cover build, concurrent build, insert, delete, vacuum,
  reindex, unlogged tables, immediate stop, and restart when PostgreSQL TAP
  modules are available.
- TAP metadata-corruption test (`test/t/004_metadata_corruption.pl`) builds a
  dense (graph metapage), a hybrid dense+tsvector (BM25 metadata), and a
  sparse-only (`sparse_ip_turbohybrid_ops`, node-map/postings) index, scribbles
  raw bytes over the relevant on-disk page(s), restarts, and asserts that a
  subsequent query and `turbohybrid_index_stats(...)` either raise a clean error
  or complete without crashing -- and that a fresh `SELECT 1` still succeeds
  after every case, proving no backend crash/PANIC. It exercises both the
  deterministic metapage-version error path and the dispatch-returns-false path
  (clobbered magic), available when PostgreSQL TAP modules are present.
- SQL fuzz/negative-input regression (`test/sql/pgturbohybrid_fuzz.sql`) feeds
  malformed input to every user-reachable parser/constructor -- text input to
  the opaque types (`turbohybrid_query`, `turbohybrid_sparse_vector`,
  `turbohybrid_multivector`), length-mismatched and non-finite (NaN/Inf) sparse
  arrays, bad jsonb build options, empty / non-divisible / zero / negative
  multivector dimensions, out-of-range multivector context offsets, and
  NaN/Inf/wrong-dimension query vectors driven through an index scan -- and
  asserts each raises a clean PostgreSQL ERROR (no crash) and that the session
  remains usable afterwards.
- Manual/nightly `hardening` workflow covers strict math with SIMD disabled,
  gcc, and clang static analysis.

## Remaining Accepted Alpha Risks

- Executor hook installation now has one owner. The unified hook manager records
  planned statements and wraps graph scans after the previous executor-start hook
  has run, then clears wrapper state and planned-statement stack entries on
  executor end and abort paths.
- On-disk corruption hardening now covers the main metadata/page readers:
  - The graph metapage reader (`PgturbohybridGraphReadMeta`) validates the
    metapage format `version` after the magic + storage-kind dispatch check and
    raises `ERRCODE_DATA_CORRUPTED` with a REINDEX hint on an unknown version,
    instead of silently misreading an incompatible layout. Sub-field counts and
    bounds were already clamped.
  - The BM25 metadata readers in `pgturbohybrid_bm25_build.c` /
    `pgturbohybrid_bm25_query.c` retain their existing dense set of
    `ERRCODE_DATA_CORRUPTED` checks (metapage pointer in range, page-kind, tuple
    presence, chain bounds).
  - The sparse node-map readers (`PgturbohybridReadNodeMap` /
    `PgturbohybridReadNodeStates`) now validate each chain pointer (in range,
    never the metapage, never self-referential) before dereferencing it, and
    refuse to read a node-map tuple's TID run that exceeds the bytes the item
    line pointer actually covers -- closing wild-read / infinite-loop / OOB-read
    risks under corruption.
  - The sparse postings decoders (`PgturbohybridSparseScoreChunk`,
    `PgturbohybridWandDecodeChunk`) reject a chunk whose posting count exceeds
    the physical block-size ceiling (`PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE`), and
    WAND decode scratch buffers are sized to that ceiling rather than the
    on-disk `meta->blockSize`, so a corrupt count can no longer overrun the
    decode buffers or the fixed-size bit-unpack stack array.
  These checks are conservative: none can fire for a validly-built current-format
  index. The crash-resistance property is proven by
  `test/t/004_metadata_corruption.pl` (see Tests Added).
- Full corruption fuzzing for *every* on-disk page type (e.g. exhaustive
  byte-level fuzzing of adjacency, correction, codebook, multivector doc-map,
  block-max, and lexicon pages, and randomized field mutation across all tuple
  types) remains a follow-up. The current coverage targets the metadata/anchor
  readers (graph metapage, BM25 metadata, sparse node-map and postings) that
  gate every scan. This branch also adds cache metadata caps and overflow checks
  on the release-facing BM25 cache/query paths.
- User-input fuzzing now covers every SQL-reachable parser/constructor (see the
  fuzz regression in Tests Added). Two input surfaces remain follow-ups: the
  binary `RECEIVE` path of `turbohybrid_multivector` with adversarial `bytea`
  (not directly constructible from plain SQL), and large-array stress at the
  documented `multivector_max_*` / sparse caps under memory pressure. Both are
  bounded by existing size validation; exhaustive fuzzing is scheduled.
- Public diagnostics now use PostgreSQL JSONB builder APIs instead of
  hand-assembled JSON text. Developer-only diagnostics remain behind
  `PGTURBOHYBRID_DEV_DIAGNOSTICS`.
- TAP restart tests are present but skipped on this local machine because the
  PostgreSQL TAP modules are unavailable in the installed PGXS tree.
- The new manual/nightly `hardening` workflow is present on this branch, but
  GitHub cannot manually dispatch workflows that are not yet known on the
  default branch. The release build matrix is green on this branch; run the
  hardening workflow after the workflow file lands on the default branch.
