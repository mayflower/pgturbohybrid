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
| Parallel and concurrent build behavior needs proof or explicit restriction | Medium priority | Fixed for alpha |
| Insert/delta node-ID and crash behavior needs stronger evidence | Medium priority | Fixed for alpha |
| BM25 large-tsquery bitmask handling needs a defined limit or dynamic bitmap | Medium priority | Fixed with 64-term cap |
| Diagnostics JSON construction should move away from manual string assembly | Alpha accepted risk | Fixed |

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
- Security regression for parallel-maintenance index build and
  `CREATE INDEX CONCURRENTLY`.
- Existing `installcheck` suite now includes `security`.
- TAP restart tests cover build, concurrent build, insert, delete, vacuum,
  reindex, unlogged tables, immediate stop, and restart when PostgreSQL TAP
  modules are available.
- Manual/nightly `hardening` workflow covers strict math with SIMD disabled,
  gcc, and clang static analysis.

## Remaining Accepted Alpha Risks

- Executor hook installation now has one owner. The unified hook manager records
  planned statements and wraps graph scans after the previous executor-start hook
  has run, then clears wrapper state and planned-statement stack entries on
  executor end and abort paths.
- Full corruption fuzzing for every on-disk BM25 and graph page type remains a
  follow-up. This branch adds cache metadata caps and overflow checks on the
  release-facing BM25 cache/query paths.
- Public diagnostics now use PostgreSQL JSONB builder APIs instead of
  hand-assembled JSON text. Developer-only diagnostics remain behind
  `PGTURBOHYBRID_DEV_DIAGNOSTICS`.
- TAP restart tests are present but skipped on this local machine because the
  PostgreSQL TAP modules are unavailable in the installed PGXS tree.
- The new manual/nightly `hardening` workflow is present on this branch, but
  GitHub cannot manually dispatch workflows that are not yet known on the
  default branch. The release build matrix is green on this branch; run the
  hardening workflow after the workflow file lands on the default branch.
