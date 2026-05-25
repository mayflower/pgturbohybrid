# Internal Readiness Checklist

This internal checklist records release-readiness criteria for the standalone
`pgturbohybrid` extension. Public release notes should use `CHANGELOG.md`,
`RELEASE.md`, and `docs/release-notes/`.

## Standalone Extension Boundary

- [x] repository is no longer a pgvector fork structurally
- [x] extension name is `pgturbohybrid`
- [x] shared library is `pgturbohybrid`
- [x] control file is `pgturbohybrid.control`
- [x] SQL install script is `sql/pgturbohybrid--0.1.0.sql`
- [x] control file has `requires = 'vector'`
- [x] no pgvector control or SQL files are modified
- [x] no pgvector types are created
- [x] no pgvector operators or opclasses are replaced
- [x] public SQL objects are owned by `pgturbohybrid` and exposed under the
  `turbohybrid` feature surface
- [x] GUCs use `turbohybrid.*`
- [x] Unix and Windows build files are scoped to `pgturbohybrid`
- [x] CI installs unmodified pgvector first

## Release-Facing Documentation

- [x] `README.md` is standalone and does not claim to be official pgvector
- [x] compatibility with pgvector versions is documented
- [x] license and pgvector attribution are documented
- [x] generated benchmark results are excluded from the source tree
- [x] benchmark methodology is documented under `benchmarks/` and
  `docs/benchmarks/`
- [x] release notes for `v0.1.0-alpha.2` are under `docs/release-notes/`

## Validation Gates

- [x] SQL regression tests pass in CI
- [x] TAP tests pass in CI
- [x] restart/WAL tests pass in CI
- [x] Windows build passes in CI
- [x] macOS matrix passes in CI
- [x] Linux matrix passes in CI
- [x] valgrind/UBSan passes in CI
- [x] perf-smoke coverage checks the default fast path
- [ ] local TAP tests require PostgreSQL TAP Perl modules to be installed

## Remaining Risks

- The on-disk format remains alpha and may require `REINDEX` across future
  incompatible releases.
- `pgturbohybrid` uses a private compatibility copy of pgvector's `Vector`
  layout, so pgvector ABI drift remains the primary compatibility risk.
- Hybrid queries with a text query are intended for indexed
  `ORDER BY ... LIMIT` retrieval; text-aware scalar fallback is intentionally
  limited.
- Source archives should be created from a clean committed tree and should not
  include generated benchmark artifacts, build outputs, regression logs, or
  local dependency checkouts.
