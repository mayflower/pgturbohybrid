# Release Policy

`pgturbohybrid` versions are independent from pgvector versions.

## Current Version

The current extension SQL version is `0.1.0`.

- Control file: `pgturbohybrid.control`
- Install script: `sql/pgturbohybrid--0.1.0.sql`
- Shared library: `pgturbohybrid`

There is no upgrade script for `0.1.0` because it is the first installable
version.

The next public alpha release tag is `v0.1.0-alpha.2`. This tag keeps the
extension SQL version at `0.1.0`; it documents changed defaults and diagnostics
without adding an extension upgrade path.

Release notes for this alpha live in
`docs/release-notes/v0.1.0-alpha.2.md`.

## Release Command

Run releases from a clean committed tree:

```sh
scripts/release-check.sh
git tag -a v0.1.0-alpha.2 -m "pgturbohybrid v0.1.0-alpha.2"
git push origin v0.1.0-alpha.2
```

The tag push triggers `.github/workflows/release.yml`, which rebuilds against
upstream pgvector, runs SQL and TAP checks, runs `make dist`, and uploads the
tracked-source archives to the GitHub Release.

## Semantic Versioning

Use semantic versioning after `0.1.0`:

- Patch releases fix bugs without changing SQL API or on-disk index format.
- Minor releases may add SQL objects, options, or index features.
- Major releases may break SQL API or require a new on-disk format.

## PostgreSQL And pgvector Support

The release target is PostgreSQL 14 through 19 (CI builds and runs the
regression tests on every version; PostgreSQL 19 is tested against pgvector
`master`, which has no tagged 0.8.2 build). The pgvector compatibility target is
pgvector 0.8.2 through current pgvector `master`.

The build and CI matrix must install unmodified pgvector before building
`pgturbohybrid`.

See `docs/compatibility.md` for the public compatibility matrix.

## Upgrade Script Policy

Future upgrade scripts must be owned by this extension and named with the
PostgreSQL extension upgrade pattern:

```text
sql/pgturbohybrid--0.1.0--0.1.1.sql
```

Do not add or edit pgvector upgrade scripts. Do not use `vector--*.sql` names.

## On-Disk Compatibility

The index metapage stores pgturbohybrid-specific magic, page id, and format
version values. If a release changes the on-disk index format incompatibly, the
release notes must say that `REINDEX` is required.

`REINDEX` is required when:

- the page layout changes,
- quantized code encoding changes,
- graph tuple layout changes,
- BM25 postings or lexicon layout changes,
- compatibility with existing index pages cannot be proven.

`REINDEX` is not required for SQL-only diagnostic changes that do not affect
stored index pages.
