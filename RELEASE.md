# Release Policy

`pgturbohybrid` versions are independent from pgvector versions.

## Versioning

The **extension SQL version** (in `pgturbohybrid.control` / the install script
name) is independent from the **Git release tag**. The SQL version may stay
`0.1.0` across several `v0.1.0-alpha.N` tags: an alpha tag can document changed
defaults and diagnostics without adding an extension upgrade path.

The planned beta freezes only the stable public core; see
[docs/beta-scope.md](docs/beta-scope.md) and [ROADMAP.md](ROADMAP.md).

Use semantic versioning for the SQL version after `0.1.0`:

- Patch releases fix bugs without changing the SQL API or on-disk index format.
- Minor releases may add SQL objects, options, or index features.
- Major releases may break the SQL API or require a new on-disk format.

## Current Release

- Extension SQL version: **`0.2.0`**
  - Control file: `pgturbohybrid.control`
  - Core install script: `sql/pgturbohybrid--0.2.0.sql`
  - Experimental install script: `sql/pgturbohybrid_experimental--0.2.0.sql`
  - Alpha migration guard: `sql/pgturbohybrid--0.1.2--0.2.0.sql`
  - Shared library: `pgturbohybrid`
- Latest published Git tag: **none yet** — the first planned tag is
  `v0.1.0-alpha.2`. The Git tag is independent of the SQL version.

`0.2.0` separates the stable core SQL surface from
`pgturbohybrid_experimental`, adds the read-only validator, hardens shared
cache publication and VACUUM/recovery behavior, and adds recovery, controller,
and soak coverage. The pre-split alpha upgrade fails transactionally with exact
reinstall guidance instead of leaving partly migrated extension ownership.

> Keep this section current: after publishing a tag, the
> [post-release checklist](#post-release-checklist) updates "Latest published
> Git tag" here. This is the single place that names the published release, so a
> new tag does not require editing hardcoded tag names elsewhere.

## Tagging Procedure

Releases are cut from a clean, committed tree. The procedure is generic — set
`TAG` to the tag you are releasing; nothing below hardcodes a specific tag:

```sh
TAG=v0.1.0-alpha.2          # the release tag

# 1. GitHub release notes for this tag must exist (release.yml requires them):
#       docs/release-notes/github-$TAG.md   (GitHub Release body)
#    and, by convention, an in-repo summary docs/release-notes/$TAG.md
# 2. Validate the tree (build, installcheck, TAP, dist, version/parity checks):
scripts/release-check.sh
# 3. Tag and push:
git tag -a "$TAG" -m "pgturbohybrid $TAG"
git push origin "$TAG"
```

The tag push triggers `.github/workflows/release.yml`, which derives the
release-notes path from the tag name (`docs/release-notes/github-$TAG.md`) and
the artifact version from `Makefile` `EXTVERSION` — neither is hardcoded —
rebuilds against upstream pgvector, runs SQL and TAP checks and `make dist`, and
uploads the tracked-source archives to the GitHub Release.

## Post-Release Checklist

Immediately after a tag is published, in one follow-up commit:

1. Update [Current Release](#current-release) above: set "Latest published Git
   tag" to the tag just released.
2. Update `CHANGELOG.md` with the release entry.
3. Confirm `scripts/check-version-consistency.sh` still passes (it requires a
   `docs/release-notes/github-<tag>.md` file when `GITHUB_REF_NAME` is set).
4. Confirm the GitHub Release was created by `release.yml` and has the expected
   archives attached.

## PostgreSQL And pgvector Support

The release target is PostgreSQL 14 through 19 (CI builds and runs the
regression tests on every version; PostgreSQL 19 is tested against pgvector
`master`, which has no tagged 0.8.2 build). The pgvector compatibility target is
pgvector 0.8.2 through current pgvector `master`.

The build and CI matrix must install unmodified pgvector before building
`pgturbohybrid`. See `docs/compatibility.md` for the public compatibility matrix
(and the Windows reduced-coverage build profile).

## Upgrade Script Policy

Upgrade scripts must be owned by this extension and named with the PostgreSQL
extension upgrade pattern. The current upgrade chain is:

```text
sql/pgturbohybrid--0.1.0--0.1.1.sql
sql/pgturbohybrid--0.1.1--0.1.2.sql
```

Each upgrade script should ship with a regression test that creates the older
version and runs `ALTER EXTENSION pgturbohybrid UPDATE`. Do not add or edit
pgvector upgrade scripts, and do not use `vector--*.sql` names.

## On-Disk Compatibility Policy

The index metapage stores pgturbohybrid-specific magic, page id, and format
version values. If a release changes the on-disk index format incompatibly, the
release notes must say that `REINDEX` is required, and readers must reject an
unknown format version with a clear `ERRCODE_DATA_CORRUPTED` error rather than
misreading it.

`REINDEX` is required when:

- the page layout changes,
- quantized code encoding changes,
- graph tuple layout changes,
- BM25 postings or lexicon layout changes,
- the sparse postings / node-map layout changes,
- compatibility with existing index pages cannot be proven.

`REINDEX` is not required for SQL-only diagnostic changes that do not affect
stored index pages. While the extension is alpha, expect a `REINDEX` across
alpha tags; see [docs/operations.md](docs/operations.md). The full format
contract (constants, page kinds, rejection rules) is in
[docs/storage-format.md](docs/storage-format.md).
