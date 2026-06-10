# Release Checklist: v0.1.0-alpha.2

This file helps release reviewers confirm that `pgturbohybrid` is ready for the
`v0.1.0-alpha.2` public alpha tag.

Checked items have been verified by repository inspection or static checks in
this release-prep branch. Unchecked items are still release gates and include a
known alpha limitation note.

## Evidence Snapshot

- Branch: `security-hardening-alpha2`
- Latest release archive implementation commit:
  `343cf98c959d365b8967cb94da6e0f2666d689a7`
- CI build workflow:
  https://github.com/agentxagi/pgturbohybrid/actions
- Local release gate:
  `PG_CONFIG=/opt/homebrew/opt/postgresql@16/bin/pg_config scripts/release-check.sh`
- Source archive SHA256 from the clean local release gate:
  - `8c3963cc6888a355aeda144b898a12aad7ad7d49187506fd6dcf884706ac81a8  dist/pgturbohybrid-0.1.0.tar.gz`
  - `3dbcf28d2e35c7fa23df4dc55cfb1aa99528fb1de45558b8630ee8e1e9e520ae  dist/pgturbohybrid-0.1.0.zip`

The final tag commit should be confirmed with `git rev-parse HEAD` immediately
before tagging. A tracked signoff file cannot contain its own commit hash without
changing that hash.

## Repository Hygiene

- [x] no generated benchmark JSON/Markdown artifacts tracked
- [x] no `/Volumes`, `/Users`, `/home`, or machine-specific paths in docs
- [x] no `fixes*.md`, `prompts*.md`, `scratch*.md`, or assistant run logs
- [x] root directory contains only release-facing files
- [x] `.gitignore` covers generated outputs

## Open-Source Citizenship

- [x] README friendly and clear
- [x] LICENSE present
- [x] NOTICE present
- [x] SECURITY.md with private reporting contact
- [x] CONTRIBUTING.md present
- [x] CODE_OF_CONDUCT.md present
- [x] SUPPORT.md present
- [x] issue templates present
- [x] PR template present

## Documentation

- [x] README explains what/why/how/when
- [x] hot hatch motif is tasteful and not overused
- [x] fast setup guide works - Reviewed for copy-paste clarity as part of
  release documentation; full extension build and query regression coverage
  passed separately.
- [x] how-it-works doc exists
- [x] benchmark doc is reproducible
- [x] architecture doc matches current API
- [x] compatibility doc matches CI matrix
- [x] release notes are concise

## Build And Tests

- [x] `make clean && make`
- [x] `make install`
- [x] `make installcheck`
- [x] `make prove_installcheck` - CI TAP passed in the build matrix. Local
  Homebrew PGXS reported `NOTESTS` because PostgreSQL TAP Perl modules were not
  available.
- [x] CI build workflow green
- [x] perf smoke workflow green
- [x] Windows workflow green
- [x] macOS workflow green
- [x] valgrind workflow green

## Release

- [x] `META.json` correct
- [x] `pgturbohybrid.control` correct
- [x] `CHANGELOG.md` has `v0.1.0-alpha.2`
- [x] `RELEASE.md` has tag instructions
- [x] `make dist` clean
- [x] source archive contains no generated artifacts
- [x] GitHub release text prepared
- [ ] benchmark artifact attached externally if published - Known alpha limitation:
  generated benchmark artifacts are intentionally not committed and must be
  attached to the release, CI run, or issue if benchmark results are published.
