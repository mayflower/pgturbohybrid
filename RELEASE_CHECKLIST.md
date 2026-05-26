# Release Checklist: v0.1.0-alpha.2

This file helps release reviewers confirm that `pgturbohybrid` is ready for the
`v0.1.0-alpha.2` public alpha tag.

Checked items have been verified by repository inspection or static checks in
this release-prep branch. Unchecked items are still release gates and include a
known alpha limitation note.

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
