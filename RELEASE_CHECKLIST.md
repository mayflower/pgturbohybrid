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
- [ ] fast setup guide works - Known alpha limitation: guide has been reviewed
  for copy-paste clarity, but the full flow still needs to be executed against a
  clean PostgreSQL installation before tagging.
- [x] how-it-works doc exists
- [x] benchmark doc is reproducible
- [x] architecture doc matches current API
- [x] compatibility doc matches CI matrix
- [x] release notes are concise

## Build And Tests

- [ ] `make clean && make` - Known alpha limitation: must be rerun on a clean
  release machine before tagging.
- [ ] `make install` - Known alpha limitation: must be rerun on a clean release
  machine before tagging.
- [ ] `make installcheck` - Known alpha limitation: must be rerun on a clean
  release machine before tagging.
- [ ] `make prove_installcheck` - Known alpha limitation: must be rerun where
  PostgreSQL TAP support is available before tagging.
- [ ] CI build workflow green - Known alpha limitation: release branch workflow
  results must be checked after pushing the final branch state.
- [ ] perf smoke workflow green - Known alpha limitation: release branch workflow
  results must be checked after pushing the final branch state.
- [ ] Windows workflow green - Known alpha limitation: release branch workflow
  results must be checked after pushing the final branch state.
- [ ] macOS workflow green - Known alpha limitation: release branch workflow
  results must be checked after pushing the final branch state.
- [ ] valgrind workflow green - Known alpha limitation: release branch workflow
  results must be checked after pushing the final branch state.

## Release

- [x] `META.json` correct
- [x] `pgturbohybrid.control` correct
- [x] `CHANGELOG.md` has `v0.1.0-alpha.2`
- [x] `RELEASE.md` has tag instructions
- [ ] `make dist` clean - Known alpha limitation: `make dist` intentionally
  requires a clean git tree, so it must run after release-prep changes are
  committed.
- [ ] source archive contains no generated artifacts - Known alpha limitation:
  archive contents must be inspected after `make dist` runs from a clean tree.
- [x] GitHub release text prepared
- [ ] benchmark artifact attached externally if published - Known alpha limitation:
  generated benchmark artifacts are intentionally not committed and must be
  attached to the release, CI run, or issue if benchmark results are published.
